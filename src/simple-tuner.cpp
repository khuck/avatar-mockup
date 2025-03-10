/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

/* https://github.com/kokkos/kokkos-tools/wiki/Profiling-Hooks
 * This page documents the interface between Kokkos and the profiling library.
 * Every function prototype on this page is an interface hook. Profiling
 * libraries may define any subset of the hooks listed here; hooks which are
 * not defined by the library will be silently ignored by Kokkos. The hooks
 * have C linkage, and we emphasize this with the extern "C" required to define
 * such symbols in C++. If the profiling library is written in C, the
 * extern "C" should be omitted.
 */

#include <sstream>
#include <unordered_map>
#include <mutex>
#include <stack>
#include <vector>
#include <set>
#include <map>
#include <iostream>
#include <fstream>
#include <random>
#include <stdlib.h>
#include <Kokkos_Core.hpp>
#include "limits.h"

#ifndef KOKKOS_ENABLE_TUNING
#error "Error! Kokkos is not configured with tuning support. Please reconfigure and rebuild Kokkos with `-DKokkos_ENABLE_TUNING=ON`."
#endif

#if defined(__GNUC__)
#define __FUNCTION__ __PRETTY_FUNCTION__
#else
#define __FUNCTION__ __func__
#endif

bool getVerbose() {
    char * tmp = getenv("KOKKOS_VERBOSE");
    if (tmp == nullptr) { return false; }
    return true;
}

class void_stream { 
public:
    std::ostream& operator()(void) {
        static bool verbose{getVerbose()};
        static std::ofstream devNull("/dev/null");
        if (verbose) {
            return std::cout;
        }
        return devNull;
    }
};

void_stream mylog;

std::string pVT(Kokkos_Tools_VariableInfo_ValueType t) {
    if (t == kokkos_value_double) {
        return std::string("double");
    }
    if (t == kokkos_value_int64) {
        return std::string("int64");
    }
    if (t == kokkos_value_string) {
        return std::string("string");
    }
    return std::string("unknown type");
}

std::string pCat(Kokkos_Tools_VariableInfo_StatisticalCategory c) {
    if (c == kokkos_value_categorical) {
        return std::string("categorical");
    }
    if (c == kokkos_value_ordinal) {
        return std::string("ordinal");
    }
    if (c == kokkos_value_interval) {
        return std::string("interval");
    }
    if (c == kokkos_value_ratio) {
        return std::string("ratio");
    }
    return std::string("unknown category");
}

std::string pCVT(Kokkos_Tools_VariableInfo_CandidateValueType t) {
    if (t == kokkos_value_set) {
        return std::string("set");
    }
    if (t == kokkos_value_range) {
        return std::string("range");
    }
    if (t == kokkos_value_unbounded) {
        return std::string("unbounded");
    }
    return std::string("unknown candidate type");
}

std::string pCan(Kokkos_Tools_VariableInfo& i) {
    std::stringstream ss;
    if (i.valueQuantity == kokkos_value_set) {
        std::string delimiter{"["};
        for (size_t index = 0 ; index < i.candidates.set.size ; index++) {
            ss << delimiter;
            if (i.type == kokkos_value_double) {
                ss << i.candidates.set.values.double_value[index];
            } else if (i.type == kokkos_value_int64) {
                ss << i.candidates.set.values.int_value[index];
            } else if (i.type == kokkos_value_string) {
                ss << i.candidates.set.values.string_value[index];
            }
            delimiter = ",";
        }
        ss << "]" << std::endl;
        std::string tmp{ss.str()};
        return tmp;
    }
    if (i.valueQuantity == kokkos_value_range) {
        ss << std::endl;
        if (i.type == kokkos_value_double) {
            ss << "    lower: " << i.candidates.range.lower.double_value << std::endl;
            ss << "    upper: " << i.candidates.range.upper.double_value << std::endl;
            ss << "    step: " << i.candidates.range.step.double_value << std::endl;
        } else if (i.type == kokkos_value_int64) {
            ss << "    lower: " << i.candidates.range.lower.int_value << std::endl;
            ss << "    upper: " << i.candidates.range.upper.int_value << std::endl;
            ss << "    step: " << i.candidates.range.step.int_value << std::endl;
        }
        ss << "    open upper: " << i.candidates.range.openUpper << std::endl;
        ss << "    open lower: " << i.candidates.range.openLower << std::endl;
        std::string tmp{ss.str()};
        return tmp;
    }
    if (i.valueQuantity == kokkos_value_unbounded) {
        return std::string("unbounded\n");
    }
    return std::string("unknown candidate values\n");
}

class Bin {
public:
    Bin(double value, size_t idx) :
        mean((double)value),
        total(value),
        min(value),
        max(value),
        count(1) {
        std::stringstream ss;
        ss << "bin_" << idx;
        name = ss.str();
    }
    double mean;
    double total;
    double min;
    double max;
    size_t count;
    std::string name;
    bool contains(double value) {
        if (value <= max && value >= min) {
            return true;
        } else if (value <= (mean * 1.25) &&
                   value >= (mean * 0.75)) {
            return true;
        }
        return false;
    }
    void add(double value) {
        count++;
        total += value;
        mean = (double)total / (double)count;
        if (value < min) min = value;
        if (value > max) max = value;
    }
    std::string getName() {
        return name;
    }
};


class Variable {
public:
    Variable(size_t _id, std::string _name, Kokkos_Tools_VariableInfo& _info, bool isOutput = true);
    void reportBest(void);
    void deepCopy(Kokkos_Tools_VariableInfo& _info);
    std::string toString() {
        std::stringstream ss;
        ss << "  hash: " << hashValue << std::endl;
        ss << "  name: " << name << std::endl;
        ss << "  id: " << id << std::endl;
        ss << "  info.type: " << pVT(info.type) << std::endl;
        ss << "  info.category: " << pCat(info.category) << std::endl;
        ss << "  info.valueQuantity: " << pCVT(info.valueQuantity) << std::endl;
        ss << "  info.candidates: " << pCan(info);
        if (info.valueQuantity == kokkos_value_unbounded) {
            ss << "  num_bins: " << bins.size() << std::endl;
            for (auto b : bins) {
                ss << "  " << b->name << ": " << std::endl;
                ss << "    min: " << std::fixed << b->min << std::endl;
                ss << "    mean: " << std::fixed << b->mean << std::endl;
                ss << "    max: " << std::fixed << b->max << std::endl;
                ss << "    count: " << std::fixed << b->count << std::endl;
            }
        }
        std::string tmp{ss.str()};
        return tmp;
    }
    size_t id;
    std::string name;
    std::string hashValue;
    Kokkos_Tools_VariableInfo info;
    std::vector<std::string> space; // enum space
    double dmin;
    double dmax;
    double dstep;
    int64_t lmin;
    int64_t lmax;
    int64_t lstep;
    int64_t lvar;
    int64_t numValues;
    void makeSpace(void);
    std::vector<Bin*> bins;
    std::string getBin(double value) {
        for (auto b : bins) {
            if (b->contains(value)) {
                b->add(value);
                return b->getName();
            }
        }
        Bin * b = new Bin(value, bins.size());
        std::string tmp{b->getName()};
        bins.push_back(b);
        return tmp;
    }
    size_t best_time;
    union Kokkos_Tools_VariableValue_ValueUnion lastValue;
    union Kokkos_Tools_VariableValue_ValueUnion bestValue;
    bool output;
    void assignNewValue(struct Kokkos_Tools_VariableValue& var) {
        // what kind is it?
        mylog() << "Setting " << name << " to ";
        if (var.metadata->type == kokkos_value_double) {
            var.value.double_value = newRandomDouble();
            lastValue.double_value = var.value.double_value;
            mylog() << var.value.double_value << std::endl;
        }
        else if (var.metadata->type == kokkos_value_int64) {
            var.value.int_value = newRandomInt();
            lastValue.int_value = var.value.int_value;
            mylog() << var.value.int_value << std::endl;
        }
        else /* if (var.metadata->type == kokkos_value_string) */ {
            strncpy(var.value.string_value, newRandomString().c_str(), KOKKOS_TOOLS_TUNING_STRING_LENGTH);
            strncpy(lastValue.string_value, var.value.string_value, KOKKOS_TOOLS_TUNING_STRING_LENGTH);
            mylog() << var.value.string_value << std::endl;
        }
    }
    int64_t newRandomInt(void) {
        size_t max = space.size();
        size_t index = rand() % max;
        return atol(space[index].c_str());
    }
    double newRandomDouble(void) {
        double r = ((double) rand() / (RAND_MAX));
        double range = dmax - dmin;
        double rando = dmin + (range * r);
        return rando;
    }
    std::string newRandomString(void) {
        size_t max = space.size();
        size_t index = rand() % max;
        return space[index];
    }
    void updateBests(size_t duration) {
        if (duration < best_time) {
            best_time = duration;
            if (info.type == kokkos_value_double) {
                bestValue.double_value = lastValue.double_value;
            }
            else if (info.type == kokkos_value_int64) {
                bestValue.int_value = lastValue.int_value;
            }
            else /* if (info.type == kokkos_value_string) */ {
                strncpy(bestValue.string_value, lastValue.string_value, KOKKOS_TOOLS_TUNING_STRING_LENGTH);
            }
        }
    }
};

/* Have to make a deep copy of the variable to use it at exit */
void Variable::deepCopy(Kokkos_Tools_VariableInfo& _info) {
    info.type = _info.type;
    info.category = _info.category;
    info.valueQuantity = _info.valueQuantity;
    switch(info.category) {
        case kokkos_value_categorical:
        case kokkos_value_ordinal:
        {
            if (info.valueQuantity == kokkos_value_set) {
                size_t size = _info.candidates.set.size;
                info.candidates.set.size = size;
                if (info.type == kokkos_value_double) {
                    info.candidates.set.values.double_value =
                        (double*)(malloc(sizeof(double) * size));
                } else if (info.type == kokkos_value_int64) {
                    info.candidates.set.values.int_value =
                        (int64_t*)(malloc(sizeof(int64_t) * size));
                } else if (info.type == kokkos_value_string) {
                    info.candidates.set.values.string_value =
                        (Kokkos_Tools_Tuning_String*)(malloc(sizeof(Kokkos_Tools_Tuning_String) * size));
                }
                for (size_t index = 0 ; index < info.candidates.set.size ; index++) {
                    if (info.type == kokkos_value_double) {
                        info.candidates.set.values.double_value[index] =
                            _info.candidates.set.values.double_value[index];
                    } else if (info.type == kokkos_value_int64) {
                        info.candidates.set.values.int_value[index] =
                            _info.candidates.set.values.int_value[index];
                    } else if (info.type == kokkos_value_string) {
                        //info.candidates.set.values.string_value[index] =
                        //    (Kokkos_Tools_Tuning_String*)(malloc(sizeof(Kokkos_Tools_Tuning_String)));
                        memcpy(&(info.candidates.set.values.string_value[index]),
                               &(_info.candidates.set.values.string_value[index]),
                               sizeof(Kokkos_Tools_Tuning_String));
                    }
                }
            }
            if (info.valueQuantity == kokkos_value_unbounded) {
            }
            break;
        }
        case kokkos_value_interval:
        case kokkos_value_ratio:
        {
            if (info.valueQuantity == kokkos_value_range) {
                if (info.type == kokkos_value_double) {
                    info.candidates.range.step.double_value =
                        _info.candidates.range.step.double_value;
                    info.candidates.range.lower.double_value =
                        _info.candidates.range.lower.double_value;
                    info.candidates.range.upper.double_value =
                        _info.candidates.range.upper.double_value;
                } else if (info.type == kokkos_value_int64) {
                    info.candidates.range.step.int_value =
                        _info.candidates.range.step.int_value;
                    info.candidates.range.lower.int_value =
                        _info.candidates.range.lower.int_value;
                    info.candidates.range.upper.int_value =
                        _info.candidates.range.upper.int_value;
                }
                info.candidates.range.openLower = _info.candidates.range.openLower;
                info.candidates.range.openUpper = _info.candidates.range.openUpper;
            }
            if (info.valueQuantity == kokkos_value_set) {
                size_t size = _info.candidates.set.size;
                info.candidates.set.size = size;
                if (info.type == kokkos_value_double) {
                    info.candidates.set.values.double_value =
                        (double*)(malloc(sizeof(double) * size));
                } else if (info.type == kokkos_value_int64) {
                    info.candidates.set.values.int_value =
                        (int64_t*)(malloc(sizeof(int64_t) * size));
                } else if (info.type == kokkos_value_string) {
                    info.candidates.set.values.string_value =
                        (Kokkos_Tools_Tuning_String*)(malloc(sizeof(Kokkos_Tools_Tuning_String) * size));
                }
                for (size_t index = 0 ; index < info.candidates.set.size ; index++) {
                    if (info.type == kokkos_value_double) {
                        info.candidates.set.values.double_value[index] =
                            _info.candidates.set.values.double_value[index];
                    } else if (info.type == kokkos_value_int64) {
                        info.candidates.set.values.int_value[index] =
                            _info.candidates.set.values.int_value[index];
                    } else if (info.type == kokkos_value_string) {
                        //info.candidates.set.values.string_value[index] =
                        //    (Kokkos_Tools_Tuning_String*)(malloc(sizeof(Kokkos_Tools_Tuning_String)));
                        memcpy(&(info.candidates.set.values.string_value[index]),
                               &(_info.candidates.set.values.string_value[index]),
                               sizeof(Kokkos_Tools_Tuning_String));
                    }
                }
            }
            break;
        }
        default:
        {
            break;
        }
    }
}

Variable::Variable(size_t _id, std::string _name,
    Kokkos_Tools_VariableInfo& _info, bool isOutput) :
        id(_id), name(_name), output(isOutput) {
        deepCopy(_info);
        // Create a hash object for strings
        std::hash<std::string> hasher;
        // Calculate the hash value of the string
        hashValue = std::to_string(hasher(name));
    /*
    if (KokkosSession::getSession().verbose) {
        mylog() << toString();
    }
    */
    best_time = INT_MAX;
}

void Variable::reportBest(void) {
    if (!output) return;
    std::cout << "Best random value for variable " << name << ": ";
    if (info.type == kokkos_value_double) {
        std::cout << bestValue.double_value;
    }
    else if (info.type == kokkos_value_int64) {
        std::cout << bestValue.int_value;
    }
    else /* if (info.type == kokkos_value_string) */ {
        std::cout << bestValue.string_value;
    }
    std::cout << std::endl;
}

void Variable::makeSpace(void) {
    switch(info.category) {
        case kokkos_value_categorical:
        case kokkos_value_ordinal:
        {
            if (info.valueQuantity == kokkos_value_set) {
                for (size_t index = 0 ; index < info.candidates.set.size ; index++) {
                    if (info.type == kokkos_value_double) {
                        std::string tmp = std::to_string(
                                info.candidates.set.values.double_value[index]);
                        space.push_back(tmp);
                    } else if (info.type == kokkos_value_int64) {
                        std::string tmp = std::to_string(
                                info.candidates.set.values.int_value[index]);
                        space.push_back(tmp);
                    } else if (info.type == kokkos_value_string) {
                        space.push_back(
                            std::string(
                                info.candidates.set.values.string_value[index]));
                    }
                }
            }
            break;
        }
        case kokkos_value_interval:
        case kokkos_value_ratio:
        {
            if (info.valueQuantity == kokkos_value_range) {
                if (info.type == kokkos_value_double) {
                    dstep = info.candidates.range.step.double_value;
                    dmin = info.candidates.range.lower.double_value;
                    dmax = info.candidates.range.upper.double_value;
                    /*
                     * [] and () denote whether the range is inclusive/exclusive of the endpoint:
                     * [ includes the endpoint
                     * ( excludes the endpoint
                     * [] = 'Closed', includes both endpoints
                     * () = 'Open', excludes both endpoints
                     * [) and (] are both 'half-open', and include only one endpoint
                     */
                    if (info.candidates.range.openLower) {
                        dmin = dmin + dstep;
                    }
                    if (info.candidates.range.openUpper) {
                        dmax = dmax - dstep;
                    }
                } else if (info.type == kokkos_value_int64) {
                    lstep = info.candidates.range.step.int_value;
                    lmin = info.candidates.range.lower.int_value;
                    lmax = info.candidates.range.upper.int_value;
                    if (info.candidates.range.openLower) {
                        lmin = lmin + lstep;
                    }
                    if (info.candidates.range.openUpper) {
                        lmax = lmax - lstep;
                    }
                }
            }
            if (info.valueQuantity == kokkos_value_set) {
                for (size_t index = 0 ; index < info.candidates.set.size ; index++) {
                    if (info.type == kokkos_value_double) {
                        std::string tmp = std::to_string(
                                info.candidates.set.values.double_value[index]);
                        space.push_back(tmp);
                    } else if (info.type == kokkos_value_int64) {
                        std::string tmp = std::to_string(
                                info.candidates.set.values.int_value[index]);
                        space.push_back(tmp);
                    } else if (info.type == kokkos_value_string) {
                        space.push_back(
                            std::string(
                                info.candidates.set.values.string_value[index]));
                    }
                }
            }
            break;
        }
        default:
        {
            break;
        }
    }
}

std::map<size_t,Variable*> variables;

class Context {
    private:
    size_t _id;
    std::vector<size_t> inputVariables;
    std::vector<size_t> outputVariables;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    public:
    Context(size_t id) : _id(id) { }
    void addInputVariables(const size_t numContextVariables,
        const Kokkos_Tools_VariableValue* contextVariableValues) {
        for (auto i = 0 ; i < numContextVariables ; i++ ) {
            inputVariables.push_back(contextVariableValues[i].type_id);
        }
    }
    void addOutputVariables(const size_t numTuningVariables,
        Kokkos_Tools_VariableValue* tuningVariableValues) {
        for (auto i = 0 ; i < numTuningVariables ; i++ ) {
            outputVariables.push_back(tuningVariableValues[i].type_id);
            // look up the variable, and assign a new value
            auto iter = variables.find(tuningVariableValues[i].type_id);
            auto var = iter->second;
            var->assignNewValue(tuningVariableValues[i]);
        }
        
    }
    void start(void) {
        start_time_ = std::chrono::high_resolution_clock::now();
    }
    void stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time_).count();
        for ( auto v : outputVariables) {
            auto iter = variables.find(v);
            auto var = iter->second;
            var->updateBests(duration);
        }
    }
};

std::map<size_t,Context*> contexts;
std::stack<Context*> contextStack;

extern "C" {
/*
 * In the past, tools have responded to the profiling hooks in Kokkos.
 * This effort adds to that, there are now a few more functions (note
 * that I'm using the C names for types. In general you can replace
 * Kokkos_Tools_ with Kokkos::Tools:: in C++ tools)
 *
 */

/* Declares a tuning variable named name with uniqueId id and all the
 * semantic information stored in info. Note that the VariableInfo
 * struct has a void* field called toolProvidedInfo. If you fill this
 * in, every time you get a value of that type you'll also get back
 * that same pointer.
 */
void kokkosp_declare_output_type(const char* name, const size_t id,
    Kokkos_Tools_VariableInfo& info) {
    mylog() << __FUNCTION__ << " " << name << std::endl;
    Variable * output = new Variable(id, name, info);
    mylog() << output->toString() << std::endl;
    output->makeSpace();
    variables[id] = output;
    return;
}

/* This is almost exactly like declaring a tuning variable. The only
 * difference is that in cases where the candidate values aren't known,
 * info.valueQuantity will be set to kokkos_value_unbounded. This is
 * fairly common, Kokkos can tell you that kernel_name is a string,
 * but we can't tell you what strings a user might provide.
 */
void kokkosp_declare_input_type(const char* name, const size_t id,
    Kokkos_Tools_VariableInfo& info) {
    mylog() << __FUNCTION__ << " " << name << std::endl;
    Variable * input = new Variable(id, name, info, false);
    mylog() << input->toString() << std::endl;
    variables[id] = input;
}

/* This starts the context pointed at by contextId. If tools use
 * measurements to drive tuning, this is where they'll do their
 * starting measurement.
 */
void kokkosp_begin_context(size_t contextId) {
    mylog() << __FUNCTION__ << "\t" << contextId << std::endl;
    contexts[contextId] = new Context(contextId);
}

/* Here Kokkos is requesting the values of tuning variables, and most
 * of the meat is here. The contextId tells us the scope across which
 * these variables were used.
 *
 * The next two arguments describe the context you're tuning in. You
 * have the number of context variables, and an array of that size
 * containing their values. Note that the Kokkos_Tuning_VariableValue
 * has a field called metadata containing all the info (type,
 * semantics, and critically, candidates) about that variable.
 *
 * The two arguments following those describe the Tuning Variables.
 * First the number of them, then an array of that size which you can
 * overwrite. Overwriting those values is how you give values back to
 * the application.
 *
 * Critically, as tuningVariableValues comes preloaded with default
 * values, if your function body is return; you will not crash Kokkos,
 * only make us use our defaults. If you don't know, you are allowed
 * to punt and let Kokkos do what it would.
 */
void kokkosp_request_values(
    const size_t contextId,
    const size_t numContextVariables,
    const Kokkos_Tools_VariableValue* contextVariableValues,
    const size_t numTuningVariables,
    Kokkos_Tools_VariableValue* tuningVariableValues) {
    // get the context
    auto iter = contexts.find(contextId);
    auto context = iter->second;
    mylog() << __FUNCTION__ << "\ncontext id: " << contextId << std::endl;
    mylog() << numContextVariables << " input variables with ids: ";
    for (auto i = 0 ; i < numContextVariables ; i++ ) {
        mylog() << contextVariableValues[i].type_id << " ";
    }
    context->addInputVariables(numContextVariables, contextVariableValues);
    mylog() << "\n" << numTuningVariables << " output variables with ids: ";
    for (auto i = 0 ; i < numTuningVariables ; i++ ) {
        mylog() << tuningVariableValues[i].type_id << " ";
    }
    mylog() << "\n" << std::endl;
    context->addOutputVariables(numTuningVariables, tuningVariableValues);
    context->start();
}

/* This simply says that the contextId in the argument is now over.
 * If you provided tuning values associated with that context, those
 * values can now be associated with a result.
 */
void kokkosp_end_context(const size_t contextId) {
    mylog() << __FUNCTION__ << "\t" << contextId << std::endl;
    auto iter = contexts.find(contextId);
    auto context = iter->second;
    context->stop();
    contexts.erase(contextId);
}

/* This function will be called only once, prior to calling any other hooks
 * in the profiling library. Currently the only argument which is non-zero
 * is version, which will specify the version of the interface (which will
 * allow future changes to the interface). The version is an integer encoding
 * a date as ((year*100)+month)*100, and the current interface version is
 * 20150628.
 */
void kokkosp_init_library(const int, const uint64_t, const uint32_t,
    struct Kokkos_Profiling_KokkosPDeviceInfo*) {
    mylog() << __FUNCTION__ << std::endl;
}

/* This function will be called only once, after all other calls to
 * profiling hooks.
 */
void kokkosp_finalize_library() {
    mylog() << __FUNCTION__ << std::endl;
    std::string banner(80, '*');
    if (variables.size() == 0) {
        std::cerr << banner << std::endl;
        std::cerr << "No variables tuned! did you configure Kokkos with `-DKokkos_ENABLE_TUNING=TRUE`?\n" << banner << std::endl;
    } else {
        std::cout << "Best values found:\n" << banner << std::endl;
        for (const auto& k : variables) {
            auto v = k.second;
            v->reportBest();
        }
        std::cout << banner << std::endl;
    }
}


} // extern "C"

