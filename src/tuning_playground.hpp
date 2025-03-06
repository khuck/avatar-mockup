#pragma once

#include<Kokkos_Core.hpp>
#include<unordered_map>
#include<iostream>
#include<Kokkos_Profiling_ScopedRegion.hpp>

namespace Impl {

constexpr const int max_iterations{1000};
struct empty {};

template <typename Tunable, template <typename...> typename TupleLike,
          typename... Components, size_t... Indices>
void invoke_benchmark_helper(const Tunable &tunable, int num_iters,
                             const TupleLike<Components...> &tup,
                             const std::index_sequence<Indices...>) {
  for (int x = 0; x < num_iters; ++x) {
    tunable(x, num_iters, std::get<Indices>(tup)...);
  }
}

template <typename Tunable, template <typename...> typename TupleLike,
          typename... Components>
void invoke_benchmark(const Tunable &tunable, int num_iters,
                      const TupleLike<Components...> &tup) {
  invoke_benchmark_helper(tunable, num_iters, tup,
                          std::make_index_sequence<sizeof...(Components)>{});
}

template <typename Setup>
auto setup_helper(const Setup &setup, int num_iters, std::false_type) {
  return setup(num_iters);
}
template <typename Setup>
auto setup_helper(const Setup &setup, int num_iters, std::true_type) {
  setup(num_iters);
  return std::make_tuple();
}

} // namespace Impl

template<typename Setup, typename Tunable>
void tuned_kernel(int argc, char* argv[], Setup setup, Tunable tunable){
  int num_iters = 1000;
  Kokkos::initialize(argc, argv);
  {
    Kokkos::print_configuration(std::cout, false);
    using emptiness =
        typename std::is_same<decltype(setup(num_iters)), void>::type;
    auto kernel_data = Impl::setup_helper(setup, num_iters, emptiness{});
    Impl::invoke_benchmark(tunable, num_iters, kernel_data);
  }
  Kokkos::finalize();
}

void fastest_of_helper(int index){
  /** error case*/
}

template<typename Head, typename... Cons>
void fastest_of_helper(int index, Head head, Cons... cons){
  if(index == 0){
    return head();
  }
  return fastest_of_helper(index-1, cons...);
}

static std::unordered_map<std::string, size_t> ids_for_kernels;
size_t create_categorical_int_tuner(std::string name, size_t num_options){
  using namespace Kokkos::Tools::Experimental;
  VariableInfo info;
  info.category = StatisticalCategory::kokkos_value_categorical;
  info.type = ValueType::kokkos_value_int64;
  info.valueQuantity = CandidateValueType::kokkos_value_set;
  std::vector<int64_t> options;
  for(size_t x=0;x<num_options;++x){
    options.push_back(x);
  }
  info.candidates = make_candidate_set(options.size(), options.data());
  return declare_output_type(name, info);
}

size_t create_fastest_implementation_id(const size_t count){
  using namespace Kokkos::Tools::Experimental;
  static size_t id;
  static bool done;
  if(!done){
    done = true;
    VariableInfo info;
    info.category = StatisticalCategory::kokkos_value_categorical;
    info.type = ValueType::kokkos_value_int64;
    info.valueQuantity = CandidateValueType::kokkos_value_unbounded;
    id = declare_input_type("fastest_implementation_of", info);
  }
  return id;
}

/* fastest_of - A convenience function
   This function will take a name for the input variable, the number of implementations to test,
   and a variable argument list of implementations to test.
   */
template<typename ... Implementations>
void fastest_of(const std::string& label, const size_t count, Implementations... implementations){
    using namespace Kokkos::Tools::Experimental;
    auto tuner_iter = [&]() {
      auto my_tuner = ids_for_kernels.find(label);
      if (my_tuner == ids_for_kernels.end()) {
        return (ids_for_kernels.emplace(label, create_categorical_int_tuner(label, sizeof...(Implementations)))
                    .first);
      }
      return my_tuner;
    }();
    auto var_id = tuner_iter->second;
    auto input_id = create_fastest_implementation_id(count);
    VariableValue picked_implementation = make_variable_value(input_id,int64_t(0));
    VariableValue which_kernel = make_variable_value(var_id,label.c_str());
    which_kernel.value.int_value = -1;
    auto context_id = get_new_context_id();
    begin_context(context_id);
    set_input_values(context_id, 1, &picked_implementation);
    request_output_values(context_id, 1, &which_kernel);
    // if we didn't get a prediction, just alternate between methods.
    if (which_kernel.value.int_value < 0) {
        static int flipper{0};
        fastest_of_helper(flipper, implementations...);
        flipper = (flipper + 1) % count;
    } else {
        fastest_of_helper(which_kernel.value.int_value, implementations...);
    }
    end_context(context_id);
}

enum schedulers{StaticSchedule, DynamicSchedule};
static const std::string scheduleNames[] = {"static", "dynamic"};
constexpr int lowerBound{100};
constexpr int upperBound{999};

// Helper function to generate tile sizes
std::vector<int64_t> factorsOf(const int &size){
    std::vector<int64_t> factors;
    for(int i=1; i<=size; i++){
        if(size % i == 0){
            factors.push_back(i);
        }
    }
    return factors;
}

// Helper function to generate a linear series
template<typename T>
std::vector<T> makeRange(const T& min, const T& max, const T& step){
    std::vector<T> range;
    for(T i=min; i<=max; i+=step){
        range.push_back(i);
    }
    return range;
}

// helper function for human output
template<typename T>
void reportOptions(const std::vector<T>& candidates,
    std::string name) {
    std::string tmpstr{"Options for "};
    tmpstr += name;
    tmpstr += " [";
    for(auto &i : candidates){ tmpstr += std::to_string(i) + ",";}
    tmpstr[tmpstr.size()-1] = ']';
    std::cout << tmpstr << std::endl;
}

// helper function for human output
void reportOptions(const std::string& name, const double& lower,
    const double& upper, const bool& openLower, const bool& openUpper) {
    std::string tmpstr{"Options for "};
    tmpstr += name;
    tmpstr += (openLower ? "(" : "[");
    tmpstr += std::to_string(lower) + "," + std::to_string(upper);
    tmpstr += (openUpper ? ")" : "]");
    std::cout << tmpstr << std::endl;
}

// helper function for declaring output tiling variables
size_t declareOutputTileSize(std::string name, std::string varname, size_t limit) {
    size_t out_value_id;
    // create a vector of potential values
    std::vector<int64_t> candidates = factorsOf(limit);
    reportOptions(candidates, name);
    // create our variable object
    Kokkos::Tools::Experimental::VariableInfo out_info;
    // set the variable details
    out_info.type = Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;
    out_info.category = Kokkos::Tools::Experimental::StatisticalCategory::kokkos_value_ordinal;
    out_info.valueQuantity = Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
    out_info.candidates = Kokkos::Tools::Experimental::make_candidate_set(candidates.size(),candidates.data());
    // declare the variable
    out_value_id = Kokkos::Tools::Experimental::declare_output_type(varname,out_info);
    // return the id
    return out_value_id;
}

// helper function for declaring input size variables
size_t declareInputViewSize(std::string varname, int64_t size) {
    size_t in_value_id;
    // create a 'vector' of value(s)
    std::vector<int64_t> candidates = {size};
    // create our variable object
    Kokkos::Tools::Experimental::VariableInfo in_info;
    // set the variable details
    in_info.type = Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;
    in_info.category = Kokkos::Tools::Experimental::StatisticalCategory::kokkos_value_ordinal;
    in_info.valueQuantity = Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
    in_info.candidates = Kokkos::Tools::Experimental::make_candidate_set(candidates.size(),candidates.data());
    // declare the variable
    in_value_id = Kokkos::Tools::Experimental::declare_input_type(varname,in_info);
    // return the id
    return in_value_id;
}

// helper function for declaring scheduler variable
size_t declareOutputSchedules(std::string varname) {
    // create a vector of potential values
    std::vector<int64_t> candidates_schedule = {StaticSchedule,DynamicSchedule};
    // create our variable object
    Kokkos::Tools::Experimental::VariableInfo schedule_out_info;
    // set the variable details
    schedule_out_info.type = Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;
    schedule_out_info.category = Kokkos::Tools::Experimental::StatisticalCategory::kokkos_value_categorical;
    schedule_out_info.valueQuantity = Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
    schedule_out_info.candidates = Kokkos::Tools::Experimental::make_candidate_set(candidates_schedule.size(),candidates_schedule.data());
    // declare the variable
    size_t schedule_out_value_id = Kokkos::Tools::Experimental::declare_output_type(varname,schedule_out_info);
    // return the id
    return schedule_out_value_id;
}

// helper function for declaring output tread count variable
size_t declareOutputThreadCount(std::string varname, size_t limit) {
    size_t out_value_id;
    // create a vector of potential values
    std::vector<int64_t> candidates = makeRange<int64_t>(2, limit, 2);
    // create our variable object
    Kokkos::Tools::Experimental::VariableInfo out_info;
    // set the variable details
    out_info.type = Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;
    out_info.category = Kokkos::Tools::Experimental::StatisticalCategory::kokkos_value_categorical;
    out_info.valueQuantity = Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
    out_info.candidates = Kokkos::Tools::Experimental::make_candidate_set(candidates.size(),candidates.data());
    // declare the variable
    out_value_id = Kokkos::Tools::Experimental::declare_output_type(varname,out_info);
    // return the id
    return out_value_id;
}

/*
// helper function for declaring scheduler variable
size_t declareOutputSchedules(std::string varname) {
    // create a vector of potential values
    std::vector<int64_t> candidates_schedule = {StaticSchedule,DynamicSchedule};
    // create our variable object
    Kokkos::Tools::Experimental::VariableInfo schedule_out_info;
    // set the variable details
    schedule_out_info.type = Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;
    schedule_out_info.category = Kokkos::Tools::Experimental::StatisticalCategory::kokkos_value_ordinal;
    schedule_out_info.valueQuantity = Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
    schedule_out_info.candidates = Kokkos::Tools::Experimental::make_candidate_set(candidates_schedule.size(),candidates_schedule.data());
    // declare the variable
    size_t schedule_out_value_id = Kokkos::Tools::Experimental::declare_output_type(varname,schedule_out_info);
    // return the id
    return schedule_out_value_id;
}
*/

// helper function for declaring range of int64_t values
size_t declareOutputRangeInt64(const std::string varname,
        const int64_t& lower, const int64_t& upper, const int64_t& step) {
    size_t out_value_id;
    // create a vector of potential values
    std::vector<int64_t> candidates = makeRange<int64_t>(lower, upper, step);
    reportOptions(candidates, varname);
    // create our variable object
    Kokkos::Tools::Experimental::VariableInfo out_info;
    // set the variable details
    out_info.type = Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;
    out_info.category = Kokkos::Tools::Experimental::StatisticalCategory::kokkos_value_ordinal;
    out_info.valueQuantity = Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
    out_info.candidates = Kokkos::Tools::Experimental::make_candidate_set(candidates.size(),candidates.data());
    // declare the variable
    out_value_id = Kokkos::Tools::Experimental::declare_output_type(varname,out_info);
    // return the id
    return out_value_id;
}

// helper function for declaring range of double values
size_t declareOutputRangeDouble(const std::string varname,
        const double& lower, const double& upper, const double& step) {
    size_t out_value_id;
    // create a vector of potential values
    std::vector<double> candidates = makeRange<double>(lower, upper, step);
    reportOptions(candidates, varname);
    // create our variable object
    Kokkos::Tools::Experimental::VariableInfo out_info;
    // set the variable details
    out_info.type = Kokkos::Tools::Experimental::ValueType::kokkos_value_double;
    out_info.category = Kokkos::Tools::Experimental::StatisticalCategory::kokkos_value_ordinal;
    out_info.valueQuantity = Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
    out_info.candidates = Kokkos::Tools::Experimental::make_candidate_set(candidates.size(),candidates.data());
    // declare the variable
    out_value_id = Kokkos::Tools::Experimental::declare_output_type(varname,out_info);
    // return the id
    return out_value_id;
}

// helper function for declaring range of double values
size_t declareOutputContinuous(const std::string varname,
        const double& lower, const double& upper, const double& step,
        bool openLower, bool openUpper) {
    size_t out_value_id;
    reportOptions(varname, lower, upper, openLower, openUpper);
    // create our variable object
    Kokkos::Tools::Experimental::VariableInfo out_info;
    // set the variable details
    out_info.type = Kokkos::Tools::Experimental::ValueType::kokkos_value_double;
    out_info.category = Kokkos::Tools::Experimental::StatisticalCategory::kokkos_value_interval;
    out_info.valueQuantity = Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_range;
    out_info.candidates = Kokkos::Tools::Experimental::make_candidate_range(lower, upper, step, openLower, openUpper);
    // declare the variable
    out_value_id = Kokkos::Tools::Experimental::declare_output_type(varname,out_info);
    // return the id
    return out_value_id;
}

// helper function for declaring generic range of values
template<typename T>
size_t declareOutputRange(const std::string varname,
        const T lower, const T upper, const T step) {
    if(typeid(T) == typeid(int64_t)) {
        return declareOutputRangeInt64(varname, lower, upper, step);
    } else if(typeid(T) == typeid(double)) {
        return declareOutputRangeDouble(varname, lower, upper, step);
    } else {
        assert(false);
    }
    return 0;
}

// helper function for matrix init
void initArray(Kokkos::View<double *, Kokkos::HostSpace>& ar, size_t d1) {
    for(size_t i=0; i<d1; i++){
        ar(i)=(rand() % (upperBound - lowerBound + 1)) + lowerBound;
    }
}

// helper function for matrix init
void initArray(Kokkos::View<double **, Kokkos::HostSpace>& ar, size_t d1, size_t d2) {
    for(size_t i=0; i<d1; i++){
        for(size_t j=0; j<d2; j++){
            ar(i,j)=(rand() % (upperBound - lowerBound + 1)) + lowerBound;
        }
    }
}

// helper function for matrix init
void initArray(Kokkos::View<int **, Kokkos::HostSpace>& ar, size_t d1, size_t d2) {
    for(size_t i=0; i<d1; i++){
        for(size_t j=0; j<d2; j++){
            ar(i,j)=(rand() % (upperBound - lowerBound + 1)) + lowerBound;
        }
    }
}

// helper function for matrix init
void initArray(Kokkos::View<double ***, Kokkos::DefaultExecutionSpace::memory_space>& ar, size_t d1, size_t d2, size_t d3) {
    const auto kernel = KOKKOS_LAMBDA(const int x, const int y, const int z) {
        ar(x,y,z)= x + y + z;
    };
    Kokkos::parallel_for("initialize",
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace,
                            Kokkos::Rank<3>>
            ({0, 0, 0}, {d1, d2, d3}), kernel);
}


