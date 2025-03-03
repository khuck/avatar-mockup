/**
 * meta-smoother
 */

#include <Kokkos_Core.hpp>
#include <unordered_map>
#include <iostream>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <cstdlib>
#include <random>
#include <tuple>
#include "tuning_playground.hpp"
#include <chrono>
#include <thread>

namespace metasmoother {
    std::vector<KTE::VariableValue> makeChebychevVariables() {
        // output variable ids
        size_t out_variables[3];
        out_variables[0] = declareOutputRange<int64_t>("C: Chebyshev Degree", 1, 6, 1);
        out_variables[1] = declareOutputContinuous("C: Eigenvalue Ratio", 10.0, 50.0, 0.1, false, false);
        out_variables[2] = declareOutputRange<int64_t>("C: Maximum Chebychev Iterations", 5, 100, 1);
        //The second argument to make_varaible_value might be a default value
        std::vector<KTE::VariableValue> answer_vector{
            KTE::make_variable_value(out_variables[0], int64_t(3)),
            KTE::make_variable_value(out_variables[1], double(25.0)),
            KTE::make_variable_value(out_variables[2], int64_t(50))
        };
        return answer_vector;
    }

    void doChebyshev(void) {
        Kokkos::Profiling::ScopedRegion region("Chebyshev");
        //std::cout << " Doing Chebyshev..." << std::endl;
        size_t context{KTE::get_new_context_id()};
        KTE::begin_context(context);

        // set the input values for the context
        static std::vector<KTE::VariableValue> input_vector{
            KTE::make_variable_value(1, "Chebyshev"),
            KTE::make_variable_value(2, "parallel_for")};
        KTE::set_input_values(context, input_vector.size(), input_vector.data());

        // set the output values for the context
        static std::vector<KTE::VariableValue> answer_vector{makeChebychevVariables()};

        // request new output values for the context
        // get the settings...
        KTE::request_output_values(context, answer_vector.size(), answer_vector.data());

        // try to converge on 5, 15, 75
        size_t delay = 1 + (std::abs(5 - answer_vector[0].value.int_value) * 750) +
                           (std::abs(15.0 - answer_vector[1].value.double_value) * 25) +
                           (std::abs(75 - answer_vector[2].value.int_value) * 10);
        // call the real solver
        std::this_thread::sleep_for(std::chrono::microseconds(delay));
        // end the context
        KTE::end_context(context);
    }

    std::vector<KTE::VariableValue> makeMultiThreadedGaussSeidelVariables() {
        // output variable ids
        size_t out_variables[3];
        out_variables[0] = declareOutputRange<int64_t>("MTGS: Number of Sweeps", 1, 2, 1);
        out_variables[1] = declareOutputContinuous("MTGS: Damping Factor", 0.8, 1.2, 0.01, false, false);
        //The second argument to make_varaible_value might be a default value
        std::vector<KTE::VariableValue> answer_vector{
            KTE::make_variable_value(out_variables[0], int64_t(2)),
            KTE::make_variable_value(out_variables[1], double(1.0)),
        };
        return answer_vector;
    }

    void MultiThreadedGaussSeidel(void) {
        Kokkos::Profiling::ScopedRegion region("Multi-threaded Gauss-Seidel");
        //std::cout << " Doing Multi-threaded Gauss-Seidel..." << std::endl;
        size_t context{KTE::get_new_context_id()};
        KTE::begin_context(context);

        // set the input values for the context
        static std::vector<KTE::VariableValue> input_vector{
            KTE::make_variable_value(1, "Multi-threaded Gauss-Seidel"),
            KTE::make_variable_value(2, "parallel_for")};
        KTE::set_input_values(context, input_vector.size(), input_vector.data());

        // set the output values for the context
        static std::vector<KTE::VariableValue> answer_vector{makeMultiThreadedGaussSeidelVariables()};

        // request new output values for the context
        // get the settings...
        KTE::request_output_values(context, answer_vector.size(), answer_vector.data());

        // try to converge on 1, 0.9
        size_t delay = 1 + (std::abs(1 - answer_vector[0].value.int_value) * 100) +
                           (std::abs(0.9 - answer_vector[1].value.double_value) * 100);
        // call the real solver
        std::this_thread::sleep_for(std::chrono::microseconds(delay));
        // end the context
        KTE::end_context(context);
    }

    std::vector<KTE::VariableValue> makeTwoStageGaussSeidelVariables() {
        // output variable ids
        size_t out_variables[3];
        out_variables[0] = declareOutputRange<int64_t>("TSGS: Number of Sweeps", 1, 2, 1);
        out_variables[1] = declareOutputContinuous("TSGS: Inner Damping Factor", 0.8, 1.2, 0.01, false, false);
        //The second argument to make_varaible_value might be a default value
        std::vector<KTE::VariableValue> answer_vector{
            KTE::make_variable_value(out_variables[0], int64_t(2)),
            KTE::make_variable_value(out_variables[1], double(1.0)),
        };
        return answer_vector;
    }

    void TwoStageGaussSeidel(void) {
        Kokkos::Profiling::ScopedRegion region("Two-Stage Gauss-Seidel");
        //std::cout << " Doing Two-Stage Gauss-Seidel..." << std::endl;
        size_t context{KTE::get_new_context_id()};
        KTE::begin_context(context);

        // set the input values for the context
        static std::vector<KTE::VariableValue> input_vector{
            KTE::make_variable_value(1, "Two-Stage Gauss-Seidel"),
            KTE::make_variable_value(2, "parallel_for")};
        KTE::set_input_values(context, input_vector.size(), input_vector.data());

        // set the output values for the context
        static std::vector<KTE::VariableValue> answer_vector{makeTwoStageGaussSeidelVariables()};

        // request new output values for the context
        // get the settings...
        KTE::request_output_values(context, answer_vector.size(), answer_vector.data());

        // call the real solver
        // try to converge on 2, 1.1
        size_t delay = 1 + (std::abs(2 - answer_vector[0].value.int_value) * 100) +
                           (std::abs(1.1 - answer_vector[1].value.double_value) * 100);
        std::this_thread::sleep_for(std::chrono::microseconds(delay));
        // end the context
        KTE::end_context(context);
    }
};

int main(int argc, char *argv[]) {

    Kokkos::initialize(argc, argv);
    {
        //Kokkos::print_configuration(std::cout, false);
        Kokkos::Profiling::ScopedRegion region("meta smoother search loop");
        for (int i = 0 ; i < 300 ; i++) {
            fastest_of("meta-smoother", 3,
                [&]() { metasmoother::doChebyshev(); },
                [&]() { metasmoother::MultiThreadedGaussSeidel(); },
                [&]() { metasmoother::TwoStageGaussSeidel(); }
            );
        }
    }
    std::cout << "\nC: Chebyshev Degree target value: 5" << std::endl;
    std::cout << "C: Eigenvalue Ratio target value: 15" << std::endl;
    std::cout << "C: Maximum Chebychev Iterations target value: 75" << std::endl;
    std::cout << "MTGS: Number of Sweeps target value: 1" << std::endl;
    std::cout << "MTGS: Damping Factor target value: 0.9" << std::endl;
    std::cout << "TSGS: Number of Sweeps target value: 2" << std::endl;
    std::cout << "TSGS: Inner Damping Factor target value: 1.1\n" << std::endl;
    Kokkos::finalize();
    
}
