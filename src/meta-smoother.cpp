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

namespace KTE = Kokkos::Tools::Experimental;

namespace metasmoother {
    std::vector<KTE::VariableValue> makeChebychevVariables() {
        // output variable ids
        size_t out_variables[3];
        // first var, an integer range from 1 to 6 with step of 1
        out_variables[0] = declareOutputRange<int64_t>("Chebyshev: Degree", 1, 6, 1);
        // second var, a continuous floating point value between 10.0 and 50.0, discrete value of 0.1 for step increments
        // the boolean values indicate whether the bounds are open or not
        out_variables[1] = declareOutputContinuous("Chebyshev: Eigenvalue Ratio", 10.0, 50.0, 0.1, false, false);
        // third var, an integer range from 5 to 100 with step of 1
        out_variables[2] = declareOutputRange<int64_t>("Chebychev: Maximum Iterations", 5, 100, 1);
        //The second argument to make_varaible_value is a default value (starting point)
        std::vector<KTE::VariableValue> answer_vector{
            KTE::make_variable_value(out_variables[0], int64_t(3)),
            KTE::make_variable_value(out_variables[1], double(25.0)),
            KTE::make_variable_value(out_variables[2], int64_t(50))
        };
        return answer_vector;
    }

    void doChebyshev(void) {
        Kokkos::Profiling::ScopedRegion region("Chebyshev");
        // create a context
        size_t context{KTE::get_new_context_id()};
        KTE::begin_context(context);

        // set the input values for the context - these two are always on the stack,
        // we are just setting the context properties for the search. Enough to make it unique.
        static std::vector<KTE::VariableValue> input_vector{
            KTE::make_variable_value(1, "Chebyshev"),
            KTE::make_variable_value(2, "parallel_for")};
        KTE::set_input_values(context, input_vector.size(), input_vector.data());

        // set the output values for the context
        static std::vector<KTE::VariableValue> answer_vector{makeChebychevVariables()};

        // request new output values for the context
        // get the settings... this will increment the search in the search space and return a suggested value.
        // once the search has converged, you will get the same value for this context until exit.
        KTE::request_output_values(context, answer_vector.size(), answer_vector.data());

        /* "run" the smoother. */
        // try to converge on 5, 15, 75
        size_t delay = 1 + (std::abs(5 - answer_vector[0].value.int_value) * 750) +
                           (std::abs(15.0 - answer_vector[1].value.double_value) * 25) +
                           (std::abs(75 - answer_vector[2].value.int_value) * 10);
        // call the real solver - not really
        std::this_thread::sleep_for(std::chrono::microseconds(delay));
        // end the context - this will end timings for the context, and set the response value for this
        // context with those properties and those suggested values.
        KTE::end_context(context);
    }

    std::vector<KTE::VariableValue> makeMultiThreadedGaussSeidelVariables() {
        // output variable ids
        size_t out_variables[3];
        // first var, an integer range from 1 to 2 with step of 1
        out_variables[0] = declareOutputRange<int64_t>("Multi-threaded Gauss-Seidel: Number of Sweeps", 1, 2, 1);
        // second var, a continuous floating point value between 0.8 and 1.2, discrete value of 0.01 for step increments
        // the boolean values indicate whether the bounds are open or not
        out_variables[1] = declareOutputContinuous("Multi-threaded Gauss-Seidel: Damping Factor", 0.8, 1.2, 0.01, false, false);
        //The second argument to make_varaible_value is a default value (starting point)
        std::vector<KTE::VariableValue> answer_vector{
            KTE::make_variable_value(out_variables[0], int64_t(2)),
            KTE::make_variable_value(out_variables[1], double(1.0)),
        };
        return answer_vector;
    }

    void MultiThreadedGaussSeidel(void) {
        Kokkos::Profiling::ScopedRegion region("Multi-threaded Gauss-Seidel");
        // create a context
        size_t context{KTE::get_new_context_id()};
        KTE::begin_context(context);

        // set the input values for the context - these two are always on the stack,
        // we are just setting the context properties for the search. Enough to make it unique.
        static std::vector<KTE::VariableValue> input_vector{
            KTE::make_variable_value(1, "Multi-threaded Gauss-Seidel"),
            KTE::make_variable_value(2, "parallel_for")};
        KTE::set_input_values(context, input_vector.size(), input_vector.data());

        // set the output values for the context
        static std::vector<KTE::VariableValue> answer_vector{makeMultiThreadedGaussSeidelVariables()};

        // request new output values for the context
        // get the settings... this will increment the search in the search space and return a suggested value.
        // once the search has converged, you will get the same value for this context until exit.
        KTE::request_output_values(context, answer_vector.size(), answer_vector.data());

        /* "run" the smoother. */
        // try to converge on 1, 0.9
        size_t delay = 1 + (std::abs(1 - answer_vector[0].value.int_value) * 100) +
                           (std::abs(0.9 - answer_vector[1].value.double_value) * 100);
        // call the real solver - not really
        std::this_thread::sleep_for(std::chrono::microseconds(delay));
        // end the context - this will end timings for the context, and set the response value for this
        // context with those properties and those suggested values.
        KTE::end_context(context);
    }

    std::vector<KTE::VariableValue> makeTwoStageGaussSeidelVariables() {
        // output variable ids
        size_t out_variables[3];
        // first var, an integer range from 1 to 2 with step of 1
        out_variables[0] = declareOutputRange<int64_t>("Two-Stage Gauss-Seidel: Number of Sweeps", 1, 2, 1);
        // second var, a continuous floating point value between 0.8 and 1.2, discrete value of 0.01 for step increments
        // the boolean values indicate whether the bounds are open or not
        out_variables[1] = declareOutputContinuous("Two-Stage Gauss-Seidel: Inner Damping Factor", 0.8, 1.2, 0.01, false, false);
        //The second argument to make_varaible_value is a default value (starting point)
        std::vector<KTE::VariableValue> answer_vector{
            KTE::make_variable_value(out_variables[0], int64_t(2)),
            KTE::make_variable_value(out_variables[1], double(1.0)),
        };
        return answer_vector;
    }

    void TwoStageGaussSeidel(void) {
        Kokkos::Profiling::ScopedRegion region("Two-Stage Gauss-Seidel");
        // create a context
        size_t context{KTE::get_new_context_id()};
        KTE::begin_context(context);

        // set the input values for the context - these two are always on the stack,
        // we are just setting the context properties for the search. Enough to make it unique.
        static std::vector<KTE::VariableValue> input_vector{
            KTE::make_variable_value(1, "Two-Stage Gauss-Seidel"),
            KTE::make_variable_value(2, "parallel_for")};
        KTE::set_input_values(context, input_vector.size(), input_vector.data());

        // set the output values for the context
        static std::vector<KTE::VariableValue> answer_vector{makeTwoStageGaussSeidelVariables()};

        // request new output values for the context
        // get the settings... this will increment the search in the search space and return a suggested value.
        // once the search has converged, you will get the same value for this context until exit.
        KTE::request_output_values(context, answer_vector.size(), answer_vector.data());

        /* "run" the smoother. */
        // try to converge on 2, 1.1
        size_t delay = 1 + (std::abs(2 - answer_vector[0].value.int_value) * 100) +
                           (std::abs(1.1 - answer_vector[1].value.double_value) * 100);
        // call the real solver - not really
        std::this_thread::sleep_for(std::chrono::microseconds(delay));
        // end the context - this will end timings for the context, and set the response value for this
        // context with those properties and those suggested values.
        KTE::end_context(context);
    }
};

int main(int argc, char *argv[]) {

    std::string banner(80, '=');
    Kokkos::initialize(argc, argv);
    /* Report the "converged" values - keeping in mind that the random search doesn't really converge */
    std::cout << "\nChebyshev: Degree target value: 5" << std::endl;
    std::cout << "Chebyshev: Eigenvalue Ratio target value: 15" << std::endl;
    std::cout << "Chebychev: Maximum Iterations target value: 75" << std::endl;
    std::cout << "Multi-threaded Gauss-Seidel: Number of Sweeps target value: 1" << std::endl;
    std::cout << "Multi-threaded Gauss-Seidel: Damping Factor target value: 0.9" << std::endl;
    std::cout << "Two-Stage Gauss-Seidel: Number of Sweeps target value: 2" << std::endl;
    std::cout << "Two-Stage Gauss-Seidel: Inner Damping Factor target value: 1.1\n" << std::endl;

    /* 
     * This implementation uses the helper function fastest_of()
     */
    {
        std::cout << banner << "\nfastest_of() method:\n" << banner << std::endl;
        Kokkos::Profiling::ScopedRegion region("meta smoother search loop");
        for (int i = 0 ; i < 300 ; i++) {
            /* fastest_of is a helper function that will set up a round-robin search
               with name "meta-smoother" and 3 implementations to test. */
            fastest_of("meta-smoother", 3,
                [&]() { metasmoother::doChebyshev(); },
                [&]() { metasmoother::MultiThreadedGaussSeidel(); },
                [&]() { metasmoother::TwoStageGaussSeidel(); }
            );
        }
        std::cout << "done.\n" << banner << "\n" << std::endl;
    }
    /* 
     * This implementation uses explicit function calls to set up the search.
     */
    {
        std::cout << banner << "\nExplicit method:\n" << banner << std::endl;
        // lambda function to help us declare/setup the output variable once
        auto makeOuterVars = []() {
            // output variable ids
            size_t out_variables[1];
            // first var, an integer range from 1 to 2 with step of 1
            out_variables[0] = declareOutputRange<int64_t>("meta smoother: implementation", 0, 2, 1);
            //The second argument to make_varaible_value is a default value (starting point)
            std::vector<KTE::VariableValue> answer_vector{
                KTE::make_variable_value(out_variables[0], int64_t(2))
            };
            return answer_vector;
        };

        Kokkos::Profiling::ScopedRegion region("meta smoother explicit search loop");
        for (int i = 0 ; i < 300 ; i++) {
            // create a context
            size_t context{KTE::get_new_context_id()};
            KTE::begin_context(context);

            // set the input values for the context - these two are always on the stack,
            // we are just setting the context properties for the search. Enough to make it unique.
            // we can declare this once, then set it when iterating
            static std::vector<KTE::VariableValue> input_vector{
                KTE::make_variable_value(1, "meta smoother explicit search loop")};
            KTE::set_input_values(context, input_vector.size(), input_vector.data());

            // we can declare this once, then set it when iterating
            static std::vector<KTE::VariableValue> answer_vector{makeOuterVars()};

            // request new output values for the context
            // get the settings... this will increment the search in the search space and return a suggested value.
            // once the search has converged, you will get the same value for this context until exit.
            KTE::request_output_values(context, answer_vector.size(), answer_vector.data());

            switch(answer_vector[0].value.int_value) {
                case 0:
                    metasmoother::doChebyshev();
                    break;
                case 1:
                    metasmoother::MultiThreadedGaussSeidel();
                    break;
                case 2:
                default:
                    metasmoother::TwoStageGaussSeidel();
                    break;
            }
        }
        std::cout << "done.\n" << banner << "\n" << std::endl;
    }
    Kokkos::finalize();
}
