
add_executable(meta-smoother meta-smoother.cpp)
add_executable(meta-smoother-discrete meta-smoother-discrete.cpp)
add_library(simple-tuner SHARED simple-tuner.cpp tuning_playground.hpp)
target_link_libraries(meta-smoother PRIVATE Kokkos::kokkos)
target_link_libraries(meta-smoother-discrete PRIVATE Kokkos::kokkos)
target_link_libraries(simple-tuner PRIVATE Kokkos::kokkos)

