
add_executable(meta-smoother meta-smoother.cpp)
add_library(simple-tuner SHARED simple-tuner.cpp)
target_link_libraries(meta-smoother PRIVATE Kokkos::kokkos)
target_link_libraries(simple-tuner PRIVATE Kokkos::kokkos)

