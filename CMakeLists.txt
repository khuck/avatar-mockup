cmake_minimum_required(VERSION 3.16)

project(avatar-mockup)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
add_compile_options(-fsanitize=address)
add_link_options(-fsanitize=address)

find_package(Kokkos 4.5 REQUIRED CONFIG) # Find Kokkos version 4.2 or later

add_subdirectory(src)


