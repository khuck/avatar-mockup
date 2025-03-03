cmake_minimum_required(VERSION 3.2)

project(avatar-mockup)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

find_package(Kokkos 4.2 REQUIRED CONFIG) # Find Kokkos version 4.2 or later

add_subdirectory(src)


