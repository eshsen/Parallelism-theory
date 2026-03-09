cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(integrate_omp VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(integrate_omp main.cpp)
target_link_libraries(integrate_omp PRIVATE OpenMP::OpenMP_CXX)