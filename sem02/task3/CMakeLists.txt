cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(lab2_task3 VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(task3_jacobi task3.cpp)

find_package(OpenMP REQUIRED)
target_link_libraries(task3_jacobi PRIVATE OpenMP::OpenMP_CXX)

if(MSVC)
  target_compile_options(task3_jacobi PRIVATE /O2)
else()
  target_compile_options(task3_jacobi PRIVATE -O2)
endif()
