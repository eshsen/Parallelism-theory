cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(task1 VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_DOUBLE "Use double precision (OFF = float)" OFF)

if(USE_DOUBLE)
    add_compile_definitions(USE_DOUBLE)
    message(STATUS "double is used")
else()
    message(STATUS "float is used")
endif()

add_executable(sinsum main.cpp)