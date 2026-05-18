cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(task1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(thr_runner thread.cpp)
add_executable(jthr_runner jthread.cpp)
