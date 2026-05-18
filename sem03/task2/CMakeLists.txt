cmake_minimum_required(VERSION 3.10)
project(task3_task2)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(task2_server server.cpp)
add_executable(task2_checker checker.cpp)

target_link_libraries(task2_server pthread)
target_link_libraries(task2_checker pthread)