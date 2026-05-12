cmake_minimum_required(VERSION 3.10)
project(iterative_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -Wall")

find_package(OpenMP REQUIRED)

find_package(Threads REQUIRED)

set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin)
file(MAKE_DIRECTORY ${OUTPUT_DIR})

file(GLOB SOURCES "*.cpp")

foreach(SRC ${SOURCES})
    get_filename_component(EXE_NAME ${SRC} NAME_WE)
    add_executable(${EXE_NAME} ${SRC})
    set_target_properties(${EXE_NAME} PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
    )
    
    target_compile_options(${EXE_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(${EXE_NAME} PRIVATE Threads::Threads OpenMP::OpenMP_CXX)
endforeach()