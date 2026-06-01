cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_COMPILER "/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/compilers/bin/pgc++" CACHE STRING "PGC++ compiler" FORCE)

project(p_6th LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_OPENACC "Enable OpenACC flags for NVHPC/PGI compilers" ON)

find_package(Boost REQUIRED COMPONENTS program_options)


function(link_common target)
  target_link_libraries(${target} PRIVATE Boost::program_options)
  target_compile_options(${target} PRIVATE -O3)
endfunction()

function(add_openacc_flags target mode)
  if(NOT ENABLE_OPENACC)
    return()
  endif()

  if(mode STREQUAL "host")
    set(_acc_flag -acc=host)
  elseif(mode STREQUAL "multicore")
    set(_acc_flag -acc=multicore)
  elseif(mode STREQUAL "gpu")
    set(_acc_flag -acc=gpu)
  else()
    message(FATAL_ERROR "Unknown OpenACC mode: ${mode}")
  endif()

  target_compile_options(${target} PRIVATE ${_acc_flag} -Minfo=accel)
  target_link_options(${target} PRIVATE ${_acc_flag})
endfunction()

add_executable(cpu_onecore eq.cpp)
link_common(cpu_onecore)
add_openacc_flags(cpu_onecore host)

add_executable(cpu_multicore eq.cpp)
link_common(cpu_multicore)
add_openacc_flags(cpu_multicore multicore)
target_compile_definitions(cpu_multicore PRIVATE ACC_MULTICORE_DEFAULTS=2)

add_executable(gpu_base eq.cpp)
link_common(gpu_base)
add_openacc_flags(gpu_base gpu)

add_executable(gpu1 gpu1.cpp)
link_common(gpu1)
add_openacc_flags(gpu1 gpu)

add_executable(gpu2 gpu2.cpp)
link_common(gpu2)
add_openacc_flags(gpu2 gpu)

add_executable(gpu3 gpu3.cpp)
link_common(gpu3)
add_openacc_flags(gpu3 gpu)
