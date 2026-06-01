cmake_minimum_required(VERSION 3.18)
project(heat2d CXX)

# ожидается pgc++/nvc++ (NVHPC)
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "PGI|NVHPC")
  message(WARNING "This project is designed for pgc++/nvc++ (NVHPC). Current compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

set(CMAKE_CXX_STANDARD 14)

set(ACC_FLAGS  -acc -Minfo=all -fast)
set(BOOST_LIB  boost_program_options)

function(add_heat_target TGT SRC ACC_MODE)
  add_executable(${TGT} ${SRC})
  target_compile_options(${TGT} PRIVATE ${ACC_FLAGS} -acc=${ACC_MODE})
  target_link_options(   ${TGT} PRIVATE ${ACC_FLAGS} -acc=${ACC_MODE})
  target_link_libraries( ${TGT} PRIVATE ${BOOST_LIB})
  foreach(DEF IN LISTS ARGN)
    target_compile_definitions(${TGT} PRIVATE ${DEF})
  endforeach()
endfunction()

# CPU: оба из heat_all.cpp
add_heat_target(cpu_onecore   heat_all.cpp host)
add_heat_target(cpu_multicore heat_all.cpp multicore)

# базовый GPU из heat_all.cpp
add_heat_target(gpu_base heat_all.cpp gpu)

# отдельные GPU-файлы
add_heat_target(gpu_1 gpu_1.cpp gpu)
add_heat_target(gpu_2 gpu_2.cpp gpu)
add_heat_target(gpu_3 gpu_3.cpp gpu VEC_LEN=128)