cmake_minimum_required(VERSION 3.18)
project(heat_equation CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------------------------------------
# Настройка компилятора NVIDIA HPC SDK (pgc++/nvc++)
# -------------------------------------------------------
# Запуск: cmake -DCMAKE_CXX_COMPILER=nvc++ ..
# Режимы:
#   CPU one-core:  -DACC_MODE=host    (или без флага)
#   CPU multicore: -DACC_MODE=multicore
#   GPU:           -DACC_MODE=gpu

set(ACC_MODE "gpu" CACHE STRING "OpenACC target mode: host | multicore | gpu")
set_property(CACHE ACC_MODE PROPERTY STRINGS host multicore gpu)

message(STATUS "OpenACC mode: ${ACC_MODE}")

# Флаги компилятора
set(ACC_FLAGS "-acc=${ACC_MODE}" "-Minfo=all" "-fast")

# Boost
find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(heat_equation src/main.cpp)

target_include_directories(heat_equation PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(heat_equation PRIVATE ${Boost_LIBRARIES})

# Применяем OpenACC флаги
foreach(flag IN LISTS ACC_FLAGS)
    target_compile_options(heat_equation PRIVATE ${flag})
    target_link_options(heat_equation PRIVATE ${flag})
endforeach()

# Вывод информации
message(STATUS "Boost version: ${Boost_VERSION}")
message(STATUS "Executable: heat_equation")
message(STATUS "Build with: nvc++ -acc=${ACC_MODE} -Minfo=all -fast")

# Удобные таргеты для разных режимов
# make heat_host     - CPU single-core
# make heat_multicore - CPU multicore
# make heat_gpu      - GPU

add_executable(heat_host      src/main.cpp)
add_executable(heat_multicore src/main.cpp)
add_executable(heat_gpu       src/main.cpp)

foreach(target IN ITEMS heat_host heat_multicore heat_gpu)
    target_include_directories(${target} PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE ${Boost_LIBRARIES})
endforeach()

set(HOST_FLAGS      "-acc=host"      "-Minfo=all" "-fast")
set(MULTICORE_FLAGS "-acc=multicore" "-Minfo=all" "-fast")
set(GPU_FLAGS       "-acc=gpu"       "-Minfo=all" "-fast" "-gpu=managed")

foreach(flag IN LISTS HOST_FLAGS)
    target_compile_options(heat_host PRIVATE ${flag})
    target_link_options(heat_host PRIVATE ${flag})
endforeach()

foreach(flag IN LISTS MULTICORE_FLAGS)
    target_compile_options(heat_multicore PRIVATE ${flag})
    target_link_options(heat_multicore PRIVATE ${flag})
endforeach()

foreach(flag IN LISTS GPU_FLAGS)
    target_compile_options(heat_gpu PRIVATE ${flag})
    target_link_options(heat_gpu PRIVATE ${flag})
endforeach()
