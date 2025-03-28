cmake_minimum_required(VERSION 3.10)
project(PDE_Solvers)

set(CMAKE_CXX_STANDARD 14)

# OpenMP configuration (explicit)
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "OpenMP found, enabling parallelization")
    add_definitions(-D_OPENMP)  # Explicitly define _OPENMP macro
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# MPI (disabled for now since you removed it)
# find_package(MPI)

include_directories(include)

add_executable(heat_solver src/heat_solver.cpp)
target_link_libraries(heat_solver OpenMP::OpenMP_CXX)

add_executable(wave_solver src/wave_solver.cpp)
target_link_libraries(wave_solver OpenMP::OpenMP_CXX)