cmake_minimum_required(VERSION 3.26)

set(CMAKE_CXX_STANDARD 20)

project(LuisaComputeSolver)


# lib
file(GLOB_RECURSE LUISA_COMPUTE_SOLVER_LIB_SOURCES CONFIGURE_DEPENDS Solver/*.cpp Solver/*.cc)
file(GLOB_RECURSE LUISA_COMPUTE_SOLVER_LIB_HEADERS CONFIGURE_DEPENDS Solver/*.h)
add_library(luisa-compute-solver-lib STATIC ${LUISA_COMPUTE_SOLVER_LIB_SOURCES} ${LUISA_COMPUTE_SOLVER_LIB_HEADERS})
target_include_directories(luisa-compute-solver-lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Solver>)
target_link_libraries(luisa-compute-solver-lib PUBLIC luisa::compute) #  ipc::toolkit

if(WIN32)
set(CMAKE_PREFIX_PATH   "C:/Applications/vcpkg/installed/x64-windows")
endif()


set(EXT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ext)

set(LUISA_COMPUTE_ENABLE_CPU OFF CACHE BOOL "" FORCE)
set(LUISA_COMPUTE_ENABLE_DX  OFF CACHE BOOL "" FORCE)
set(LUISA_COMPUTE_ENABLE_CUDA  ON CACHE BOOL "" FORCE)
set(LUISA_COMPUTE_ENABLE_VULKAN OFF CACHE BOOL "" FORCE)
set(LUISA_COMPUTE_ENABLE_FALLBACK OFF CACHE BOOL "" FORCE)

option(LUISA_COMPUTE_SOLVER_USE_GUI "Enable GUI support" OFF)
option(LUISA_COMPUTE_SOLVER_USE_SYSTEM_PARALLEL_FOR "Enable system parallel_for" OFF)

add_subdirectory(ext)

set(LCSV_RESOURCE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Resources)
target_compile_definitions(luisa-compute-solver-lib PUBLIC LCSV_RESOURCE_PATH="${LCSV_RESOURCE_PATH}")

# test
add_executable(test-xtypes UnitTest/test_xtypes.cpp)
target_link_libraries(test-xtypes PRIVATE luisa::compute luisa-compute-solver-lib)

# app
add_executable(app-simulation Application/app_simulation.cpp Application/app_simulation_demo_config.cpp Application/app_simulation_demo_config.h)
target_link_libraries(app-simulation PRIVATE 
    luisa::compute 
    luisa-compute-solver-lib)
if (LUISA_COMPUTE_SOLVER_USE_GUI)
    target_link_libraries(app-simulation PRIVATE polyscope)
    target_compile_definitions(app-simulation PRIVATE SIMULATION_APP_USE_GUI=1)
endif()

add_executable(app-test-features Application/app_test_features.cpp)
target_link_libraries(app-test-features PRIVATE 
    luisa::compute 
    luisa-compute-solver-lib)
