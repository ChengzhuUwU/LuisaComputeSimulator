cmake_minimum_required(VERSION 3.26)

set(CMAKE_CXX_STANDARD 20)

project(LuisaComputeSolver)


# Global options
option(LCS_ENABLE_GUI "Enable GUI support" OFF)
option(LCS_ENABLE_TEST "Enable Unit Tests" OFF)
option(LCS_USE_SYSTEM_PARALLEL_FOR "Enable system parallel_for" OFF)
option(LCS_BUILD_MAIN_APPLICATION "Build Excutable Simulation Application" ON)

option(LUISA_COMPUTE_ENABLE_CUDA "Enable CUDA backend" ON)
option(LUISA_COMPUTE_ENABLE_METAL "Enable Metal backend" ON)
option(LUISA_COMPUTE_ENABLE_VULKAN "Enable Vulkan backend" OFF)
option(LUISA_COMPUTE_ENABLE_DX "Enable DirectX backend" OFF)
option(LUISA_COMPUTE_ENABLE_FALLBACK "Enable Fallback (CPU TBB) backend" OFF)

# Solver
file(GLOB_RECURSE LUISA_COMPUTE_SOLVER_LIB_SOURCES CONFIGURE_DEPENDS Solver/*.cpp Solver/*.cc)
file(GLOB_RECURSE LUISA_COMPUTE_SOLVER_LIB_HEADERS CONFIGURE_DEPENDS Solver/*.h CONFIGURE_DEPENDS Solver/*.hpp)
add_library(luisa-compute-solver-lib STATIC ${LUISA_COMPUTE_SOLVER_LIB_SOURCES} ${LUISA_COMPUTE_SOLVER_LIB_HEADERS})
target_include_directories(luisa-compute-solver-lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Solver>)



# External
set(EXT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ext)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext)
include(${LuisaCompute_SOURCE_DIR}/scripts/setup_output_dirs.cmake)

# Rousource
set(LCSV_RESOURCE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Resources)
target_compile_definitions(luisa-compute-solver-lib PUBLIC LCSV_RESOURCE_PATH="${LCSV_RESOURCE_PATH}")


# Main Application
if(LCS_BUILD_MAIN_APPLICATION)
    add_executable(app-simulation Application/app_simulation.cpp Application/app_simulation_demo_config.cpp Application/app_simulation_demo_config.h)
    target_link_libraries(app-simulation PRIVATE 
        luisa::compute 
        luisa-compute-solver-lib
        luisa-compute-solver-ext-yyjson
        )
    if (UNIX AND NOT APPLE)
        target_link_libraries(app-simulation PRIVATE atomic)
    endif()

    if (LCS_ENABLE_GUI)
        target_link_libraries(app-simulation PRIVATE polyscope)
        target_compile_definitions(app-simulation PRIVATE SIMULATION_APP_USE_GUI=1)
    endif()
endif()


if(LCS_ENABLE_TEST)
    add_subdirectory(UnitTest)
endif()
