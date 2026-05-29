cmake_minimum_required(VERSION 3.26)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(LuisaComputeSimulator)

# Ensure compile config exists
set(LCS_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/lcs_config.ini")
set(LCS_CONFIG_TEMPLATE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/lcs_config_template.ini")
if(NOT EXISTS "${LCS_CONFIG_FILE}")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/lcs_config_template.ini" "${LCS_CONFIG_FILE}" COPYONLY)
endif()

# Global options
option(LCS_ENABLE_GUI "Enable GUI support" OFF)
option(LCS_ENABLE_TEST "Enable Unit Tests" OFF)
option(LCS_USE_SYSTEM_PARALLEL_FOR "Enable system parallel_for" OFF)
option(LCS_BUILD_MAIN_APPLICATION "Build Executable Simulation Application" ON)
option(LCS_BUILD_PYBINDINGS "Build Python bindings (requires pybind11)" OFF)
set(LCS_PYTHON_EXECUTABLE "" CACHE FILEPATH "Path to the Python executable to build python bindings for (optional)")

# In scikit-build-core (pip install / pip wheel) context, override defaults
# for Python package builds. This avoids duplicating these settings in
# pyproject.toml's [tool.scikit-build.cmake.define].
if(SKBUILD)
    set(LCS_BUILD_PYBINDINGS ON CACHE BOOL "" FORCE)
endif()

if(LCS_BUILD_PYBINDINGS AND LCS_PYTHON_EXECUTABLE STREQUAL "")
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    set(LCS_PYTHON_EXECUTABLE "${Python_EXECUTABLE}" CACHE FILEPATH "Path to the Python executable to build python bindings for (optional)" FORCE)
endif()

# LuisaCompute backends
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
set_target_properties(luisa-compute-solver-lib PROPERTIES POSITION_INDEPENDENT_CODE ON)



# External
set(EXT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ext)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext)
include(${LuisaCompute_SOURCE_DIR}/scripts/setup_output_dirs.cmake)

# Rousource
set(LCSV_RESOURCE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Resources)
target_compile_definitions(luisa-compute-solver-lib PUBLIC LCSV_RESOURCE_PATH="${LCSV_RESOURCE_PATH}")


# Executable application
if(LCS_BUILD_MAIN_APPLICATION)
    add_executable(app_simulation Application/app_simulation.cpp Application/app_simulation_demo_config.cpp Application/app_simulation_demo_config.h)
    add_executable(app_integration Application/app_integration.cpp)
    target_link_libraries(app_simulation PRIVATE 
        luisa::compute 
        luisa-compute-solver-lib
        luisa-compute-solver-ext-yyjson
        )
    target_link_libraries(app_integration PRIVATE 
        luisa::compute 
        luisa-compute-solver-lib
        )
    if (UNIX AND NOT APPLE)
        target_link_libraries(app_simulation PRIVATE atomic)
    endif()

    if (LCS_ENABLE_GUI)
        target_link_libraries(app_simulation PRIVATE polyscope)
        target_compile_definitions(app_simulation PRIVATE SIMULATION_APP_USE_GUI=1)
    endif()
endif()


if(LCS_ENABLE_TEST)
    add_subdirectory(UnitTest)
endif()

if(LCS_BUILD_PYBINDINGS)
    add_subdirectory(PythonBindings)
endif()
