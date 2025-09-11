cmake_minimum_required(VERSION 3.26)

set(CMAKE_CXX_STANDARD 20)

project(LuisaComputeSolver)

set(EXT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/External)
include(FetchContent)
if (NOT TARGET luisa::compute)
    # download LuisaCompute from github.com/LuisaGroup/LuisaCompute.git to the build directory

    # FetchContent_Populate(luisa_compute
    #         # GIT_REPOSITORY "https://github.com/LuisaGroup/LuisaCompute.git"
    #         # GIT_TAG stable)
    #         GIT_REPOSITORY "https://gitee.com/celeste-huohuo/LuisaCompute.git"
    #         GIT_TAG stable)
    # set(LuisaCompute_SOURCE_DIR ${luisa_compute_SOURCE_DIR})
    set(LuisaCompute_SOURCE_DIR ${EXT_PATH}/LuisaCompute)
    message(STATUS "LuisaCompute source dir: ${LuisaCompute_SOURCE_DIR}")

    include(${LuisaCompute_SOURCE_DIR}/scripts/setup_output_dirs.cmake)

    # add LuisaCompute as a subdirectory
    set(LUISA_COMPUTE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_DOWNLOAD_NVCOMP OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_USE_SYSTEM_STL ON CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_RUST OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_CPU OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_DX  OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_CUDA  ON CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_VULKAN OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_REMOTE OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_GUI ON CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_WAYLAND ON CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_FALLBACK OFF CACHE BOOL "" FORCE)
    add_subdirectory(${LuisaCompute_SOURCE_DIR})
endif ()

if (NOT TARGET polyscope)

    set(POLYSCOPE_USE_EXTERNAL_GLFW ON CACHE BOOL "" FORCE)
    set(POLYSCOPE_BACKEND_OPENGL3_GLFW ON CACHE BOOL "" FORCE)
    set(POLYSCOPE_BACKEND_OPENGL_MOCK ON CACHE BOOL "" FORCE)
    add_subdirectory(${EXT_PATH}/polyscope)

    # FetchContent_Declare(
    #         polyscope
    #         GIT_REPOSITORY https://gitee.com/celeste-huohuo/polyscope.git
    #         GIT_TAG master
    #         # GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
    #         # GIT_TAG 2cd38b7496a1e8ff79f615920de2980295bedef7
    # )
    # FetchContent_GetProperties(polyscope)
    # if(NOT polyscope_POPULATED)
    #     # Fetch the content using previously declared details
    #     FetchContent_Populate(polyscope)
    #     message(STATUS "polyscope_SOURCE_DIR: ${polyscope_SOURCE_DIR}")
    #     message(STATUS "polyscope_BINARY_DIR: ${polyscope_BINARY_DIR}")
    #     add_subdirectory(${polyscope_SOURCE_DIR} ${polyscope_BINARY_DIR})
    # endif()
    # FetchContent_MakeAvailable(polyscope)
endif()

# if (NOT TARGET ipc::toolkit)
#     set(IPC_TOOLKIT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
#     set(IPC_TOOLKIT_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
#     set(IPC_TOOLKIT_WITH_CUDA OFF CACHE BOOL "" FORCE)
#     set(IPC_TOOLKIT_WITH_RATIONAL_INTERSECTION OFF CACHE BOOL "" FORCE)
#     set(IPC_TOOLKIT_WITH_ROBIN_MAP OFF CACHE BOOL "" FORCE)
#     set(IPC_TOOLKIT_WITH_ABSEIL OFF CACHE BOOL "" FORCE)
#     set(IPC_TOOLKIT_WITH_FILIB OFF CACHE BOOL "" FORCE)
#     set(IPC_TOOLKIT_WITH_SIMD OFF CACHE BOOL "" FORCE)
#     set(IPC_TOOLKIT_WITH_CODE_COVERAGE OFF CACHE BOOL "" FORCE)
#     add_subdirectory(${EXT_PATH}/ipc-toolkit)
# endif()

# if (NOT TARGET amgcl)
#     add_subdirectory(${EXT_PATH}/amgcl)
# endif()

# if (NOT TARGET TBB::tbb)
#     set(TBB_TEST OFF CACHE BOOL "" FORCE)
#     add_subdirectory(${EXT_PATH}/oneTBB)
# endif()

# lib
file(GLOB_RECURSE LUISA_COMPUTE_SOLVER_LIB_SOURCES CONFIGURE_DEPENDS Solver/*.cpp Solver/*.cc)
file(GLOB_RECURSE LUISA_COMPUTE_SOLVER_LIB_HEADERS CONFIGURE_DEPENDS Solver/*.h)
add_library(luisa-compute-solver-lib STATIC ${LUISA_COMPUTE_SOLVER_LIB_SOURCES} ${LUISA_COMPUTE_SOLVER_LIB_HEADERS})
# set_target_properties(luisa-compute-solver-lib PROPERTIES PUBLIC_HEADER "${LUISA_COMPUTE_SOLVER_LIB_HEADERS}")
# target_compile_definitions(luisa-compute-solver-lib PRIVATE LS_DLL_EXPORTS=1)
target_include_directories(luisa-compute-solver-lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Solver>)
# target_include_directories(luisa-compute-solver-lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Solver/Core>)
# target_include_directories(luisa-compute-solver-lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Solver/Energy>)
# target_include_directories(luisa-compute-solver-lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Solver/Initializer>)
# target_include_directories(luisa-compute-solver-lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Solver/LinearSolver>)
# target_include_directories(luisa-compute-solver-lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Solver/MeshOperation>)
# target_include_directories(luisa-compute-solver-lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Solver/SimulationCore>)
# target_include_directories(luisa-compute-solver-lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Solver/SimulationSolver>)
# target_include_directories(luisa-compute-solver-lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Solver/Utils>)
target_link_libraries(luisa-compute-solver-lib PUBLIC luisa::compute) #  ipc::toolkit

if(WIN32)
set(CMAKE_PREFIX_PATH   "C:/Applications/vcpkg/installed/x64-windows")
endif()

find_package(Eigen3 REQUIRED)
find_package(TBB CONFIG REQUIRED)
target_link_libraries(luisa-compute-solver-lib PUBLIC TBB::tbb Eigen3::Eigen)

set(LCSV_RESOURCE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Resources)
target_compile_definitions(luisa-compute-solver-lib PUBLIC LCSV_RESOURCE_PATH="${LCSV_RESOURCE_PATH}")

# test
# file(GLOB_RECURSE LUISA_SOLVER_TEST_SOURCES CONFIGURE_DEPENDS test/*.cpp)
# add_executable(luisa-solver-test ${LUISA_SOLVER_TEST_SOURCES})
# target_include_directories(luisa-solver-test PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test/_framework>)
# target_link_libraries(luisa-solver-test PRIVATE luisa::compute luisa-compute-solver-lib)
add_executable(test-xtypes UnitTest/test_xtypes.cpp)
target_link_libraries(test-xtypes PRIVATE luisa::compute luisa-compute-solver-lib)

# app
add_executable(app-simulation Application/app_simulation.cpp Application/app_simulation_demo_config.cpp Application/app_simulation_demo_config.h)
# add_executable(app-simulation Application/app_simulation_v2.cpp)
target_link_libraries(app-simulation PRIVATE 
    luisa::compute polyscope 
    luisa-compute-solver-lib)

# T

# file(GLOB_RECURSE SIMULATION CONFIGURE_DEPENDS Application/app_simulation.cpp)
# add_executable(app-simulation ${LUISA_SOLVER_APP_SOURCES})
