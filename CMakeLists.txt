cmake_minimum_required(VERSION 3.26)

set(CMAKE_CXX_STANDARD 20)

project(LuisaComputeSolver)

# set(EXT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ext)
include(FetchContent)
if (NOT TARGET luisa::compute)
    # download LuisaCompute from github.com/LuisaGroup/LuisaCompute.git to the build directory

    FetchContent_Populate(luisa_compute
            # GIT_REPOSITORY "https://github.com/LuisaGroup/LuisaCompute.git"
            # GIT_TAG stable)
            GIT_REPOSITORY "https://gitee.com/celeste-huohuo/LuisaCompute.git"
            GIT_TAG stable)
    set(LuisaCompute_SOURCE_DIR ${luisa_compute_SOURCE_DIR})
    message(STATUS "LuisaCompute source dir: ${LuisaCompute_SOURCE_DIR}")

    include(${LuisaCompute_SOURCE_DIR}/scripts/setup_output_dirs.cmake)

    # add LuisaCompute as a subdirectory
    set(LUISA_COMPUTE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_DOWNLOAD_NVCOMP OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_USE_SYSTEM_STL ON CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_RUST OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_CPU OFF CACHE BOOL "" FORCE)
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

# Remember to set the following variables in polyscope's CMakeLists.txt
# if("${POLYSCOPE_BACKEND_OPENGL3_GLFW}")
#   ## Glad
#   add_subdirectory(glad)
#   ## GLFW
#   if (POLYSCOPE_USE_EXTERNAL_GLFW)
#     find_package(glfw3 REQUIRED)
#   else()
#     set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
#     set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
#     set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
#     set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
#     add_subdirectory(glfw)
#   endif()
# endif()

    FetchContent_Declare(
            polyscope
            # GIT_REPOSITORY https://gitee.com/yunchengliu/polyscope.git
            # GIT_TAG 2cd38b7496a1e8ff79f615920de2980295bedef7
            GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
            GIT_TAG 2cd38b7496a1e8ff79f615920de2980295bedef7
    )
    FetchContent_GetProperties(polyscope)
    if(NOT polyscope_POPULATED)
        # Fetch the content using previously declared details
        FetchContent_Populate(polyscope)
        message(STATUS "polyscope_SOURCE_DIR: ${polyscope_SOURCE_DIR}")
        message(STATUS "polyscope_BINARY_DIR: ${polyscope_BINARY_DIR}")
        add_subdirectory(${polyscope_SOURCE_DIR} ${polyscope_BINARY_DIR})
    endif()
    FetchContent_MakeAvailable(polyscope)
endif()



# lib
file(GLOB_RECURSE LUISA_COMPUTE_SOLVER_LIB_SOURCES CONFIGURE_DEPENDS Solver/*.cpp Solver/*.cc)
file(GLOB_RECURSE LUISA_COMPUTE_SOLVER_LIB_HEADERS CONFIGURE_DEPENDS Solver/*.h)
add_library(luisa-compute-solver-lib SHARED ${LUISA_COMPUTE_SOLVER_LIB_SOURCES})
set_target_properties(luisa-compute-solver-lib PROPERTIES PUBLIC_HEADER "${LUISA_COMPUTE_SOLVER_LIB_HEADERS}")
target_compile_definitions(luisa-compute-solver-lib PRIVATE LS_DLL_EXPORTS=1)
target_include_directories(luisa-compute-solver-lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Solver>)
# target_include_directories(luisa-compute-solver-lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Solver/Core>)
# target_include_directories(luisa-compute-solver-lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Solver/Utils>)
# target_include_directories(luisa-compute-solver-lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Solver/Initializer>)
# target_include_directories(luisa-compute-solver-lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Solver/MeshOperation>)
# target_include_directories(luisa-compute-solver-lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Solver/PrecondConjugateGradient>)
target_link_libraries(luisa-compute-solver-lib PUBLIC luisa::compute)

find_package(TBB REQUIRED)
target_link_libraries(luisa-compute-solver-lib PUBLIC TBB::tbb)

# test
# file(GLOB_RECURSE LUISA_SOLVER_TEST_SOURCES CONFIGURE_DEPENDS test/*.cpp)
# add_executable(luisa-solver-test ${LUISA_SOLVER_TEST_SOURCES})
# target_include_directories(luisa-solver-test PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test/_framework>)
# target_link_libraries(luisa-solver-test PRIVATE luisa::compute luisa-compute-solver-lib)

# app
add_executable(app-simulation Application/app_simulation.cpp)
# add_executable(app-simulation Application/app_simulation_v2.cpp)
target_link_libraries(app-simulation PRIVATE 
    luisa::compute polyscope 
    luisa-compute-solver-lib)

# file(GLOB_RECURSE SIMULATION CONFIGURE_DEPENDS Application/app_simulation.cpp)
# add_executable(app-simulation ${LUISA_SOLVER_APP_SOURCES})
