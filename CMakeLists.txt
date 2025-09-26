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
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_VERBOSE ON)
include(FetchContent)
if (NOT TARGET luisa::compute)
    FetchContent_Declare(
        luisa_compute
        GIT_REPOSITORY https://github.com/LuisaGroup/LuisaCompute.git # https://gitee.com/celeste-huohuo/LuisaCompute.git
        GIT_TAG next
        # GIT_SUBMODULES ""
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/ext/luisaCompute
    )
    FetchContent_Populate(luisa_compute)
    set(LuisaCompute_SOURCE_DIR ${EXT_PATH}/LuisaCompute)
    # set(LuisaCompute_SOURCE_DIR ${luisa_compute_SOURCE_DIR})
    message(STATUS "LuisaCompute source dir: ${LuisaCompute_SOURCE_DIR}")
    include(${LuisaCompute_SOURCE_DIR}/scripts/setup_output_dirs.cmake)

    # FetchContent_Declare(
    #     spdlog
    #     GIT_REPOSITORY https://github.com/LuisaGroup/spdlog.git
    #     SOURCE_DIR ${LuisaCompute_SOURCE_DIR}/src/ext/spdlog
    # )
    # FetchContent_Populate(spdlog)

    # FetchContent_Declare(
    #     imgui
    #     GIT_REPOSITORY https://github.com/ocornut/imgui.git
    #     GIT_TAG docking
    #     SOURCE_DIR ${LuisaCompute_SOURCE_DIR}/src/ext/imgui
    # )
    # FetchContent_Populate(imgui)

    # FetchContent_Declare(
    #     glfw
    #     GIT_REPOSITORY https://github.com/glfw/glfw.git
    #     SOURCE_DIR ${LuisaCompute_SOURCE_DIR}/src/ext/glfw
    # )
    # FetchContent_Populate(glfw)

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
    set(LUISA_COMPUTE_ENABLE_GUI OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_WAYLAND ON CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_FALLBACK OFF CACHE BOOL "" FORCE)
    add_subdirectory(${LuisaCompute_SOURCE_DIR})
endif ()

if (NOT TARGET polyscope)
    FetchContent_Declare(
        polyscope
        # https://github.com/nmwsharp/polyscope.git
        # https://gitee.com/celeste-huohuo/polyscope.git
        GIT_REPOSITORY https://gitee.com/celeste-huohuo/polyscope.git
        SOURCE_DIR ${EXT_PATH}/polyscope
    )
    set(POLYSCOPE_USE_EXTERNAL_GLFW ON CACHE BOOL "" FORCE)
    set(POLYSCOPE_BACKEND_OPENGL3_GLFW ON CACHE BOOL "" FORCE)
    set(POLYSCOPE_BACKEND_OPENGL_MOCK ON CACHE BOOL "" FORCE)
    FetchContent_Populate(polyscope)
    add_subdirectory(${EXT_PATH}/polyscope)
endif()

if (NOT TARGET libdispatch)
    FetchContent_Declare(
        libdispatch
        GIT_REPOSITORY https://github.com/MaxwellGengYF/libdispatch.git
        SOURCE_DIR ${EXT_PATH}/libdispatch
    )
    FetchContent_Populate(libdispatch)
endif()


option(LUISA_COMPUTE_SOLVER_USE_SYSTEM_PARALLEL_FOR "Enable system parallel_for" ON)
if (LUISA_COMPUTE_SOLVER_USE_SYSTEM_PARALLEL_FOR)
    target_compile_definitions(luisa-compute-solver-lib PUBLIC LUISA_COMPUTE_SOLVER_USE_SYSTEM_PARALLEL_FOR=1)
    find_package(TBB CONFIG)
    if (TBB_FOUND)
        message("Host parallel function use TBB")
        target_compile_definitions(luisa-compute-solver-lib PUBLIC LUISA_COMPUTE_SOLVER_ENABLE_TBB=1)
        target_link_libraries(luisa-compute-solver-lib PUBLIC TBB::tbb)
    else()
        # use libdispatch or TBB for parallel_for on non-Apple Unix systems if available
        if (UNIX AND NOT APPLE)
            # libdispatch
            find_library(DISPATCH_LIB dispatch)
            find_path(DISPATCH_INCLUDE_DIR dispatch/dispatch.h) # Apple use native libdispatch
            if (DISPATCH_LIB AND DISPATCH_INCLUDE_DIR)
                message("Host parallel function use package libdispatch")
                target_compile_definitions(luisa-compute-solver-lib PUBLIC LUISA_COMPUTE_SOLVER_ENABLE_LIBDISPATCH=1)
                target_link_libraries(luisa-compute-solver-lib PUBLIC ${DISPATCH_LIB})
                target_include_directories(luisa-compute-solver-lib PUBLIC ${DISPATCH_INCLUDE_DIR})
            endif ()
        else() 
            if (APPLE)
                # Use native libdispatch
                target_compile_definitions(luisa-compute-solver-lib PUBLIC LUISA_COMPUTE_SOLVER_ENABLE_LIBDISPATCH=1)
                message("Host parallel function use native libdispatch")
            else ()
                message(WARNING "TBB and libdispatch not found. Performance of the fallback backend may be affected.")
            endif ()
        endif()
    endif()
endif ()


find_package(Eigen3 REQUIRED)
# find_package(TBB CONFIG REQUIRED)
target_link_libraries(luisa-compute-solver-lib PUBLIC 
    Eigen3::Eigen)
if (USE_AMGCL_FOR_SIM)
    target_compile_definitions(luisa-compute-solver-lib PUBLIC USE_AMGCL_FOR_SIM)
    target_link_libraries(luisa-compute-solver-lib PUBLIC amgcl::amgcl)
endif()

set(LCSV_RESOURCE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Resources)
target_compile_definitions(luisa-compute-solver-lib PUBLIC LCSV_RESOURCE_PATH="${LCSV_RESOURCE_PATH}")

# test
add_executable(test-xtypes UnitTest/test_xtypes.cpp)
target_link_libraries(test-xtypes PRIVATE luisa::compute luisa-compute-solver-lib)

# app
add_executable(app-simulation Application/app_simulation.cpp Application/app_simulation_demo_config.cpp Application/app_simulation_demo_config.h)
target_link_libraries(app-simulation PRIVATE 
    luisa::compute 
    polyscope 
    luisa-compute-solver-lib)



add_executable(app-test-features Application/app_test_features.cpp)
target_link_libraries(app-test-features PRIVATE 
    luisa::compute 
    luisa-compute-solver-lib)
