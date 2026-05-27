set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_VERBOSE ON)
include(FetchContent)

#
# LuisaCompute
#
if (NOT TARGET luisa::compute)
    FetchContent_Declare(
        luisa_compute
        GIT_REPOSITORY https://github.com/LuisaGroup/LuisaCompute.git # https://gitee.com/celeste-huohuo/LuisaCompute.git
        GIT_TAG 8fda2e0268c09bc3051cde913a612836036397d9
        # GIT_TAG next
        # GIT_SUBMODULES ""
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/ext/luisaCompute
        GIT_SHALLOW TRUE
    )
    FetchContent_Populate(luisa_compute)
    set(LuisaCompute_SOURCE_DIR ${luisa_compute_SOURCE_DIR})
    message(STATUS "LuisaCompute source dir: ${LuisaCompute_SOURCE_DIR}")
    # include(${LuisaCompute_SOURCE_DIR}/scripts/setup_output_dirs.cmake)

    set(LUISA_COMPUTE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_DOWNLOAD_NVCOMP OFF CACHE BOOL "" FORCE)
    # Use the system STL on every path (plain cmake build, scikit-build wheel,
    # editable install). The previous SKBUILD-only override forced embedded STL
    # for `pip install -e .`, which produced a Metal backend that loaded but
    # failed device probing on macOS. Keeping a single setting means dev and
    # wheel builds share the same ABI as the C++ application.
    set(LUISA_COMPUTE_USE_SYSTEM_STL ON CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_RUST OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_REMOTE OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_GUI OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_WAYLAND ON CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_CPU OFF CACHE BOOL "" FORCE)
    if (LCS_BUILD_PYBINDINGS)
        # scikit-build-core defines SKBUILD for the top-level project, but the
        # embedded LuisaCompute project also treats SKBUILD as a request to build
        # and install its own Python package. We only need LuisaCompute as a C++
        # dependency for lcs_py here, so hide SKBUILD while adding that subproject.
        set(_LCS_SAVED_SKBUILD ${SKBUILD})
        set(SKBUILD OFF)
        add_subdirectory(${LuisaCompute_SOURCE_DIR})
        set(SKBUILD ${_LCS_SAVED_SKBUILD})
        unset(_LCS_SAVED_SKBUILD)
    else()
        add_subdirectory(${LuisaCompute_SOURCE_DIR})
    endif()
endif ()
target_link_libraries(luisa-compute-solver-lib PUBLIC luisa::compute) #  ipc::toolkit

message(STATUS "Use LUISA_COMPUTE_ENABLE_EXPERIMENTAL_CUDA_LLVM_CODEGEN : ${LUISA_COMPUTE_ENABLE_EXPERIMENTAL_CUDA_LLVM_CODEGEN} ")

#
# lcpp
#
if (NOT TARGET lcpp)
    FetchContent_Declare(
        lcpp
        GIT_REPOSITORY https://github.com/Ligo04/lc_parallel_primitive.git
        GIT_TAG 346650626098ba236297df152d2974f2c0618f44
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/ext/lcpp
    )
    set(LCPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Populate(lcpp)
    add_subdirectory(${EXT_PATH}/lcpp)
endif ()
target_link_libraries(luisa-compute-solver-lib PUBLIC lcpp)

message(STATUS "Use LUISA_COMPUTE_ENABLE_EXPERIMENTAL_CUDA_LLVM_CODEGEN : ${LUISA_COMPUTE_ENABLE_EXPERIMENTAL_CUDA_LLVM_CODEGEN} ")


#
# Eigen3
#
find_package(Eigen3 QUIET)
if (NOT TARGET Eigen3::Eigen)
    message(STATUS "Can not find eigen3 package, download...")
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        SOURCE_DIR ${EXT_PATH}/eigen
        GIT_SHALLOW TRUE
    )
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_LEAVE_TEST_IN_ALL_TARGET OFF CACHE BOOL "" FORCE)
    set(EIGEN_MPL2_ONLY ON CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_TEST_CXX11 OFF CACHE BOOL "" FORCE)
    set(EIGEN_TEST_COMPILER_FEATURES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(eigen)
else()
    message(STATUS "Use system Eigen3")
endif()
target_link_libraries(luisa-compute-solver-lib PUBLIC Eigen3::Eigen)

#
# Polyscope
#
if(LCS_ENABLE_GUI)
    if (NOT TARGET polyscope)
        FetchContent_Declare(
            polyscope
            # https://github.com/nmwsharp/polyscope.git
            # https://gitee.com/celeste-huohuo/polyscope.git
            GIT_REPOSITORY https://gitee.com/celeste-huohuo/polyscope.git
            SOURCE_DIR ${EXT_PATH}/polyscope
            GIT_SHALLOW TRUE
        )
        set(POLYSCOPE_USE_EXTERNAL_GLFW ON CACHE BOOL "" FORCE)
        set(POLYSCOPE_BACKEND_OPENGL3_GLFW ON CACHE BOOL "" FORCE)
        set(POLYSCOPE_BACKEND_OPENGL_MOCK ON CACHE BOOL "" FORCE)
        FetchContent_Populate(polyscope)
        add_subdirectory(${EXT_PATH}/polyscope)
    endif()
endif()


#
# TBB or libdispatch
#

if (LCS_BUILD_PYBINDINGS)

    set(PYBIND11_FINDPYTHON ON CACHE BOOL "Force pybind11 to use modern FindPython" FORCE)
    set(Python_EXECUTABLE ${LCS_PYTHON_EXECUTABLE} CACHE FILEPATH "Python executable" FORCE)
    set(Python3_EXECUTABLE ${LCS_PYTHON_EXECUTABLE} CACHE FILEPATH "Python3 executable" FORCE)
    set(PYTHON_EXECUTABLE ${LCS_PYTHON_EXECUTABLE} CACHE FILEPATH "Legacy Python executable" FORCE)
    set(PYBIND11_PYTHON_EXECUTABLE ${LCS_PYTHON_EXECUTABLE} CACHE FILEPATH "Python executable for pybind11" FORCE)
    message(STATUS "Python binding executable path: ${PYBIND11_PYTHON_EXECUTABLE}")

    # find_package(pybind11 QUIET)
    # if(pybind11_FOUND)
    #     message(STATUS "Found pybind11: ${pybind11_DIR}")
    # else()
        # message(STATUS "pybind11 not found; fetching pybind11 via FetchContent...")
        # Using a specific commit of pybind11 to ensure stability
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.11.1
            SOURCE_DIR ${EXT_PATH}/pybind11
        )
        FetchContent_MakeAvailable(pybind11)
    # endif()
endif()

#
# TBB or libdispatch
#
if (LCS_USE_SYSTEM_PARALLEL_FOR)
    target_compile_definitions(luisa-compute-solver-lib PRIVATE LCS_USE_SYSTEM_PARALLEL_FOR=1)
    find_package(TBB CONFIG QUIET)
    if (TBB_FOUND AND TARGET TBB::tbb)
        message(STATUS "Host parallel function use TBB")
        target_compile_definitions(luisa-compute-solver-lib PRIVATE LUISA_COMPUTE_SOLVER_ENABLE_TBB=1)
        target_link_libraries(luisa-compute-solver-lib PRIVATE TBB::tbb)
    else()
        # use libdispatch or TBB for parallel_for on non-Apple Unix systems if available
        if (UNIX AND NOT APPLE)

            if (NOT TARGET libdispatch)
                FetchContent_Declare(
                    libdispatch
                    GIT_REPOSITORY https://github.com/MaxwellGengYF/libdispatch.git
                    SOURCE_DIR ${EXT_PATH}/libdispatch
                )
                FetchContent_Populate(libdispatch)
            endif()

            # libdispatch
            find_library(DISPATCH_LIB dispatch)
            find_path(DISPATCH_INCLUDE_DIR dispatch/dispatch.h) # Apple use native libdispatch
            if (DISPATCH_LIB AND DISPATCH_INCLUDE_DIR)
                message(STATUS "Host parallel function use package libdispatch")
                target_compile_definitions(luisa-compute-solver-lib PRIVATE LUISA_COMPUTE_SOLVER_ENABLE_LIBDISPATCH=1)
                target_link_libraries(luisa-compute-solver-lib PRIVATE ${DISPATCH_LIB})
                target_include_directories(luisa-compute-solver-lib PRIVATE ${DISPATCH_INCLUDE_DIR})
            endif ()
        else() # Windows or Apple
            if (APPLE)
                # Use native libdispatch
                target_compile_definitions(luisa-compute-solver-lib PRIVATE LUISA_COMPUTE_SOLVER_ENABLE_LIBDISPATCH=1)
                message(STATUS "Host parallel function use native libdispatch")
            else ()
                target_compile_definitions(luisa-compute-solver-lib PRIVATE LUISA_COMPUTE_SOLVER_USE_LUISA_FIBER=1)
                message(STATUS "Host parallel function use luisa::fiber")
            endif ()
        endif()
    endif()
else()
    target_compile_definitions(luisa-compute-solver-lib PRIVATE LUISA_COMPUTE_SOLVER_USE_LUISA_FIBER=1)
    message(STATUS "Host parallel function use luisa::fiber")
endif()




# if (USE_AMGCL_FOR_SIM)
#     target_compile_definitions(luisa-compute-solver-lib PRIVATE USE_AMGCL_FOR_SIM)
#     target_link_libraries(luisa-compute-solver-lib PRIVATE amgcl::amgcl)
# endif()


#
# yyjson
#
if (NOT TARGET yyjson)
    find_package(yyjson QUIET)
    if (yyjson_FOUND)
        target_link_libraries(luisa-compute-solver-lib PRIVATE yyjson::yyjson)
    else ()
        add_library(luisa-compute-solver-ext-yyjson-interface INTERFACE)
        target_include_directories(luisa-compute-solver-ext-yyjson-interface INTERFACE
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/embed_yyjson>)
        add_library(luisa-compute-solver-ext-yyjson OBJECT embed_yyjson/yyjson.c embed_yyjson/yyjson.h)
        target_link_libraries(luisa-compute-solver-ext-yyjson PUBLIC luisa-compute-solver-ext-yyjson-interface)
        if (UNIX AND NOT APPLE)
            set_target_properties(luisa-compute-solver-ext-yyjson PROPERTIES POSITION_INDEPENDENT_CODE ON)
        endif ()
    endif ()
endif()
