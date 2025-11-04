set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_VERBOSE ON)
include(FetchContent)

if (NOT TARGET luisa::compute)
    FetchContent_Declare(
        luisa_compute
        GIT_REPOSITORY https://github.com/LuisaGroup/LuisaCompute.git # https://gitee.com/celeste-huohuo/LuisaCompute.git
        GIT_TAG 0a8ea6c33573e5f099a9278b661bbe195cbe0380
        # GIT_TAG next
        # GIT_SUBMODULES ""
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/ext/luisaCompute
    )
    FetchContent_Populate(luisa_compute)
    set(LuisaCompute_SOURCE_DIR ${luisa_compute_SOURCE_DIR})
    message(STATUS "LuisaCompute source dir: ${LuisaCompute_SOURCE_DIR}")
    # include(${LuisaCompute_SOURCE_DIR}/scripts/setup_output_dirs.cmake)

    # set(LUISA_COMPUTE_ENABLE_CUDA ON CACHE BOOL "" FORCE)
    # set(LUISA_COMPUTE_ENABLE_METAL ON CACHE BOOL "" FORCE)
    # set(LUISA_COMPUTE_ENABLE_VULKAN OFF CACHE BOOL "" FORCE)
    # set(LUISA_COMPUTE_ENABLE_DX OFF CACHE BOOL "" FORCE)
    # set(LUISA_COMPUTE_ENABLE_FALLBACK OFF CACHE BOOL "" FORCE)

    set(LUISA_COMPUTE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_DOWNLOAD_NVCOMP OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_USE_SYSTEM_STL ON CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_RUST OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_REMOTE OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_GUI OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_WAYLAND ON CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_CPU OFF CACHE BOOL "" FORCE)
    add_subdirectory(${LuisaCompute_SOURCE_DIR})
endif ()
target_link_libraries(luisa-compute-solver-lib PUBLIC luisa::compute) #  ipc::toolkit

find_package(Eigen3 QUIET)
if (NOT TARGET Eigen3::Eigen)
    message(STATUS "Can not find eigen3 package, download...")
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        SOURCE_DIR ${EXT_PATH}/eigen
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


if(LCS_ENABLE_GUI)
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
endif()


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



# # yyjson
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