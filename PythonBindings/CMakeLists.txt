if (NOT DEFINED LCS_PYTHON_EXECUTABLE OR LCS_PYTHON_EXECUTABLE STREQUAL "")
    message(FATAL_ERROR "LCS_PYTHON_EXECUTABLE is required to build Python bindings (lcs_py). Configure with -D LCS_PYTHON_EXECUTABLE=/path/to/python")
endif()

pybind11_add_module(lcs_py
    src/python_bindings.cpp
    ../Application/app_simulation_demo_config.cpp
)

target_include_directories(lcs_py PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../Application
)

target_link_libraries(lcs_py PRIVATE
    luisa-compute-solver-lib
    luisa-compute-solver-ext-yyjson
    luisa::compute
)
