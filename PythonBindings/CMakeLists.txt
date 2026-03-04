if (NOT DEFINED LCS_PYTHON_EXECUTABLE OR LCS_PYTHON_EXECUTABLE STREQUAL "")
    message(FATAL_ERROR "LCS_PYTHON_EXECUTABLE is required to build Python bindings (lcs_py). Configure with -D LCS_PYTHON_EXECUTABLE=/path/to/python")
endif()

pybind11_add_module(lcs_py python_bindings.cpp)

target_link_libraries(lcs_py PRIVATE
    luisa-compute-solver-lib
    luisa::compute
)
