if (NOT DEFINED LCS_PYTHON_EXECUTABLE OR LCS_PYTHON_EXECUTABLE STREQUAL "")
    message(FATAL_ERROR "LCS_PYTHON_EXECUTABLE is required to build Python bindings (lcs_py). Configure with -D LCS_PYTHON_EXECUTABLE=/path/to/python")
endif()

set(PYBIND11_PYTHON_EXECUTABLE ${LCS_PYTHON_EXECUTABLE} CACHE FILEPATH "Python executable for pybind11")
message(STATUS "Using Python executable for bindings: ${PYBIND11_PYTHON_EXECUTABLE}")

pybind11_add_module(lcs_py python_bindings.cpp)

target_link_libraries(lcs_py PRIVATE
    luisa-compute-solver-lib
    luisa::compute
)
