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

# RPATH so lcs_py.so finds libluisa-*.dylib next to it inside the installed
# wheel (or in the build/bin layout for editable installs).
if (APPLE)
    set(_LCS_PYTHON_ORIGIN_RPATH "@loader_path")
elseif (UNIX)
    set(_LCS_PYTHON_ORIGIN_RPATH "$ORIGIN")
else()
    set(_LCS_PYTHON_ORIGIN_RPATH "")
endif()
if (_LCS_PYTHON_ORIGIN_RPATH)
    set_target_properties(lcs_py PROPERTIES
        INSTALL_RPATH
            "${_LCS_PYTHON_ORIGIN_RPATH}/lib;${_LCS_PYTHON_ORIGIN_RPATH}/bin;${_LCS_PYTHON_ORIGIN_RPATH}"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()
unset(_LCS_PYTHON_ORIGIN_RPATH)

install(TARGETS lcs_py
    LIBRARY DESTINATION .
)

install(FILES ${CMAKE_SOURCE_DIR}/PythonBindings/python/lcs_py/__init__.pyi
    DESTINATION .
    RENAME lcs_py.pyi
)

install(FILES
    ${CMAKE_SOURCE_DIR}/PythonBindings/python/lcs_gui/__init__.py
    DESTINATION lcs_gui
)
install(FILES
    ${CMAKE_SOURCE_DIR}/PythonBindings/tests/utils/polyscope_gui.py
    DESTINATION lcs_gui
)

# ---------------------------------------------------------------------------
# Bundle LuisaCompute runtime into the wheel
# ---------------------------------------------------------------------------
#
# In editable installs the .so stays in build/.../bin/ alongside the
# LuisaCompute libraries, so @loader_path resolves them at import time.
# For a real wheel (`pip install .` or `pip wheel .`) those libraries
# are not in the wheel by default, and the import would fail with
#   "Library not loaded: @rpath/libluisa-core.dylib".
# Install them next to lcs_py so the same @loader_path RPATH picks them up.
if (SKBUILD)
    set(_LCS_LUISA_RUNTIME_TARGETS
        luisa-compute-core
        luisa-compute-ast
        luisa-compute-dsl
        luisa-compute-runtime
        luisa-compute-xir
        luisa-compute-osl
    )
    foreach(_lcs_t IN LISTS _LCS_LUISA_RUNTIME_TARGETS)
        if (TARGET ${_lcs_t})
            install(TARGETS ${_lcs_t}
                LIBRARY DESTINATION .
                RUNTIME DESTINATION .
            )
        endif()
    endforeach()
    unset(_lcs_t)
    unset(_LCS_LUISA_RUNTIME_TARGETS)

    # Backend modules (loaded via dlopen at runtime). Only the targets that
    # actually exist for the current LUISA_COMPUTE_ENABLE_* settings will be
    # installed.
    set(_LCS_LUISA_BACKEND_NAMES metal cuda dx vulkan fallback)
    foreach(_lcs_b IN LISTS _LCS_LUISA_BACKEND_NAMES)
        if (TARGET luisa-compute-backend-${_lcs_b})
            install(TARGETS luisa-compute-backend-${_lcs_b}
                LIBRARY DESTINATION .
                RUNTIME DESTINATION .
            )
        endif()
    endforeach()
    unset(_lcs_b)
    unset(_LCS_LUISA_BACKEND_NAMES)

    if (TARGET luisa-compute-validation-layer)
        install(TARGETS luisa-compute-validation-layer
            LIBRARY DESTINATION .
            RUNTIME DESTINATION .
        )
    endif()
endif()

# ---------------------------------------------------------------------------
# pybind11 stub generation
# ---------------------------------------------------------------------------
#
# The lcs_py/__init__.pyi stub package under stubPath (PythonBindings/python/)
# ensures Pyright resolves types with highest priority, bypassing the
# lcs_py.cpython-*.so in site-packages (which carries no type information).
#
# Regenerate stubs after C++ binding changes with:
#   cmake --build build --target stubs
#
# Requires pybind11-stubgen on LCS_PYTHON_EXECUTABLE. Install via:
#   pip install -e .[dev]   # in your project venv
add_custom_target(stubs DEPENDS lcs_py
    COMMAND ${CMAKE_COMMAND} -E env
        "PYTHONPATH=$<TARGET_FILE_DIR:lcs_py>"
        ${LCS_PYTHON_EXECUTABLE} -m pybind11_stubgen lcs_py
        -o ${CMAKE_SOURCE_DIR}/PythonBindings/python/
    COMMAND ${CMAKE_COMMAND}
        -DGEN_DIR=${CMAKE_SOURCE_DIR}/PythonBindings/python
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/normalize_stub.cmake
    COMMENT "Generating pybind11 stubs and syncing to lcs_py/__init__.pyi"
)
