# Normalize pybind11-stubgen output into a stub package layout.
#
# Different versions of pybind11-stubgen emit either:
#   ${GEN_DIR}/lcs_py.pyi              (single-file mode, common case)
# or
#   ${GEN_DIR}/lcs_py/__init__.pyi     (package mode)
#
# We always end up with ${GEN_DIR}/lcs_py/__init__.pyi so the package layout
# (with py.typed and the hand-maintained lcs/ subclass overrides) stays valid.
#
# Inputs (set via cmake -D):
#   GEN_DIR  Absolute path of the directory pybind11-stubgen wrote into.

if (NOT DEFINED GEN_DIR)
    message(FATAL_ERROR "normalize_stub.cmake: GEN_DIR is not set")
endif()

set(_single_file "${GEN_DIR}/lcs_py.pyi")
set(_pkg_init    "${GEN_DIR}/lcs_py/__init__.pyi")

if (EXISTS "${_single_file}")
    file(MAKE_DIRECTORY "${GEN_DIR}/lcs_py")
    # rename overwrites the destination on POSIX/Windows, but be explicit so
    # the previous package-mode output (if any) doesn't linger.
    file(REMOVE "${_pkg_init}")
    file(RENAME "${_single_file}" "${_pkg_init}")
    message(STATUS "Normalized stub: ${_single_file} -> ${_pkg_init}")
elseif (EXISTS "${_pkg_init}")
    message(STATUS "Stub already in package form: ${_pkg_init}")
else()
    message(FATAL_ERROR
        "normalize_stub.cmake: pybind11-stubgen produced neither "
        "'${_single_file}' nor '${_pkg_init}'. "
        "Check that pybind11-stubgen is installed in LCS_PYTHON_EXECUTABLE."
    )
endif()

# ---------------------------------------------------------------------------
# Translate C++ wrapper type names back to their Python class names.
# pybind11-stubgen resolves the underlying C++ type for method return values
# (e.g. PySceneParams) which don't match the Python name registered with
# py::class_<> (e.g. SceneParams).  This replacement keeps the stubs
# consistent so that IDE type-checkers correctly resolve the types.
# ---------------------------------------------------------------------------
file(READ "${_pkg_init}" _stub_content)
string(REGEX REPLACE
    "PySceneParams"
    "SceneParams"
    _stub_content "${_stub_content}"
)
file(WRITE "${_pkg_init}" "${_stub_content}")
