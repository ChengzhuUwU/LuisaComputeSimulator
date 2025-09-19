set_xmakever("3.0.0")
add_rules("mode.release", "mode.debug", "mode.releasedbg")
set_languages("c++20")

includes("Solver")
includes("Application")

if is_os("windows") then
    add_cxxflags("/utf-8")
end