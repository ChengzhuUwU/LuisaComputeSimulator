set_allowedplats("windows")
set_allowedarchs("x64")
set_allowedmodes("debug", "release", "releasedbg")
-- windows flags
if (is_host("windows")) then 
    add_defines("NOMINMAX")
    add_defines("_GAMING_DESKTOP")
    add_defines("_CRT_SECURE_NO_WARNINGS")
    add_defines("_ENABLE_EXTENDED_ALIGNED_STORAGE")
    add_defines("_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR") -- for preventing std::mutex crash when lock
    if (is_mode("release")) then
        set_runtimes("MD")
    elseif (is_mode("asan")) then
        add_defines("_DISABLE_VECTOR_ANNOTATION")
    else
        set_runtimes("MDd")
    end
end

includes("ext")

option("dev", {default = true})

-- fixed config
set_languages("c++20")
add_rules("mode.debug", "mode.release", "mode.releasedbg")

-- dynamic config
if has_config("dev") then
    set_policy("build.ccache", true)

    add_rules("plugin.compile_commands.autoupdate", {lsp = "clangd", outputdir = "build"})

    set_warnings("all")

    if is_plat("windows") then
        set_runtimes("MD")
        add_cxflags("/permissive-", {tools = "cl"})
    end
end
-- add_requires("luisa-compute", "eigen", "tbb", "polyscope")
-- add_requires("luisa-compute[cuda]", "eigen", "tbb", "polyscope")

target("luisa-compute-solver-lib")
    add_rules("lc_basic_settings", {
        project_kind = "static",
        enable_exception = true
    })
    add_files("Solver/**.cpp", "Solver/**.cc")
    add_includedirs("Solver", {public = true})
    add_defines(format([[LCSV_RESOURCE_PATH="%s"]], path.unix(path.join(os.scriptdir(), "Resources"))), {public = true})
    add_deps("lc-dsl", "lc-runtime", "lc-backends-dummy", "lc-vstl", "eigen")
    set_pcxxheader("Solver/zzpch.h")

target("app-simulation")
    add_rules("lc_basic_settings", {
        project_kind = "binary",
        enable_exception = true
    })
    add_files("Application/*.cpp|app_test_features.cpp")

    add_deps("luisa-compute-solver-lib", "polyscope", "lc-yyjson")
    set_pcxxheader("Application/zzpch.h")