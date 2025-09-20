set_allowedplats("windows")
set_allowedarchs("x64")
set_allowedmodes("debug", "release", "releasedbg")

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

add_requires("luisa-compute", "eigen", "tbb", "polyscope")
-- add_requires("luisa-compute[cuda]", "eigen", "tbb", "polyscope")

target("luisa-compute-solver-lib")
    set_kind("static")
    add_files("Solver/**.cpp", "Solver/**.cc")
    add_includedirs("Solver", {public = true})

    add_defines(format([[LCSV_RESOURCE_PATH="%s"]], path.unix(path.join(os.scriptdir(), "Resources"))), {public = true})
    add_packages("luisa-compute", "tbb")
    add_packages("eigen", {public = true})

target("app-simulation")
    set_kind("binary")
    add_rules("luisa-compute-runtime")
    add_files("Application/*.cpp|app_test_features.cpp")

    add_deps("luisa-compute-solver-lib")

rule("luisa-compute-runtime")
    on_config(function (target)
        if target:is_binary() then
            target:add("packages", "luisa-compute")
            target:add("runargs", path.join(target:pkg("luisa-compute"):installdir(), "bin"))
            -- os.vcp(path.join(target:pkg("luisa-compute"):installdir(), "bin/*.dll"), target:targetdir())
        end
    end)

package("luisa-compute")
    set_homepage("https://luisa-render.com/")
    set_description("High-Performance Rendering Framework on Stream Architectures")
    set_license("Apache-2.0")

    add_urls("https://github.com/LuisaGroup/LuisaCompute.git", {submodules = false})
    add_versions("2025.09.19", "5a1cbcc861ba413e6243e70e19bb188f3388302d")

    add_configs("cuda", {description = "Enable CUDA backend", default = false, type = "boolean"})
    add_configs("vulkan", {description = "Enable Vulkan backend", default = false, type = "boolean"})
    add_configs("cpu", {description = "Enable CPU backend", default = false, type = "boolean"})
    add_configs("gui", {description = "Enable GUI support", default = false, type = "boolean"})
    add_configs("shared", {description = "Build shared library.", default = true, type = "boolean", readonly = true})

    if is_host("windows") then
        set_policy("platform.longpaths", true)
    end

    add_includedirs("include", "include/luisa/ext")

    add_defines(
        "LUISA_USE_SYSTEM_SPDLOG=1",
        "LUISA_USE_SYSTEM_XXHASH=1",
        "LUISA_USE_SYSTEM_MAGIC_ENUM=1",
        "LUISA_USE_SYSTEM_STL=1",
        "LUISA_USE_SYSTEM_REPROC=1",
        "LUISA_USE_SYSTEM_YYJSON=1",
        "MARL_USE_SYSTEM_STL=1",
        "LUISA_USE_SYSTEM_MARL=1",
        "LUISA_USE_SYSTEM_LMDB=1",

        "LUISA_ENABLE_DSL=1",
        "LUISA_ENABLE_XIR=1"
    )
    if is_plat("windows") then
        add_defines("LUISA_PLATFORM_WINDOWS=1", "_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR")
    elseif is_plat("macosx") then
        add_defines("LUISA_PLATFORM_APPLE=1")
    else
        add_defines("LUISA_PLATFORM_UNIX=1")
    end

    add_deps("cmake", "pkgconf")
    add_deps("spdlog", {configs = {header_only = false, fmt_external = true}})
    add_deps("lmdb", "reproc", "xxhash", "yyjson", "magic_enum", "marl", "stb") -- TODO: half

    on_check(function (package)
        assert(package:is_arch64(), "package(luisa-compute) only support 64 bit")
    end)

    on_load(function (package)
        if package:config("gui") then
            package:add("deps", "glfw")
            package:add("defines", "LUISA_USE_SYSTEM_GLFW=1")
        end
        if package:config("cuda") then
            package:add("deps", "cuda")
        end
        if package:config("vulkan") then
            package:add("deps", "vulkansdk", "volk")
            package:add("defines", "LUISA_USE_SYSTEM_VULKAN=1")
        end
    end)

    on_install("windows|x64", "macosx", function (package)
        if package:has_tool("cxx", "cl") then
            package:add("cxflags", "/Zc:preprocessor", "/Zc:__cplusplus")
        end

        local configs = {
            "-DLUISA_COMPUTE_ENABLE_SCCACHE=OFF",
            "-DLUISA_COMPUTE_BUILD_TESTS=OFF",
            "-DLUISA_COMPUTE_ENABLE_UNITY_BUILD=ON",

            "-DLUISA_COMPUTE_USE_SYSTEM_LIBS=ON",

            "-DLUISA_COMPUTE_ENABLE_RUST=OFF",
            "-DLUISA_COMPUTE_ENABLE_REMOTE=OFF",
        }
        table.insert(configs, "-DCMAKE_BUILD_TYPE=" .. (package:is_debug() and "Debug" or "Release"))
        table.insert(configs, "-DBUILD_SHARED_LIBS=" .. (package:config("shared") and "ON" or "OFF"))
        table.insert(configs, "-DLUISA_COMPUTE_ENABLE_LTO=" .. (package:config("lto") and "ON" or "OFF"))
        table.insert(configs, "-DLUISA_COMPUTE_ENABLE_SANITIZERS=" .. (package:config("asan") and "ON" or "OFF"))
        if package:is_plat("windows") and package:is_debug() then
            table.insert(configs, "-DCMAKE_COMPILE_PDB_OUTPUT_DIRECTORY=")
        end

        table.insert(configs, "-DLUISA_COMPUTE_ENABLE_CUDA=" .. (package:config("cuda") and "ON" or "OFF"))
        table.insert(configs, "-DLUISA_COMPUTE_ENABLE_VULKAN=" .. (package:config("vulkan") and "ON" or "OFF"))
        table.insert(configs, "-DLUISA_COMPUTE_ENABLE_CPU=" .. (package:config("cpu") and "ON" or "OFF"))
        table.insert(configs, "-DLUISA_COMPUTE_ENABLE_GUI=" .. (package:config("gui") and "ON" or "OFF"))

        os.vcp(package:dep("stb"):installdir("include/stb"), "src/ext/stb/")
        import("package.tools.cmake").install(package, configs)

        if package:is_plat("windows") and package:is_debug() then
            os.vcp("build/lib/*.pdb", package:installdir("bin"))
        end
    end)

    on_test(function (package)
        assert(package:check_cxxsnippets({test = [[
            #include <luisa/luisa-compute.h>
            #include <luisa/dsl/sugar.h>

            void test(int argc, char *argv[]) {
                luisa::compute::Context context{argv[0]};
                luisa::compute::Device device = context.create_device("cuda");
                luisa::compute::Stream stream = device.create_stream();
            }
        ]]}, {configs = {languages = "c++20"}}))
    end)
