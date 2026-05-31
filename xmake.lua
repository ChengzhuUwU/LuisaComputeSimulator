set_allowedplats("windows", "macosx", "linux")
set_allowedarchs("x64", "arm64", "x86_64")
set_allowedmodes("debug", "release", "releasedbg")
set_targetdir(path.join(os.projectdir(), "build", "bin"))

local function cfg_bool(default, ...)
    local keys = {...}
    for _, key in ipairs(keys) do
        local value = get_config(key)
        if value ~= nil then
            if type(value) == "boolean" then
                return value
            end
            if type(value) == "number" then
                return value ~= 0
            end
            if type(value) == "string" then
                local v = value:lower()
                if v == "y" or v == "yes" or v == "true" or v == "on" or v == "1" then
                    return true
                end
                if v == "n" or v == "no" or v == "false" or v == "off" or v == "0" then
                    return false
                end
            end
        end
    end
    return default
end

option("lcs_enable_gui")
    set_default(false)
    set_showmenu(true)
    set_description("Enable GUI support (polyscope)")
option_end()

option("lcs_enable_test")
    set_default(false)
    set_showmenu(true)
    set_description("Enable unit tests")
option_end()

option("lcs_build_pybindings")
    set_default(false)
    set_showmenu(true)
    set_description("Build Python bindings (lcs_py)")
option_end()

option("lcs_python_executable")
    set_default("")
    set_showmenu(true)
    set_description("Path to Python executable for building lcs_py")
option_end()



option("lc_cuda_backend")
    set_default(true)
    set_showmenu(true)
    set_description("Enable CUDA backend")
option_end()

option("lc_metal_backend")
    set_default(true)
    set_showmenu(true)
    set_description("Enable Metal backend")
option_end()

option("lc_vk_backend")
    set_default(false)
    set_showmenu(true)
    set_description("Enable Vulkan backend")
option_end()

option("lc_dx_backend")
    set_default(false)
    set_showmenu(true)
    set_description("Enable DirectX backend")
option_end()

option("lc_fallback_backend")
    set_default(false)
    set_showmenu(true)
    set_description("Enable Fallback backend")
option_end()

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
    add_deps("lc-dsl", "lc-runtime", "lc-backends-dummy", "lc-vstl", "eigen", "lcpp")
    set_pcxxheader("Solver/zzpch.h")

target("app_simulation")
    add_rules("lc_basic_settings", {
        project_kind = "binary",
        enable_exception = true
    })
    add_files("Application/*.cpp|app_integration.cpp")
    add_deps("luisa-compute-solver-lib", "lc-yyjson")
    if cfg_bool(false, "lcs_enable_gui") then
        add_deps("polyscope")
        add_defines("SIMULATION_APP_USE_GUI=1")
    end
    set_pcxxheader("Application/zzpch.h")

target("app_integration")
    add_rules("lc_basic_settings", {
        project_kind = "binary",
        enable_exception = true
    })
    add_files("Application/app_integration.cpp")
    add_deps("luisa-compute-solver-lib")
    set_pcxxheader("Application/zzpch.h")

if cfg_bool(false, "lcs_enable_test") then
    target("test_xtypes")
        add_rules("lc_basic_settings", {project_kind = "binary", enable_exception = true})
        add_files("UnitTest/test_xtypes.cpp")
        add_deps("luisa-compute-solver-lib")

    target("test_lc_features")
        add_rules("lc_basic_settings", {project_kind = "binary", enable_exception = true})
        add_files("UnitTest/test_lc_features.cpp")
        add_deps("luisa-compute-solver-lib")

    target("test_lbvh_refit")
        add_rules("lc_basic_settings", {project_kind = "binary", enable_exception = true})
        add_files("UnitTest/test_lbvh_refit.cpp")
        add_deps("luisa-compute-solver-lib")

    target("test_jit")
        add_rules("lc_basic_settings", {project_kind = "binary", enable_exception = true})
        add_files("UnitTest/test_jit.cpp")
        add_deps("luisa-compute-solver-lib")

    target("test_host_functions")
        add_rules("lc_basic_settings", {project_kind = "binary", enable_exception = true})
        add_files("UnitTest/test_host_functions.cpp")
        add_deps("luisa-compute-solver-lib")

    target("test_abd_energy")
        add_rules("lc_basic_settings", {project_kind = "binary", enable_exception = true})
        add_files("UnitTest/test_abd_energy.cpp")
        add_deps("luisa-compute-solver-lib")

    target("test_gradient_hessian")
        add_rules("lc_basic_settings", {project_kind = "binary", enable_exception = true})
        add_files("UnitTest/test_gradient_hessian.cpp")
        add_deps("luisa-compute-solver-lib")

    target("test_newton_solver_integration")
        add_rules("lc_basic_settings", {project_kind = "binary", enable_exception = true})
        add_files("UnitTest/test_newton_solver_integration.cpp")
        add_deps("luisa-compute-solver-lib")

    target("test_profiling")
        add_rules("lc_basic_settings", {project_kind = "binary", enable_exception = true})
        add_files("UnitTest/test_profiling.cpp")
        add_deps("luisa-compute-solver-lib")

    -- ===========================================================================
    -- New modular unit tests (IPC framework testing suite)
    -- ===========================================================================

    target("test_lbvh")
        add_rules("lc_basic_settings", {project_kind = "binary", enable_exception = true})
        add_files("UnitTest/test_lbvh.cpp")
        add_deps("luisa-compute-solver-lib")

    target("test_narrow_phase")
        add_rules("lc_basic_settings", {project_kind = "binary", enable_exception = true})
        add_files("UnitTest/test_narrow_phase.cpp")
        add_deps("luisa-compute-solver-lib")

    target("test_energy_assembly")
        add_rules("lc_basic_settings", {project_kind = "binary", enable_exception = true})
        add_files("UnitTest/test_energy_assembly.cpp")
        add_deps("luisa-compute-solver-lib")

    target("test_pcg_solver")
        add_rules("lc_basic_settings", {project_kind = "binary", enable_exception = true})
        add_files("UnitTest/test_pcg_solver.cpp")
        add_deps("luisa-compute-solver-lib")

    target("test_ccd")
        add_rules("lc_basic_settings", {project_kind = "binary", enable_exception = true})
        add_files("UnitTest/test_ccd.cpp")
        add_deps("luisa-compute-solver-lib")

    target("test_integration")
        add_rules("lc_basic_settings", {project_kind = "binary", enable_exception = true})
        add_files("UnitTest/test_integration.cpp")
        add_deps("luisa-compute-solver-lib")
end

if cfg_bool(false, "lcs_build_pybindings") then
    target("lcs_py")
        add_rules("lc_basic_settings", {
            project_kind = "shared",
            enable_exception = true
        })
        if is_plat("windows") then
            set_filename("lcs_py.pyd")
        else
            set_filename("lcs_py.so")
            set_prefixname("")
        end

        -- Source files (mirrors PythonBindings/CMakeLists.txt)
        add_files("PythonBindings/src/python_bindings.cpp")
        add_files("Application/app_simulation_demo_config.cpp")

        -- Include directories
        add_includedirs("Application")

        -- Dependencies (mirrors CMake: luisa-compute-solver-lib, yyjson, luisa::compute)
        add_deps("luisa-compute-solver-lib", "lc-yyjson")

        after_load(function(target)
            local pyexe = get_config("lcs_python_executable")
            if not pyexe or pyexe == "" then
                raise("lcs_build_pybindings requires --lcs_python_executable=<path-to-python>")
            end
            if not os.isfile(pyexe) then
                raise("lcs_python_executable does not exist: " .. pyexe)
            end

            -- Set correct Python extension suffix (e.g. lcs_py.cpython-313-darwin.so)
            local ext_suffix = os.iorunv(pyexe, {"-c", "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"}):trim()
            if ext_suffix and ext_suffix ~= "" then
                target:set("filename", "lcs_py" .. ext_suffix)
            end

            -- RPATH so lcs_py.so finds runtime libs next to it (build/lib, build/bin)
            if target:is_plat("macosx") then
                target:add("shflags", "-Wl,-rpath,@loader_path/lib", "-Wl,-rpath,@loader_path/bin", "-Wl,-rpath,@loader_path", {force = true})
                target:add("shflags", "-undefined", "dynamic_lookup", {force = true})
                target:add("ldflags", "-Wl,-rpath,@loader_path/lib", "-Wl,-rpath,@loader_path/bin", "-Wl,-rpath,@loader_path", {force = true})
                target:add("ldflags", "-undefined", "dynamic_lookup", {force = true})
            elseif target:is_plat("linux") then
                target:add("shflags", "-Wl,-rpath,$ORIGIN/lib", "-Wl,-rpath,$ORIGIN/bin", "-Wl,-rpath,$ORIGIN", {force = true})
                target:add("ldflags", "-Wl,-rpath,$ORIGIN/lib", "-Wl,-rpath,$ORIGIN/bin", "-Wl,-rpath,$ORIGIN", {force = true})
            end

            local probe = [[import sys,sysconfig,pathlib;v=f"{sys.version_info[0]}{sys.version_info[1]}";inc=sysconfig.get_path("include") or "";platinc=sysconfig.get_path("platinclude") or "";base=pathlib.Path(sys.base_prefix);libdir=(base/"libs");print("INCLUDE="+inc);print("PLATINCLUDE="+platinc);print("LIBDIR="+str(libdir));print("LIB=python"+v)]]
            local out = os.iorunv(pyexe, {"-c", probe})

            local include_dir = out:match("INCLUDE=([^\r\n]+)")
            local plat_include_dir = out:match("PLATINCLUDE=([^\r\n]+)")
            local lib_dir = out:match("LIBDIR=([^\r\n]+)")
            local py_lib = out:match("LIB=([^\r\n]+)")

            if include_dir and include_dir ~= "" then
                target:add("includedirs", include_dir)
            end
            if plat_include_dir and plat_include_dir ~= "" and plat_include_dir ~= include_dir then
                target:add("includedirs", plat_include_dir)
            end
            target:add("includedirs", path.join(os.projectdir(), "ext/pybind11/include"))

            if target:is_plat("windows") then
                if lib_dir and lib_dir ~= "" then
                    target:add("linkdirs", lib_dir)
                end
                if py_lib and py_lib ~= "" then
                    target:add("links", py_lib)
                end
            end
        end)

        after_build(function(target)
            local bindir = path.join(os.projectdir(), "build", "bin")
            local gui_dir = path.join(bindir, "lcs_gui")
            os.mkdir(gui_dir)

            local src_dir = path.join(os.projectdir(), "PythonBindings")

            os.cp(path.join(src_dir, "python", "lcs_gui", "__init__.py"), gui_dir)
            os.cp(path.join(src_dir, "tests", "utils", "polyscope_gui.py"), gui_dir)

            cprint("${bright green}lcs_gui Python files copied to: ${bright cyan}%s", gui_dir)
        end)
    target_end()

    -- Stub generation target (equivalent to cmake --build build --target stubs)
    target("stubs")
        set_kind("phony")
        add_deps("lcs_py")
        on_build(function(target)
            local pyexe = get_config("lcs_python_executable")
            if not pyexe or pyexe == "" then
                raise("stubs target requires --lcs_python_executable=<path-to-python>")
            end
            local bindir = path.join(os.projectdir(), "build", "bin")
            local stubout = path.join(os.projectdir(), "PythonBindings", "python")
            os.execv(pyexe, {"-m", "pybind11_stubgen", "lcs_py", "-o", stubout}, {envs = {PYTHONPATH = bindir}})
            local single_stub = path.join(stubout, "lcs_py.pyi")
            local package_dir = path.join(stubout, "lcs_py")
            local package_init = path.join(package_dir, "__init__.pyi")
            if os.isfile(single_stub) then
                os.mkdir(package_dir)
                if os.isfile(package_init) then
                    os.rm(package_init)
                end
                os.mv(single_stub, package_init)
                print("Normalized stub: " .. single_stub .. " -> " .. package_init)
            elseif os.isfile(package_init) then
                print("Stub already in package form: " .. package_init)
            else
                raise("pybind11_stubgen produced neither " .. single_stub .. " nor " .. package_init)
            end

            -- Translate C++ wrapper type names back to Python class names.
            -- pybind11-stubgen resolves the underlying C++ type (PySceneParams)
            -- but the Python name registered with py::class_<> is SceneParams.
            --
            -- Mirror of PythonBindings/cmake/normalize_stub.cmake lines 38-51.
            local stub_content = io.readfile(package_init)
            local fixed, count = stub_content:gsub("PySceneParams", "SceneParams")
            if count > 0 then
                io.writefile(package_init, fixed)
                print(string.format("Replaced PySceneParams -> SceneParams (%d occurrence(s))", count))
            end

            print("Stubs generated to: " .. stubout)
        end)
    target_end()
end