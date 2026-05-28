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

lc_options = {
    lc_fallback_backend = cfg_bool(false, "lc_fallback_backend"),
    lc_cuda_backend = cfg_bool(true, "lc_cuda_backend") and not is_host("macosx") ,
    lc_dx_backend = cfg_bool(false, "lc_dx_backend"),
    lc_vk_backend = cfg_bool(false, "lc_vk_backend"),
    lc_vk_support = cfg_bool(false, "lc_vk_support"),
    lc_metal_backend = cfg_bool(true, "lc_metal_backend") and is_host("macosx"),
    lc_enable_mimalloc = true,
    lc_enable_api = false,
    lc_enable_clangcxx = false,
    lc_enable_dsl = true,
    lc_enable_gui = cfg_bool(false, "lcs_enable_gui"),
    lc_enable_imgui = cfg_bool(false, "lcs_enable_gui"),
    lc_enable_osl = false,
    lc_enable_ir = false,
    lc_enable_tests = false,
    lc_backend_lto = false,
    lc_sdk_dir = path.join(os.scriptdir(), "LuisaCompute/SDKs"),
    lc_win_runtime = "MD",
    lc_dx_cuda_interop = false,
    lc_vk_cuda_interop = false,
    lc_cuda_ext_lcub = false
    -- lc_toy_c_backend = true
}
includes("LuisaCompute")

-- lcpp: disable tests to avoid pulling in boost_ut/cpptrace
option("lcpp_test")
    set_default(false)
option_end()
includes("lcpp")

-- target("tbb")
--     add_rules("lc_basic_settings", {
--         project_kind = "static"
--     })
--     add_includedirs("tbb/include", { public = true })
--     add_files("tbb/src/**.cpp")
--     add_defines("__TBB_PREVIEW_PARALLEL_PHASE")
-- target_end()
add_defines("LUISA_COMPUTE_SOLVER_USE_LUISA_FIBER")

target("eigen")
set_kind("headeronly")
add_includedirs("eigen", {
    public = true
})
add_defines("EIGEN_HAS_STD_RESULT_OF=0", {
    public = true
})
on_config(function(target)
    local _, cc = target:tool("cxx")
    if (cc == "clang" or cc == "clangxx") then
        target:add("defines", "EIGEN_DISABLE_AVX", {
            public = true
        })
    end
end)
target_end()

if cfg_bool(false, "lcs_enable_gui") then
    target("glfw")
    set_kind("static")
    add_includedirs("LuisaCompute/src/ext/glfw/include", {
        public = true
    })
    add_files("LuisaCompute/src/ext/glfw/src/**.c")
    if is_host("windows") then
        add_defines("_GLFW_WIN32")
        add_syslinks("User32", "Gdi32", "Shell32")
    elseif is_host("macosx") then
        add_defines("_GLFW_COCOA")
        add_files("LuisaCompute/src/ext/glfw/src/**.m")
        add_mflags("-fno-objc-arc")
        add_frameworks("Cocoa", "IOKit", "CoreVideo", "CoreFoundation", "QuartzCore", {public = true})
    end
    target_end()

    target("glm")
    set_kind("headeronly")
    add_includedirs("glm", {
        public = true
    })
    add_defines("GLM_ENABLE_EXPERIMENTAL", {
        public = true
    })
    target_end()

    target("polyscope")
    add_rules("lc_basic_settings", {
        project_kind = "static",
        enable_exception = true,
        rtti = true
    })
    add_includedirs("polyscope/include", "polyscope/deps/MarchingCubeCpp/include", {
        public = true
    })
    add_files("polyscope/src/**.cpp")
    add_includedirs("LuisaCompute/src/ext/stb/stb")
    add_defines("GLAD_GLAPI_EXPORT", {
        public = true
    })
    add_defines("GLAD_GLAPI_EXPORT_BUILD", "POLYSCOPE_BACKEND_OPENGL3_GLFW_ENABLED", "POLYSCOPE_BACKEND_OPENGL3_ENABLED")
    add_deps("glm", "implot", "stb-image", "nlohmann_json", "glad", "glfw")
    if is_host("macosx") then
        add_frameworks("OpenGL", {public = true})
    end
    set_pcxxheader("polyscope_pch.h")
    target_end()

    target("nlohmann_json")
    set_kind("headeronly")
    add_includedirs("nlohmann_json/include", {
        public = true
    })
    target_end()

    target("implot")
    add_rules("lc_basic_settings", {
        project_kind = "shared"
    })
    add_files("implot/implot.cpp", "implot/implot_items.cpp")
    add_deps("imgui")
    add_includedirs("implot", {
        public = true
    })
    on_load(function(target)
        if is_host("windows") then
            target:add("defines", "IMPLOT_API=__declspec(dllexport)");
            target:add("defines", "IMPLOT_API=__declspec(dllimport)", {
                interface = true
            });
        end
    end)
    target_end()

    target("glad")
    add_rules("lc_basic_settings", {
        project_kind = "static"
    })
    add_files("glad_ogl33/src/**.c")
    add_includedirs("glad_ogl33/include", {
        public = true
    })
    target_end()

    target("imgui")
    set_kind("static")
    add_deps("glfw")
    add_includedirs("LuisaCompute/src/ext/imgui", {
        public = true
    })
    add_includedirs("LuisaCompute/src/ext/imgui/backends")
    add_files(
        "LuisaCompute/src/ext/imgui/imgui.cpp",
        "LuisaCompute/src/ext/imgui/imgui_draw.cpp",
        "LuisaCompute/src/ext/imgui/imgui_tables.cpp",
        "LuisaCompute/src/ext/imgui/imgui_widgets.cpp",
        "LuisaCompute/src/ext/imgui/backends/imgui_impl_glfw.cpp",
        "LuisaCompute/src/ext/imgui/backends/imgui_impl_opengl3.cpp"
    )
    target_end()
    add_defines("SIMULATION_APP_USE_GUI")

end
