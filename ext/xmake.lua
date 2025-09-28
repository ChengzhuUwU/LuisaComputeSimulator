lc_options = {
    lc_cpu_backend = false,
    lc_cuda_backend = is_host("windows"),
    lc_dx_backend = is_host("windows"),
    lc_vk_backend = is_host("windows"),
    lc_vk_support = is_host("windows"),
    lc_enable_mimalloc = true,
    lc_enable_api = false,
    lc_enable_clangcxx = false,
    lc_enable_dsl = true,
    lc_enable_gui = true,
    lc_enable_osl = false,
    lc_enable_ir = false,
    lc_enable_tests = false,
    lc_backend_lto = false,
    lc_sdk_dir = path.join(os.scriptdir(), "ext/LuisaCompute/SDKs"),
    lc_win_runtime = "MD",
    lc_metal_backend = is_host("macosx"),
    lc_dx_cuda_interop = is_host("windows")
    -- lc_toy_c_backend = true
}
includes("LuisaCompute")

-- target("tbb")
--     add_rules("lc_basic_settings", {
--         project_kind = "static"
--     })
--     add_includedirs("tbb/include", { public = true })
--     add_files("tbb/src/**.cpp")
--     add_defines("__TBB_PREVIEW_PARALLEL_PHASE")
-- target_end()

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

target("glm")
add_rules("lc_basic_settings", {
    project_kind = "static"
})
add_includedirs("glm", {
    public = true
})
add_files("glm/glm/detail/glm.cpp")
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
add_deps("glm", "implot", "stb-image", "nlohmann_json", "glad")
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
add_files("glad/src/**.c")
add_includedirs("glad/include", {
    public = true
})
on_load(function(target)
    if (not os.exists(path.join(os.scriptdir(), "glad/src"))) or
        (not os.exists(path.join(os.scriptdir(), "glad/include"))) then
        os.execv("python", {path.join(os.scriptdir(), "glad/glad"), "--api", "gl:core=3.3", "--out-path",
                            path.join(os.scriptdir(), "glad")})
    end
    --  python glad --api gl:core=3.3 --out-path .
end)
target_end()

target("imgui")
add_files("LuisaCompute/src/ext/imgui/backends/imgui_impl_opengl3.cpp")
target_end()
