add_requires("luisa-compute 3ab5163b1736b96b61627eff9f81b9c49f1e2918", {configs = {cuda = true}})
add_requires("eigen","tbb","glfw","amgcl")

add_requireconfs("amgcl.boost", {
    override = true,
    configs = {
        cmake = true,
        serialization = true,
        program_options = true,
        container = true,
        regex = true,
        thread = true,
    }
})
-- add_requireconfs("luisa-compute.spdlog.fmt", {override = true, version = "<12"})


target("luisa-compute-solver")
    set_kind("shadred")
    add_packages("luisa-compute", {public = true})
    add_packages("eigen", "tbb", "glfw", "amgcl", {public = true})
    add_files("**.cpp")
    add_headerfiles("**.h", "**.hpp")
    add_defines("USE_AMGCL_FOR_SIM","LCGS_DLL_EXPORTS")
    add_includedirs("./", {public = true})
    local resource_path = path.join(os.projectdir(), "Resources")
    resource_path = resource_path:gsub("\\", "/")
    add_defines("LCSV_RESOURCE_PATH=\"" .. resource_path .. "\"", {public = true})
    on_config(function (target)
        target:add("runargs", path.join(target:pkg("luisa-compute"):installdir(), "bin"))
    end)