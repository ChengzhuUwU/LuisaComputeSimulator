import("core.base.scheduler")
import("private.async.jobpool")
import("async.jobgraph")
import("async.runjobs")

local function is_empty_folder(dir)
    if os.exists(dir) and not os.isfile(dir) then
        for _, v in ipairs(os.filedirs(path.join(dir, '*'))) do
            return false
        end
        return true
    else
        return true
    end
end
local function git_clone_or_pull(git_address, subdir, branch)
    local args
    if is_empty_folder(subdir) then
        args = {'clone', git_address}
        if branch then
            table.insert(args, '-b')
            table.insert(args, branch)
        end
        table.insert(args, path.translate(subdir))
    else
        args = {'-C', subdir, 'pull'}
    end
    local done = false
    print("pulling " .. git_address)
    for i = 1, 4, 1 do
        try {function()
            os.execv('git', args)
            done = true
        end}
        if done then
            return
        end
    end
    utils.error("git clone error.")
    os.exit(1)
end

local ext_path = path.join(os.projectdir(), "ext")
if not os.exists(ext_path) then
    os.mkdir(ext_path)
end
local lc_path = path.join(ext_path, "LuisaCompute")
function install_lc()
    ------------------------------ git ------------------------------
    print("Clone extensions? (y/n)")
    local clone_lc = io.read()
    if clone_lc == 'Y' or clone_lc == 'y' then
        print("git...")
        local jobs = jobgraph.new()
        local function add_git_job(job_name, subdir, git_address, branch)
            return jobs:add(job_name, function(index, total, opt)
                git_clone_or_pull(git_address, subdir, branch)
            end)
        end
        local function add_lc_ext_git_job(job_name, subdir, git_address, branch)
            local job = add_git_job(job_name, subdir, git_address, branch)
            jobs:add_orders("LuisaCompute", job_name)
            return job
        end
        ------------------------------ LuisaCompute And Extensions ------------------------------
        add_git_job("LuisaCompute", lc_path, "https://github.com/LuisaGroup/LuisaCompute.git", "next")
        add_lc_ext_git_job("xxhash", path.join(lc_path, "src/ext/xxhash"), "https://github.com/Cyan4973/xxHash.git")
        add_lc_ext_git_job("spdlog", path.join(lc_path, "src/ext/spdlog"), "https://github.com/LuisaGroup/spdlog.git")
        add_lc_ext_git_job("mimalloc", path.join(lc_path, "src/ext/EASTL/packages/mimalloc"), "https://github.com/LuisaGroup/mimalloc.git")
        add_lc_ext_git_job("eabase", path.join(lc_path, "src/ext/EASTL/packages/EABase"), "https://github.com/LuisaGroup/EABase.git")
        add_lc_ext_git_job("eastl", path.join(lc_path, "src/ext/EASTL"), "https://github.com/LuisaGroup/EASTL.git")
        jobs:add_orders("eastl", "eabase")
        jobs:add_orders("eastl", "mimalloc")
        add_lc_ext_git_job("marl", path.join(lc_path, "src/ext/marl"), "https://github.com/LuisaGroup/marl.git")
        add_lc_ext_git_job("glfw", path.join(lc_path, "src/ext/glfw"), "https://github.com/glfw/glfw.git")
        add_lc_ext_git_job("magic_enum", path.join(lc_path, "src/ext/magic_enum"), "https://github.com/Neargye/magic_enum")
        add_lc_ext_git_job("imgui", path.join(lc_path, "src/ext/imgui"), "https://github.com/ocornut/imgui.git", "docking")
        add_lc_ext_git_job("reproc", path.join(lc_path, "src/ext/reproc"), "https://github.com/LuisaGroup/reproc.git")
        add_lc_ext_git_job("stb", path.join(lc_path, "src/ext/stb/stb"), "https://github.com/nothings/stb.git")
        add_lc_ext_git_job("yyjson", path.join(lc_path, "src/ext/yyjson"), "https://github.com/ibireme/yyjson.git")
        ------------------------------ Extensions ------------------------------
        add_git_job("implot", path.join(ext_path, "implot"), "https://github.com/epezent/implot.git")
        add_git_job("nlohmann_json", path.join(ext_path, "nlohmann_json"), "https://github.com/nlohmann/json.git")
        add_git_job("eigen", path.join(ext_path, "eigen"), "https://github.com/MaxwellGengYF/eigen.git", "3.4")
        add_git_job("polyscope", path.join(ext_path, "polyscope"), "https://github.com/MaxwellGengYF/polyscope.git")
        add_git_job("glm", path.join(ext_path, "glm"), "https://github.com/icaven/glm.git")


        runjobs("git", jobs, {
            comax = 1000,
            timeout = -1,
            timer = function(running_jobs_indices)
                utils.error("git timeout.")
            end
        })
    end
    if is_empty_folder(lc_path) then
        utils.error("LuisaCompute not installed.")
    end
    ------------------------------ llvm ------------------------------
    local builddir
    if os.is_host("windows") then
        builddir = path.absolute("build/windows/x64/release")
    elseif os.is_host("macosx") then
        builddir = path.absolute("build/macosx/x64/release")
    end
    local lib = import("lib", {
        rootdir = path.join(lc_path, "scripts")
    })
    print("install lc sdk")
    local setup = import("setup", {rootdir = lc_path})
    setup.main(path.join(lc_path, "SDKs"))
    
end

function main(...)
    install_lc(...)
end