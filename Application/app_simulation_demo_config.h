#pragma once

#include <vector>
#include "Initializer/init_mesh_data.h"

namespace Demo
{

namespace Simulation
{

    void load_scene(std::vector<lcs::Initializer::ShellInfo>& shell_list);
    void load_scene_params_from_json(std::vector<lcs::Initializer::ShellInfo>& shell_list);

}  // namespace Simulation

}  // namespace Demo