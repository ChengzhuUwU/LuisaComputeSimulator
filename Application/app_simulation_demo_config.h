#pragma once

#include <vector>
#include "Initializer/init_mesh_data.h"

namespace Demo
{

	namespace Simulation
	{
		void load_scene_params_from_json(std::vector<lcs::Initializer::WorldData>& shell_list, const std::string& json_path);
	} // namespace Simulation

} // namespace Demo