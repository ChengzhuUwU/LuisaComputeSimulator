#pragma once

#include <vector>
#include "Initializer/init_mesh_data.h"
#include <functional>

namespace Demo
{

	namespace Simulation
	{
		void load_scene_params_from_json(const std::function<void(const lcs::Initializer::WorldData&)>& fn_register_mesh, const std::string& json_path);
	} // namespace Simulation

} // namespace Demo