#include <memory>
#include <stdexcept>
#include "scene_params.h"

namespace lcs
{

	static std::weak_ptr<SceneParams> g_scene_params_ptr;

	void set_scene_params_ptr(const std::shared_ptr<SceneParams>& scene_params_ptr)
	{
		g_scene_params_ptr = scene_params_ptr;
	}

	std::shared_ptr<SceneParams> get_scene_params_ptr()
	{
		return g_scene_params_ptr.lock();
	}

	SceneParams& get_scene_params()
	{
		auto ptr = g_scene_params_ptr.lock();
		if (!ptr)
			throw std::runtime_error(
				"SceneParams is not initialized. Create a SolverInterface/NewtonSolver instance first.");
		return *ptr;
	}

	// std::vector<SceneParams>& get_scene_params_array()
	// {
	//     return scene_params;
	// }

} // namespace lcs