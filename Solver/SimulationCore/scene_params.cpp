#include <vector>
#include "scene_params.h"

namespace lcsv 
{

// std::vector<SceneParams> scene_params;
SceneParams scene_params;
// std::shared_ptr<SceneParams> scene_params = std::make_shared<SceneParams>();

void init_scene_params()
{
    // scene_params.resize(1);
    // scene_params[0] = SceneParams();
}

SceneParams& get_scene_params()
{
    return scene_params;
}

// std::vector<SceneParams>& get_scene_params_array()
// {
//     return scene_params;
// }

}