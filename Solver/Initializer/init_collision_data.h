#pragma once

#include "Initializer/init_collision_data.h"
#include "Initializer/init_mesh_data.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/collision_data.h"
#include "SimulationCore/simulation_data.h"
#include "Initializer/initializer_utils.h"

namespace lcs::Initializer
{
void init_collision_data(std::vector<lcs::Initializer::WorldData>& shell_infos,
                         lcs::MeshData<std::vector>*               mesh_data,
                         lcs::SimulationData<std::vector>*         sim_data,
                         lcs::CollisionData<std::vector>*          collision_data);
void upload_collision_buffers(luisa::compute::Device&                      device,
                              luisa::compute::Stream&                      stream,
                              lcs::SimulationData<std::vector>*            input_sim_data,
                              lcs::SimulationData<luisa::compute::Buffer>* output_sim_data,
                              lcs::CollisionData<std::vector>*             input_collision_data,
                              lcs::CollisionData<luisa::compute::Buffer>*  output_collision_data);

}  // namespace lcs::Initializer