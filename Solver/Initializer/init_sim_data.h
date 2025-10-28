#pragma once

#include "Initializer/init_mesh_data.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/simulation_data.h"


namespace lcs::Initializer
{


void init_sim_data(std::vector<lcs::Initializer::ShellInfo>& shell_infos,
                   lcs::MeshData<std::vector>*               mesh_data,
                   lcs::SimulationData<std::vector>*         sim_data);
void upload_sim_buffers(luisa::compute::Device&                      device,
                        luisa::compute::Stream&                      stream,
                        lcs::SimulationData<std::vector>*            input_data,
                        lcs::SimulationData<luisa::compute::Buffer>* output_data);

void resize_pcg_data(luisa::compute::Device&                      device,
                     luisa::compute::Stream&                      stream,
                     lcs::MeshData<std::vector>*                  mesh_data,
                     lcs::SimulationData<std::vector>*            host_data,
                     lcs::SimulationData<luisa::compute::Buffer>* device_data);

}  // namespace lcs::Initializer