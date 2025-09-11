#pragma once

#include "SimulationCore/base_mesh.h"
#include "SimulationCore/simulation_data.h"


namespace lcs::Initializer
{


void init_xpbd_data(lcs::MeshData<std::vector>* mesh_data, lcs::SimulationData<std::vector>* xpbd_data);
void upload_xpbd_buffers(
    luisa::compute::Device& device, 
    luisa::compute::Stream& stream, 
    lcs::SimulationData<std::vector>* input_data, 
    lcs::SimulationData<luisa::compute::Buffer>* output_data);
    
void resize_pcg_data(
    luisa::compute::Device& device, 
    luisa::compute::Stream& stream, 
    lcs::MeshData<std::vector>* mesh_data, 
    lcs::SimulationData<std::vector>* host_data, 
    lcs::SimulationData<luisa::compute::Buffer>* device_data
);

} // namespace lcs::Initializer