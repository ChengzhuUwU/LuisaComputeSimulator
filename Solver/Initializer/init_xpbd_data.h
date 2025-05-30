#pragma once

#include "SimulationCore/base_mesh.h"
#include "SimulationCore/simulation_data.h"


namespace lcsv::Initializater
{


void init_xpbd_data(lcsv::MeshData<std::vector>* mesh_data, lcsv::SimulationData<std::vector>* xpbd_data);
void upload_xpbd_buffers(
    luisa::compute::Device& device, 
    luisa::compute::Stream& stream, 
    lcsv::SimulationData<std::vector>* input_data, 
    lcsv::SimulationData<luisa::compute::Buffer>* output_data);
    
void resize_pcg_data(
    luisa::compute::Device& device, 
    luisa::compute::Stream& stream, 
    lcsv::MeshData<std::vector>* mesh_data, 
    lcsv::SimulationData<std::vector>* host_data, 
    lcsv::SimulationData<luisa::compute::Buffer>* device_data
);

} // namespace lcsv::Initializater