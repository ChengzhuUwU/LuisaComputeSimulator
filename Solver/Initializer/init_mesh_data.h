#pragma once

#include "SimulationCore/base_mesh.h"

namespace lcsv 
{

namespace Initializater
{

void init_mesh_data(lcsv::MeshData<std::vector>* mesh_data);
void upload_mesh_buffers(
    luisa::compute::Device& device, 
    luisa::compute::Stream& stream, 
    lcsv::MeshData<std::vector>* input_data, 
    lcsv::MeshData<luisa::compute::Buffer>* output_data);
    
}


}