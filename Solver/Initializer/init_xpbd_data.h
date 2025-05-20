#pragma once

#include "SimulationCore/base_mesh.h"
#include "SimulationCore/simulation_data.h"


namespace lcsv 
{

namespace Initializater
{

void init_xpbd_data(lcsv::MeshData<std::vector>* mesh_data, lcsv::XpbdData<std::vector>* xpbd_data);
void upload_xpbd_buffers(
    luisa::compute::Device& device, 
    luisa::compute::Stream& stream, 
    lcsv::XpbdData<std::vector>* input_data, 
    lcsv::XpbdData<luisa::compute::Buffer>* output_data);
    
}


}