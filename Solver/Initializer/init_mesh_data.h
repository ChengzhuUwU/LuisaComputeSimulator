#pragma once

#include "SimulationCore/base_mesh.h"

namespace lcsv 
{

namespace Initializater
{

using IsFixedPointFunc = std::function<bool(const float3&)>;
struct FixedPointInfo
{
    IsFixedPointFunc is_fixed_point_func;
    bool use_translate = false;
    float3 translate;
    bool use_scale = false;
    float3 scale;
    bool use_rotate = false;
    float3 rotCenter;
    float3 rotAxis;
    float3 rotDeg;
};
struct ShellInfo
{
    std::string model_name = "square8K.obj";
    float3 transform = luisa::make_float3(0.0f, 0.0f, 0.0f);
    float3 rotation = luisa::make_float3(0.0f * lcsv::Pi);
    float3 scale = luisa::make_float3(1.0f);
    std::vector<FixedPointInfo> fixed_point_info;
};

void init_mesh_data(
    const std::vector<lcsv::Initializater::ShellInfo>& shell_list, 
    lcsv::MeshData<std::vector>* mesh_data);
void upload_mesh_buffers(
    luisa::compute::Device& device, 
    luisa::compute::Stream& stream, 
    lcsv::MeshData<std::vector>* input_data, 
    lcsv::MeshData<luisa::compute::Buffer>* output_data);
    
}


}