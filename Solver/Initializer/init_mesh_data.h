#pragma once

#include "SimulationCore/base_mesh.h"

namespace lcs
{

namespace Initializer
{

    using IsFixedPointFunc = std::function<bool(const float3&)>;
    struct FixedPointInfo
    {
        IsFixedPointFunc  is_fixed_point_func;
        bool              use_translate = false;
        float3            translate     = luisa::make_float3(0.0f);
        bool              use_scale     = false;
        float3            scale         = luisa::make_float3(1.0f);
        bool              use_rotate    = false;
        float3            rotCenter;
        float3            rotAxis;
        float             rotAngVelDeg = 0.0f;
        std::vector<uint> fixed_point_verts;
    };
    enum ShellType
    {
        ShellTypeCloth,
        ShellTypeTetrahedral,
        ShellTypeRigid,
    };
    struct ShellInfo
    {
        std::string model_name  = "square8K.obj";
        float3      translation = luisa::make_float3(0.0f, 0.0f, 0.0f);
        float3 rotation = luisa::make_float3(0.0f * lcs::Pi);  // Rotation in x-channel means rotate along with x-axis
        float3                      scale = luisa::make_float3(1.0f);
        std::vector<FixedPointInfo> fixed_point_list;
        ShellType                   shell_type = ShellTypeCloth;
    };

    void init_mesh_data(std::vector<lcs::Initializer::ShellInfo>& shell_list, lcs::MeshData<std::vector>* mesh_data);
    void upload_mesh_buffers(luisa::compute::Device&                device,
                             luisa::compute::Stream&                stream,
                             lcs::MeshData<std::vector>*            input_data,
                             lcs::MeshData<luisa::compute::Buffer>* output_data);

}  // namespace Initializer


}  // namespace lcs