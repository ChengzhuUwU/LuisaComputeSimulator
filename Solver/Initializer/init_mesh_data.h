#pragma once

#include "MeshOperation/mesh_reader.h"
#include "SimulationCore/base_mesh.h"

namespace lcs
{

namespace Initializer
{

    struct FixedPointInfo
    {
        // IsFixedPointFunc  is_fixed_point_func;
        // std::vector<uint> fixed_point_verts;
        // std::vector<float3> fixed_point_target_positions;
        // uint   fixed_vid;
        // float3 target_position;

        std::function<bool(const float3&)> is_fixed_point_func;

        bool   use_translate = false;
        float3 translate     = luisa::make_float3(0.0f);

        bool   use_scale = false;
        float3 scale     = luisa::make_float3(1.0f);

        bool   use_rotate = false;
        float3 rotCenter;
        float3 rotAxis;
        float  rotAngVelDeg = 0.0f;

        bool   use_setting_position = false;
        float3 setting_position;

        static float3 fn_affine_position(const lcs::Initializer::FixedPointInfo& fixed_point,
                                         const float                             time,
                                         const lcs::float3&                      pos)
        {
            auto fn_scale =
                [](const lcs::Initializer::FixedPointInfo& fixed_point, const float time, const lcs::float3& pos)
            { return (luisa::scaling(fixed_point.scale * time) * luisa::make_float4(pos, 1.0f)).xyz(); };
            auto fn_rotate =
                [](const lcs::Initializer::FixedPointInfo& fixed_point, const float time, const lcs::float3& pos)
            {
                const float rotAngRad    = time * fixed_point.rotAngVelDeg / 180.0f * float(lcs::Pi);
                const auto  relative_vec = pos - fixed_point.rotCenter;
                auto        matrix       = luisa::rotation(fixed_point.rotAxis, rotAngRad);
                const auto  rotated_pos  = matrix * luisa::make_float4(relative_vec, 1.0f);
                return fixed_point.rotCenter + rotated_pos.xyz();
            };
            auto fn_translate = [](const lcs::Initializer::FixedPointInfo& fixed_point,
                                   const float                             time,
                                   const lcs::float3&                      pos) {
                return (luisa::translation(fixed_point.translate * time) * luisa::make_float4(pos, 1.0f)).xyz();
            };
            auto new_pos = pos;
            if (fixed_point.use_scale)
                new_pos = fn_scale(fixed_point, time, new_pos);
            if (fixed_point.use_rotate)
                new_pos = fn_rotate(fixed_point, time, new_pos);
            if (fixed_point.use_translate)
                new_pos = fn_translate(fixed_point, time, new_pos);
            return new_pos;
        };
    };
    enum ShellType
    {
        ShellTypeCloth,
        ShellTypeTetrahedral,
        ShellTypeRigid,
        ShellTypeRod,
    };

    enum class ConstitutiveStretchModelCloth
    {
        // None     = 0,
        Spring   = 0,  // Impl
        FEM_BW98 = 1,  // Impl
    };
    enum class ConstitutiveBendingModelCloth
    {
        None             = 0,
        QuadraticBending = 1,  // Impl
        DihedralAngle    = 2,  // Impl
    };
    enum class ConstitutiveModelTet
    {
        // None             = 0,
        Spring           = 0,  // Impl
        StVK             = 1,
        StableNeoHookean = 2,
        Corotated        = 3,
        ARAP             = 4,
    };
    enum class ConstitutiveModelRigid
    {
        // None          = 0,
        Spring           = 0,
        Orthogonality    = 1,  // Impl
        ARAP             = 2,
        StableNeoHookean = 3,  // Full space simulation
    };
    enum class ConstitutiveModelRod
    {
        Spring = 0,
    };

    struct ClothMaterial
    {
        ConstitutiveStretchModelCloth stretch_model  = ConstitutiveStretchModelCloth::FEM_BW98;
        ConstitutiveBendingModelCloth bending_model  = ConstitutiveBendingModelCloth::DihedralAngle;
        float                         thickness      = 1e-3f;
        float                         density        = 1e3f;
        float                         youngs_modulus = 1e5f;
        float                         poisson_ratio  = 0.25f;
        float                         area_bending_stiffness = 5e-3f;
        // float                         area_youngs_modulus = 1e3f;
    };
    struct TetMaterial
    {
        ConstitutiveModelTet model          = ConstitutiveModelTet::Spring;
        float                density        = 1e3f;
        float                youngs_modulus = 1e6f;
        float                poisson_ratio  = 0.35f;
    };
    struct RigidMaterial
    {
        ConstitutiveModelRigid model           = ConstitutiveModelRigid::Orthogonality;
        bool                   is_solid        = false;
        float                  density         = 1e3f;
        float                  shell_thickness = 3e-3f;
        float                  stiffness       = 1e6f;
        // float                  youngs_modulus  = 1e9f;
        // float                  poisson_ratio   = 0.35f;
    };
    struct RodMaterial
    {
        ConstitutiveModelRod model              = ConstitutiveModelRod::Spring;
        float                density            = 1e3f;
        float                radius             = 1e-3f;
        float                bending_stiffness  = 1e4f;
        float                twisting_stiffness = 1e4f;
    };

    using MaterialVariant = std::variant<ClothMaterial, TetMaterial, RigidMaterial, RodMaterial>;

    struct ShellInfo
    {
        std::string model_name  = "square8K.obj";
        float3      translation = luisa::make_float3(0.0f, 0.0f, 0.0f);
        float3 rotation = luisa::make_float3(0.0f * lcs::Pi);  // Rotation in x-channel means rotate along with x-axis
        float3 scale = luisa::make_float3(1.0f);

        float mass    = 0.0f;  // If mass > 0, use mass to compute density
        float density = 1e3f;

        // For cloth, rod, non-solid rigid body, is_shell = true. For solid rigid body and tet mesh, is_shell = false
        bool  is_shell  = true;
        float thickness = 1e-3f;  // defulat 1mm for shell, ignored for volumn mesh

        MaterialVariant physics_material;

        template <typename T>
        bool holds() const
        {
            return std::holds_alternative<T>(physics_material);
        }
        template <typename T>
        T& get()
        {
            return std::get<T>(physics_material);
        }
        template <typename T>
        const T& get() const
        {
            return std::get<T>(physics_material);
        }
        template <typename T>
        T* get_if()
        {
            return std::get_if<T>(&physics_material);
        }

        std::vector<FixedPointInfo> fixed_point_info;
        std::vector<uint>           fixed_point_list;
        std::vector<float3>         fixed_point_target_positions;
        ShellType                   shell_type = ShellTypeCloth;
        SimMesh::TriangleMeshData   input_mesh;
        std::vector<float3>         simulated_positions;

        std::vector<uint> set_pinned_verts_from_norm_position(const std::function<bool(const float3&)>& func,
                                                              const FixedPointInfo& info = FixedPointInfo());
        void set_pinned_vert_fixed_info(const uint vid, const FixedPointInfo& info);
        void update_pinned_verts(const float time);
        void update_pinned_verts(const std::vector<float3>& new_positions);

        // template <typename T>
        void get_rest_positions(std::vector<std::array<float, 3>>& rest_positions);

        ShellInfo& load_mesh_data()
        {
            bool second_read = SimMesh::read_mesh_file(model_name, input_mesh);
            return *this;
        }
    };

    void init_mesh_data(std::vector<lcs::Initializer::ShellInfo>& shell_list, lcs::MeshData<std::vector>* mesh_data);
    void upload_mesh_buffers(luisa::compute::Device&                device,
                             luisa::compute::Stream&                stream,
                             lcs::MeshData<std::vector>*            input_data,
                             lcs::MeshData<luisa::compute::Buffer>* output_data);

}  // namespace Initializer


}  // namespace lcs