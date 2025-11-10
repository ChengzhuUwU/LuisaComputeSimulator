#pragma once

#include "MeshOperation/mesh_reader.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/physical_material.h"

namespace lcs
{

namespace Initializer
{

    enum struct FixedPointsType
    {
        None,
        FromIndices,
        FromFunction,
        Left,
        Right,
        Front,
        Back,
        Up,
        Down,
        LeftUp,
        LeftDown,
        LeftFront,
        LeftBack,
        RightUp,
        RightDown,
        RightFront,
        RightBack,
        FrontUp,
        FrontDown,
        BackUp,
        BackDown,
        All,
    };

    struct FixedPointAnimationInfo
    {
        // IsFixedPointFunc  is_fixed_point_func;
        // std::vector<uint> fixed_point_verts;
        // std::vector<float3> fixed_point_target_positions;
        // uint   fixed_vid;
        // float3 target_position;

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

        static float3 fn_affine_position(const lcs::Initializer::FixedPointAnimationInfo& fixed_point,
                                         const float                                      time,
                                         const lcs::float3&                               pos)
        {
            auto fn_scale = [](const lcs::Initializer::FixedPointAnimationInfo& fixed_point,
                               const float                                      time,
                               const lcs::float3&                               pos)
            { return (luisa::scaling(fixed_point.scale * time) * luisa::make_float4(pos, 1.0f)).xyz(); };
            auto fn_rotate = [](const lcs::Initializer::FixedPointAnimationInfo& fixed_point,
                                const float                                      time,
                                const lcs::float3&                               pos)
            {
                const float rotAngRad    = time * fixed_point.rotAngVelDeg / 180.0f * float(lcs::Pi);
                const auto  relative_vec = pos - fixed_point.rotCenter;
                auto        matrix       = luisa::rotation(fixed_point.rotAxis, rotAngRad);
                const auto  rotated_pos  = matrix * luisa::make_float4(relative_vec, 1.0f);
                return fixed_point.rotCenter + rotated_pos.xyz();
            };
            auto fn_translate = [](const lcs::Initializer::FixedPointAnimationInfo& fixed_point,
                                   const float                                      time,
                                   const lcs::float3&                               pos) {
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

    struct MakeFixedPointsInterface
    {
        FixedPointsType         method = FixedPointsType::All;
        FixedPointAnimationInfo fixed_info;
        std::vector<float>      range    = {0.001f};
        void*                   data_ptr = nullptr;
    };


    enum SimulationType
    {
        SimulationTypeCloth,
        SimulationTypeTetrahedral,
        SimulationTypeRigid,
        SimulationTypeRod,
    };

    struct WorldData
    {
        std::string model_name  = "square8K.obj";
        float3      translation = luisa::make_float3(0.0f, 0.0f, 0.0f);
        float3 rotation = luisa::make_float3(0.0f * lcs::Pi);  // Rotation in x-channel means rotate along with x-axis
        float3 scale = luisa::make_float3(1.0f);

        MaterialVariant physics_material;  // Maybe we can use Polymorphism

        std::vector<MakeFixedPointsInterface> fixed_point_range_info;
        std::vector<uint>                     fixed_point_indices;
        std::vector<FixedPointAnimationInfo>  fixed_point_animations;
        std::vector<float3>                   fixed_point_target_positions;

        SimulationType            simulation_type = SimulationTypeCloth;
        SimMesh::TriangleMeshData input_mesh;

        template <typename T>
        bool holds() const
        {
            return std::holds_alternative<T>(physics_material);
        }
        template <typename T>
        T& get_material()
        {
            return std::get<T>(physics_material);
        }
        template <typename T>
        const T& get_material() const
        {
            return std::get<T>(physics_material);
        }


        WorldData()
            : model_name("sim_object")
            , simulation_type(SimulationTypeCloth)
        {
        }
        WorldData(const std::string& model_name, const SimulationType& mesh_type)
            : model_name(model_name)
            , simulation_type(mesh_type)
        {
        }
        WorldData& set_name(const std::string& model_name)
        {
            this->model_name = model_name;
            return *this;
        }
        WorldData& set_simulation_type(const SimulationType& sim_type)
        {
            this->simulation_type = sim_type;
            return *this;
        }
        WorldData& add_fixed_point_info(const MakeFixedPointsInterface& info)
        {
            this->fixed_point_range_info.emplace_back(info);
            return *this;
        }
        WorldData& load_mesh_data();
        WorldData& load_mesh_from_path(const std::string& path);

        WorldData& set_translation(const float3& t)
        {
            this->translation = t;
            return *this;
        }
        WorldData& set_translation(const float x, const float y, const float z)
        {
            this->translation = luisa::make_float3(x, y, z);
            return *this;
        }
        WorldData& set_rotation(const float3& r)
        {
            this->rotation = r;
            return *this;
        }
        WorldData& set_rotation(const float axis_x, const float axis_y, const float axis_z)
        {
            this->rotation = luisa::make_float3(axis_x, axis_y, axis_z);
            return *this;
        }
        WorldData& set_scale(const float3& s)
        {
            this->scale = s;
            return *this;
        }
        WorldData& set_scale(const float scale_x, const float scale_y, const float scale_z)
        {
            this->scale = luisa::make_float3(scale_x, scale_y, scale_z);
            return *this;
        }
        WorldData& set_scale(const float s)
        {
            this->scale = luisa::make_float3(s);
            return *this;
        }
        WorldData& set_physics_material(const MaterialVariant& mat)
        {
            this->physics_material = mat;
            return *this;
        }


        // template <typename T>
        // T* get_if()
        // {
        //     return std::get_if<T>(&physics_material);
        // }

        // Return whether the contained material should be treated as a shell.
        // We can't call std::get<MaterialBase>() because MaterialBase is not
        // a variant alternative; instead expose this helper which visits the
        // active alternative and reads its `is_shell` member.
        bool get_is_shell() const
        {
            return std::visit([](auto const& m) noexcept { return m.is_shell; }, physics_material);
        }
        float get_mass() const
        {
            return std::visit([](auto const& m) noexcept { return m.mass; }, physics_material);
        }
        float get_density() const
        {
            return std::visit([](auto const& m) noexcept { return m.density; }, physics_material);
        }
        float get_d_hat() const
        {
            return std::visit([](auto const& m) noexcept { return m.d_hat; }, physics_material);
        }
        float get_thickness() const
        {
            if (get_is_shell())
            {
                if (holds<ClothMaterial>())
                {
                    return get_material<ClothMaterial>().thickness;
                }
                else if (holds<RigidMaterial>())
                {
                    return get_material<RigidMaterial>().thickness;
                }
                else if (holds<RodMaterial>())
                {
                    return get_material<RodMaterial>().radius * 2.0f;
                }
                else
                {
                    LUISA_ERROR("Mesh {} should not have thickness", model_name);
                    return 0.0f;
                }
            }
            else
            {
                return 0.0f;
            }
        }
        std::string get_model_name() const
        {
            return std::filesystem::path(model_name).filename().string();
        }


        WorldData& load_fixed_points();

        void set_pinned_verts_from_norm_position(const std::function<bool(const float3&)>& func,
                                                 const FixedPointAnimationInfo& info = FixedPointAnimationInfo());
        void set_pinned_verts_from_functions(const std::function<bool(uint)>& func,
                                             const FixedPointAnimationInfo& info = FixedPointAnimationInfo());
        void set_pinned_verts_from_indices(const std::vector<uint>& indices,
                                           const FixedPointAnimationInfo& info = FixedPointAnimationInfo());
        void set_pinned_vert_fixed_info(const uint vid, const FixedPointAnimationInfo& info);
        void update_pinned_verts(const std::vector<float3>& new_positions);

        std::vector<float3> get_fixed_point_target_positions(const float time);
        void                get_rest_positions(std::vector<std::array<float, 3>>& rest_positions);
    };

    void init_mesh_data(std::vector<lcs::Initializer::WorldData>& shell_list, lcs::MeshData<std::vector>* mesh_data);
    void upload_mesh_buffers(luisa::compute::Device&                device,
                             luisa::compute::Stream&                stream,
                             lcs::MeshData<std::vector>*            input_data,
                             lcs::MeshData<luisa::compute::Buffer>* output_data);

}  // namespace Initializer


}  // namespace lcs