#include <MacTypes.h>
#include <iostream>
#include <filesystem>
#include <stb/stb_image_write.h>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
// #include <luisa/gui/input.h>
// #include <luisa/gui/window.h>

// #include "common/cornell_box.h"
// "projectionMode" is "Perspective", so keep perspective logic
// Load camera parameters from JSON file
// Requires nlohmann/json (https://github.com/nlohmann/json)
#include <fstream>
#include "../Solver/Utils/json.hpp"

#ifndef TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#endif

#include "../Solver/MeshOperation/tiny_obj_loader.h"

using namespace luisa;
using namespace luisa::compute;

struct Onb
{
    float3 tangent;
    float3 binormal;
    float3 normal;
};

struct Camera
{
    float3 position;
    float3 front;
    float3 up;
    float3 right;
    float  fov;
};

// clang-format off
LUISA_STRUCT(Onb, tangent, binormal, normal){
    [[nodiscard]] Float3 to_world(Expr<float3> v) const noexcept {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
};

LUISA_STRUCT(Camera, position, front, up, right, fov) {
    [[nodiscard]] auto generate_ray(Expr<float2> p/* normalized pixel coordinate */) const noexcept {
        auto fov_radians = radians(fov);
        auto wi_local = make_float3(p * tan(0.5f * fov_radians), -1.0f);
        auto wi_world = normalize(wi_local.x * right + wi_local.y * up - wi_local.z * front);
        return make_ray(position, wi_world);
    }
};
// clang-format on

class FPVCameraController
{

  private:
    Camera& _camera;
    float   _move_speed;
    float   _rotate_speed;
    float   _zoom_speed;

  public:
    explicit FPVCameraController(Camera& camera, float move_speed, float rotate_speed, float zoom_speed) noexcept
        : _camera{camera}
        , _move_speed{move_speed}
        , _rotate_speed{rotate_speed}
        , _zoom_speed{zoom_speed}
    {
        // make sure the camera is valid
        _camera.front = normalize(_camera.front);
        _camera.right = normalize(cross(_camera.front, _camera.up));
        _camera.up    = normalize(cross(_camera.right, _camera.front));
        _camera.fov   = std::clamp(_camera.fov, 1.f, 179.f);
    }
    void zoom(float scale) noexcept
    {
        _camera.fov = std::clamp(_camera.fov * std::pow(2.f, -scale * _zoom_speed), 1.f, 179.f);
    }
    void move_right(float dx) noexcept { _camera.position += _camera.right * dx * _move_speed; }
    void move_up(float dy) noexcept { _camera.position += _camera.up * dy * _move_speed; }
    void move_forward(float dz) noexcept { _camera.position += _camera.front * dz * _move_speed; }
    void rotate_roll(float angle) noexcept
    {
        auto m        = make_float3x3(rotation(_camera.front, radians(_rotate_speed * angle)));
        _camera.up    = normalize(m * _camera.up);
        _camera.right = normalize(m * _camera.right);
    }
    void rotate_yaw(float angle) noexcept
    {
        auto m        = make_float3x3(rotation(_camera.up, radians(_rotate_speed * angle)));
        _camera.front = normalize(m * _camera.front);
        _camera.right = normalize(m * _camera.right);
    }
    void rotate_pitch(float angle) noexcept
    {
        auto m        = make_float3x3(rotation(_camera.right, radians(_rotate_speed * angle)));
        _camera.front = normalize(m * _camera.front);
        _camera.up    = normalize(m * _camera.up);
    }
    [[nodiscard]] auto move_speed() const noexcept { return _move_speed; }
    [[nodiscard]] auto rotate_speed() const noexcept { return _rotate_speed; }
    [[nodiscard]] auto zoom_speed() const noexcept { return _zoom_speed; }
    void               set_move_speed(float speed) noexcept { _move_speed = speed; }
    void               set_rotate_speed(float speed) noexcept { _rotate_speed = speed; }
    void               set_zoom_speed(float speed) noexcept { _zoom_speed = speed; }
};

static const char* cornel_box_string = R"(
g floor
v  -1.01  -1.00   0.99
v   1.00  -1.00   0.99
v   1.00  -1.00  -1.04
v  -0.99  -1.00  -1.04
f -4 -3 -2 -1

g backWall
v  -0.99  -1.00  -1.04
v   1.00  -1.00  -1.04
v   1.00  1.99  -1.04
v  -1.02  1.99  -1.04
f -4 -3 -2 -1

g light
v	-0.24  1.98   0.16
v	-0.24  1.98  -0.22
v	 0.23  1.98  -0.22
v	 0.23  1.98   0.16
f -4 -3 -2 -1)";

int main(int argc, char* argv[])
{

    log_level_verbose();

    Context context{argv[0]};

#if defined(__APPLE__)
    std::string backend = "metal";
#else
    std::string backend = "cuda";
#endif

    Device device = context.create_device(backend);

    // load the Cornell Box scene
    const std::string obj_mesh_path   = std::string(LCSV_RESOURCE_PATH) + "/OutputMesh/";
    const std::string img_output_path = std::string(LCSV_RESOURCE_PATH) + "/OutputImage/";
    std::vector<std::pair<tinyobj::ObjReader, tinyobj::ObjReaderConfig>> obj_readers;

    std::vector<float3> materials_color;

    auto load_mesh = [&](const std::string& path, const float3& color)
    {
        tinyobj::ObjReaderConfig obj_reader_config;
        tinyobj::ObjReader       obj_reader;

        if (!obj_reader.ParseFromFile(path, obj_reader_config))
        {
            luisa::string_view error_message = "unknown error.";
            if (auto&& e = obj_reader.Error(); !e.empty())
            {
                error_message = e;
            }
            LUISA_ERROR_WITH_LOCATION("Failed to load OBJ file: {}", error_message);
        }
        if (auto&& e = obj_reader.Warning(); !e.empty())
        {
            LUISA_WARNING_WITH_LOCATION("{}", e);
        }
        obj_readers.push_back(std::make_pair(obj_reader, obj_reader_config));
        materials_color.push_back(color);
    };

    // const std::string obj_full_path = obj_mesh_path + "frame_1801.obj";
    const std::string obj_full_path = std::string(LCSV_RESOURCE_PATH) + "/InputMesh/Cylinder/cylinder7K.obj";

    load_mesh(obj_full_path, make_float3(0.2, 0.408, 0.8));
    load_mesh(std::string(LCSV_RESOURCE_PATH) + "/InputMesh/Background/background copy.obj", make_float3(1.0f));
    load_mesh(std::string(LCSV_RESOURCE_PATH) + "/InputMesh/Background/light.obj", make_float3(1.0f));

    constexpr float3 light_position = make_float3(-0.24f, 1.98f, 0.16f);
    // constexpr float3 light_position = make_float3(0.4f, -2.3f, 0.1f);
    constexpr float3 light_u        = make_float3(-0.24f, 1.98f, -0.22f) - light_position;
    constexpr float3 light_v        = make_float3(0.23f, 1.98f, 0.16f) - light_position;
    constexpr float3 light_emission = make_float3(15.0f, 15.0f, 15.0f);

    if (false)
    {
        tinyobj::ObjReaderConfig obj_reader_config;
        tinyobj::ObjReader       obj_reader;
        if (!obj_reader.ParseFromString(cornel_box_string, "", obj_reader_config))
        {
            luisa::string_view error_message = "unknown error.";
            if (auto&& e = obj_reader.Error(); !e.empty())
            {
                error_message = e;
            }
            LUISA_ERROR_WITH_LOCATION("Failed to load OBJ file: {}", error_message);
        }
        if (auto&& e = obj_reader.Warning(); !e.empty())
        {
            LUISA_WARNING_WITH_LOCATION("{}", e);
        }
        obj_readers.push_back(std::make_pair(obj_reader, obj_reader_config));

        materials_color.push_back(make_float3(1, 1, 1));                 // floor
        materials_color.push_back(make_float3(1, 1, 1));                 // back wall
        materials_color.push_back(make_float3(0.000f, 0.000f, 0.000f));  // light
    }


    luisa::vector<float3> vertices;
    // vertices.reserve(p.size() / 3u);
    for (auto& pair : obj_readers)
    {
        auto&  obj_reader = pair.first;
        auto&& p          = obj_reader.GetAttrib().vertices;
        LUISA_INFO("Loaded mesh with {} shape(s) and {} vertices.", obj_reader.GetShapes().size(), p.size());
        for (uint i = 0u; i < p.size(); i += 3u)
        {
            vertices.emplace_back(float3{p[i + 0u], p[i + 1u], p[i + 2u]});
        }
    }
    Buffer<float3> vertex_buffer = device.create_buffer<float3>(vertices.size());
    Stream         stream        = device.create_stream(StreamTag::GRAPHICS);
    stream << vertex_buffer.copy_from(vertices.data());


    BindlessArray                   heap = device.create_bindless_array();
    luisa::vector<Mesh>             meshes;
    luisa::vector<Buffer<Triangle>> triangle_buffers;
    uint                            prefix_vid = 0;
    for (auto& pair : obj_readers)
    {
        LUISA_INFO("Current prefix to {}", prefix_vid);
        auto& obj_reader = pair.first;
        for (auto&& shape : obj_reader.GetShapes())
        {
            uint                                 index          = static_cast<uint>(meshes.size());
            std::vector<tinyobj::index_t> const& t              = shape.mesh.indices;
            uint                                 triangle_count = t.size() / 3u;
            LUISA_INFO("Processing shape '{}' at index {} with {} triangle(s).", shape.name, index, triangle_count);
            luisa::vector<uint> indices;
            for (tinyobj::index_t i : t)
            {
                indices.emplace_back(prefix_vid + i.vertex_index);
            }
            Buffer<Triangle>& triangle_buffer =
                triangle_buffers.emplace_back(device.create_buffer<Triangle>(triangle_count));
            Mesh& mesh = meshes.emplace_back(device.create_mesh(vertex_buffer, triangle_buffer));
            heap.emplace_on_update(index, triangle_buffer);
            stream << triangle_buffer.copy_from(indices.data()) << mesh.build();
        }
        prefix_vid += obj_reader.GetAttrib().vertices.size() / 3u;
    }

    Accel accel = device.create_accel({});
    for (Mesh& m : meshes)
    {
        accel.emplace_back(m, make_float4x4(1.0f));
    }
    stream << heap.update() << accel.build() << synchronize();

    Camera camera;
    {
        std::string    json_path = obj_mesh_path + "scene_3.json";
        nlohmann::json scene_json;
        {
            std::ifstream ifs(json_path);
            if (!ifs)
            {
                std::cerr << "Failed to open JSON file: " << json_path << std::endl;
                std::exit(1);
            }
            ifs >> scene_json;
        }

        // Set camera parameters from JSON
        // Default values
        camera.position = make_float3(0.0f, 0.0f, 2.0f);
        // camera.position = make_float3(0.0f, 1.0f, 5.0f);
        camera.front = make_float3(0.0f, 0.0f, -1.0f);
        camera.up    = make_float3(0.0f, 1.0f, 0.0f);
        camera.right = normalize(cross(camera.front, camera.up));
        camera.fov   = 45.0f;
        // camera.fov = 27.8;

        // Use values from JSON if available
        if (false && scene_json.contains("fov"))
        {
            camera.fov = scene_json["fov"].get<float>();
        }

        // If viewMat is present and valid, extract camera orientation and position
        if (false && scene_json.contains("viewMat") && scene_json["viewMat"].is_array()
            && scene_json["viewMat"].size() == 16)
        {
            LUISA_INFO("Reading json...");
            // Read the view matrix (row-major order)
            float viewMat[16];
            for (int i = 0; i < 16; ++i)
            {
                if (scene_json["viewMat"][i].is_null())
                    viewMat[i] = (i % 5 == 0) ? 1.0f : 0.0f;  // identity fallback
                else
                    viewMat[i] = scene_json["viewMat"][i].get<float>();
            }
            // Invert the view matrix to get the camera-to-world matrix
            // (Assume the matrix is affine, so we can invert rotation/translation separately)
            float3 right = make_float3(viewMat[0], viewMat[4], viewMat[8]);
            float3 up    = make_float3(viewMat[1], viewMat[5], viewMat[9]);
            float3 front = -make_float3(viewMat[2], viewMat[6], viewMat[10]);
            float3 trans = make_float3(viewMat[12], viewMat[13], viewMat[14]);
            // The camera position in world space is -R^T * T
            float3 position = make_float3(-(right.x * trans.x + up.x * trans.y + front.x * trans.z),
                                          -(right.y * trans.x + up.y * trans.y + front.y * trans.z),
                                          -(right.z * trans.x + up.z * trans.y + front.z * trans.z));
            camera.position = position;
            camera.right    = normalize(right);
            camera.up       = normalize(up);
            camera.front    = normalize(front);
        }
    }

    Constant materials(materials_color.data(), static_cast<uint>(materials_color.size()));

    Callable linear_to_srgb = [](Var<float3> x) noexcept
    {
        return clamp(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f, 12.92f * x, x <= 0.00031308f), 0.0f, 1.0f);
    };

    Callable tea = [](UInt v0, UInt v1) noexcept
    {
        UInt s0 = def(0u);
        for (uint n = 0u; n < 4u; n++)
        {
            s0 += 0x9e3779b9u;
            v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
    };

    Kernel2D make_sampler_kernel = [&](ImageUInt seed_image) noexcept
    {
        UInt2 p     = dispatch_id().xy();
        UInt  state = tea(p.x, p.y);
        seed_image.write(p, make_uint4(state));
    };

    Callable lcg = [](UInt& state) noexcept
    {
        constexpr uint lcg_a = 1664525u;
        constexpr uint lcg_c = 1013904223u;
        state                = lcg_a * state + lcg_c;
        return cast<float>(state & 0x00ffffffu) * (1.0f / static_cast<float>(0x01000000u));
    };

    Callable make_onb = [](const Float3& normal) noexcept
    {
        Float3 binormal = normalize(ite(abs(normal.x) > abs(normal.z),
                                        make_float3(-normal.y, normal.x, 0.0f),
                                        make_float3(0.0f, -normal.z, normal.y)));
        Float3 tangent  = normalize(cross(binormal, normal));
        return def<Onb>(tangent, binormal, normal);
    };

    Callable cosine_sample_hemisphere = [](Float2 u) noexcept
    {
        Float r   = sqrt(u.x);
        Float phi = 2.0f * constants::pi * u.y;
        return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0f - u.x));
    };

    Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept
    { return pdf_a / max(pdf_a + pdf_b, 1e-4f); };

    Kernel2D raytracing_kernel =
        [&](ImageFloat image, ImageUInt seed_image, Var<Camera> camera, AccelVar accel, UInt2 resolution) noexcept
    {
        set_block_size(16u, 16u, 1u);
        UInt2    coord        = dispatch_id().xy();
        Float    frame_size   = min(resolution.x, resolution.y).cast<float>();
        UInt     state        = seed_image.read(coord).x;
        Float    rx           = lcg(state);
        Float    ry           = lcg(state);
        Float2   pixel        = (make_float2(coord) + make_float2(rx, ry)) / frame_size * 2.0f - 1.0f;
        Float3   radiance     = def(make_float3(0.0f));
        Var<Ray> ray          = camera->generate_ray(pixel * make_float2(1.0f, -1.0f));
        Float3   beta         = def(make_float3(1.0f));
        Float    pdf_bsdf     = def(0.0f);
        Float    light_area   = length(cross(light_u, light_v));
        Float3   light_normal = normalize(cross(light_u, light_v));
        $for(depth, 10u)
        {
            // trace
            Var<TriangleHit> hit = accel.intersect(ray, {});
            $if(hit->miss())
            {
                $break;
            };
            Var<Triangle> triangle = heap->buffer<Triangle>(hit.inst).read(hit.prim);
            Float3        p0       = vertex_buffer->read(triangle.i0);
            Float3        p1       = vertex_buffer->read(triangle.i1);
            Float3        p2       = vertex_buffer->read(triangle.i2);
            Float3        p        = triangle_interpolate(hit.bary, p0, p1, p2);
            Float3        n        = normalize(cross(p1 - p0, p2 - p0));
            Float         cos_wo   = dot(-ray->direction(), n);
            $if(cos_wo < 1e-4f)
            {
                $break;
            };

            // hit light
            $if(hit.inst == static_cast<uint>(meshes.size() - 1u))
            {
                $if(depth == 0u)
                {
                    radiance += light_emission;
                }
                $else
                {
                    Float pdf_light  = length_squared(p - ray->origin()) / (light_area * cos_wo);
                    Float mis_weight = balanced_heuristic(pdf_bsdf, pdf_light);
                    radiance += mis_weight * beta * light_emission;
                };
                $break;
            };

            // sample light
            Float    ux_light     = lcg(state);
            Float    uy_light     = lcg(state);
            Float3   p_light      = light_position + ux_light * light_u + uy_light * light_v;
            Float3   pp           = offset_ray_origin(p, n);
            Float3   pp_light     = offset_ray_origin(p_light, light_normal);
            Float    d_light      = distance(pp, pp_light);
            Float3   wi_light     = normalize(pp_light - pp);
            Var<Ray> shadow_ray   = make_ray(offset_ray_origin(pp, n), wi_light, 0.f, d_light);
            Bool     occluded     = accel.intersect_any(shadow_ray, {});
            Float    cos_wi_light = dot(wi_light, n);
            Float    cos_light    = -dot(light_normal, wi_light);
            Float3   albedo       = materials.read(hit.inst);
            $if(!occluded & cos_wi_light > 1e-4f & cos_light > 1e-4f)
            {
                Float  pdf_light  = (d_light * d_light) / (light_area * cos_light);
                Float  pdf_bsdf   = cos_wi_light * inv_pi;
                Float  mis_weight = balanced_heuristic(pdf_light, pdf_bsdf);
                Float3 bsdf       = albedo * inv_pi * cos_wi_light;
                radiance += beta * bsdf * mis_weight * light_emission / max(pdf_light, 1e-4f);
            };

            // sample BSDF
            Var<Onb> onb           = make_onb(n);
            Float    ux            = lcg(state);
            Float    uy            = lcg(state);
            Float3   wi_local      = cosine_sample_hemisphere(make_float2(ux, uy));
            Float    cos_wi        = abs(wi_local.z);
            Float3   new_direction = onb->to_world(wi_local);
            ray                    = make_ray(pp, new_direction);
            pdf_bsdf               = cos_wi * inv_pi;
            beta *= albedo;  // * cos_wi * inv_pi / pdf_bsdf => * 1.f

            // rr
            Float l = dot(make_float3(0.212671f, 0.715160f, 0.072169f), beta);
            $if(l == 0.0f)
            {
                $break;
            };
            Float q = max(l, 0.05f);
            Float r = lcg(state);
            $if(r >= q)
            {
                $break;
            };
            beta *= 1.0f / q;
        };
        seed_image.write(coord, make_uint4(state));
        $if(any(dsl::isnan(radiance)))
        {
            radiance = make_float3(0.0f);
        };
        image.write(dispatch_id().xy(), make_float4(clamp(radiance, 0.0f, 30.0f), 1.0f));
    };

    Kernel2D accumulate_kernel = [&](ImageFloat accum_image, ImageFloat curr_image) noexcept
    {
        UInt2  p     = dispatch_id().xy();
        Float4 accum = accum_image.read(p);
        Float3 curr  = curr_image.read(p).xyz();
        accum_image.write(p, accum + make_float4(curr, 1.f));
    };

    Callable aces_tonemapping = [](Float3 x) noexcept
    {
        static constexpr float a = 2.51f;
        static constexpr float b = 0.03f;
        static constexpr float c = 2.43f;
        static constexpr float d = 0.59f;
        static constexpr float e = 0.14f;
        return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
    };

    Kernel2D clear_kernel = [](ImageFloat image) noexcept
    { image.write(dispatch_id().xy(), make_float4(0.0f)); };

    Kernel2D hdr2ldr_kernel = [&](ImageFloat hdr_image, ImageFloat ldr_image, Float scale, Bool is_hdr) noexcept
    {
        UInt2  coord = dispatch_id().xy();
        Float4 hdr   = hdr_image.read(coord);
        Float3 ldr   = hdr.xyz() / hdr.w * scale;
        $if(!is_hdr)
        {
            ldr = linear_to_srgb(ldr);
        };
        ldr_image.write(coord, make_float4(ldr, 1.0f));
    };

    auto clear_shader        = device.compile(clear_kernel);
    auto hdr2ldr_shader      = device.compile(hdr2ldr_kernel);
    auto accumulate_shader   = device.compile(accumulate_kernel);
    auto raytracing_shader   = device.compile(raytracing_kernel);
    auto make_sampler_shader = device.compile(make_sampler_kernel);

    static constexpr uint2 resolution  = make_uint2(1024u);
    Image<float>           framebuffer = device.create_image<float>(PixelStorage::HALF4, resolution);
    Image<float>           accum_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    luisa::vector<std::array<uint8_t, 4u>> host_image(resolution.x * resolution.y);
    Image<uint>  seed_image = device.create_image<uint>(PixelStorage::INT1, resolution);
    Image<float> ldr_image  = device.create_image<float>(PixelStorage::BYTE4, resolution);

    double last_time   = 0.0;
    uint   frame_count = 0u;
    Clock  clock;

    auto        is_dirty   = true;
    auto        delta_time = 0.;
    CommandList cmd_list;
    cmd_list << make_sampler_shader(seed_image).dispatch(resolution);

    for (uint iter = 0; iter < 100; iter++)
    {
        LUISA_INFO("Rendering iter {}", iter);
        cmd_list << raytracing_shader(framebuffer, seed_image, camera, accel, resolution).dispatch(resolution)
                 << accumulate_shader(accum_image, framebuffer).dispatch(resolution);
        cmd_list << hdr2ldr_shader(accum_image, ldr_image, 1.0f, false).dispatch(resolution);
        stream << cmd_list.commit();
        delta_time = clock.toc() - last_time;
        last_time  = clock.toc();
    }

    stream << ldr_image.copy_to(host_image.data()) << synchronize();

    LUISA_INFO("FPS: {}", frame_count / clock.toc() * 1000);

    std::string obj_filename  = std::filesystem::path(obj_full_path).filename().string();
    std::string img_filename  = obj_filename.substr(0, obj_filename.find_last_of('.')) + ".png";
    std::string img_full_path = img_output_path + img_filename;
    auto        output_dir    = std::filesystem::path(img_full_path).parent_path();
    if (!std::filesystem::exists(output_dir))
    {
        if (!std::filesystem::create_directories(output_dir))
        {
            std::cerr << "Failed to create output directory: " << output_dir << std::endl;
            std::exit(1);
        }
    }
    stbi_write_png(img_full_path.c_str(), resolution.x, resolution.y, 4, host_image.data(), 0);
    LUISA_INFO("Saved image to {}", img_full_path);
}
