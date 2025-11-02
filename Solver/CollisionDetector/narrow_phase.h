#pragma once

#include "CollisionDetector/lbvh.h"
#include "Core/scalar.h"
#include "SimulationCore/simulation_data.h"
#include "SimulationCore/collision_data.h"
#include "SimulationCore/simulation_type.h"
#include <vector>
#include <string>
#include <luisa/luisa-compute.h>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Utils/async_compiler.h>

namespace lcs
{

enum class ContactEnergyType
{
    Quadratic,
    Barrier,
};
// constexpr ContactEnergyType contact_energy_type = ContactEnergyType::Quadratic; // Quadratic or Barrier

class NarrowPhasesDetector
{
    template <typename T>
    using Buffer = luisa::compute::Buffer<T>;
    template <typename T>
    using BufferView = luisa::compute::BufferView<T>;
    using Stream     = luisa::compute::Stream;
    using Device     = luisa::compute::Device;

  private:
    void compile_ccd(AsyncCompiler& compiler);
    void compile_dcd(AsyncCompiler& compiler, const ContactEnergyType contact_energy_type);
    void compile_energy(AsyncCompiler& compiler, const ContactEnergyType contact_energy_type);
    void compile_construct_pervert_adj_collision_list(AsyncCompiler& compiler);
    void compile_make_contact_triplet(AsyncCompiler& compiler);
    void compile_assemble_atomic(AsyncCompiler& compiler);
    void compile_assemble_non_conflict(AsyncCompiler& compiler);
    void compile_SpMV(AsyncCompiler& compiler);

  public:
    void unit_test(luisa::compute::Device& device, luisa::compute::Stream& stream);
    void compile(AsyncCompiler& compiler);
    void set_collision_data(CollisionData<std::vector>* host_ccd_ptr, CollisionData<luisa::compute::Buffer>* ccd_ptr)
    {
        host_collision_data = host_ccd_ptr;
        collision_data      = ccd_ptr;
    }

  public:
    void  reset_energy(Stream& stream);
    float download_energy(Stream& stream);

    void  reset_toi(Stream& stream);
    void  host_reset_toi(Stream& stream);
    void  reset_broadphase_count(Stream& stream);
    void  reset_narrowphase_count(Stream& stream);
    void  reset_pervert_collision_count(Stream& stream);
    float get_global_toi(Stream& stream);
    void  download_broadphase_collision_count(Stream& stream);
    void  download_narrowphase_collision_count(Stream& stream);
    void  download_narrowphase_list(Stream& stream);
    void  download_contact_triplet(Stream& stream);
    void  upload_spd_narrowphase_list(Stream& stream);
    void  post_resize_buffers(Device& decice, Stream& stream);

  public:
    // CCD
    void vf_ccd_query(Stream&               stream,
                      const Buffer<float3>& sa_x_begin_left,
                      const Buffer<float3>& sa_x_begin_right,
                      const Buffer<float3>& sa_x_end_left,
                      const Buffer<float3>& sa_x_end_right,
                      const Buffer<uint3>&  sa_faces_right,
                      const Buffer<float>&  d_hat,
                      const Buffer<float>&  thickness);

    void ee_ccd_query(Stream&               stream,
                      const Buffer<float3>& sa_x_begin_left,
                      const Buffer<float3>& sa_x_begin_right,
                      const Buffer<float3>& sa_x_end_left,
                      const Buffer<float3>& sa_x_end_right,
                      const Buffer<uint2>&  sa_edges_left,
                      const Buffer<uint2>&  sa_edges_right,
                      const Buffer<float>&  d_hat,
                      const Buffer<float>&  thickness);

    void host_vf_ccd_query(Stream&                    stream,
                           const std::vector<float3>& sa_x_begin_left,
                           const std::vector<float3>& sa_x_begin_right,
                           const std::vector<float3>& sa_x_end_left,
                           const std::vector<float3>& sa_x_end_right,
                           const std::vector<uint3>&  sa_faces_right,
                           const float                d_hat,
                           const float                thickness);

    void host_ee_ccd_query(Stream&                    stream,
                           const std::vector<float3>& sa_x_begin_left,
                           const std::vector<float3>& sa_x_begin_right,
                           const std::vector<float3>& sa_x_end_left,
                           const std::vector<float3>& sa_x_end_right,
                           const std::vector<uint2>&  sa_edges_left,
                           const std::vector<uint2>&  sa_edges_right,
                           const float                d_hat,
                           const float                thickness);

  public:
    // DCD
    void vf_dcd_query_repulsion(Stream&               stream,
                                const Buffer<float3>& sa_x_left,
                                const Buffer<float3>& sa_x_right,
                                const Buffer<float3>& sa_rest_x_left,
                                const Buffer<float3>& sa_rest_x_right,
                                const Buffer<float>&  sa_rest_area_left,
                                const Buffer<float>&  sa_rest_area_right,
                                const Buffer<uint3>&  sa_faces_right,
                                const Buffer<uint>&   sa_vert_affine_bodies_id_left,
                                const Buffer<uint>&   sa_vert_affine_bodies_id_right,
                                const Buffer<float>&  d_hat,
                                const Buffer<float>&  thickness,
                                const float           kappa);

    void ee_dcd_query_repulsion(Stream&               stream,
                                const Buffer<float3>& sa_x_left,
                                const Buffer<float3>& sa_x_right,
                                const Buffer<float3>& sa_rest_x_left,
                                const Buffer<float3>& sa_rest_x_right,
                                const Buffer<float>&  sa_rest_area_left,
                                const Buffer<float>&  sa_rest_area_right,
                                const Buffer<uint2>&  sa_edges_left,
                                const Buffer<uint2>&  sa_edges_right,
                                const Buffer<uint>&   sa_vert_affine_bodies_id_left,
                                const Buffer<uint>&   sa_vert_affine_bodies_id_right,
                                const Buffer<float>&  d_hat,
                                const Buffer<float>&  thickness,
                                const float           kappa);

    void device_perPair_evaluate_gradient_hessian(Stream&               stream,
                                                  const Buffer<float3>& sa_x_left,
                                                  const Buffer<float3>& sa_x_right,
                                                  const float           d_hat,
                                                  const float           thickness,
                                                  const Buffer<uint>&   sa_vert_affine_bodies_id,
                                                  const Buffer<float3>& sa_scaled_model_x,
                                                  const uint            prefix_abd,
                                                  Buffer<float3>&       sa_cgB,
                                                  Buffer<float3x3>&     sa_cgA_diag);
    void device_perVert_evaluate_gradient_hessian(Stream&               stream,
                                                  const Buffer<float3>& sa_x_left,
                                                  const Buffer<float3>& sa_x_right,
                                                  const float           d_hat,
                                                  const float           thickness,
                                                  Buffer<float3>&       sa_cgB,
                                                  Buffer<float3x3>&     sa_cgA_diag);
    void construct_pervert_adj_list(Stream& stream, Buffer<uint>& sa_vert_affine_bodies_id, const uint prefix_abd);
    void host_perPair_spmv(Stream& stream, const std::vector<float3>& input_array, std::vector<float3>& output_array);
    void host_perVert_spmv(Stream& stream, const std::vector<float3>& input_array, std::vector<float3>& output_array);
    void device_perVert_spmv(Stream& stream, const Buffer<float3>& input_array, Buffer<float3>& output_array);
    void device_perPair_spmv(Stream& stream, const Buffer<float3>& input_array, Buffer<float3>& output_array);
    void host_sort_contact_triplet(Stream& stream);
    void device_sort_contact_triplet(Stream& stream);
    void device_assemble_contact_triplet(Stream& stream, Buffer<float3>& sa_scaled_model_x, const uint prefix_abd);

  public:
    // Compute barrier energy
    void compute_contact_energy_from_iter_start_list(Stream&               stream,
                                                     const Buffer<float3>& sa_x_left,
                                                     const Buffer<float3>& sa_x_right,
                                                     const Buffer<float3>& sa_rest_x_left,
                                                     const Buffer<float3>& sa_rest_x_right,
                                                     const Buffer<float>&  sa_rest_area_left,
                                                     const Buffer<float>&  sa_rest_area_right,
                                                     const Buffer<uint3>&  sa_faces_right,
                                                     const float           d_hat,
                                                     const float           thickness,
                                                     const float           kappa);

  public:
  public:
    CollisionData<luisa::compute::Buffer>* collision_data;
    CollisionData<std::vector>*            host_collision_data;

  private:
    CollisionData<luisa::compute::Buffer>& get_collision_data() { return *collision_data; }
    using CDBG = lcs::CollisionData<luisa::compute::Buffer>;  // CollisionData Binding Group

    luisa::compute::Shader<1,
                           CDBG,
                           luisa::compute::Buffer<float3>,
                           luisa::compute::Buffer<float3>,
                           luisa::compute::Buffer<float3>,
                           luisa::compute::Buffer<float3>,
                           luisa::compute::Buffer<uint3>,
                           luisa::compute::Buffer<float>,
                           luisa::compute::Buffer<float>>
        fn_narrow_phase_vf_ccd_query;

    luisa::compute::Shader<1,
                           CDBG,
                           luisa::compute::Buffer<float3>,
                           luisa::compute::Buffer<float3>,
                           luisa::compute::Buffer<float3>,
                           luisa::compute::Buffer<float3>,
                           luisa::compute::Buffer<uint2>,
                           luisa::compute::Buffer<uint2>,
                           luisa::compute::Buffer<float>,
                           luisa::compute::Buffer<float>>
        fn_narrow_phase_ee_ccd_query;

    luisa::compute::Shader<1, luisa::compute::BufferView<float>> fn_reset_toi;
    luisa::compute::Shader<1, luisa::compute::BufferView<float>> fn_reset_energy;
    luisa::compute::Shader<1, luisa::compute::BufferView<uint>>  fn_reset_uint;
    luisa::compute::Shader<1, luisa::compute::BufferView<float>> fn_reset_float;


    luisa::compute::Shader<1,
                           CDBG,
                           luisa::compute::Buffer<float3>,
                           luisa::compute::Buffer<float3>,
                           luisa::compute::Buffer<float3>,
                           luisa::compute::Buffer<float3>,
                           luisa::compute::Buffer<float>,
                           luisa::compute::Buffer<float>,
                           luisa::compute::Buffer<uint3>,
                           luisa::compute::Buffer<uint>,
                           luisa::compute::Buffer<uint>,
                           luisa::compute::Buffer<float>,
                           luisa::compute::Buffer<float>,
                           float>
        fn_narrow_phase_vf_dcd_query;

    luisa::compute::Shader<1,
                           CDBG,
                           luisa::compute::Buffer<float3>,
                           luisa::compute::Buffer<float3>,
                           luisa::compute::Buffer<float3>,
                           luisa::compute::Buffer<float3>,
                           luisa::compute::Buffer<float>,
                           luisa::compute::Buffer<float>,
                           luisa::compute::Buffer<uint2>,
                           luisa::compute::Buffer<uint2>,
                           luisa::compute::Buffer<uint>,
                           luisa::compute::Buffer<uint>,
                           luisa::compute::Buffer<float>,
                           luisa::compute::Buffer<float>,
                           float>
        fn_narrow_phase_ee_dcd_query;

    luisa::compute::Shader<1, CDBG, luisa::compute::BufferView<float3>, luisa::compute::BufferView<float3>, float, float, float> fn_compute_repulsion_energy;

    // Scan
    luisa::compute::Shader<1, CDBG>                     fn_preprocess_for_affine_bodies;
    luisa::compute::Shader<1, CDBG, Buffer<uint>, uint> fn_calc_pervert_collion_count;
    luisa::compute::Shader<1, CDBG>                     fn_calc_pervert_prefix_adj_pairs;
    luisa::compute::Shader<1, CDBG>                     fn_calc_pervert_prefix_adj_verts;
    luisa::compute::Shader<1, CDBG, Buffer<uint>, uint> fn_fill_in_pairs_in_vert_adjacent;
    luisa::compute::Shader<1, CDBG, Buffer<uint2>, Buffer<uint>, uint> fn_block_level_sort_contact_triplet;
    luisa::compute::Shader<1, CDBG> fn_specify_target_slot;
    luisa::compute::Shader<1, CDBG> fn_block_level_second_sort_contact_triplet_fill_in;
    luisa::compute::Shader<1, CDBG> fn_specify_target_slot_2_level;
    luisa::compute::Shader<1, CDBG, Buffer<float3>, uint> fn_assemble_triplet_sorted;
    luisa::compute::Shader<1, CDBG, Buffer<float3>, Buffer<uint>, uint> fn_assemble_triplet_sorted_perPair;
    luisa::compute::Shader<1, CDBG, Buffer<MatrixTriplet3x3>> fn_reset_triplet;

    // Assemble
    luisa::compute::Shader<1, CDBG, Buffer<float3>, Buffer<float3>, float, float, Buffer<uint>, Buffer<float3>, uint, Buffer<float3>, Buffer<float3x3>> fn_perPair_assemble_gradient_hessian;
    luisa::compute::Shader<1, CDBG, Buffer<float3>, Buffer<float3x3>> fn_perVert_assemble_gradient_hessian;

    // SpMV
    luisa::compute::Shader<1, CDBG, Buffer<float3>, Buffer<float3>> fn_perPair_spmv;
    luisa::compute::Shader<1, CDBG, Buffer<float3>, Buffer<float3>> fn_perVert_spmv;
};


// class AccdDetector
// {
// private:
//     CollisionDataCCD<luisa::compute::Buffer>* ccd_data;
// };


}  // namespace lcs