#pragma once

#include <luisa/luisa-compute.h>
#include <cstdint>
#include <string>
#include "CollisionDetector/lbvh.h"
#include "CollisionDetector/narrow_phase.h"
#include "Core/xbasic_types.h"
#include "Initializer/init_mesh_data.h"
#include "LinearSolver/precond_cg.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/simulation_data.h"
#include "SimulationCore/collision_data.h"
#include "Utils/buffer_filler.h"
#include "Utils/device_parallel.h"
#include "luisa/runtime/buffer.h"
#include "luisa/runtime/device.h"
#include "luisa/runtime/shader.h"
#include "luisa/runtime/stream.h"
#include <Utils/async_compiler.h>
#include <memory>
#include "Energies/soft_inertia_energy.h"
#include "Energies/ground_collision_energy.h"
#include "Energies/spring_energy.h"
#include "Energies/stretch_face_energy.h"
#include "Energies/bending_energy_kernel.h"
#include "Energies/abd_inertia_energy.h"
#include "Energies/abd_ortho_energy.h"
#include "SimulationCore/scene_params.h"

namespace lcs
{

	// Global state for luisa device/context created from Python.
	// Supports two modes:
	//   - Owned:    created by device_init(), resources released by cleanup()
	//   - Borrowed: set by device_set(), caller retains ownership
	struct GlobalState
	{
		// Owned resources (only populated when we create them ourselves)
		std::unique_ptr<luisa::compute::Context> owned_context;
		std::unique_ptr<luisa::compute::Device>	 owned_device;
		std::unique_ptr<luisa::compute::Stream>	 owned_stream;

		// Active pointers (always valid when initialized == true;
		// point to owned_* or external objects depending on mode)
		luisa::compute::Device* device = nullptr;
		luisa::compute::Stream* stream = nullptr;

		bool initialized = false;
		bool owns_resources = false;

		void cleanup()
		{
			device = nullptr;
			stream = nullptr;
			if (owns_resources)
			{
				owned_stream.reset();
				owned_device.reset();
				owned_context.reset();
			}
			initialized = false;
			owns_resources = false;
		}
	};

	class SolverInterface
	{
	private:
		struct SolverData
		{
			lcs::MeshData<std::vector>			  host_mesh_data;
			lcs::MeshData<luisa::compute::Buffer> mesh_data;

			lcs::SimulationData<std::vector>			host_sim_data;
			lcs::SimulationData<luisa::compute::Buffer> sim_data;

			lcs::LbvhData<luisa::compute::Buffer> lbvh_data_face;
			lcs::LbvhData<luisa::compute::Buffer> lbvh_data_edge;

			lcs::CollisionData<std::vector>			   host_collision_data;
			lcs::CollisionData<luisa::compute::Buffer> collision_data;
		};

		struct SolverHelper
		{
			lcs::BufferFiller	buffer_filler;
			lcs::DeviceParallel device_parallel;

			lcs::LBVH lbvh_face;
			lcs::LBVH lbvh_edge;

			lcs::NarrowPhasesDetector	 narrow_phase_detector;
			lcs::ConjugateGradientSolver pcg_solver;
		};

	public:
		SolverInterface()
		{
			scene_params_ptr = lcs::get_scene_params_ptr();
			if (!scene_params_ptr)
			{
				scene_params_ptr = std::make_shared<SceneParams>();
				lcs::set_scene_params_ptr(scene_params_ptr);
			}
		}
		~SolverInterface() {}

	private:
		void init_world_data();
		void init_animation();

	protected:
		void init_data(luisa::compute::Device& device, luisa::compute::Stream& stream);
		void compile(AsyncCompiler& compiler);
		void set_data_pointer(SolverData& solver_data, SolverHelper& solver_helper);

	public:
		void restart_system();
		void device_compute_elastic_energy(luisa::compute::Stream& stream, std::map<std::string, double>& energy_list);
		void compile_compute_energy(AsyncCompiler& compiler);

	public:
		void save_current_frame_state_to_host(const std::string_view& full_path);
		void load_saved_state_from_host(const std::string_view& full_path);
		void save_mesh_to_obj(const std::string_view& full_path);

	public:
		void get_curr_vertices_to_host(std::vector<std::vector<std::array<float, 3>>>& output_positions);
		void get_rest_vertices_to_host(std::vector<std::vector<std::array<float, 3>>>& output_positions);
		void get_triangles_to_host(std::vector<std::vector<std::array<uint, 3>>>& output_triangles);
		uint query_sorted_index_by_registration_id(uint registration_id) const;
		uint query_registration_id_by_sorted_index(uint sorted_idx) const;
		void get_object_sim_result_by_registration_id(uint registration_id,
			std::vector<std::array<float, 3>>&			   output_positions,
			std::vector<std::array<uint, 3>>&			   output_triangles);
		void update_per_vertex_animation(const uint meshIdx, const uint local_vid, const std::array<float, 3>& target_position);
		void update_per_body_animation(const uint body_id, const std::array<float, 3>& target_translation, const std::array<float, 3>& target_rotation);
		void update_default_animations();

	protected:
		void physics_step_prev_operation();
		void physics_step_post_operation();

	private:
		std::vector<Initializer::WorldData> world_data;

		SolverData	 solver_data;
		SolverHelper solver_helper;

		// TODO: Impl with vector
		std::unordered_map<uint, uint>			   vid_to_animation_idx_map;
		std::unordered_map<uint, uint>			   body_to_animation_idx_map;
		std::vector<Animation::PerVertexAnimation> per_vertex_animations;
		std::vector<Animation::PerBodyAnimation>   per_body_animations;

	protected:
		MeshData<std::vector>*			  host_mesh_data;
		MeshData<luisa::compute::Buffer>* mesh_data;

		SimulationData<std::vector>*			host_sim_data;
		SimulationData<luisa::compute::Buffer>* sim_data;

		lcs::LbvhData<luisa::compute::Buffer>* lbvh_data_face;
		lcs::LbvhData<luisa::compute::Buffer>* lbvh_data_edge;

		CollisionData<std::vector>*			   host_collision_data;
		CollisionData<luisa::compute::Buffer>* collision_data;

		BufferFiller*			 buffer_filler;
		DeviceParallel*			 device_parallel;
		LBVH*					 lbvh_face;
		LBVH*					 lbvh_edge;
		NarrowPhasesDetector*	 narrow_phase_detector;
		ConjugateGradientSolver* pcg_solver;
		// lcs::LBVH* collision_detector_narrow_phase;

	public:
		MeshData<std::vector>&				   get_host_mesh_data() const { return *host_mesh_data; }
		SimulationData<std::vector>&		   get_host_sim_data() const { return *host_sim_data; }
		CollisionData<std::vector>&			   get_host_collision_data() const { return *host_collision_data; }
		CollisionData<luisa::compute::Buffer>& get_device_collision_data() const { return *collision_data; }

		// Note that: world_data will be sorted after calling `init_solver()`
		const std::vector<Initializer::WorldData>& get_sorted_world_data() const { return world_data; }
		const Initializer::WorldData&			   get_object_by_registration_id(uint registration_id) const
		{
			const uint sorted_idx = query_sorted_index_by_registration_id(registration_id);
			return world_data[sorted_idx];
		}
		Initializer::WorldData create_world_data()
		{
			return Initializer::WorldData();
		}
		uint register_world_data(const Initializer::WorldData& wd)
		{
			uint new_registration_id = static_cast<uint>(world_data.size());
			world_data.emplace_back(wd).registration_index = new_registration_id;
			return new_registration_id;
		}
		// uint register_world_data_from_file_path(const std::string_view& name, const std::string_view& file_path)
		// {
		// 	auto& data = world_data.emplace_back().load_mesh_from_path(file_path).set_name(name);
		// 	data.registration_index = static_cast<uint>(world_data.size() - 1);
		// 	return data;
		// }
		// uint register_world_data_from_array(const std::string_view& name, const std::vector<std::array<float, 3>>& vertices, const std::vector<std::array<uint, 3>>& faces)
		// {
		// 	auto& data = world_data.emplace_back().load_mesh_from_array(vertices, faces).set_name(name);
		// 	data.registration_index = static_cast<uint>(world_data.size() - 1);
		// 	return data;
		// }

		// Device management: create and own a luisa device/stream.
		// binary_path : argv[0], used by luisa::compute::Context. Empty = use current executable path.
		// backend_name: e.g. "metal", "cuda", "dx". Empty = platform default.
		void create_device(const std::string& binary_path, const std::string& backend_name = "");

		// Device management: borrow an external device/stream (non-owning).
		// The caller must ensure the objects outlive this solver.
		void set_device_from_pointers(uintptr_t device_ptr, uintptr_t stream_ptr);

		// Release owned device resources (no-op when using borrowed device).
		void cleanup_device();

		// Accessors for inter-module sharing.
		uintptr_t	 get_device_ptr() const;
		uintptr_t	 get_stream_ptr() const;
		SceneParams& get_config() const { return *scene_params_ptr; }

	protected:
		// Accessors for energy objects for derived solvers
		SoftInertiaEnergy*	   get_inertia_energy() const { return inertia_energy.get(); }
		AbdInertiaEnergy*	   get_abd_inertia_energy() const { return abd_inertia_energy.get(); }
		GroundCollisionEnergy* get_ground_collision_energy() const { return ground_collision_energy.get(); }
		SpringEnergy*		   get_spring_energy() const { return spring_energy.get(); }
		StretchFaceEnergy*	   get_stretch_face_energy() const { return stretch_face_energy.get(); }
		BendingEnergy*		   get_bending_energy() const { return bending_energy.get(); }
		AbdOrthoEnergy*		   get_abd_ortho_energy() const { return abd_ortho_energy.get(); }

	private:
		luisa::compute::Shader<1, luisa::compute::BufferView<float>> fn_reset_float;

		std::unique_ptr<SoftInertiaEnergy>	   inertia_energy;
		std::unique_ptr<AbdInertiaEnergy>	   abd_inertia_energy;
		std::unique_ptr<GroundCollisionEnergy> ground_collision_energy;
		std::unique_ptr<SpringEnergy>		   spring_energy;
		std::unique_ptr<StretchFaceEnergy>	   stretch_face_energy;
		std::unique_ptr<BendingEnergy>		   bending_energy;
		std::unique_ptr<AbdOrthoEnergy>		   abd_ortho_energy;
		// luisa::compute::Shader<1,
		//     luisa::compute::BufferView<float3>,
		//     luisa::compute::BufferView<float3>,
		//     float,
		//     float,
		//     float
		//     > fn_compute_repulsion_energy_from_vf;
		// luisa::compute::Shader<1,
		//     luisa::compute::BufferView<float3>,
		//     luisa::compute::BufferView<float3>,
		//     float,
		//     float,
		//     float
		//     >  fn_compute_repulsion_energy_from_ee;

	protected:
		luisa::fiber::scheduler scheduler;

		// Device/stream state owned or borrowed by this solver instance.
		GlobalState					 device_state;
		std::shared_ptr<SceneParams> scene_params_ptr;

		SceneParams& get_scene_params() const { return *scene_params_ptr; }
	};

} // namespace lcs