#pragma once

#include <luisa/luisa-compute.h>
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

namespace lcs
{

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

		// public:
		//     template<typename T>
		//     using Buffer = luisa::compute::Buffer<T>;

	public:
		SolverInterface() {}
		~SolverInterface() {}

	protected:
		void init_data(luisa::compute::Device&		  device,
			luisa::compute::Stream&					  stream,
			std::vector<lcs::Initializer::WorldData>& shell_list);
		void compile(AsyncCompiler& compiler);
		void set_data_pointer(SolverData& solver_data, SolverHelper& solver_helper);

	public:
		void restart_system();
		void save_current_frame_state_to_host(const uint frame, const std::string& addition_str);
		void load_saved_state_from_host(const uint frame, const std::string& addition_str);
		void save_mesh_to_obj(const uint frame, const std::string& addition_str = "");
		void device_compute_elastic_energy(luisa::compute::Stream& stream, std::map<std::string, double>& energy_list);
		void compile_compute_energy(AsyncCompiler& compiler);

	public:
		void get_simulation_results_to_host(std::vector<std::vector<std::array<float, 3>>>& output_positions);
		void update_pinned_verts_position(const uint meshIdx,
			const uint								 local_vid,
			const std::array<float, 3>&				 pinned_verts_target_position);
		void update_pinned_body_state(const uint body_id,
			const std::array<float, 3>&			 translation = { 0.0f, 0.0f, 0.0f },
			const std::array<float, 4>&			 rotation = { 0.0f, 0.0f, 0.0f, 0.0f });

	protected:
		void physics_step_prev_operation();
		void physics_step_post_operation();

	private:
		SolverData	 solver_data;
		SolverHelper solver_helper;

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
	};

} // namespace lcs