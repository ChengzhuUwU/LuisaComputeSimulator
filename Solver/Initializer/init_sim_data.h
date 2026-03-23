#pragma once

#include "Initializer/init_mesh_data.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/simulation_data.h"

namespace lcs::Initializer
{

	void init_sim_data(const std::vector<lcs::Initializer::WorldData>& shell_infos,
		lcs::MeshData<std::vector>*									   mesh_data,
		lcs::SimulationData<std::vector>*							   sim_data,
		const std::vector<lcs::FixedJointConstraintDesc>&			   fixed_joint_descs,
		const std::vector<lcs::PrismaticJointConstraintDesc>&		   prismatic_joint_descs,
		const std::vector<lcs::RevoluteJointConstraintDesc>&		   revolute_joint_descs);
	void upload_sim_buffers(luisa::compute::Device&	 device,
		luisa::compute::Stream&						 stream,
		lcs::SimulationData<std::vector>*			 input_data,
		lcs::SimulationData<luisa::compute::Buffer>* output_data);

	void resize_pcg_data(luisa::compute::Device&	 device,
		luisa::compute::Stream&						 stream,
		lcs::MeshData<std::vector>*					 mesh_data,
		lcs::SimulationData<std::vector>*			 host_data,
		lcs::SimulationData<luisa::compute::Buffer>* device_data);

	void init_colored_data(lcs::SimulationData<std::vector>* sim_data);
	void upload_colored_data(luisa::compute::Device& device,
		luisa::compute::Stream&						 stream,
		lcs::SimulationData<std::vector>*			 input_data,
		lcs::SimulationData<luisa::compute::Buffer>* output_data);

} // namespace lcs::Initializer