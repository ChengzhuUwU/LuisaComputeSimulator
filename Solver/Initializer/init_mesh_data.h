#pragma once

#include "Core/affine_position.h"
#include "MeshOperation/mesh_reader.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/physical_material.h"
#include "SimulationCore/world_data.h"
#include <limits>

namespace lcs
{

	namespace Initializer
	{
		void init_mesh_data(const std::vector<lcs::Initializer::WorldData>& shell_list, lcs::MeshData<std::vector>* mesh_data);
		void upload_mesh_buffers(luisa::compute::Device& device,
			luisa::compute::Stream&						 stream,
			lcs::MeshData<std::vector>*					 input_data,
			lcs::MeshData<luisa::compute::Buffer>*		 output_data);

	} // namespace Initializer

} // namespace lcs