#include "bending_energy_kernel.h"
#include "SimulationCore/scene_params.h"
#include "Utils/cpu_parallel.h"
#include "Utils/reduce_helper.h"

using namespace luisa::compute;

namespace lcs
{
	BendingEnergy::BendingEnergy(BufferView<float> sa_system_energy) noexcept
		: _sa_system_energy(sa_system_energy)
	{
	}

	void BendingEnergy::compile(AsyncCompiler& compiler)
	{
		luisa::compute::ShaderOption default_option = { .enable_debug_info = false };
		compiler.compile<1>(
			_shader,
			[sa_system_energy = _sa_system_energy](Var<Constitutions::BendingEdge<luisa::compute::Buffer>> constraint,
				Var<BufferView<float3>>																	   sa_x,
				Float																					   scaling)
			{
				auto& sa_edges = constraint.constraint_indices;
				auto& sa_bending_edges_rest_angle = constraint.sa_bending_edges_rest_angle;
				auto& sa_bending_edges_rest_area = constraint.sa_bending_edges_rest_area;
				auto& sa_bending_edges_stiffness = constraint.sa_bending_edges_stiffness;

				const Uint eid = dispatch_id().x;
				Float	   energy = 0.0f;
				{
					const Uint4 edge = sa_edges->read(eid);

					Float3 vert_pos[4] = { sa_x.read(edge[0]), sa_x.read(edge[1]), sa_x.read(edge[2]), sa_x.read(edge[3]) };
					Float  rest_angle = sa_bending_edges_rest_angle->read(eid);
					Float  angle =
						BendingEnergyUtils::compute_theta(vert_pos[0], vert_pos[1], vert_pos[2], vert_pos[3]);
					Float delta_angle = angle - rest_angle;
					Float area = sa_bending_edges_rest_area->read(eid);
					energy = 0.5f * sa_bending_edges_stiffness->read(eid) * scaling * area * delta_angle * delta_angle;
				};

				energy = ParallelIntrinsic::block_intrinsic_reduce(eid, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
				$if(eid % 256 == 0)
				{
					sa_system_energy->atomic(offset_bending).fetch_add(energy);
				};
			},
			default_option);

		// evaluate gradient/hessian shader
		compiler.compile<1>(
			_eval_shader,
			[](Var<Constitutions::BendingEdge<luisa::compute::Buffer>> constraint, Var<BufferView<float3>> sa_x, Float scaling)
			{
				auto& sa_edges = constraint.constraint_indices;
				auto& sa_bending_edges_rest_angle = constraint.sa_bending_edges_rest_angle;
				auto& sa_bending_edges_rest_area = constraint.sa_bending_edges_rest_area;
				auto& sa_bending_edges_stiffness = constraint.sa_bending_edges_stiffness;
				auto& output_gradient_ptr = constraint.constraint_gradients;
				auto& output_hessian_ptr = constraint.constraint_hessians;

				const UInt	eid = dispatch_id().x;
				const UInt4 edge = sa_edges->read(eid);

				Float3 vert_pos[4] = {
					sa_x->read(edge[0]),
					sa_x->read(edge[1]),
					sa_x->read(edge[2]),
					sa_x->read(edge[3]),
				};
				Float3 gradients[4] = {
					make_float3(0.0f),
					make_float3(0.0f),
					make_float3(0.0f),
					make_float3(0.0f),
				};

				const Float rest_angle = sa_bending_edges_rest_angle->read(eid);
				const Float angle =
					BendingEnergyUtils::compute_d_theta_d_x(vert_pos[0], vert_pos[1], vert_pos[2], vert_pos[3], gradients);
				const Float delta_angle = angle - rest_angle;

				const Float area = sa_bending_edges_rest_area->read(eid);
				const Float stiff = sa_bending_edges_stiffness->read(eid) * scaling * area;

				{
					output_gradient_ptr->write(eid * 4 + 0, stiff * delta_angle * gradients[0]);
					output_gradient_ptr->write(eid * 4 + 1, stiff * delta_angle * gradients[1]);
					output_gradient_ptr->write(eid * 4 + 2, stiff * delta_angle * gradients[2]);
					output_gradient_ptr->write(eid * 4 + 3, stiff * delta_angle * gradients[3]);

					auto outer = [&](const uint ii, const uint jj) -> Float3x3
					{ return stiff * outer_product(gradients[ii], gradients[jj]); };
					for (uint ii = 0; ii < 4; ii++)
					{
						for (uint jj = 0; jj < 4; jj++)
						{
							output_hessian_ptr->write(eid * 16 + ii * 4 + jj, outer(ii, jj));
						}
					}
				}
			},
			default_option);
	}

	void BendingEnergy::device_compute_energy(luisa::compute::Stream& stream)
	{
	}

	void BendingEnergy::device_compute_energy(luisa::compute::Stream& stream,
		const Constitutions::BendingEdge<luisa::compute::Buffer>&	  constraint,
		const luisa::compute::Buffer<float3>&						  sa_x,
		float														  scaling,
		size_t														  dispatch_count)
	{
		stream << _shader(constraint, sa_x, scaling).dispatch(dispatch_count);
	}

	void BendingEnergy::device_evaluate(luisa::compute::Stream&	  stream,
		const Constitutions::BendingEdge<luisa::compute::Buffer>& constraint,
		const luisa::compute::Buffer<float3>&					  sa_x,
		float													  scaling,
		size_t													  dispatch_count)
	{
		stream << _eval_shader(constraint, sa_x.view(), scaling).dispatch(dispatch_count);
	}

	double BendingEnergy::host_evaluate(const std::vector<float>& host_energy)
	{
		return host_energy[offset_bending];
	}

	void BendingEnergy::host_evaluate(lcs::SimulationData<std::vector>& host_sim_data, lcs::MeshData<std::vector>& host_mesh_data)
	{
		auto& bending_edges = host_sim_data.get_bending_edge_data();

		CpuParallel::parallel_for(
			0,
			bending_edges.get_num_indices(),
			[sa_x = std::span(host_sim_data.sa_x),
				sa_bending_edges = std::span(bending_edges.constraint_indices),
				sa_bending_edges_Q = std::span(bending_edges.sa_bending_edges_Q),
				sa_bending_edges_rest_angle = std::span(bending_edges.sa_bending_edges_rest_angle),
				sa_bending_edges_rest_area = std::span(bending_edges.sa_bending_edges_rest_area),
				sa_bending_edges_stiffness = std::span(bending_edges.sa_bending_edges_stiffness),
				output_gradient_ptr = std::span(bending_edges.constraint_gradients),
				output_hessian_ptr = std::span(bending_edges.constraint_hessians),
				scaling = get_scene_params().get_bending_stiffness_scaling()](const uint eid)
			{
				uint4  edge = sa_bending_edges[eid];
				float3 vert_pos[4] = { sa_x[edge[0]], sa_x[edge[1]], sa_x[edge[2]], sa_x[edge[3]] };
				float3 gradients[4] = { Zero3, Zero3, Zero3, Zero3 };

				const float rest_angle = sa_bending_edges_rest_angle[eid];
				const float angle =
					BendingEnergyUtils::compute_d_theta_d_x(vert_pos[0], vert_pos[1], vert_pos[2], vert_pos[3], gradients);
				const float delta_angle = angle - rest_angle;

				const float area = sa_bending_edges_rest_area[eid];
				const float stiff = sa_bending_edges_stiffness[eid] * scaling * area;
				output_gradient_ptr[eid * 4 + 0] = stiff * delta_angle * gradients[0];
				output_gradient_ptr[eid * 4 + 1] = stiff * delta_angle * gradients[1];
				output_gradient_ptr[eid * 4 + 2] = stiff * delta_angle * gradients[2];
				output_gradient_ptr[eid * 4 + 3] = stiff * delta_angle * gradients[3];

				auto outer = [&gradients, stiff](uint ii, uint jj) -> float3x3
				{ return outer_product(stiff * gradients[ii], gradients[jj]); };

				for (uint ii = 0; ii < 4; ii++)
				{
					for (uint jj = 0; jj < 4; jj++)
					{
						output_hessian_ptr[eid * 16 + ii * 4 + jj] = outer(ii, jj);
					}
				}
			});
	}

} // namespace lcs

namespace lcs
{
	namespace BendingEnergyUtils
	{
		using Float3 = luisa::compute::Float3;
		using Float = luisa::compute::Float;

		namespace detail
		{
			using HostVector12 = std::array<float3, 4>;
			static inline HostVector12 face_dihedral_angle_grad(const float3& v2, const float3& v0, const float3& v1, const float3& v3)
			{
				const float3 e0 = v1 - v0;
				const float3 e1 = v2 - v0;
				const float3 e2 = v3 - v0;
				const float3 e3 = v2 - v1;
				const float3 e4 = v3 - v1;
				const float3 n1 = luisa::cross(e0, e1);
				const float3 n2 = luisa::cross(e2, e0);
				const float	 n1_sqnm = luisa::dot(n1, n1);
				const float	 n2_sqnm = luisa::dot(n2, n2);
				const float	 e0_norm = luisa::length(e0);
				assert(n1_sqnm > 0.0f);
				assert(n2_sqnm > 0.0f);
				assert(e0_norm > 0.0f);

				HostVector12 grad;
				grad[0] = -e0_norm / n1_sqnm * n1;
				grad[1] = -luisa::dot(e0, e3) / (e0_norm * n1_sqnm) * n1 - luisa::dot(e0, e4) / (e0_norm * n2_sqnm) * n2;
				grad[2] = luisa::dot(e0, e1) / (e0_norm * n1_sqnm) * n1 + luisa::dot(e0, e2) / (e0_norm * n2_sqnm) * n2;
				grad[3] = -e0_norm / n2_sqnm * n2;
				return grad;
			}

			using DeviceVector12 = luisa::compute::ArrayFloat3<4>;
			static inline DeviceVector12 face_dihedral_angle_grad(const Float3& v2,
				const Float3&													v0,
				const Float3&													v1,
				const Float3&													v3)
			{

				const Float3 e0 = v1 - v0;
				const Float3 e1 = v2 - v0;
				const Float3 e2 = v3 - v0;
				const Float3 e3 = v2 - v1;
				const Float3 e4 = v3 - v1;
				const Float3 n1 = luisa::compute::cross(e0, e1);
				const Float3 n2 = luisa::compute::cross(e2, e0);
				const Float	 n1_sqnm = luisa::compute::dot(n1, n1);
				const Float	 n2_sqnm = luisa::compute::dot(n2, n2);
				const Float	 e0_norm = luisa::compute::length(e0);
				luisa::compute::device_assert(n1_sqnm > 0.0f);
				luisa::compute::device_assert(n2_sqnm > 0.0f);
				luisa::compute::device_assert(e0_norm > 0.0f);

				DeviceVector12 grad;
				grad[0] = -e0_norm / n1_sqnm * n1;
				grad[1] = -luisa::compute::dot(e0, e3) / (e0_norm * n1_sqnm) * n1
					- luisa::compute::dot(e0, e4) / (e0_norm * n2_sqnm) * n2;
				grad[2] = luisa::compute::dot(e0, e1) / (e0_norm * n1_sqnm) * n1
					+ luisa::compute::dot(e0, e2) / (e0_norm * n2_sqnm) * n2;
				grad[3] = -e0_norm / n2_sqnm * n2;
				return grad;
			}

			static inline float face_dihedral_angle(const float3& v0, const float3& v1, const float3& v2, const float3& v3)
			{
				const float3 n1 = luisa::cross(v1 - v0, v2 - v0);
				const float3 n2 = luisa::cross(v2 - v3, v1 - v3);
				float		 dot = luisa::dot(n1, n2) / luisa::sqrt(luisa::dot(n1, n1) * luisa::dot(n2, n2));
				float		 angle = luisa::acos(luisa::max(-1.0f, luisa::min(1.0f, dot)));
				float		 sign = luisa::sign(luisa::dot(luisa::cross(n2, n1), v1 - v2));
				angle = sign * angle;
				return angle;
			}

			static inline Float face_dihedral_angle(const Float3& v0, const Float3& v1, const Float3& v2, const Float3& v3)
			{
				const Float3 n1 = luisa::compute::cross(v1 - v0, v2 - v0);
				const Float3 n2 = luisa::compute::cross(v2 - v3, v1 - v3);
				Float		 dot = luisa::compute::dot(n1, n2)
					/ luisa::compute::sqrt(luisa::compute::dot(n1, n1) * luisa::compute::dot(n2, n2));
				Float angle = luisa::compute::acos(luisa::compute::max(-1.0f, luisa::compute::min(1.0f, dot)));
				Float sign = luisa::compute::sign(luisa::compute::dot(luisa::compute::cross(n2, n1), v1 - v2));
				angle = sign * angle;
				return angle;
			}

			inline uint4 remap(const uint4& hinge)
			{
				return luisa::make_uint4(hinge[2], hinge[1], hinge[0], hinge[3]);
			}
			inline luisa::compute::Uint4 remap(const luisa::compute::Uint4& hinge)
			{
				return luisa::compute::make_uint4(hinge[2], hinge[1], hinge[0], hinge[3]);
			}
		} // namespace detail

		float compute_d_theta_d_x(const float3& x2, const float3& x1, const float3& x0, const float3& x3, float3 gradient[4])
		{
			const auto angle = detail::face_dihedral_angle(x0, x1, x2, x3);
			const auto angle_grad = detail::face_dihedral_angle_grad(x0, x1, x2, x3);
			gradient[2] = angle_grad[0];
			gradient[1] = angle_grad[1];
			gradient[0] = angle_grad[2];
			gradient[3] = angle_grad[3];
			return angle;
		}
		Float compute_d_theta_d_x(const Float3& x2, const Float3& x1, const Float3& x0, const Float3& x3, Float3 gradient[4])
		{
			const auto angle = detail::face_dihedral_angle(x0, x1, x2, x3);
			const auto angle_grad = detail::face_dihedral_angle_grad(x0, x1, x2, x3);
			gradient[2] = angle_grad[0];
			gradient[1] = angle_grad[1];
			gradient[0] = angle_grad[2];
			gradient[3] = angle_grad[3];
			return angle;
		}
		float compute_theta(const float3& x2, const float3& x1, const float3& x0, const float3& x3)
		{
			return detail::face_dihedral_angle(x0, x1, x2, x3);
		}
		Float compute_theta(const Float3& x2, const Float3& x1, const Float3& x0, const Float3& x3)
		{
			return detail::face_dihedral_angle(x0, x1, x2, x3);
		}

	}; // namespace BendingEnergyUtils

} // namespace lcs