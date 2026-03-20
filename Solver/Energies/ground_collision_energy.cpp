#include "ground_collision_energy.h"
#include "Energies/detail/ground_collision_energy.hpp"
#include "CollisionDetector/cipc_kernel.hpp"
#include "CollisionDetector/friction_kernel.hpp"
#include "SimulationCore/scene_params.h"
#include "Utils/buffer_adder.h"
#include "Utils/cpu_parallel.h"

using namespace luisa::compute;

namespace lcs
{
	GroundCollisionEnergy::GroundCollisionEnergy(BufferView<float> sa_rest_vert_area,
		BufferView<uint>										   sa_is_fixed,
		BufferView<uint>										   sa_contact_active_verts,
		BufferView<float>										   sa_contact_active_verts_offset,
		BufferView<float>										   sa_contact_active_verts_d_hat,
		BufferView<float>										   sa_contact_active_verts_friction_coeff,
		BufferView<float3>										   sa_x_step_start,
		BufferView<float3>										   sa_x,
		BufferView<float3>										   sa_scaled_model_x,
		BufferView<VertexToDofMap>								   sa_x_to_dof_map,
		BufferView<float>										   sa_system_energy) noexcept
		: _sa_rest_vert_area(sa_rest_vert_area)
		, _sa_is_fixed(sa_is_fixed)
		, _sa_contact_active_verts(sa_contact_active_verts)
		, _sa_contact_active_verts_offset(sa_contact_active_verts_offset)
		, _sa_contact_active_verts_d_hat(sa_contact_active_verts_d_hat)
		, _sa_contact_active_verts_friction_coeff(sa_contact_active_verts_friction_coeff)
		, _sa_x_step_start(sa_x_step_start)
		, _sa_x(sa_x)
		, _sa_scaled_model_x(sa_scaled_model_x)
		, _sa_x_to_dof_map(sa_x_to_dof_map)
		, _sa_system_energy(sa_system_energy)
	{
	}

	void GroundCollisionEnergy::device_compute_energy(luisa::compute::Stream& stream,
		const Constitutions::SoftInertia<luisa::compute::Buffer>&			  constraint,
		float																  floor_y,
		bool																  use_ground_collision,
		float																  stiffness,
		uint																  collision_type,
		size_t																  dispatch_count)
	{
		stream << _eval_soft_shader(constraint, floor_y, use_ground_collision, stiffness, collision_type).dispatch(_sa_contact_active_verts.size());
	}

	void GroundCollisionEnergy::device_compute_energy(luisa::compute::Stream& stream,
		const Constitutions::AbdInertia<luisa::compute::Buffer>&			  constraint,
		float																  floor_y,
		bool																  use_ground_collision,
		float																  stiffness,
		uint																  vid_start,
		uint																  collision_type,
		size_t																  dispatch_count)
	{
		stream << _eval_abd_shader(constraint, floor_y, use_ground_collision, stiffness, vid_start, collision_type)
					  .dispatch(_sa_contact_active_verts.size());
	}

	void GroundCollisionEnergy::compile(AsyncCompiler& compiler)
	{
		luisa::compute::ShaderOption default_option = compiler.default_option();
		compiler.compile<1>(
			_shader,
			[sa_rest_vert_area = _sa_rest_vert_area,
				sa_is_fixed = _sa_is_fixed,
				sa_contact_active_verts_offset = _sa_contact_active_verts_offset,
				sa_contact_active_verts_d_hat = _sa_contact_active_verts_d_hat,
				sa_contact_active_verts_friction_coeff = _sa_contact_active_verts_friction_coeff,
				sa_x_step_start = _sa_x_step_start,
				sa_system_energy = _sa_system_energy](
				Var<BufferView<float3>> sa_x, Float floor_y, Bool use_ground_collision, Float stiffness, Uint collision_type)
			{
				const Uint vid = dispatch_id().x;

				Float energy_repulsive = 0.0f;
				Float energy_friction = 0.0f;
				Bool  is_fixed = sa_is_fixed->read(vid) != 0;

				$if(use_ground_collision & !is_fixed)
				{
					Float d_hat = sa_contact_active_verts_d_hat->read(vid);
					Float thickness = sa_contact_active_verts_offset->read(vid);
					Float area = sa_rest_vert_area->read(vid);
					Float stiff = stiffness * area;

					Float3 normal = make_float3(0.0f, 1.0f, 0.0f);

					Float3 x_k = sa_x->read(vid);
					Float3 x_0 = sa_x_step_start->read(vid);

					Float curr_dist = x_k.y - floor_y;
					$if(curr_dist - thickness < d_hat)
					{
						$if(collision_type == 0)
						{
							energy_repulsive = detail::ground_collision_energy::repulsive_energy(
								curr_dist, thickness, d_hat, stiff, 0u);
						}
						$else
						{
							energy_repulsive = detail::ground_collision_energy::repulsive_energy(
								curr_dist, thickness, d_hat, stiff, 1u);
						};
					};

					Float init_dist = x_0.y - floor_y;
					$if(init_dist - thickness < d_hat)
					{
						Float3 rel_dx = x_k - x_0;

						Float k1 = 0.0f;
						$if(collision_type == 0)
						{
							k1 = detail::ground_collision_energy::repulsive_first_derivative(
								init_dist, thickness, d_hat, stiff, 0u);
						}
						$else
						{
							k1 = detail::ground_collision_energy::repulsive_first_derivative(
								init_dist, thickness, d_hat, stiff, 1u);
						};

						Float friction_mu = sa_contact_active_verts_friction_coeff->read(vid);
						Float friction_eps = Friction::ando_barrier::friction_eps;

						auto lambda = -k1 * friction_mu;
						energy_friction = detail::ground_collision_energy::friction_energy(
							lambda, normal, rel_dx, friction_eps);
					};
				};

				Float2 energy =
					ParallelIntrinsic::block_intrinsic_reduce(
						make_float2(energy_repulsive, energy_friction),
						ParallelIntrinsic::warp_reduce_op_sum<float2>);
				$if(vid % 256 == 0)
				{
					sa_system_energy->atomic(offset_ground_collision).fetch_add(energy.x);
					sa_system_energy->atomic(offset_ground_friction).fetch_add(energy.y);
				};
			},
			default_option);

		auto calculate_per_vert_grad_hess_template =
			[sa_x = _sa_x,
				sa_x_step_start = _sa_x_step_start,
				sa_rest_vert_area = _sa_rest_vert_area,
				sa_is_fixed = _sa_is_fixed,
				sa_contact_active_verts_offset = _sa_contact_active_verts_offset,
				sa_contact_active_verts_d_hat = _sa_contact_active_verts_d_hat,
				sa_contact_active_verts_friction_coeff =
					_sa_contact_active_verts_friction_coeff](const Uint vid,
				Float3&													out_gradient,
				Float3x3&												out_hessian,
				const Float												floor_y,
				const Bool												use_ground_collision,
				const Float												stiffness,
				const Uint												collision_type)
		{
			Bool collide = false;
			$if(!sa_is_fixed->read(vid) & use_ground_collision)
			{
				Float3 x_k = sa_x->read(vid);
				Float3 x_0 = sa_x_step_start->read(vid);

				Float d_hat = sa_contact_active_verts_d_hat->read(vid);
				Float thickness = sa_contact_active_verts_offset->read(vid);

				float3 normal = luisa::make_float3(0, 1, 0);
				Float  area = sa_rest_vert_area->read(vid);
				Float  stiff = stiffness * area;

				// Repulsion
				Float curr_dist = x_k.y - floor_y;
				$if(curr_dist - thickness < d_hat)
				{
					Float k1;
					Float k2;
					$if(collision_type == 0)
					{
						k1 = detail::ground_collision_energy::repulsive_first_derivative(
							curr_dist, thickness, d_hat, stiff, 0u);
						k2 = detail::ground_collision_energy::repulsive_second_derivative(
							curr_dist, thickness, d_hat, stiff, 0u);
					}
					$else
					{
						k1 = detail::ground_collision_energy::repulsive_first_derivative(
							curr_dist, thickness, d_hat, stiff, 1u);
						k2 = detail::ground_collision_energy::repulsive_second_derivative(
							curr_dist, thickness, d_hat, stiff, 1u);
					};

					out_gradient = k1 * normal;
					out_hessian = k2 * outer_product(normal, normal);
					collide = true;
				};

				// Friction
				Float init_dist = x_0.y - floor_y;
				$if(init_dist - thickness < d_hat)
				{
					Float k1;
					$if(collision_type == 0)
					{
						k1 = detail::ground_collision_energy::repulsive_first_derivative(
							init_dist, thickness, d_hat, stiff, 0u);
					}
					$else
					{
						k1 = detail::ground_collision_energy::repulsive_first_derivative(
							init_dist, thickness, d_hat, stiff, 1u);
					};

					Float3 rel_dx = x_k - x_0;
					Float  friction_mu = sa_contact_active_verts_friction_coeff->read(vid);
					Float  friction_eps = Friction::ando_barrier::friction_eps;
					auto   lambda_mu = -k1 * friction_mu;
					auto   friction_grad_hess = detail::ground_collision_energy::friction_gradient_hessian<Float, Float3, Float3x3>(
						  lambda_mu, normal, rel_dx, friction_eps);
					out_gradient += friction_grad_hess.first;
					out_hessian += friction_grad_hess.second;
					collide = true;
				};
			};
			return collide;
		};

		compiler.compile<1>(
			_eval_soft_shader,
			[calculate_per_vert_grad_hess_template,
				sa_contact_active_verts = _sa_contact_active_verts,
				sa_x_to_dof_map = _sa_x_to_dof_map](Var<Constitutions::SoftInertia<luisa::compute::Buffer>> contraint,
				Float																						floor_y,
				Bool																						use_ground_collision,
				Float																						stiffness,
				Uint																						collision_type)
			{
				auto& output_gradient = contraint.constraint_gradients;
				auto& output_hessian = contraint.constraint_hessians;

				const UInt index = dispatch_id().x;
				const UInt vid = sa_contact_active_verts->read(index);
				const auto dof_info = sa_x_to_dof_map->read(vid);
				$if(dof_info->is_soft_body())
				{
					const Uint dof_idx = dof_info->get_dof_idx();

					Float3	 grad = make_float3(0.0f);
					Float3x3 hess = make_float3x3(0.0f);
					Bool	 collide = calculate_per_vert_grad_hess_template(
						vid, grad, hess, floor_y, use_ground_collision, stiffness, collision_type);

					$if(collide)
					{
						BufferOp::buffer_add(output_gradient, vid, grad);
						BufferOp::buffer_add(output_hessian, vid, hess);
					};
				};
			},
			default_option);

		compiler.compile<1>(
			_eval_abd_shader,
			[sa_contact_active_verts = _sa_contact_active_verts,
				sa_scaled_model_x = _sa_scaled_model_x,
				sa_x_to_dof_map = _sa_x_to_dof_map,
				calculate_per_vert_grad_hess_template](Var<Constitutions::AbdInertia<luisa::compute::Buffer>> constraint,
				Float																						  floor_y,
				Bool																						  use_ground_collision,
				Float																						  stiffness,
				Uint																						  vid_start,
				Uint																						  collision_type)
			{
				auto& abd_gradients = constraint.constraint_gradients;
				auto& abd_hessians = constraint.constraint_hessians;

				// const UInt vid = vid_start + dispatch_id().x;
				const UInt index = dispatch_id().x;
				const UInt vid = sa_contact_active_verts->read(index);

				const auto dof_info = sa_x_to_dof_map->read(vid);
				$if(dof_info->is_rigid_body())
				{
					const Uint dof_idx = dof_info->get_dof_idx();
					const Uint body_idx = (dof_idx - vid_start) / 4;

					Float3	 grad = make_float3(0.0f);
					Float3x3 hess = make_float3x3(0.0f);
					Bool	 collide = calculate_per_vert_grad_hess_template(
						vid, grad, hess, floor_y, use_ground_collision, stiffness, collision_type);

					const Float4 weight = make_float4(1.0f, sa_scaled_model_x->read(vid));

					for (uint ii = 0; ii < 4; ii++)
					{
						Float  wi = weight[ii];
						Float3 affine_grad = wi * grad;
						BufferOp::atomic_buffer_add(abd_gradients, 4 * body_idx + ii, affine_grad);
						for (uint jj = 0; jj < 4; jj++)
						{
							Float	 wj = weight[jj];
							Float3x3 affine_hess = wi * wj * hess;
							BufferOp::atomic_buffer_add(abd_hessians, 16 * body_idx + ii * 4 + jj, affine_hess);
						}
					}
				};
			},
			default_option);
	}

	void GroundCollisionEnergy::device_compute_energy(luisa::compute::Stream& stream)
	{
		// left empty — use overload below with explicit args
	}

	void GroundCollisionEnergy::device_compute_energy(luisa::compute::Stream& stream,
		const luisa::compute::Buffer<float3>&								  sa_x,
		float																  floor_y,
		bool																  use_ground_collision,
		float																  stiffness,
		uint																  collision_type,
		size_t																  dispatch_count)
	{
		stream << _shader(sa_x, floor_y, use_ground_collision, stiffness, collision_type).dispatch(dispatch_count);
	}

	double GroundCollisionEnergy::host_evaluate(const std::vector<float>& host_energy)
	{
		return host_energy[offset_ground_collision];
	}

	void GroundCollisionEnergy::host_evaluate(lcs::SimulationData<std::vector>& host_sim_data,
		lcs::MeshData<std::vector>&												host_mesh_data)
	{
		if (!get_scene_params().use_floor)
			return;

		auto calculate_per_vert_grad_hess_template =
			[sa_x = std::span(host_sim_data.sa_x),
				sa_x_step_start = std::span(host_sim_data.sa_x_step_start),
				sa_contact_active_verts_offset = std::span(host_sim_data.sa_contact_active_verts_offset),
				sa_contact_active_verts_d_hat = std::span(host_sim_data.sa_contact_active_verts_d_hat),
				sa_contact_active_verts_friction_coeff = std::span(host_sim_data.sa_contact_active_verts_friction_coeff),
				sa_is_fixed = std::span(host_mesh_data.sa_is_fixed),
				sa_rest_vert_area = std::span(host_mesh_data.sa_rest_vert_area),
				floor_y = get_scene_params().floor.y,
				stiffness_ground = get_scene_params().stiffness_collision,
				collision_type = get_scene_params().contact_energy_type](
				const uint vid, float3& out_gradient, float3x3& out_hessian) -> bool
		{
			bool collide = false;
			if (!sa_is_fixed[vid] && get_scene_params().use_floor)
			{
				float3 x_k = sa_x[vid];
				float3 x_0 = sa_x_step_start[vid];

				float thickness = sa_contact_active_verts_offset[vid];
				float d_hat = sa_contact_active_verts_d_hat[vid];
				float curr_dist = x_k.y - floor_y;
				float init_dist = x_0.y - floor_y;

				float3 normal = luisa::make_float3(0, 1, 0);
				float  area = sa_rest_vert_area[vid];
				float  stiff = stiffness_ground * area;

				if (curr_dist - thickness < d_hat)
				{
					float k1;
					float k2;
					if (collision_type == 0)
					{
						k1 = detail::ground_collision_energy::repulsive_first_derivative(curr_dist, thickness, d_hat, stiff, 0u);
						k2 = detail::ground_collision_energy::repulsive_second_derivative(curr_dist, thickness, d_hat, stiff, 0u);
					}
					else
					{
						k1 = detail::ground_collision_energy::repulsive_first_derivative(curr_dist, thickness, d_hat, stiff, 1u);
						k2 = detail::ground_collision_energy::repulsive_second_derivative(curr_dist, thickness, d_hat, stiff, 1u);
					}
					if (luisa::isnan(k1 * k2) || luisa::isinf(k1 * k2))
					{
						LUISA_ERROR("NaN detected in ground collision computation");
					}
					out_gradient = k1 * normal;
					out_hessian = k2 * outer_product(normal, normal);
					collide = true;
				}
				if (init_dist - thickness < d_hat)
				{
					float k1;
					if (collision_type == 0)
					{
						k1 = detail::ground_collision_energy::repulsive_first_derivative(init_dist, thickness, d_hat, stiff, 0u);
					}
					else
					{
						k1 = detail::ground_collision_energy::repulsive_first_derivative(init_dist, thickness, d_hat, stiff, 1u);
					}
					float3 rel_dx = x_k - x_0;
					float  friction_mu = sa_contact_active_verts_friction_coeff[vid];
					float  friction_eps = Friction::ando_barrier::friction_eps;
					auto   lambda_mu = -k1 * friction_mu;
					auto   friction_grad_hess = detail::ground_collision_energy::friction_gradient_hessian<float, float3, float3x3>(
						  lambda_mu, normal, rel_dx, friction_eps);
					out_gradient += friction_grad_hess.first;
					out_hessian = out_hessian + friction_grad_hess.second;
					collide = true;
				}
			}
			return collide;
		};

		auto& inertia_data = host_sim_data.get_soft_inertia_data();
		auto  active_verts = std::span(host_sim_data.sa_contact_active_verts);
		if (inertia_data.is_valid())
		{
			CpuParallel::parallel_for(0,
				active_verts.size(),
				[output_gradient = std::span(inertia_data.constraint_gradients),
					output_hessian = std::span(inertia_data.constraint_hessians),
					sa_contact_active_verts = active_verts,
					sa_x_to_dof_map = std::span(host_sim_data.sa_x_to_dof_map),
					&calculate_per_vert_grad_hess_template](const uint index)
				{
					const uint vid = sa_contact_active_verts[index];
					if (!sa_x_to_dof_map[vid].is_soft_body())
						return;

					float3	 gradient = Zero3;
					float3x3 hessian = zero3x3;
					bool	 collide = calculate_per_vert_grad_hess_template(vid, gradient, hessian);
					if (collide)
					{
						BufferOp::buffer_add(output_gradient, vid, gradient);
						BufferOp::buffer_add(output_hessian, vid, hessian);
					}
				});
		}

		auto& abd_data = host_sim_data.get_abd_inertia_data();

		if (abd_data.is_valid())
		{
			auto& mtx_array = host_sim_data.sa_cgMutex;
			auto  mtx_view = std::span(reinterpret_cast<luisa::spin_mutex*>(mtx_array.data()), mtx_array.size());

			CpuParallel::parallel_for(
				0,
				active_verts.size(),
				[output_gradient = std::span(abd_data.constraint_gradients),
					output_hessian = std::span(abd_data.constraint_hessians),
					sa_contact_active_verts = active_verts,
					sa_scaled_model_x = std::span(host_sim_data.sa_scaled_model_x),
					sa_x_to_dof_map = std::span(host_sim_data.sa_x_to_dof_map),
					prefix_vid = host_sim_data.num_verts_soft,
					&calculate_per_vert_grad_hess_template,
					mtx_view](const uint index)
				{
					float3	   gradient = Zero3;
					float3x3   hessian = zero3x3;
					const uint vid = sa_contact_active_verts[index];
					if (!sa_x_to_dof_map[vid].is_rigid_body())
						return;

					bool collide = calculate_per_vert_grad_hess_template(vid, gradient, hessian);
					if (collide)
					{
						const uint dof_idx = sa_x_to_dof_map[vid].get_dof_idx();
						const uint body_idx = (dof_idx - prefix_vid) / 4;
						float3	   model_x = sa_scaled_model_x[vid];
						float4	   weight = luisa::make_float4(1.0f, model_x);
						for (uint ii = 0; ii < 4; ii++)
						{
							float  wi = weight[ii];
							float3 affine_grad = wi * gradient;
							BufferOp::atomic_buffer_add(output_gradient, mtx_view, 4 * body_idx + ii, affine_grad);
							for (uint jj = 0; jj < 4; jj++)
							{
								float	 wj = weight[jj];
								float3x3 affine_hess = wi * wj * hessian;
								BufferOp::atomic_buffer_add(output_hessian,
									mtx_view,
									16 * body_idx + ii * 4 + jj,
									affine_hess);
							}
						}
					}
				});
		}
	}

} // namespace lcs
