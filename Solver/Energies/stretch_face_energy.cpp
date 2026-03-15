#include "stretch_face_energy.h"
#include "SimulationCore/base_mesh.h"
#include "Utils/cpu_parallel.h"
#include "Utils/reduce_helper.h"

namespace lcs
{
	namespace StretchEnergy
	{
		namespace detail
		{
			// using namespace FemUtils;
			template <typename T>
			inline T sqr(T x)
			{
				return x * x;
			}
			float stretch_energy(const float2x3& F, float mu)
			{
				const auto i5u = luisa::dot(F[0], F[0]);
				const auto i5v = luisa::dot(F[1], F[1]);
				return 0.5f * mu * (sqr(luisa::sqrt(i5u) - 1.0f) + sqr(luisa::sqrt(i5v) - 1.0f));
			}
			float shear_energy(const float2x3& F, float lmd)
			{
				const auto i6 = luisa::dot(F[0], F[1]);
				return 0.5f * lmd * sqr(i6);
			}
			Var<float> stretch_energy(const Var<float2x3>& F, Var<float> mu)
			{
				const auto i5u = luisa::compute::dot(F.cols[0], F.cols[0]);
				const auto i5v = luisa::compute::dot(F.cols[1], F.cols[1]);
				return 0.5f * mu * (sqr(luisa::compute::sqrt(i5u) - 1.0f) + sqr(luisa::compute::sqrt(i5v) - 1.0f));
			}
			Var<float> shear_energy(const Var<float2x3>& F, Var<float> lmd)
			{
				const auto i6 = luisa::compute::dot(F.cols[0], F.cols[1]);
				return 0.5f * lmd * sqr(i6);
			}

			float2x3 stretch_gradient(const float2x3& F, const float mu)
			{
				const float3& Fu = F.cols[0];
				const float3& Fv = F.cols[1];

				const auto I5u = luisa::dot(Fu, Fu);
				const auto I5v = luisa::dot(Fv, Fv);

				float sqrtI5u = luisa::sqrt(I5u);
				float sqrtI5v = luisa::sqrt(I5v);
				float invSqrtI5u = 1.0f / sqrtI5u;
				float invSqrtI5v = 1.0f / sqrtI5v;

				float2x3 result;
				result.cols[0] = (sqrtI5u - 1.0f) * invSqrtI5u * Fu;
				result.cols[1] = (sqrtI5v - 1.0f) * invSqrtI5v * Fv;
				return mu * result;
			}
			float2x3 shear_gradient(const float2x3& F, const float lmd)
			{
				float	 w = luisa::dot(F.cols[0], F.cols[1]);
				float2x3 result;
				result.cols[0] = w * F.cols[1];
				result.cols[1] = w * F.cols[0];
				return lmd * result;
			}

			float6x6 stretch_hessian(const float2x3& F, float mu)
			{
				float6x6 H = float6x6::zero();

				const float3& Fu = F.cols[0];
				const float3& Fv = F.cols[1];

				const auto I5u = luisa::dot(Fu, Fu);
				const auto I5v = luisa::dot(Fv, Fv);

				float sqrtI5u = luisa::sqrt(I5u);
				float sqrtI5v = luisa::sqrt(I5v);
				float invSqrtI5u = 1.0f / sqrtI5u;
				float invSqrtI5v = 1.0f / sqrtI5v;

				H.scalar(0, 0) = H.scalar(1, 1) = H.scalar(2, 2) = luisa::max(0.0f, 1.0f - invSqrtI5u);
				H.scalar(3, 3) = H.scalar(4, 4) = H.scalar(5, 5) = luisa::max(0.0f, 1.0f - invSqrtI5v);

				auto fu = F.cols[0] * invSqrtI5u;
				auto fv = F.cols[1] * invSqrtI5v;

				float uCoeff = (invSqrtI5u < 1.0f) ? invSqrtI5u : 1.0f;
				float vCoeff = (invSqrtI5v < 1.0f) ? invSqrtI5v : 1.0f;
				H.block(0, 0) = H.block(0, 0) + uCoeff * outer_product(fu, fu);
				H.block(1, 1) = H.block(1, 1) + vCoeff * outer_product(fv, fv);
				return mu * H;
			}
			float6x6 shear_hessian(const float2x3& F, float mu)
			{
				float6x6 H = float6x6::zero();

				const float3& Fu = F.cols[0];
				const float3& Fv = F.cols[1];

				const float I6 = luisa::dot(Fu, Fv);
				const float signI6 = luisa::sign(I6);

				H.scalar<3, 0>() = H.scalar<4, 1>() = H.scalar<5, 2>() = H.scalar<0, 3>() =
					H.scalar<1, 4>() = H.scalar<2, 5>() = 1.0f;

				const float6 g = FemUtils::flatten(F * luisa::make_float2x2(0, 1, 1, 0)); // F * (a b^T + b a^T)

				const float I2 = luisa::dot(Fu, Fu) + luisa::dot(Fv, Fv); // F.squaredNorm();
				const float lambda0 = 0.5f * (I2 + luisa::sqrt(I2 * I2 + 12.0f * I6 * I6));

				const float6 q0 = (I6 * H * g + lambda0 * g).normalize();
				float6x6	 T = float6x6::identity();
				T = 0.5f * (T + signI6 * H);
				const float6 Tq = T * q0;
				const float	 normTq = Tq.squared_norm();

				H = luisa::abs(I6) * (T - float6x6::outer_product(Tq, Tq) / normTq)
					+ lambda0 * float6x6::outer_product(q0, q0);

				return mu * H;
			}

			Var<float2x3> stretch_gradient(const Var<float2x3>& F, const Var<float> mu)
			{
				const auto& Fu = F.cols[0];
				const auto& Fv = F.cols[1];

				const auto I5u = luisa::compute::dot(Fu, Fu);
				const auto I5v = luisa::compute::dot(Fv, Fv);

				const auto sqrtI5u = luisa::compute::sqrt(I5u);
				const auto sqrtI5v = luisa::compute::sqrt(I5v);
				const auto invSqrtI5u = 1.0f / sqrtI5u;
				const auto invSqrtI5v = 1.0f / sqrtI5v;

				Var<float2x3> result;
				result.cols[0] = (sqrtI5u - 1.0f) * invSqrtI5u * Fu;
				result.cols[1] = (sqrtI5v - 1.0f) * invSqrtI5v * Fv;
				return mu * result;
			}
			Var<float2x3> shear_gradient(const Var<float2x3>& F, const Var<float> lmd)
			{
				Var<float>	  w = luisa::compute::dot(F.cols[0], F.cols[1]);
				Var<float2x3> result;
				result.cols[0] = w * F.cols[1];
				result.cols[1] = w * F.cols[0];
				return lmd * result;
			}

			Var<float6x6> stretch_hessian(const Var<float2x3>& F, Var<float> mu)
			{
				Var<float6x6> H;
				H->set_zero();

				const auto& Fu = F.cols[0];
				const auto& Fv = F.cols[1];

				const auto I5u = luisa::compute::dot(Fu, Fu);
				const auto I5v = luisa::compute::dot(Fv, Fv);

				const auto sqrtI5u = luisa::compute::sqrt(I5u);
				const auto sqrtI5v = luisa::compute::sqrt(I5v);
				const auto invSqrtI5u = 1.0f / sqrtI5u;
				const auto invSqrtI5v = 1.0f / sqrtI5v;

				H->scalar(0, 0) = H->scalar(1, 1) = H->scalar(2, 2) = luisa::compute::max(0.0f, 1.0f - invSqrtI5u);
				H->scalar(3, 3) = H->scalar(4, 4) = H->scalar(5, 5) = luisa::compute::max(0.0f, 1.0f - invSqrtI5v);

				auto fu = F.cols[0] * invSqrtI5u;
				auto fv = F.cols[1] * invSqrtI5v;

				Var<float> uCoeff = luisa::compute::min(invSqrtI5u, 1.0f);
				Var<float> vCoeff = luisa::compute::min(invSqrtI5v, 1.0f);
				H->block(0, 0) = H->block(0, 0) + uCoeff * outer_product(fu, fu);
				H->block(1, 1) = H->block(1, 1) + vCoeff * outer_product(fv, fv);
				return mu * H;
			}
			Var<float6x6> shear_hessian(const Var<float2x3>& F, Var<float> mu)
			{
				using Float = Var<float>;
				using Float3 = Var<float3>;
				using Float6 = Var<float6>;
				using Float6x6 = Var<float6x6>;

				Float6x6 H;
				H->set_zero();

				const Float3& Fu = F.cols[0];
				const Float3& Fv = F.cols[1];

				const Float I6 = luisa::compute::dot(Fu, Fv);
				const Float signI6 = luisa::compute::sign(I6);

				H->scalar<3, 0>() = H->scalar<4, 1>() = H->scalar<5, 2>() = H->scalar<0, 3>() =
					H->scalar<1, 4>() = H->scalar<2, 5>() = 1.0f;

				Var<float2x2> tmp = luisa::compute::make_float2x2(luisa::compute::make_float2(0.0f, 1.0f),
					luisa::compute::make_float2(1.0f, 0.0f));
				const Float6  g = FemUtils::flatten(F * tmp); // F * (a b^T + b a^T)

				const Float I2 = luisa::compute::dot(Fu, Fu) + luisa::compute::dot(Fv, Fv); // F.squaredNorm();
				const Float lambda0 = 0.5f * (I2 + luisa::compute::sqrt(I2 * I2 + 12.0f * I6 * I6));

				const Float6 q0 = (I6 * H * g + lambda0 * g)->normalize();

				Float6x6 T;
				T->set_identity();
				T = 0.5f * (T + signI6 * H);
				const Float6 Tq = T * q0;
				const auto	 normTq = Tq->squared_norm();

				H = luisa::compute::abs(I6) * (T - outer_product(Tq, Tq) / normTq) + lambda0 * outer_product(q0, q0);

				return mu * H;
			}
		} // namespace detail

		//  float2x2 get_Dm_inv(const float3& x_0, const float3& x_1, const float3& x_2)
		// {
		// 	float3		   r_1 = x_1 - x_0;
		// 	float3		   r_2 = x_2 - x_0;
		// 	float3		   cross = cross_vec(r_1, r_2);
		// 	float3		   axis_1 = normalize_vec(r_1);
		// 	float3		   axis_2 = normalize_vec(cross_vec(cross, axis_1));
		// 	float2		   uv0 = float2(dot_vec(axis_1, x_0), dot_vec(axis_2, x_0));
		// 	float2		   uv1 = float2(dot_vec(axis_1, x_1), dot_vec(axis_2, x_1));
		// 	float2		   uv2 = float2(dot_vec(axis_1, x_2), dot_vec(axis_2, x_2));
		// 	float2		   duv0 = uv1 - uv0;
		// 	float2		   duv1 = uv2 - uv0;
		// 	const float2x2 duv = float2x2(duv0, duv1);
		// 	const float2x2 inv_duv = luisa::inverse(duv);
		// 	return inv_duv;
		// }

		void compute_gradient_hessian(const float3& x0,
			const float3&							x1,
			const float3&							x2,
			const float2x2&							Dm,
			const float								mu,
			const float								lambda,
			const float								area,
			float3x3&								dedx,
			float9x9&								d2edx2)
		{
			dedx = luisa::make_float3x3(0.0f);
			d2edx2.set_zero();

			float2x3 F = makeFloat2x3(x1 - x0, x2 - x0) * Dm;

			float2x3 de0dF = detail::stretch_gradient(F, mu);
			float6x6 d2e0dF2 = detail::stretch_hessian(F, mu);

			float2x3 de1dF = detail::shear_gradient(F, lambda);
			float6x6 d2e1dF2 = detail::shear_hessian(F, lambda);

			float2x3 dedF = de0dF + de1dF;
			float6x6 d2edF2 = d2e0dF2 + d2e1dF2;

			dedx = area * FemUtils::convert_force(dedF, Dm);
			d2edx2 = area * FemUtils::convert_hessian(d2edF2, Dm);

			// LUISA_INFO("BW98 Info: F = [{}, {}], Area = {}", F.cols[0], F.cols[1], area);
			// LUISA_INFO("		 : Stretch Force = {}", detail::convert_force(de0dF, Dm));
			// LUISA_INFO("		 : Shear   Force = {}", detail::convert_force(de1dF, Dm));

			// gradient[0] = dedx[0];
			// gradient[1] = dedx[1];
			// gradient[2] = dedx[2];
			// for (uint ii = 0; ii < 3; ii++)
			// {
			//     for (uint jj = 0; jj < 3; jj++)
			//     {
			//         hessian[jj][ii] = d2edx2.block(ii, jj);
			//     }
			// }
		}
		void compute_gradient_hessian(const Var<float3>& x0,
			const Var<float3>&							 x1,
			const Var<float3>&							 x2,
			const Var<float2x2>&						 Dm,
			const Var<float>							 mu,
			const Var<float>							 lambda,
			const Var<float>							 area,
			Var<float3x3>&								 dedx,
			Var<float9x9>&								 d2edx2)
		{
			// dedx = luisa::make_float3x3(0.0f);
			// d2edx2->set_zero();

			Var<float2x3> F = makeFloat2x3(x1 - x0, x2 - x0) * Dm;

			// float2x3 de0dF   = libuipc::stretch_gradient(F, mu, 1.0f);
			// float6x6 d2e0dF2 = libuipc::stretch_hessian(F, mu, 1.0f);
			auto de0dF = detail::stretch_gradient(F, mu);
			auto d2e0dF2 = detail::stretch_hessian(F, mu);

			auto de1dF = detail::shear_gradient(F, lambda);
			auto d2e1dF2 = detail::shear_hessian(F, lambda);

			auto dedF = de0dF + de1dF;
			auto d2edF2 = d2e0dF2 + d2e1dF2;

			dedx = area * FemUtils::convert_force(dedF, Dm);
			d2edx2 = area * FemUtils::convert_hessian(d2edF2, Dm);
		}

		float compute_energy(const float3& x0,
			const float3&				   x1,
			const float3&				   x2,
			const float2x2&				   Dm,
			const float					   mu,
			const float					   lambda,
			const float					   area)
		{

			const float2x3 F = makeFloat2x3(x1 - x0, x2 - x0) * Dm;
			auto		   energy = detail::stretch_energy(F, mu) + detail::shear_energy(F, lambda);
			return area * energy;
		}
		Var<float> compute_energy(const Var<float3>& x0,
			const Var<float3>&						 x1,
			const Var<float3>&						 x2,
			const Var<float2x2>&					 Dm,
			const Var<float>						 mu,
			const Var<float>						 lambda,
			const Var<float>						 area)
		{

			const Var<float2x3> F = makeFloat2x3(x1 - x0, x2 - x0) * Dm;
			auto				energy = detail::stretch_energy(F, mu) + detail::shear_energy(F, lambda);
			return area * energy;
		}

	}; // namespace StretchEnergy

} // namespace lcs

using namespace luisa::compute;

namespace lcs
{
	StretchFaceEnergy::StretchFaceEnergy(BufferView<float> sa_system_energy) noexcept
		: _sa_system_energy(sa_system_energy)
	{
	}

	void StretchFaceEnergy::compile(AsyncCompiler& compiler)
	{
		luisa::compute::ShaderOption default_option = { .enable_debug_info = false };
		compiler.compile<1>(
			_shader,
			[sa_system_energy = _sa_system_energy](Var<Constitutions::StretchFace<luisa::compute::Buffer>> constraint,
				Var<BufferView<float3>>																	   sa_x)
			{
				auto& sa_faces = constraint.constraint_indices;
				auto& sa_stretch_faces_rest_area = constraint.sa_stretch_faces_rest_area;
				auto& sa_stretch_faces_Dm_inv = constraint.sa_stretch_faces_Dm_inv;
				auto& sa_stretch_faces_mu_lambda = constraint.sa_stretch_faces_mu_lambda;

				const Uint fid = dispatch_id().x;
				Float	   energy = 0.0f;
				{
					const Uint3 face = sa_faces->read(fid);
					Float3		vert_pos[3] = { sa_x->read(face[0]), sa_x->read(face[1]), sa_x->read(face[2]) };

					Float2x2 Dm_inv = sa_stretch_faces_Dm_inv->read(fid);
					Float	 area = sa_stretch_faces_rest_area->read(fid);

					Float2 mu_lambda = sa_stretch_faces_mu_lambda->read(fid);
					Float  mu_cloth = mu_lambda[0];
					Float  lambda_cloth = mu_lambda[1];
					// lambda_cloth = 0.0f;

					energy = StretchEnergy::compute_energy(
						vert_pos[0], vert_pos[1], vert_pos[2], Dm_inv, mu_cloth, lambda_cloth, area);
				};

				energy = ParallelIntrinsic::block_intrinsic_reduce(fid, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
				$if(fid % 256 == 0)
				{
					sa_system_energy->atomic(offset_stretch_face).fetch_add(energy);
				};
			},
			default_option);

		// gradient/hessian evaluate shader
		compiler.compile<1>(
			_eval_shader,
			[](Var<Constitutions::StretchFace<luisa::compute::Buffer>> constraint, Var<BufferView<float3>> sa_x)
			{
				auto& sa_faces = constraint.constraint_indices;
				auto& sa_stretch_faces_Dm_inv = constraint.sa_stretch_faces_Dm_inv;
				auto& sa_stretch_faces_rest_area = constraint.sa_stretch_faces_rest_area;
				auto& sa_stretch_faces_mu_lambda = constraint.sa_stretch_faces_mu_lambda;
				auto& output_gradient_ptr = constraint.constraint_gradients;
				auto& output_hessian_ptr = constraint.constraint_hessians;

				const UInt	fid = dispatch_id().x;
				const UInt3 face = sa_faces->read(fid);

				Float3	 vert_pos[3] = { sa_x->read(face[0]), sa_x->read(face[1]), sa_x->read(face[2]) };
				Float3x3 gradients;
				Float9x9 hessians;

				Float2x2 Dm_inv = sa_stretch_faces_Dm_inv->read(fid);
				Float	 area = sa_stretch_faces_rest_area->read(fid);

				Float2 mu_lambda = sa_stretch_faces_mu_lambda->read(fid);
				Float  mu_cloth = mu_lambda[0];
				Float  lambda_cloth = mu_lambda[1];
				// lambda_cloth = 0.0f;

				StretchEnergy::compute_gradient_hessian(
					vert_pos[0], vert_pos[1], vert_pos[2], Dm_inv, mu_cloth, lambda_cloth, area, gradients, hessians);

				// Output
				{
					output_gradient_ptr->write(fid * 3 + 0, gradients[0]);
					output_gradient_ptr->write(fid * 3 + 1, gradients[1]);
					output_gradient_ptr->write(fid * 3 + 2, gradients[2]);
				}
				{
					output_hessian_ptr->write(fid * 9 + 0, hessians->block(0, 0));
					output_hessian_ptr->write(fid * 9 + 1, hessians->block(1, 0));
					output_hessian_ptr->write(fid * 9 + 2, hessians->block(2, 0));
					output_hessian_ptr->write(fid * 9 + 3, hessians->block(0, 1));
					output_hessian_ptr->write(fid * 9 + 4, hessians->block(1, 1));
					output_hessian_ptr->write(fid * 9 + 5, hessians->block(2, 1));
					output_hessian_ptr->write(fid * 9 + 6, hessians->block(0, 2));
					output_hessian_ptr->write(fid * 9 + 7, hessians->block(1, 2));
					output_hessian_ptr->write(fid * 9 + 8, hessians->block(2, 2));
				}
			},
			default_option);
	}

	void StretchFaceEnergy::device_compute_energy(luisa::compute::Stream& stream)
	{
	}

	void StretchFaceEnergy::device_compute_energy(luisa::compute::Stream& stream,
		const Constitutions::StretchFace<luisa::compute::Buffer>&		  constraint,
		const luisa::compute::Buffer<float3>&							  sa_x,
		size_t															  dispatch_count)
	{
		stream << _shader(constraint, sa_x).dispatch(dispatch_count);
	}

	void StretchFaceEnergy::device_evaluate(luisa::compute::Stream& stream,
		const Constitutions::StretchFace<luisa::compute::Buffer>&	constraint,
		const luisa::compute::Buffer<float3>&						sa_x,
		size_t														dispatch_count)
	{
		stream << _eval_shader(constraint, sa_x.view()).dispatch(dispatch_count);
	}

	double StretchFaceEnergy::host_evaluate(const std::vector<float>& host_energy)
	{
		return host_energy[offset_stretch_face];
	}

	void StretchFaceEnergy::host_evaluate(lcs::SimulationData<std::vector>& host_sim_data,
		lcs::MeshData<std::vector>&											host_mesh_data)
	{
		auto& stretch_faces = host_sim_data.get_stretch_face_data();

		CpuParallel::parallel_for(
			0,
			stretch_faces.get_num_indices(),
			[sa_x = std::span(host_sim_data.sa_x),
				sa_faces = std::span(stretch_faces.constraint_indices),
				sa_stretch_faces_rest_area = std::span(stretch_faces.sa_stretch_faces_rest_area),
				sa_stretch_faces_Dm_inv = std::span(stretch_faces.sa_stretch_faces_Dm_inv),
				sa_stretch_faces_mu_lambda = std::span(stretch_faces.sa_stretch_faces_mu_lambda),
				output_gradient_ptr = std::span(stretch_faces.constraint_gradients),
				output_hessian_ptr = std::span(stretch_faces.constraint_hessians)](const uint fid)
			{
				uint3 face = sa_faces[fid];

				float3	 vert_pos[3] = { sa_x[face[0]], sa_x[face[1]], sa_x[face[2]] };
				float3x3 gradients;
				float9x9 hessians;
				float2x2 Dm_inv = sa_stretch_faces_Dm_inv[fid];
				float	 area = sa_stretch_faces_rest_area[fid];

				auto [mu_cloth, lambda_cloth] = sa_stretch_faces_mu_lambda[fid];
				// lambda_cloth = 0.0f;

				// LUISA_INFO("BW98 Info: Fid = {}, Face = {}, lambda = {}, mu = {}", fid, face, lambda_cloth, mu_cloth);
				StretchEnergy::compute_gradient_hessian(
					vert_pos[0], vert_pos[1], vert_pos[2], Dm_inv, mu_cloth, lambda_cloth, area, gradients, hessians);

				output_gradient_ptr[fid * 3 + 0] = gradients[0];
				output_gradient_ptr[fid * 3 + 1] = gradients[1];
				output_gradient_ptr[fid * 3 + 2] = gradients[2];

				output_hessian_ptr[fid * 9 + 0] = hessians.block(0, 0);
				output_hessian_ptr[fid * 9 + 1] = hessians.block(1, 0);
				output_hessian_ptr[fid * 9 + 2] = hessians.block(2, 0);
				output_hessian_ptr[fid * 9 + 3] = hessians.block(0, 1);
				output_hessian_ptr[fid * 9 + 4] = hessians.block(1, 1);
				output_hessian_ptr[fid * 9 + 5] = hessians.block(2, 1);
				output_hessian_ptr[fid * 9 + 6] = hessians.block(0, 2);
				output_hessian_ptr[fid * 9 + 7] = hessians.block(1, 2);
				output_hessian_ptr[fid * 9 + 8] = hessians.block(2, 2);
			});
	}

} // namespace lcs
