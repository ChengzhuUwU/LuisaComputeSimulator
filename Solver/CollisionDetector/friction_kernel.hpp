#pragma once

#include "CollisionDetector/cipc_kernel.hpp"
#include "Core/scalar.h"
#include "Core/xbasic_types.h"
#include "luisa/dsl/sugar.h"
#include "distance.hpp"

namespace lcs
{

	namespace Friction
	{
		namespace ando_barrier
		{
			//
			// Modified from [https://github.com/st-tech/ppf-contact-solver/blob/main/src/cpp/energy/model/friction.hpp]
			//
			constexpr float friction_eps = 1e-4f; // minVelocity = 0.1mm/s, need to be multiplied by h (time step)

			template <typename T>
			inline auto get_projection(const T& normal)
			{
				return Identity3x3 - outer_product(normal, normal); // P*dv = dv - dot(dv, n)*n
			}
			inline std::pair<float, float3x3> get_friction_lambda_P(const float3& grad_contact,
				const float3&													  rel_dx,
				const float3&													  normal,
				const float														  mu,
				const float														  min_dx)
			{
				float	 contact = -dot(grad_contact, normal);
				float3x3 P = get_projection(normal);

				float lambda;
				if (mu > 0.0f)
				{
					float denom = length_squared_vec(P * rel_dx);
					if (denom > 0.0f)
					{
						lambda = mu * contact / luisa::max(min_dx, luisa::sqrt(denom));
					}
					else
					{
						lambda = mu * contact / min_dx;
					}
				}
				else
				{
					lambda = 0.0f;
				}
				return std::make_pair(lambda, P);
			}
			inline std::pair<Var<float>, Var<float3x3>> get_friction_lambda_P(const Var<float3>& grad_contact,
				const Var<float3>&																 rel_dx,
				const Var<float3>&																 normal,
				const Var<float>																 mu,
				const Var<float>																 min_dx)
			{
				Var<float>	  contact = -luisa::compute::dot(grad_contact, normal);
				Var<float3x3> P = get_projection(normal);

				Var<float> lambda;
				$if(mu > 0.0f)
				{
					Var<float> denom = length_squared_vec(P * rel_dx);
					$if(denom > 0.0f)
					{
						lambda = mu * contact / luisa::compute::max(min_dx, luisa::compute::sqrt(denom));
					}
					$else
					{
						lambda = mu * contact / min_dx;
					};
				}
				$else
				{
					lambda = 0.0f;
				};
				return std::make_pair(lambda, P);
			}
			inline std::pair<float, float3x3> get_friction_lambda_P(
				const float dbdd, const float3& rel_dx, const float3& normal, const float mu, const float min_dx)
			{
				float	 contact = -dbdd;
				float3x3 P = get_projection(normal);

				float lambda;
				if (mu > 0.0f)
				{
					float denom = length_squared_vec(P * rel_dx);
					if (denom > 0.0f)
					{
						lambda = mu * contact / luisa::max(min_dx, luisa::sqrt(denom));
					}
					else
					{
						lambda = mu * contact / min_dx;
					}
				}
				else
				{
					lambda = 0.0f;
				}
				return std::make_pair(lambda, P);
			}

			template <typename PairType, typename Vec>
			inline auto compute_gradient_hessian(const PairType& lambda_P, const Vec& rel_dx)
			{
				const auto& lambda = lambda_P.first;
				const auto& P = lambda_P.second;
				auto		gradient = lambda * (P * rel_dx);
				auto		hessian = lambda * P;
				return std::make_pair(gradient, hessian);
			}
			template <typename FloatType, typename Vec>
			inline auto compute_gradient_hessian(const FloatType& lambda, const Vec& normal, const Vec& rel_dx)
			{
				const auto P = Identity3x3 - outer_product(normal, normal);
				auto	   gradient = lambda * (P * rel_dx);
				auto	   hessian = lambda * P;
				return std::make_pair(gradient, hessian);
			}
			template <typename FloatType, typename Vec>
			inline auto compute_hessian(const FloatType& lambda, const Vec& normal)
			{
				const auto P = Identity3x3 - outer_product(normal, normal);
				auto	   hessian = lambda * P;
				return hessian;
			}
		} // namespace ando_barrier
	}	  // namespace Friction

	namespace Friction
	{
		namespace ipc_barrier
		{

			// E = h * eps_v  (IPC: eps_v is friction velocity threshold; multiply by h)
			template <typename T>
			inline T f0_scalar(const T& y, const T& E)
			{
				// for y < E : y^2/E - y^3/(3 E^2)
				//      else : y - E/3
				return lcs::select(y < E,
					(y * y) / E - (y * y * y) / (3.0f * E * E), //
					y - E / 3.0f								//
				);
			}

			template <typename T>
			inline T f1_scalar(const T& y, const T& E)
			{
				// for y < E : 2*y/E - y^2/(E^2)
				//      else : 1
				return lcs::select(y < E,
					(2.0f * y) / E - (y * y) / (E * E), //
					T(1.0f)								//
				);
			}

			template <typename T>
			inline T f1_scalar_devide_s(const T& y, const T& E)
			{
				// f1/y
				// for y < E : 2/E - y/(E^2)
				//      else : 1/y
				return lcs::select(y < E,
					(2.0f) / E - (y) / (E * E), //
					T(1.0f) / y					//
				);
			}

			template <typename T>
			inline T f2_scalar(const T& y, const T& E)
			{
				// for y < E : 2/E - 2y/E^2
				//      else : 0
				return lcs::select(y < E,
					(2.0f / E) - (2.0f * y) / (E * E), //
					T(0.0f)							   //
				);
			}

			static constexpr float small_s = 1e-6f;

			// Compute friction energy D = mu * lambda_n * f0(||u||)
			inline auto compute_friction_energy(const Var<float>& mu_lambda,
				const Var<float3>&								  normal,
				const Var<float3>&								  rel_dx,
				const Var<float>&								  E /* = h * eps_v */)
			{
				using Float = Var<float>;
				using Float3 = Var<float3>;
				using Float3x3 = Var<float3x3>;

				const Float3x3 nnT = outer_product(normal, normal);
				const Float3x3 I = Identity3x3;
				const Float3x3 P = I - nnT;
				const Float3   u = P * rel_dx;

				Float s2 = dot_vec(u, u);
				Float s = sqrt_scalar(s2);
				Float f0_val = f0_scalar(s, E);
				Float e_u = lcs::select(s2 < small_s * small_s, Float(0.0f), mu_lambda * f0_val);
				// Float e_u = mu_lambda * f0_val;
				return e_u;
			}

			// Compute friction gradient and hessian (to t, where t = sum(wi*xi), wi are weights for vertex i):
			// let s = ||u||, u = P * rel_dx = P * (t-t0)
			//        D = mu * lambda * f0(s)
			//     dDdt = mu * lambda * f1(s) * dsdt, dsdt = dsdu * dudt = u/s * P
			//   d2Ddt2 = mu * lambda * f2(s) * ddsddt,
			// returns gradient (3), hessian (3x3)
			inline auto compute_friction_gradient_hessian(const Var<float>& mu_lambda,
				const Var<float3>&											normal,
				const Var<float3>&											rel_dx,
				const Var<float>&											E /* = h * eps_v */)
			{
				using Float = Var<float>;
				using Float3 = Var<float3>;
				using Float3x3 = Var<float3x3>;

				// tangential projector P = I - n n^T
				const Float3x3 nnT = outer_product(normal, normal);
				const Float3x3 I = Identity3x3;
				const Float3x3 P = I - nnT;

				const Float3 u = P * rel_dx;
				const Float	 s2 = dot_vec(u, u);
				const Float	 s = sqrt_scalar(s2);
				Float3		 u_bar = u / s;
				u_bar = normalize_vec(u_bar);
				const Float3x3 uuT = outer_product(u, u);

				Float f1_devide_s = f1_scalar_devide_s(s, E);
				Float f2_val = f2_scalar(s, E);

				Float3	 grad_u;
				Float3x3 hess_u;
				$if(s2 < small_s * small_s)
				{
					grad_u = Float3(0.0f, 0.0f, 0.0f);
					hess_u = f1_devide_s * P;
				}
				$else
				{
					grad_u = f1_devide_s * u;
					hess_u = (f2_val - f1_devide_s) * outer_product(u_bar, u_bar) + f1_devide_s * P;
				};

				Float3	 grad = mu_lambda * grad_u;
				Float3x3 hess = mu_lambda * hess_u;

				return std::make_pair(grad, hess);
			}
			inline auto compute_friction_gradient_hessian(const float& mu_lambda,
				const float3&										   normal,
				const float3&										   rel_dx,
				const float&										   E /* = h * eps_v */)
			{
				using Float = float;
				using Float3 = float3;
				using Float3x3 = float3x3;

				// tangential projector P = I - n n^T
				const Float3x3 nnT = outer_product(normal, normal);
				const Float3x3 I = Identity3x3;
				const Float3x3 P = I - nnT;

				const Float3   u = P * rel_dx;
				const Float	   s2 = dot_vec(u, u);
				const Float	   s = sqrt_scalar(s2);
				const Float3   u_bar = u / s;
				const Float3x3 uuT = outer_product(u, u);

				Float f1_devide_s = f1_scalar_devide_s(s, E);
				Float f2_val = f2_scalar(s, E);

				Float3	 grad_u;
				Float3x3 hess_u;
				if (s2 < small_s * small_s)
				{
					grad_u = Float3(0.0f, 0.0f, 0.0f);
					hess_u = f1_devide_s * P;
				}
				else // s >= E
				{
					grad_u = f1_devide_s * u;
					hess_u = (f2_val - f1_devide_s) * outer_product(u_bar, u_bar) + f1_devide_s * P;
				};

				Float3	 grad = mu_lambda * grad_u;
				Float3x3 hess = mu_lambda * hess_u;

				return std::make_pair(grad, hess);
			}

		} // namespace ipc_barrier
	}	  // namespace Friction

} // namespace lcs