#pragma once

#include <variant>

namespace lcs
{

	namespace Initializer
	{
		enum class ConstitutiveStretchModelCloth
		{
			// None     = 0,
			Spring = 0,	  // Impl
			FEM_BW98 = 1, // Impl
		};
		enum class ConstitutiveBendingModelCloth
		{
			None = 0,
			QuadraticBending = 1, // Impl
			DihedralAngle = 2,	  // Impl
		};
		enum class ConstitutiveModelTet
		{
			// None             = 0,
			Spring = 0, // Impl
			StVK = 1,
			StableNeoHookean = 2,
			Corotated = 3,
			ARAP = 4,
		};
		enum class ConstitutiveModelRigid
		{
			// None          = 0,
			Spring = 0,
			Orthogonality = 1, // Impl
			ARAP = 2,
			StableNeoHookean = 3, // Full space simulation
		};
		enum class ConstitutiveModelRod
		{
			Spring = 0,
		};

		enum class MaterialType
		{
			None,
			Cloth,
			Tetrahedral,
			Rigid,
			Rod,
		};

		struct MaterialBase
		{
			static constexpr float default_mass() { return 0.0f; }
			static constexpr float default_density() { return 1e3f; }
			static constexpr float default_d_hat() { return 1e-3f; }
			static constexpr float default_friction_mu() { return 0.5f; }

			float mass = default_mass();
			float density = default_density();
			float d_hat = default_d_hat();
			float friction_mu = default_friction_mu();
			bool  is_shell = true;
		};

		struct ClothMaterial : MaterialBase
		{
			static constexpr ConstitutiveStretchModelCloth default_stretch_model() { return ConstitutiveStretchModelCloth::FEM_BW98; }
			static constexpr ConstitutiveBendingModelCloth default_bending_model() { return ConstitutiveBendingModelCloth::DihedralAngle; }
			static constexpr float						   default_thickness() { return 1e-3f; }
			static constexpr float						   default_youngs_modulus() { return 1e6f; }
			static constexpr float						   default_poisson_ratio() { return 0.35f; }
			static constexpr float						   default_area_bending_stiffness() { return 5e-3f; }

			ConstitutiveStretchModelCloth stretch_model = default_stretch_model();
			ConstitutiveBendingModelCloth bending_model = default_bending_model();
			float						  thickness = default_thickness();
			float						  youngs_modulus = default_youngs_modulus();
			float						  poisson_ratio = default_poisson_ratio();
			float						  area_bending_stiffness = default_area_bending_stiffness();
			// float                         area_youngs_modulus = 1e3f;
		};
		struct TetMaterial : MaterialBase
		{
			static constexpr ConstitutiveModelTet default_model() { return ConstitutiveModelTet::Spring; }
			static constexpr float				  default_youngs_modulus() { return 1e6f; }
			static constexpr float				  default_poisson_ratio() { return 0.35f; }

			ConstitutiveModelTet model = default_model();
			float				 youngs_modulus = default_youngs_modulus();
			float				 poisson_ratio = default_poisson_ratio();
		};
		struct RigidMaterial : MaterialBase
		{
			static constexpr ConstitutiveModelRigid default_model() { return ConstitutiveModelRigid::Orthogonality; }
			static constexpr float					default_thickness() { return 1e-3f; }
			static constexpr float					default_stiffness() { return 1e6f; }

			ConstitutiveModelRigid model = default_model();
			bool				   is_solid = true;
			float				   thickness = default_thickness();
			float				   stiffness = default_stiffness();
			// float                  youngs_modulus  = 1e9f;
			// float                  poisson_ratio   = 0.35f;
		};
		struct RodMaterial : MaterialBase
		{
			static constexpr ConstitutiveModelRod default_model() { return ConstitutiveModelRod::Spring; }
			static constexpr float				  default_radius() { return 1e-3f; }
			static constexpr float				  default_bending_stiffness() { return 1e4f; }
			static constexpr float				  default_twisting_stiffness() { return 1e4f; }

			ConstitutiveModelRod model = default_model();
			float				 radius = default_radius();
			float				 bending_stiffness = default_bending_stiffness();
			float				 twisting_stiffness = default_twisting_stiffness();
		};

		using MaterialVariant = std::variant<ClothMaterial, TetMaterial, RigidMaterial, RodMaterial>;

		constexpr MaterialType material_type_from_variant(const MaterialVariant& var)
		{
			return std::visit([](auto const& m) noexcept
				{
					using T = std::decay_t<decltype(m)>;
					if constexpr (std::is_same_v<T, ClothMaterial>)
						return MaterialType::Cloth;
					else if constexpr (std::is_same_v<T, TetMaterial>)
						return MaterialType::Tetrahedral;
					else if constexpr (std::is_same_v<T, RigidMaterial>)
						return MaterialType::Rigid;
					else if constexpr (std::is_same_v<T, RodMaterial>)
						return MaterialType::Rod;
					else
						return MaterialType::None; },
				var);
		}

	} // namespace Initializer

} // namespace lcs