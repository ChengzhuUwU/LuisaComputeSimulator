#include "physical_material.h"

#include <array>
#include <string_view>

namespace lcs::Material
{
	namespace
	{
		template <typename EnumT, size_t N>
		constexpr std::string_view enum_to_string_in_table(
			EnumT													 value,
			const std::array<std::pair<EnumT, std::string_view>, N>& table,
			std::string_view										 fallback)
		{
			for (const auto& [key, name] : table)
			{
				if (key == value)
					return name;
			}
			return fallback;
		}

		template <typename EnumT, size_t N>
		constexpr const EnumT* find_enum_by_string_in_table(
			std::string_view										 key,
			const std::array<std::pair<EnumT, std::string_view>, N>& table)
		{
			for (const auto& [value, name] : table)
			{
				if (name == key)
					return &value;
			}
			return nullptr;
		}

		constexpr std::array<std::pair<ConstitutiveStretchModelCloth, std::string_view>, 3>
			cloth_stretch_model_table = {
				std::pair{ ConstitutiveStretchModelCloth::Empty, "Empty" },
				std::pair{ ConstitutiveStretchModelCloth::Spring, "Spring" },
				std::pair{ ConstitutiveStretchModelCloth::FEM_BW98, "FEM_BW98" },
			};

		constexpr std::array<std::pair<ConstitutiveBendingModelCloth, std::string_view>, 3>
			cloth_bending_model_table = {
				std::pair{ ConstitutiveBendingModelCloth::Empty, "Empty" },
				std::pair{ ConstitutiveBendingModelCloth::QuadraticBending, "QuadraticBending" },
				std::pair{ ConstitutiveBendingModelCloth::DihedralAngle, "DihedralAngle" },
			};

		constexpr std::array<std::pair<ConstitutiveModelTet, std::string_view>, 6> tet_model_table = {
			std::pair{ ConstitutiveModelTet::Empty, "Empty" },
			std::pair{ ConstitutiveModelTet::Spring, "Spring" },
			std::pair{ ConstitutiveModelTet::StVK, "StVK" },
			std::pair{ ConstitutiveModelTet::StableNeoHookean, "StableNeoHookean" },
			std::pair{ ConstitutiveModelTet::Corotated, "Corotated" },
			std::pair{ ConstitutiveModelTet::ARAP, "ARAP" },
		};

		constexpr std::array<std::pair<ConstitutiveModelRigid, std::string_view>, 5> rigid_model_table = {
			std::pair{ ConstitutiveModelRigid::Empty, "Empty" },
			std::pair{ ConstitutiveModelRigid::Spring, "Spring" },
			std::pair{ ConstitutiveModelRigid::Orthogonality, "Orthogonality" },
			std::pair{ ConstitutiveModelRigid::ARAP, "ARAP" },
			std::pair{ ConstitutiveModelRigid::StableNeoHookean, "StableNeoHookean" },
		};

		constexpr std::array<std::pair<ConstitutiveModelRod, std::string_view>, 2> rod_model_table = {
			std::pair{ ConstitutiveModelRod::Empty, "Empty" },
			std::pair{ ConstitutiveModelRod::Spring, "Spring" },
		};
	} // namespace

	std::string_view cloth_stretch_model_to_string(ConstitutiveStretchModelCloth model)
	{
		return enum_to_string_in_table(model, cloth_stretch_model_table, "FEM_BW98");
	}

	std::string_view cloth_bending_model_to_string(ConstitutiveBendingModelCloth model)
	{
		return enum_to_string_in_table(model, cloth_bending_model_table, "DihedralAngle");
	}

	std::string_view tet_model_to_string(ConstitutiveModelTet model)
	{
		return enum_to_string_in_table(model, tet_model_table, "Spring");
	}

	std::string_view rigid_model_to_string(ConstitutiveModelRigid model)
	{
		return enum_to_string_in_table(model, rigid_model_table, "Orthogonality");
	}

	std::string_view rod_model_to_string(ConstitutiveModelRod model)
	{
		return enum_to_string_in_table(model, rod_model_table, "Spring");
	}

	ConstitutiveStretchModelCloth parse_cloth_stretch_model(const std::string_view& s)
	{
		if (const auto* model = find_enum_by_string_in_table(std::string_view{ s }, cloth_stretch_model_table))
		{
			return *model;
		}
		return ClothMaterial::default_stretch_model();
	}

	ConstitutiveBendingModelCloth parse_cloth_bending_model(const std::string_view& s)
	{
		if (const auto* model = find_enum_by_string_in_table(std::string_view{ s }, cloth_bending_model_table))
		{
			return *model;
		}
		return ClothMaterial::default_bending_model();
	}

	ConstitutiveModelTet parse_tet_model(const std::string_view& s)
	{
		if (const auto* model = find_enum_by_string_in_table(std::string_view{ s }, tet_model_table))
		{
			return *model;
		}
		return TetMaterial::default_model();
	}

	ConstitutiveModelRigid parse_rigid_model(const std::string_view& s)
	{
		if (const auto* model = find_enum_by_string_in_table(std::string_view{ s }, rigid_model_table))
		{
			return *model;
		}
		return RigidMaterial::default_model();
	}

	ConstitutiveModelRod parse_rod_model(const std::string_view& s)
	{
		if (const auto* model = find_enum_by_string_in_table(std::string_view{ s }, rod_model_table))
		{
			return *model;
		}
		return RodMaterial::default_model();
	}
} // namespace lcs::Material
