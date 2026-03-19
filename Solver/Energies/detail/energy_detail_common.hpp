#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

namespace lcs::detail
{
	template <size_t GradientCount, size_t HessianCount, typename GradientT, typename HessianT>
	struct EnergyEvalResult
	{
		using gradient_type = GradientT;
		using hessian_type = HessianT;
		static constexpr size_t gradient_count = GradientCount;
		static constexpr size_t hessian_count = HessianCount;

		std::array<GradientT, GradientCount> gradients{};
		std::array<HessianT, HessianCount>	 hessians{};
	};

	template <typename GradientT, typename HessianT>
	using SingleVertexEvalResult = EnergyEvalResult<1, 1, GradientT, HessianT>;

	template <typename GradientT, typename HessianT>
	using EdgeEvalResult = EnergyEvalResult<2, 4, GradientT, HessianT>;

} // namespace lcs::detail
