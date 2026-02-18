#pragma once

#include "Utils/async_compiler.h"
#include <luisa/runtime/stream.h>
#include <vector>

namespace lcs
{
	class Energy
	{
	public:
		virtual ~Energy() = default;
		virtual void   compile(AsyncCompiler& compiler) = 0;
		virtual void   device_compute_energy(luisa::compute::Stream& stream) = 0;
		virtual double host_evaluate(const std::vector<float>& host_energy) = 0;
	};

} // namespace lcs
