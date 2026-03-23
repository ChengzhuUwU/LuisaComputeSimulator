#pragma once

#include <luisa/core/basic_types.h>
#include <cstdint>
#include <cstddef>

namespace lcs
{
	struct JointIndices8
	{
		uint32_t i0 = 0u;
		uint32_t i1 = 0u;
		uint32_t i2 = 0u;
		uint32_t i3 = 0u;
		uint32_t i4 = 0u;
		uint32_t i5 = 0u;
		uint32_t i6 = 0u;
		uint32_t i7 = 0u;

		[[nodiscard]] uint32_t operator[](size_t idx) const
		{
			switch (idx)
			{
				case 0:
					return i0;
				case 1:
					return i1;
				case 2:
					return i2;
				case 3:
					return i3;
				case 4:
					return i4;
				case 5:
					return i5;
				case 6:
					return i6;
				default:
					return i7;
			}
		}
	};

	enum class JointConstraintType : uint32_t
	{
		Fixed,
		Prismatic,
		Revolute
	};

	struct FixedJointConstraintDesc
	{
		uint32_t	  body_a_registration = 0u;
		uint32_t	  body_b_registration = 0u;
		luisa::float3 anchor_a_local = luisa::make_float3(0.0f);
		luisa::float3 anchor_b_local = luisa::make_float3(0.0f);
		float		  stiffness_pos = 1.0e4f;
		float		  stiffness_rot = 1.0e3f;
	};

	struct PrismaticJointConstraintDesc
	{
		uint32_t	  body_a_registration = 0u;
		uint32_t	  body_b_registration = 0u;
		luisa::float3 anchor_a_local = luisa::make_float3(0.0f);
		luisa::float3 anchor_b_local = luisa::make_float3(0.0f);
		luisa::float3 axis_world = luisa::make_float3(1.0f, 0.0f, 0.0f);
		float		  stiffness_pos = 1.0e4f;
		float		  stiffness_rot = 1.0e3f;
	};

	struct RevoluteJointConstraintDesc
	{
		uint32_t	  body_a_registration = 0u;
		uint32_t	  body_b_registration = 0u;
		luisa::float3 anchor_a_local = luisa::make_float3(0.0f);
		luisa::float3 anchor_b_local = luisa::make_float3(0.0f);
		luisa::float3 axis_world = luisa::make_float3(1.0f, 0.0f, 0.0f);
		luisa::float3 axis_a_local = luisa::make_float3(1.0f, 0.0f, 0.0f);
		luisa::float3 axis_b_local = luisa::make_float3(1.0f, 0.0f, 0.0f);
		float		  stiffness_pos = 1.0e4f;
		float		  stiffness_axis = 1.0e3f;
	};

} // namespace lcs
