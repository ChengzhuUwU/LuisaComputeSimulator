#pragma once

#include <luisa/luisa-compute.h>

namespace lcs
{

namespace BendingEnergy
{

using FVector3f = luisa::float3;
constexpr float FLOAT_SMALL_NUMBER = 1e-8f;

static inline FVector3f SafeDivide(const luisa::float3& Numerator, const float Denominator)
{
	if (Denominator <= FLOAT_SMALL_NUMBER) return luisa::make_float3(0.0f);
	else return Numerator / Denominator;
}
// static inline FVector3f VectorGetSafeNormal(const luisa::float3& vec)
// {
// 	const float squaredLength = length_squared_vec(vec);
// 	if (squaredLength <= FLOAT_SMALL_NUMBER) return Zero3;
// 	else return vec / sqrt_scalar(squaredLength);
// }
static inline FVector3f VectorGetSafeNormal(const luisa::float3& Vector)
{
	const float SquareSum = luisa::dot(Vector, Vector);

	// Not sure if it's safe to add tolerance in there. Might introduce too many errors
	if (SquareSum == 1.f)
	{
		return Vector;
	}
	else if (SquareSum < FLOAT_SMALL_NUMBER)
	{
		return luisa::make_float3(0.0f);
	}
	const float Scale = 1.0f / luisa::sqrt(SquareSum);
	return Vector * Scale;
}

inline float CalcGradientsAndAngle(const luisa::float3& P1, const luisa::float3& P2, const luisa::float3& P3, const luisa::float3& P4, 
	luisa::float3& Grad1, luisa::float3& Grad2, luisa::float3& Grad3, luisa::float3& Grad4)
{
	const FVector3f SharedEdgeNormalized = VectorGetSafeNormal(P2 - P1);

	const FVector3f P13CrossP23 = luisa::cross(P1 - P3, P2 - P3);
	const float Normal1Len = luisa::length(P13CrossP23);
	const FVector3f Normal1 = SafeDivide(P13CrossP23, Normal1Len);

	const FVector3f P24CrossP14 = luisa::cross(P2 - P4, P1 - P4);
	const float Normal2Len = luisa::length(P24CrossP14);
	const FVector3f Normal2 = SafeDivide(P24CrossP14, Normal2Len);

	const FVector3f N2CrossN1 = luisa::cross(Normal2, Normal1);

	const float CosPhi = luisa::clamp(luisa::dot(Normal1, Normal2), (float)(-1), (float)(1));
	const float SinPhi = luisa::clamp(luisa::dot(N2CrossN1, SharedEdgeNormalized), (float)(-1), (float)(1));
	// const float SinPhi = luisa::clamp(luisa::dot(N2CrossN1, SharedEdgeNormalized), 1e-8, (float)(1));

	const float Angle = luisa::atan2(SinPhi, CosPhi); // if CosPhi == 0, atan(sin/cos) -> nan, so we use safe atan2
	// const float Angle = acos_scalar(CosPhi);

	const FVector3f DPhiDN1_OverNormal1Len = SafeDivide(CosPhi * luisa::cross(SharedEdgeNormalized, Normal2) - SinPhi * Normal2, Normal1Len);
	const FVector3f DPhiDN2_OverNormal2Len = SafeDivide(CosPhi * luisa::cross(Normal1, SharedEdgeNormalized) - SinPhi * Normal1, Normal2Len);

	const FVector3f DPhiDP13 = luisa::cross(P2 - P3, DPhiDN1_OverNormal1Len);
	const FVector3f DPhiDP23 = luisa::cross(DPhiDN1_OverNormal1Len, P1 - P3);
	const FVector3f DPhiDP24 = luisa::cross(P1 - P4, DPhiDN2_OverNormal2Len);
	const FVector3f DPhiDP14 = luisa::cross(DPhiDN2_OverNormal2Len, P2 - P4);

	// 梯度分配到四个点
	Grad1 = DPhiDP13 + DPhiDP14;
	Grad2 = DPhiDP23 + DPhiDP24;
	Grad3 = -1.0f * DPhiDP13 - DPhiDP23;
	Grad4 = -1.0f * DPhiDP14 - DPhiDP24;

	return Angle;
}



};


};