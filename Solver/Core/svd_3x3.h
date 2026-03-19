// Modified from [https://github.com/ericjang/svd3/blob/master/svd3.h]

#pragma once

#include "Core/scalar.h"
#include "Core/float_n.h"
#include "Core/float_nxn.h"

namespace lcs
{

	namespace Local
	{

		// Generic rsqrt that works for both float and Var<float>
		template <typename T>
		inline T rsqrt(const T& x)
		{
			return rsqrt_scalar(x);
		}

		constexpr float _gamma = 5.828427124f;	// FOUR_GAMMA_SQUARED = sqrt(8)+3;
		constexpr float _cstar = 0.923879532f;	// cos(pi/8)
		constexpr float _sstar = 0.3826834323f; // sin(p/8)
		constexpr float EPSILON = 1e-6f;

		// #include <cuda.h>
		// #include "math.h" // CUDA math library
		// CUDA's rsqrt seems to be faster than the inlined approximation?

		// Accurate sqrt using rsqrt; templated for float / Var<float>
		template <typename T>
		static inline T accurateSqrt(const T& x)
		{
			return x * rsqrt(x);
		}

		// Conditional swap that works with bool and Var<bool>
		template <typename CondT, typename T>
		static inline void condSwap(const CondT& c, T& X, T& Y)
		{
			T newX = select(c, Y, X);
			T newY = select(c, X, Y);
			X = newX;
			Y = newY;
		}

		template <typename CondT, typename T>
		static inline void condNegSwap(const CondT& c, T& X, T& Y)
		{
			// used in step 2 and 3
			T Z = -X;
			T newX = select(c, Y, X);
			T newY = select(c, Z, Y);
			X = newX;
			Y = newY;
		}

		// matrix multiplication M = A * B
		template <typename T>
		static inline void multAB(T a11,
			T						a12,
			T						a13,
			T						a21,
			T						a22,
			T						a23,
			T						a31,
			T						a32,
			T						a33,
			//
			T b11,
			T b12,
			T b13,
			T b21,
			T b22,
			T b23,
			T b31,
			T b32,
			T b33,
			//
			T& m11,
			T& m12,
			T& m13,
			T& m21,
			T& m22,
			T& m23,
			T& m31,
			T& m32,
			T& m33)
		{

			m11 = a11 * b11 + a12 * b21 + a13 * b31;
			m12 = a11 * b12 + a12 * b22 + a13 * b32;
			m13 = a11 * b13 + a12 * b23 + a13 * b33;
			m21 = a21 * b11 + a22 * b21 + a23 * b31;
			m22 = a21 * b12 + a22 * b22 + a23 * b32;
			m23 = a21 * b13 + a22 * b23 + a23 * b33;
			m31 = a31 * b11 + a32 * b21 + a33 * b31;
			m32 = a31 * b12 + a32 * b22 + a33 * b32;
			m33 = a31 * b13 + a32 * b23 + a33 * b33;
		}

		// matrix multiplication M = Transpose[A] * B
		template <typename T>
		static inline void multAtB(T a11,
			T						 a12,
			T						 a13,
			T						 a21,
			T						 a22,
			T						 a23,
			T						 a31,
			T						 a32,
			T						 a33,
			//
			T b11,
			T b12,
			T b13,
			T b21,
			T b22,
			T b23,
			T b31,
			T b32,
			T b33,
			//
			T& m11,
			T& m12,
			T& m13,
			T& m21,
			T& m22,
			T& m23,
			T& m31,
			T& m32,
			T& m33)
		{
			m11 = a11 * b11 + a21 * b21 + a31 * b31;
			m12 = a11 * b12 + a21 * b22 + a31 * b32;
			m13 = a11 * b13 + a21 * b23 + a31 * b33;
			m21 = a12 * b11 + a22 * b21 + a32 * b31;
			m22 = a12 * b12 + a22 * b22 + a32 * b32;
			m23 = a12 * b13 + a22 * b23 + a32 * b33;
			m31 = a13 * b11 + a23 * b21 + a33 * b31;
			m32 = a13 * b12 + a23 * b22 + a33 * b32;
			m33 = a13 * b13 + a23 * b23 + a33 * b33;
		}

		template <typename T>
		static inline void quatToMat3(
			const T* qV, T& m11, T& m12, T& m13, T& m21, T& m22, T& m23, T& m31, T& m32, T& m33)
		{
			T w = qV[3];
			T x = qV[0];
			T y = qV[1];
			T z = qV[2];

			T qxx = x * x;
			T qyy = y * y;
			T qzz = z * z;
			T qxz = x * z;
			T qxy = x * y;
			T qyz = y * z;
			T qwx = w * x;
			T qwy = w * y;
			T qwz = w * z;

			m11 = T(1.0f) - T(2.0f) * (qyy + qzz);
			m12 = T(2.0f) * (qxy - qwz);
			m13 = T(2.0f) * (qxz + qwy);
			m21 = T(2.0f) * (qxy + qwz);
			m22 = T(1.0f) - T(2.0f) * (qxx + qzz);
			m23 = T(2.0f) * (qyz - qwx);
			m31 = T(2.0f) * (qxz - qwy);
			m32 = T(2.0f) * (qyz + qwx);
			m33 = T(1.0f) - T(2.0f) * (qxx + qyy);
		}

		template <typename T>
		static inline void approximateGivensQuaternion(T a11, T a12, T a22, T& ch, T& sh)
		{
			/*
			 * Given givens angle computed by approximateGivensAngles,
			 * compute the corresponding rotation quaternion.
			 */
			ch = T(2.0f) * (a11 - a22);
			sh = a12;
			auto b = T(_gamma) * sh * sh < ch * ch;
			T	 w = rsqrt(ch * ch + sh * sh);
			ch = select(b, w * ch, T(_cstar));
			sh = select(b, w * sh, T(_sstar));
		}

		template <typename T>
		static inline void jacobiConjugation(
			const int x, const int y, const int z, T& s11, T& s21, T& s22, T& s31, T& s32, T& s33, T* qV)
		{
			T ch, sh;
			approximateGivensQuaternion(s11, s21, s22, ch, sh);

			T scale = ch * ch + sh * sh;
			T a = (ch * ch - sh * sh) / scale;
			T b = (T(2.0f) * sh * ch) / scale;

			// make temp copy of S
			T _s11 = s11;
			T _s21 = s21;
			T _s22 = s22;
			T _s31 = s31;
			T _s32 = s32;
			T _s33 = s33;

			// perform conjugation S = Q'*S*Q
			// Q already implicitly solved from a, b
			s11 = a * (a * _s11 + b * _s21) + b * (a * _s21 + b * _s22);
			s21 = a * (-b * _s11 + a * _s21) + b * (-b * _s21 + a * _s22);
			s22 = -b * (-b * _s11 + a * _s21) + a * (-b * _s21 + a * _s22);
			s31 = a * _s31 + b * _s32;
			s32 = -b * _s31 + a * _s32;
			s33 = _s33;

			// update cumulative rotation qV
			T tmp[3];
			tmp[0] = qV[0] * sh;
			tmp[1] = qV[1] * sh;
			tmp[2] = qV[2] * sh;
			sh *= qV[3];

			qV[0] *= ch;
			qV[1] *= ch;
			qV[2] *= ch;
			qV[3] *= ch;

			// (x,y,z) corresponds to ((0,1,2),(1,2,0),(2,0,1))
			// for (p,q) = ((0,1),(1,2),(0,2))
			qV[z] += sh;
			qV[3] -= tmp[z]; // w
			qV[x] += tmp[y];
			qV[y] -= tmp[x];

			// re-arrange matrix for next iteration
			_s11 = s22;
			_s21 = s32;
			_s22 = s33;
			_s31 = s21;
			_s32 = s31;
			_s33 = s11;
			s11 = _s11;
			s21 = _s21;
			s22 = _s22;
			s31 = _s31;
			s32 = _s32;
			s33 = _s33;
		}

		template <typename T>
		static inline T dist2(const T& x, const T& y, const T& z)
		{
			return x * x + y * y + z * z;
		}

		// finds transformation that diagonalizes a symmetric matrix
		template <typename T>
		static inline void jacobiEigenanlysis( // symmetric matrix
			T& s11,
			T& s21,
			T& s22,
			T& s31,
			T& s32,
			T& s33,
			// quaternion representation of V
			T* qV)
		{
			qV[3] = T(1.0f);
			qV[0] = T(0.0f);
			qV[1] = T(0.0f);
			qV[2] = T(0.0f); // follow same indexing convention as GLM
			for (int i = 0; i < 10; i++)
			{
				// we wish to eliminate the maximum off-diagonal element
				// on every iteration, but cycling over all 3 possible rotations
				// in fixed order (p,q) = (1,2) , (2,3), (1,3) still retains
				//  asymptotic convergence
				jacobiConjugation(0, 1, 2, s11, s21, s22, s31, s32, s33, qV); // p,q = 0,1
				jacobiConjugation(1, 2, 0, s11, s21, s22, s31, s32, s33, qV); // p,q = 1,2
				jacobiConjugation(2, 0, 1, s11, s21, s22, s31, s32, s33, qV); // p,q = 0,2
			}
		}

		template <typename T>
		static inline void sortSingularValues( // matrix that we want to decompose
			T& b11,
			T& b12,
			T& b13,
			T& b21,
			T& b22,
			T& b23,
			T& b31,
			T& b32,
			T& b33,
			// sort V simultaneously
			T& v11,
			T& v12,
			T& v13,
			T& v21,
			T& v22,
			T& v23,
			T& v31,
			T& v32,
			T& v33)
		{
			T	 rho1 = dist2(b11, b21, b31);
			T	 rho2 = dist2(b12, b22, b32);
			T	 rho3 = dist2(b13, b23, b33);
			auto c = rho1 < rho2;
			c = rho1 < rho2;
			condNegSwap(c, b11, b12);
			condNegSwap(c, v11, v12);
			condNegSwap(c, b21, b22);
			condNegSwap(c, v21, v22);
			condNegSwap(c, b31, b32);
			condNegSwap(c, v31, v32);
			condSwap(c, rho1, rho2);
			c = rho1 < rho3;
			condNegSwap(c, b11, b13);
			condNegSwap(c, v11, v13);
			condNegSwap(c, b21, b23);
			condNegSwap(c, v21, v23);
			condNegSwap(c, b31, b33);
			condNegSwap(c, v31, v33);
			condSwap(c, rho1, rho3);
			c = rho2 < rho3;
			condNegSwap(c, b12, b13);
			condNegSwap(c, v12, v13);
			condNegSwap(c, b22, b23);
			condNegSwap(c, v22, v23);
			condNegSwap(c, b32, b33);
			condNegSwap(c, v32, v33);
		}

		template <typename T>
		static inline void QRGivensQuaternion(T a1, T a2, T& ch, T& sh)
		{
			// a1 = pivot point on diagonal
			// a2 = lower triangular entry we want to annihilate
			T epsilon = T(EPSILON);
			T rho = accurateSqrt(a1 * a1 + a2 * a2);

			auto c1 = rho > epsilon;
			sh = select(c1, a2, T(0.0f));
			ch = abs_scalar(a1) + max_scalar(rho, epsilon);
			auto b = a1 < T(0.0f);
			condSwap(b, sh, ch);
			T w = rsqrt(ch * ch + sh * sh);
			ch *= w;
			sh *= w;
		}

		template <typename T>
		static inline void QRDecomposition( // matrix that we want to decompose
			T b11,
			T b12,
			T b13,
			T b21,
			T b22,
			T b23,
			T b31,
			T b32,
			T b33,
			// output Q
			T& q11,
			T& q12,
			T& q13,
			T& q21,
			T& q22,
			T& q23,
			T& q31,
			T& q32,
			T& q33,
			// output R
			T& r11,
			T& r12,
			T& r13,
			T& r21,
			T& r22,
			T& r23,
			T& r31,
			T& r32,
			T& r33)
		{
			T ch1, sh1, ch2, sh2, ch3, sh3;
			T a, b;

			// first givens rotation (ch,0,0,sh)
			QRGivensQuaternion(b11, b21, ch1, sh1);
			a = T(1.0f) - T(2.0f) * sh1 * sh1;
			b = T(2.0f) * ch1 * sh1;
			// apply B = Q' * B
			r11 = a * b11 + b * b21;
			r12 = a * b12 + b * b22;
			r13 = a * b13 + b * b23;
			r21 = -b * b11 + a * b21;
			r22 = -b * b12 + a * b22;
			r23 = -b * b13 + a * b23;
			r31 = b31;
			r32 = b32;
			r33 = b33;

			// second givens rotation (ch,0,-sh,0)
			QRGivensQuaternion(r11, r31, ch2, sh2);
			a = T(1.0f) - T(2.0f) * sh2 * sh2;
			b = T(2.0f) * ch2 * sh2;
			// apply B = Q' * B;
			b11 = a * r11 + b * r31;
			b12 = a * r12 + b * r32;
			b13 = a * r13 + b * r33;
			b21 = r21;
			b22 = r22;
			b23 = r23;
			b31 = -b * r11 + a * r31;
			b32 = -b * r12 + a * r32;
			b33 = -b * r13 + a * r33;

			// third givens rotation (ch,sh,0,0)
			QRGivensQuaternion(b22, b32, ch3, sh3);
			a = T(1.0f) - T(2.0f) * sh3 * sh3;
			b = T(2.0f) * ch3 * sh3;
			// R is now set to desired value
			r11 = b11;
			r12 = b12;
			r13 = b13;
			r21 = a * b21 + b * b31;
			r22 = a * b22 + b * b32;
			r23 = a * b23 + b * b33;
			r31 = -b * b21 + a * b31;
			r32 = -b * b22 + a * b32;
			r33 = -b * b23 + a * b33;

			// construct the cumulative rotation Q=Q1 * Q2 * Q3
			// the number of floating point operations for three quaternion multiplications
			// is more or less comparable to the explicit form of the joined matrix.
			// certainly more memory-efficient!
			T sh12 = sh1 * sh1;
			T sh22 = sh2 * sh2;
			T sh32 = sh3 * sh3;

			q11 = (T(-1.0f) + T(2.0f) * sh12) * (T(-1.0f) + T(2.0f) * sh22);
			q12 = T(4.0f) * ch2 * ch3 * (T(-1.0f) + T(2.0f) * sh12) * sh2 * sh3 + T(2.0f) * ch1 * sh1 * (T(-1.0f) + T(2.0f) * sh32);
			q13 = T(4.0f) * ch1 * ch3 * sh1 * sh3 - T(2.0f) * ch2 * (T(-1.0f) + T(2.0f) * sh12) * sh2 * (T(-1.0f) + T(2.0f) * sh32);

			q21 = T(2.0f) * ch1 * sh1 * (T(1.0f) - T(2.0f) * sh22);
			q22 = T(-8.0f) * ch1 * ch2 * ch3 * sh1 * sh2 * sh3 + (T(-1.0f) + T(2.0f) * sh12) * (T(-1.0f) + T(2.0f) * sh32);
			q23 = -T(2.0f) * ch3 * sh3 + T(4.0f) * sh1 * (ch3 * sh1 * sh3 + ch1 * ch2 * sh2 * (T(-1.0f) + T(2.0f) * sh32));

			q31 = T(2.0f) * ch2 * sh2;
			q32 = T(2.0f) * ch3 * (T(1.0f) - T(2.0f) * sh22) * sh3;
			q33 = (T(-1.0f) + T(2.0f) * sh22) * (T(-1.0f) + T(2.0f) * sh32);
		}

		template <typename T>
		static inline void svd_built_in( // input A
			T a11,
			T a12,
			T a13,
			T a21,
			T a22,
			T a23,
			T a31,
			T a32,
			T a33,
			// output U
			T& u11,
			T& u12,
			T& u13,
			T& u21,
			T& u22,
			T& u23,
			T& u31,
			T& u32,
			T& u33,
			// output S
			T& s11,
			T& s12,
			T& s13,
			T& s21,
			T& s22,
			T& s23,
			T& s31,
			T& s32,
			T& s33,
			// output V
			T& v11,
			T& v12,
			T& v13,
			T& v21,
			T& v22,
			T& v23,
			T& v31,
			T& v32,
			T& v33)
		{
			// normal equations matrix
			T ATA11, ATA12, ATA13;
			T ATA21, ATA22, ATA23;
			T ATA31, ATA32, ATA33;

			multAtB(a11, a12, a13, a21, a22, a23, a31, a32, a33, a11, a12, a13, a21, a22, a23, a31, a32, a33, ATA11, ATA12, ATA13, ATA21, ATA22, ATA23, ATA31, ATA32, ATA33);

			// symmetric eigenalysis
			T qV[4];
			jacobiEigenanlysis(ATA11, ATA21, ATA22, ATA31, ATA32, ATA33, qV);
			quatToMat3(qV, v11, v12, v13, v21, v22, v23, v31, v32, v33);

			T b11, b12, b13;
			T b21, b22, b23;
			T b31, b32, b33;
			multAB(a11, a12, a13, a21, a22, a23, a31, a32, a33, v11, v12, v13, v21, v22, v23, v31, v32, v33, b11, b12, b13, b21, b22, b23, b31, b32, b33);

			// sort singular values and find V
			sortSingularValues(b11, b12, b13, b21, b22, b23, b31, b32, b33, v11, v12, v13, v21, v22, v23, v31, v32, v33);

			// QR decomposition
			QRDecomposition(
				b11, b12, b13, b21, b22, b23, b31, b32, b33, u11, u12, u13, u21, u22, u23, u31, u32, u33, s11, s12, s13, s21, s22, s23, s31, s32, s33);
		}

	} // namespace Local

	// Generic SVD wrapper that works for both float3x3 and Var<float3x3>.
	// Mat3T is expected to be lcs::float3x3 or Var<lcs::float3x3>.
	template <typename Mat3T, typename Vec3T>
	inline void svd(const Mat3T& F, Mat3T& U, Vec3T& Sigma, Mat3T& V)
	{
		using ScalarT = std::decay_t<decltype(F[0][0])>;

		ScalarT u11, u12, u13, u21, u22, u23, u31, u32, u33;
		ScalarT s11, s12, s13, s21, s22, s23, s31, s32, s33;
		ScalarT v11, v12, v13, v21, v22, v23, v31, v32, v33;

		// float3x3 uses column-major indexing M[col][row], while svd_built_in expects row-major scalars.
		Local::svd_built_in<ScalarT>(F[0][0],
			F[1][0],
			F[2][0],
			F[0][1],
			F[1][1],
			F[2][1],
			F[0][2],
			F[1][2],
			F[2][2],
			u11,
			u12,
			u13,
			u21,
			u22,
			u23,
			u31,
			u32,
			u33,
			s11,
			s12,
			s13,
			s21,
			s22,
			s23,
			s31,
			s32,
			s33,
			v11,
			v12,
			v13,
			v21,
			v22,
			v23,
			v31,
			v32,
			v33);

		U[0][0] = u11;
		U[1][0] = u12;
		U[2][0] = u13;
		U[0][1] = u21;
		U[1][1] = u22;
		U[2][1] = u23;
		U[0][2] = u31;
		U[1][2] = u32;
		U[2][2] = u33;

		V[0][0] = v11;
		V[1][0] = v12;
		V[2][0] = v13;
		V[0][1] = v21;
		V[1][1] = v22;
		V[2][1] = v23;
		V[0][2] = v31;
		V[1][2] = v32;
		V[2][2] = v33;

		Sigma[0] = s11;
		Sigma[1] = s22;
		Sigma[2] = s33;
	}

} // namespace lcs