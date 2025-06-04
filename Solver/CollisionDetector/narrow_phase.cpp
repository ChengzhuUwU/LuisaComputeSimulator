#include "CollisionDetector/narrow_phase.h"
#include "CollisionDetector/accd.hpp"
#include "Utils/cpu_parallel.h"
#include <Eigen/Dense>
#include <iostream>
#include "Utils/reduce_helper.h"

namespace lcsv 
{

using EigenFloat3 = Eigen::Vector<float, 3>;
using EigenFloat4x3 = Eigen::Matrix<float, 3, 4, Eigen::ColMajor>;
static inline auto float3_to_eigen3(const float3& input) { EigenFloat3 vec; vec << input[0], input[1], input[2]; return vec; };
static inline auto eigen3_to_float3(const EigenFloat3& input) { return luisa::make_float3(input(0, 0), input(1, 0), input(2, 0)); };
static inline auto make_eigen4x3(const EigenFloat3& c0, const EigenFloat3& c1, const EigenFloat3& c2, const EigenFloat3& c3) 
{ 
    EigenFloat4x3 mat; mat << c0, c1, c2, c3; return mat;
}



namespace ipc 
{

template <typename T>
T barrier(const T d, const T dhat) // E
{
    const T d_minus_dhat = (d - dhat);
    // b(d) = -(d-d̂)²ln(d / d̂)
    return -d_minus_dhat * d_minus_dhat * log_scalar(d / dhat);
}
template <typename T>
T barrier_first_derivative(const T d, const T dhat) // \frac{ \partial E }{ \partial d }
{
    // b(d) = -(d - d̂)²ln(d / d̂)
    // b'(d) = -2(d - d̂)ln(d / d̂) - (d-d̂)²(1 / d)
    //       = (d - d̂) * (-2ln(d/d̂) - (d - d̂) / d)
    //       = (d̂ - d) * (2ln(d/d̂) - d̂/d + 1)
    return (dhat - d) * (2.0f * log_scalar(d / dhat) - dhat / d + 1.0f);
}
template <typename T>
T barrier_second_derivative(const T d, const T dhat) // \frac{ \partial^2 E }{ \partial d^2 }
{
    const T dhat_d = dhat / d;
    return (dhat_d + 2.0f) * dhat_d - 2.0f * log_scalar(d / dhat) - 3.0f;
}

} // namespace ipc

namespace cipc
{
    
// template <typename T>
// inline void KappaBarrier(T& R, const T& kappa, const T& d2, const T& dHat, const T& xi)
// {
//     auto x0 = square_scalar(xi);
//     auto x1 = square_scalar(dHat) + T(2)*dHat*xi;
//     /* Simplified Expr */
//     R = -kappa*square_scalar(d2 - x0 - x1)*log_scalar((d2 - x0)/x1);
// }

template <typename T>
inline void NoKappa_Barrier(T& R, const T& d2, const T& dHat, const T& xi)
{
    auto x0 = square_scalar(xi);
    auto x1 = square_scalar(dHat) + T(2)*dHat*xi;
    /* Simplified Expr */
    R = -square_scalar(d2 - x0 - x1)*log_scalar((d2 - x0)/x1);
}
template <typename T>
inline void NoKappa_dBarrier_dD(T& R, const T& d2, const T& dHat, const T& xi)
{
    auto x0 = square_scalar(xi);
    auto x1 = d2 - x0;
    auto x2 = square_scalar(dHat);
    auto x3 = dHat*xi;
    auto x4 = x2 + 2*x3;
    /* Simplified Expr */
    R = -(2*d2 - 2*x0 - 2*x2 - 4*x3)*log_scalar(x1/x4) - square_scalar(d2 - x0 - x4)/x1;
}
template <typename T>
inline void NoKappa_ddBarrier_ddD(T& R, const T& d2, const T& dHat, const T& xi)
{
    auto x0 = square_scalar(xi);
    auto x1 = d2 - x0;
    auto x2 = square_scalar(dHat);
    auto x3 = dHat*xi;
    auto x4 = x2 + 2*x3;
    auto x5 = 2;
    /* Simplified Expr */
    R = square_scalar(d2 - x0 - x4)/square_scalar(x1) 
    - x5*log_scalar(x1/x4) 
    - x5*(2*d2 - 2*x0 - 2*x2 - 4*x3)/x1;
}

} // namespace cipc

namespace DistanceGradient
{
   
namespace details
{

template <uint idx, typename LargeVector>
constexpr lcsv::Float& getVec(LargeVector& G)
{
    constexpr uint outer_row_idx = idx / 3;
    constexpr uint inner_row_idx = idx % 3;
    return G.vec[outer_row_idx][inner_row_idx];
}
template <uint idx, typename LargeVector>
constexpr void setVec(LargeVector& G, const lcsv::Float& value)
{
    constexpr uint outer_row_idx = idx / 3;
    constexpr uint inner_row_idx = idx % 3;
    G.vec[outer_row_idx][inner_row_idx] = value;
}
template <uint idx, typename LargeMatrix>
constexpr lcsv::Float& getMat9x9(LargeMatrix& H)
{
    constexpr uint col_idx = idx % 9;
    constexpr uint row_idx = idx / 9;
    constexpr uint outer_col_idx = col_idx / 3;
    constexpr uint outer_row_idx = row_idx / 3;
    constexpr uint inner_col_idx = col_idx % 3;
    constexpr uint inner_row_idx = row_idx % 3;
    return H.mat[outer_row_idx][outer_col_idx][inner_row_idx][inner_col_idx];
}
template <uint idx, typename LargeMatrix>
constexpr lcsv::Float& getMat12x12(LargeMatrix& H)
{
    constexpr uint col_idx = idx % 16;
    constexpr uint row_idx = idx / 16;
    constexpr uint outer_col_idx = col_idx / 3;
    constexpr uint outer_row_idx = row_idx / 3;
    constexpr uint inner_col_idx = col_idx % 3;
    constexpr uint inner_row_idx = row_idx % 3;
    return H.mat[outer_row_idx][outer_col_idx][inner_row_idx][inner_col_idx];
}
template <uint idx, typename LargeMatrix>
constexpr void setMat9x9(LargeMatrix& H, const lcsv::Float& value)
{
    constexpr uint col_idx = idx % 9;
    constexpr uint row_idx = idx / 9;
    constexpr uint outer_col_idx = col_idx / 3;
    constexpr uint outer_row_idx = row_idx / 3;
    constexpr uint inner_col_idx = col_idx % 3;
    constexpr uint inner_row_idx = row_idx % 3;
    H.mat[outer_row_idx][outer_col_idx][inner_row_idx][inner_col_idx] = value;
}
template <uint idx, typename LargeMatrix>
constexpr void setMat12x12(LargeMatrix& H, const lcsv::Float& value)
{
    constexpr uint col_idx = idx % 16;
    constexpr uint row_idx = idx / 16;
    constexpr uint outer_col_idx = col_idx / 3;
    constexpr uint outer_row_idx = row_idx / 3;
    constexpr uint inner_col_idx = col_idx % 3;
    constexpr uint inner_row_idx = row_idx % 3;
    H.mat[outer_row_idx][outer_col_idx][inner_row_idx][inner_col_idx] = value;
}

template <class T, class Vec>
inline void g_PE3D(T v01, T v02, T v03, T v11, T v12, T v13, T v21, T v22, T v23, Vec& g)
{
    T t17;
    T t18;
    T t19;
    T t20;
    T t21;
    T t22;
    T t23;
    T t24;
    T t25;
    T t42;
    T t44;
    T t45;
    T t46;
    T t43;
    T t50;
    T t51;
    T t52;

    /* G_PE */
    /*     G = G_PE(V01,V02,V03,V11,V12,V13,V21,V22,V23) */
    /*     This function was generated by the Symbolic Math Toolbox version 8.3. */
    /*     10-Jun-2019 18:02:37 */
    t17  = -v11 + v01;
    t18  = -v12 + v02;
    t19  = -v13 + v03;
    t20  = -v21 + v01;
    t21  = -v22 + v02;
    t22  = -v23 + v03;
    t23  = -v21 + v11;
    t24  = -v22 + v12;
    t25  = -v23 + v13;
    t42  = 1.0f / ((t23 * t23 + t24 * t24) + t25 * t25);
    t44  = t17 * t21 + -(t18 * t20);
    t45  = t17 * t22 + -(t19 * t20);
    t46  = t18 * t22 + -(t19 * t21);
    t43  = t42 * t42;
    t50  = (t44 * t44 + t45 * t45) + t46 * t46;
    t51  = (v11 * 2.0f + -(v21 * 2.0f)) * t43 * t50;
    t52  = (v12 * 2.0f + -(v22 * 2.0f)) * t43 * t50;
    t43  = (v13 * 2.0f + -(v23 * 2.0f)) * t43 * t50;
    getVec<0>(g) =  t42 * (t24 * t44 * 2.0f + t25 * t45 * 2.0f);
    getVec<1>(g) = -t42 * (t23 * t44 * 2.0f - t25 * t46 * 2.0f);
    getVec<2>(g) = -t42 * (t23 * t45 * 2.0f + t24 * t46 * 2.0f);
    getVec<3>(g) = -t51 - t42 * (t21 * t44 * 2.0f + t22 * t45 * 2.0f);
    getVec<4>(g) = -t52 + t42 * (t20 * t44 * 2.0f - t22 * t46 * 2.0f);
    getVec<5>(g) = -t43 + t42 * (t20 * t45 * 2.0f + t21 * t46 * 2.0f);
    getVec<6>(g) = t51 + t42 * (t18 * t44 * 2.0f + t19 * t45 * 2.0f);
    getVec<7>(g) = t52 - t42 * (t17 * t44 * 2.0f - t19 * t46 * 2.0f);
    getVec<8>(g) = t43 - t42 * (t17 * t45 * 2.0f + t18 * t46 * 2.0f);
}

template <class T, class Vec>
inline void g_PT(
    T v01, T v02, T v03, T v11, T v12, T v13, T v21, T v22, T v23, T v31, T v32, T v33, Vec& g)
{
    T t11;
    T t12;
    T t13;
    T t14;
    T t15;
    T t16;
    T t17;
    T t18;
    T t19;
    T t20;
    T t21;
    T t22;
    T t32;
    T t33;
    T t34;
    T t43;
    T t45;
    T t44;
    T t46;

    /* G_PT */
    /*     G = G_PT(V01,V02,V03,V11,V12,V13,V21,V22,V23,V31,V32,V33) */
    /*     This function was generated by the Symbolic Math Toolbox version 8.3. */
    /*     10-Jun-2019 17:42:16 */
    t11  = -v11 + v01;
    t12  = -v12 + v02;
    t13  = -v13 + v03;
    t14  = -v21 + v11;
    t15  = -v22 + v12;
    t16  = -v23 + v13;
    t17  = -v31 + v11;
    t18  = -v32 + v12;
    t19  = -v33 + v13;
    t20  = -v31 + v21;
    t21  = -v32 + v22;
    t22  = -v33 + v23;
    t32  = t14 * t18 + -(t15 * t17);
    t33  = t14 * t19 + -(t16 * t17);
    t34  = t15 * t19 + -(t16 * t18);
    t43  = 1.0f / ((t32 * t32 + t33 * t33) + t34 * t34);
    t45  = (t13 * t32 + t11 * t34) + -(t12 * t33);
    t44  = t43 * t43;
    t46  = t45 * t45;
    getVec<0>(g) = t34 * t43 * t45 *  2.0f;
    getVec<1>(g) = t33 * t43 * t45 * -2.0f;
    getVec<2>(g) = t32 * t43 * t45 *  2.0f; t45 *= t43;
    getVec<3>(g) =  - t44 * t46 * (t21 * t32 * 2.0f + t22 * t33 * 2.0f) - t45 * ((t34 + t12 * t22) - t13 * t21) * 2.0f; t43  = t44 * t46;
    getVec<4>(g) = t43 * (t20 * t32 * 2.0f - t22 * t34 * 2.0f) + t45 * ((t33 + t11 * t22) - t13 * t20) * 2.0f;
    getVec<5>(g) = t43 * (t20 * t33 * 2.0f + t21 * t34 * 2.0f) - t45 * ((t32 + t11 * t21) - t12 * t20) * 2.0f;
    getVec<6>(g) = t45 * (t12 * t19 - t13 * t18) * 2.0f + t43 * (t18 * t32 * 2.0f + t19 * t33 * 2.0f);
    getVec<7>(g) = t45 * (t11 * t19 - t13 * t17) * -2.0f - t43 * (t17 * t32 * 2.0f - t19 * t34 * 2.0f);
    getVec<8>(g) = t45 * (t11 * t18 - t12 * t17) * 2.0f - t43 * (t17 * t33 * 2.0f + t18 * t34 * 2.0f);
    getVec<9>(g) = t45 * (t12 * t16 - t13 * t15) * -2.0f - t43 * (t15 * t32 * 2.0f + t16 * t33 * 2.0f);
    getVec<10>(g) = t45 * (t11 * t16 - t13 * t14) * 2.0f + t43 * (t14 * t32 * 2.0f - t16 * t34 * 2.0f);
    getVec<11>(g) = t45 * (t11 * t15 - t12 * t14) * -2.0f + t43 * (t14 * t33 * 2.0f + t15 * t34 * 2.0f);
}

template <class T, class Vec>
inline void g_EE(
    T v01, T v02, T v03, T v11, T v12, T v13, T v21, T v22, T v23, T v31, T v32, T v33, Vec& g)
{
    T t11;
    T t12;
    T t13;
    T t14;
    T t15;
    T t16;
    T t17;
    T t18;
    T t19;
    T t32;
    T t33;
    T t34;
    T t35;
    T t36;
    T t37;
    T t44;
    T t45;
    T t46;
    T t75;
    T t77;
    T t76;
    T t78;
    T t79;
    T t80;
    T t81;
    T t83;

    /* G_EE */
    /*     G = G_EE(V01,V02,V03,V11,V12,V13,V21,V22,V23,V31,V32,V33) */
    /*     This function was generated by the Symbolic Math Toolbox version 8.3. */
    /*     14-Jun-2019 13:58:25 */
    t11   = -v11 + v01;
    t12   = -v12 + v02;
    t13   = -v13 + v03;
    t14   = -v21 + v01;
    t15   = -v22 + v02;
    t16   = -v23 + v03;
    t17   = -v31 + v21;
    t18   = -v32 + v22;
    t19   = -v33 + v23;
    t32   = t14 * t18;
    t33   = t15 * t17;
    t34   = t14 * t19;
    t35   = t16 * t17;
    t36   = t15 * t19;
    t37   = t16 * t18;
    t44   = t11 * t18 + -(t12 * t17);
    t45   = t11 * t19 + -(t13 * t17);
    t46   = t12 * t19 + -(t13 * t18);
    t75   = 1.0f / ((t44 * t44 + t45 * t45) + t46 * t46);
    t77   = (t16 * t44 + t14 * t46) + -(t15 * t45);
    t76   = t75 * t75;
    t78   = t77 * t77;
    t79   = (t12 * t44 * 2.0f +   t13 * t45 * 2.0f) * t76 * t78;
    t80   = (t11 * t45 * 2.0f +   t12 * t46 * 2.0f) * t76 * t78;
    t81   = (t18 * t44 * 2.0f +   t19 * t45 * 2.0f) * t76 * t78;
    t18   = (t17 * t45 * 2.0f +   t18 * t46 * 2.0f) * t76 * t78;
    t83   = (t11 * t44 * 2.0f + -(t13 * t46 * 2.0f)) * t76 * t78;
    t19   = (t17 * t44 * 2.0f + -(t19 * t46 * 2.0f)) * t76 * t78;
    t76   = t75 * t77;
    getVec<0>(g)  = -t81 + t76 * ((-t36 + t37) + t46) * 2.0f;
    getVec<1>(g)  =  t19 - t76 * ((-t34 + t35) + t45) * 2.0f;
    getVec<2>(g)  =  t18 + t76 * ((-t32 + t33) + t44) * 2.0f;
    getVec<3>(g)  =  t81 + t76 * (t36 - t37) * 2.0f;
    getVec<4>(g)  = -t19 - t76 * (t34 - t35) * 2.0f;
    getVec<5>(g)  = -t18 + t76 * (t32 - t33) * 2.0f; t17   = t12 * t16 + -(t13 * t15);
    getVec<6>(g)  = t79 - t76 * (t17 + t46) * 2.0f; t18   = t11 * t16 + -(t13 * t14);
    getVec<7>(g)  = -t83 + t76 * (t18 + t45) * 2.0f; t19   = t11 * t15 + -(t12 * t14);
    getVec<8>(g)  = -t80 - t76 * (t19 + t44) * 2.0f;
    getVec<9>(g) = -t79 + t76 * t17 * 2.0f;
    getVec<10>(g) =  t83 - t76 * t18 * 2.0f;
    getVec<11>(g) =  t80 + t76 * t19 * 2.0f;
}

} // namespace details

namespace details
{

// template <uint idx, typename Mat>
// constexpr void setMat9x9(Mat H[3][3], const lcsv::Float& value)
// {
//     constexpr uint col_idx = idx % 9;
//     constexpr uint row_idx = idx / 9;
//     constexpr uint outer_col_idx = col_idx / 3;
//     constexpr uint outer_row_idx = row_idx / 3;
//     constexpr uint inner_col_idx = col_idx % 3;
//     constexpr uint inner_row_idx = row_idx % 3;
//     H[outer_row_idx][outer_col_idx][inner_row_idx][inner_col_idx] = value;
// }
// template <uint idx, typename Mat>
// constexpr void setMat12x12(Mat H[4][4], const lcsv::Float& value)
// {
//     constexpr uint col_idx = idx % 16;
//     constexpr uint row_idx = idx / 16;
//     constexpr uint outer_col_idx = col_idx / 4;
//     constexpr uint outer_row_idx = row_idx / 4;
//     constexpr uint inner_col_idx = col_idx % 4;
//     constexpr uint inner_row_idx = row_idx % 4;
//     H[outer_row_idx][outer_col_idx][inner_row_idx][inner_col_idx] = value;
// }


template <class T>
inline void H_PE3D(T v01, T v02, T v03, T v11, T v12, T v13, T v21, T v22, T v23, Float9x9& mat)
{
    T t17;
    T t18;
    T t19;
    T t20;
    T t21;
    T t22;
    T t23;
    T t24;
    T t25;
    T t26;
    T t27;
    T t28;
    T t35;
    T t36;
    T t37;
    T t50;
    T t51;
    T t52;
    T t53;
    T t54;
    T t55;
    T t56;
    T t62;
    T t70;
    T t71;
    T t75;
    T t79;
    T t80;
    T t84;
    T t88;
    T t38;
    T t39;
    T t40;
    T t41;
    T t42;
    T t43;
    T t44;
    T t46;
    T t48;
    T t57;
    T t58;
    T t60;
    T t63;
    T t65;
    T t67;
    T t102;
    T t103;
    T t104;
    T t162;
    T t163;
    T t164;
    T t213;
    T t214;
    T t215;
    T t216;
    T t217;
    T t218;
    T t225;
    T t226;
    T t227;
    T t229;
    T t230;
    T t311;
    T t231;
    T t232;
    T t233;
    T t234;
    T t235;
    T t236;
    T t237;
    T t238;
    T t239;
    T t240;
    T t245;
    T t279;
    T t281;
    T t282;
    T t283;
    T t287;
    T t289;
    T t247;
    T t248;
    T t249;
    T t250;
    T t251;
    T t252;
    T t253;
    T t293;
    T t295;
    T t299;
    T t300;
    T t303;
    T t304;
    T t294;
    T t297;
    T t301;
    T t302;

    /* H_PE */
    /*     H = H_PE(V01,V02,V03,V11,V12,V13,V21,V22,V23) */
    /*     This function was generated by the Symbolic Math Toolbox version 8.3. */
    /*     10-Jun-2019 18:02:39 */
    t17   = -v11 + v01;
    t18   = -v12 + v02;
    t19   = -v13 + v03;
    t20   = -v21 + v01;
    t21   = -v22 + v02;
    t22   = -v23 + v03;
    t23   = -v21 + v11;
    t24   = -v22 + v12;
    t25   = -v23 + v13;
    t26   = v11 * 2.0f + -(v21 * 2.0f);
    t27   = v12 * 2.0f + -(v22 * 2.0f);
    t28   = v13 * 2.0f + -(v23 * 2.0f);
    t35   = t23 * t23;
    t36   = t24 * t24;
    t37   = t25 * t25;
    t50   = t17 * t21;
    t51   = t18 * t20;
    t52   = t17 * t22;
    t53   = t19 * t20;
    t54   = t18 * t22;
    t55   = t19 * t21;
    t56   = t17 * t20 * 2.0f;
    t62   = t18 * t21 * 2.0f;
    t70   = t19 * t22 * 2.0f;
    t71   = t17 * t23 * 2.0f;
    t75   = t18 * t24 * 2.0f;
    t79   = t19 * t25 * 2.0f;
    t80   = t20 * t23 * 2.0f;
    t84   = t21 * t24 * 2.0f;
    t88   = t22 * t25 * 2.0f;
    t38   = t17 * t17 * 2.0f;
    t39   = t18 * t18 * 2.0f;
    t40   = t19 * t19 * 2.0f;
    t41   = t20 * t20 * 2.0f;
    t42   = t21 * t21 * 2.0f;
    t43   = t22 * t22 * 2.0f;
    t44   = t35 * 2.0f;
    t46   = t36 * 2.0f;
    t48   = t37 * 2.0f;
    t57   = t50 * 2.0f;
    t58   = t51 * 2.0f;
    t60   = t52 * 2.0f;
    t63   = t53 * 2.0f;
    t65   = t54 * 2.0f;
    t67   = t55 * 2.0f;
    t102  = 1.0f / ((t35 + t36) + t37);
    t36   = t50 + -t51;
    t35   = t52 + -t53;
    t37   = t54 + -t55;
    t103  = t102 * t102;
    t104  = pow(t102, 3.0f);
    t162  = -(t23 * t24 * t102 * 2.0f);
    t163  = -(t23 * t25 * t102 * 2.0f);
    t164  = -(t24 * t25 * t102 * 2.0f);
    t213  = t18 * t36 * 2.0f + t19 * t35 * 2.0f;
    t214  = t17 * t35 * 2.0f + t18 * t37 * 2.0f;
    t215  = t21 * t36 * 2.0f + t22 * t35 * 2.0f;
    t216  = t20 * t35 * 2.0f + t21 * t37 * 2.0f;
    t217  = t24 * t36 * 2.0f + t25 * t35 * 2.0f;
    t218  = t23 * t35 * 2.0f + t24 * t37 * 2.0f;
    t35   = (t36 * t36 + t35 * t35) + t37 * t37;
    t225  = t17 * t36 * 2.0f + -(t19 * t37 * 2.0f);
    t226  = t20 * t36 * 2.0f + -(t22 * t37 * 2.0f);
    t227  = t23 * t36 * 2.0f + -(t25 * t37 * 2.0f);
    t36   = t26 * t103;
    t229  = t36 * t213;
    t37   = t27 * t103;
    t230  = t37 * t213;
    t311  = t28 * t103;
    t231  = t311 * t213;
    t232  = t36 * t214;
    t233  = t37 * t214;
    t234  = t311 * t214;
    t235  = t36 * t215;
    t236  = t37 * t215;
    t237  = t311 * t215;
    t238  = t36 * t216;
    t239  = t37 * t216;
    t240  = t311 * t216;
    t214  = t36 * t217;
    t215  = t37 * t217;
    t216  = t311 * t217;
    t217  = t36 * t218;
    t245  = t37 * t218;
    t213  = t311 * t218;
    t279  = t103 * t35 * 2.0f;
    t281  = t26 * t26 * t104 * t35 * 2.0f;
    t282  = t27 * t27 * t104 * t35 * 2.0f;
    t283  = t28 * t28 * t104 * t35 * 2.0f;
    t287  = t26 * t27 * t104 * t35 * 2.0f;
    t218  = t26 * t28 * t104 * t35 * 2.0f;
    t289  = t27 * t28 * t104 * t35 * 2.0f;
    t247  = t36 * t225;
    t248  = t37 * t225;
    t249  = t311 * t225;
    t250  = t36 * t226;
    t251  = t37 * t226;
    t252  = t311 * t226;
    t253  = t36 * t227;
    t35   = t37 * t227;
    t36   = t311 * t227;
    t293  = t102 * (t75 + t79) + t214;
    t295  = -(t102 * (t80 + t84)) + t213;
    t299  = t102 * ((t63 + t22 * t23 * 2.0f) + -t60) + t217;
    t300  = t102 * ((t67 + t22 * t24 * 2.0f) + -t65) + t245;
    t303  = -(t102 * ((t57 + t17 * t24 * 2.0f) + -t58)) + t215;
    t304  = -(t102 * ((t60 + t17 * t25 * 2.0f) + -t63)) + t216;
    t294  = t102 * (t71 + t75) + -t213;
    t297  = -(t102 * (t80 + t88)) + t35;
    t88   = -(t102 * (t84 + t88)) + -t214;
    t301  = t102 * ((t58 + t21 * t23 * 2.0f) + -t57) + t253;
    t302  = t102 * ((t65 + t21 * t25 * 2.0f) + -t67) + t36;
    t84   = t102 * ((t57 + t20 * t24 * 2.0f) + -t58) + -t215;
    t80   = t102 * ((t60 + t20 * t25 * 2.0f) + -t63) + -t216;
    t75   = -(t102 * ((t63 + t19 * t23 * 2.0f) + -t60)) + -t217;
    t227  = -(t102 * ((t67 + t19 * t24 * 2.0f) + -t65)) + -t245;
    t311  = ((-(t17 * t19 * t102 * 2.0f) + t231) + -t232) + t218;
    t245  = ((-(t20 * t22 * t102 * 2.0f) + t237) + -t238) + t218;
    t226  = ((-t102 * (t67 - t54 * 4.0f) + t233) + t252) + -t289;
    t28   = ((-t102 * (t63 - t52 * 4.0f) + t232) + -t237) + -t218;
    t27   = ((-t102 * (t58 - t50 * 4.0f) + t247) + -t236) + -t287;
    t225  = ((-(t102 * (t65 + -(t55 * 4.0f))) + t239) + t249) + -t289;
    t26   = ((-(t102 * (t60 + -(t53 * 4.0f))) + t238) + -t231) + -t218;
    t103  = ((-(t102 * (t57 + -(t51 * 4.0f))) + t250) + -t230) + -t287;
    t104  = (((-(t102 * (t56 + t62)) + t234) + t240) + t279) + -t283;
    t218  = (((-(t102 * (t56 + t70)) + t248) + t251) + t279) + -t282;
    t217  = (((-(t102 * (t62 + t70)) + -t229) + -t235) + t279) + -t281;
    t216  = t102 * (t71 + t79) + -t35;
    t215  = -(t102 * ((t58 + t18 * t23 * 2.0f) + -t57)) + -t253;
    t214  = -(t102 * ((t65 + t18 * t25 * 2.0f) + -t67)) + -t36;
    t213  = ((-(t17 * t18 * t102 * 2.0f) + t230) + -t247) + t287;
    t37   = ((-(t20 * t21 * t102 * 2.0f) + t236) + -t250) + t287;
    t36   = ((-(t18 * t19 * t102 * 2.0f) + -t233) + -t249) + t289;
    t35   = ((-(t21 * t22 * t102 * 2.0f) + -t239) + -t252) + t289;
    setMat9x9<0>(mat, t102 * (t46 + t48));
    setMat9x9<1>(mat, t162);
    setMat9x9<2>(mat, t163);
    setMat9x9<3>(mat, t88);
    setMat9x9<4>(mat, t84);
    setMat9x9<5>(mat, t80);
    setMat9x9<6>(mat, t293);
    setMat9x9<7>(mat, t303);
    setMat9x9<8>(mat, t304);
    setMat9x9<9>(mat, t162);
    setMat9x9<10>(mat, t102 * (t44 + t48));
    setMat9x9<11>(mat, t164);
    setMat9x9<12>(mat, t301);
    setMat9x9<13>(mat, t297);
    setMat9x9<14>(mat, t302);
    setMat9x9<15>(mat, t215);
    setMat9x9<16>(mat, t216);
    setMat9x9<17>(mat, t214);
    setMat9x9<18>(mat, t163);
    setMat9x9<19>(mat, t164);
    setMat9x9<20>(mat, t102 * (t44 + t46));
    setMat9x9<21>(mat, t299);
    setMat9x9<22>(mat, t300);
    setMat9x9<23>(mat, t295);
    setMat9x9<24>(mat, t75);
    setMat9x9<25>(mat, t227);
    setMat9x9<26>(mat, t294);
    setMat9x9<27>(mat, t88);
    setMat9x9<28>(mat, t301);
    setMat9x9<29>(mat, t299);
    setMat9x9<30>(mat, ((t235 * 2.0f + -t279) + t281) + t102 * (t42 + t43));
    setMat9x9<31>(mat, t37);
    setMat9x9<32>(mat, t245);
    setMat9x9<33>(mat, t217);
    setMat9x9<34>(mat, t27);
    setMat9x9<35>(mat, t28);
    setMat9x9<36>(mat, t84);
    setMat9x9<37>(mat, t297);
    setMat9x9<38>(mat, t300);
    setMat9x9<39>(mat, t37);
    setMat9x9<40>(mat, ((t251 * -2.0f + -t279) + t282) + t102 * (t41 + t43));
    setMat9x9<41>(mat, t35);
    setMat9x9<42>(mat, t103);
    setMat9x9<43>(mat, t218);
    setMat9x9<44>(mat, t226);
    setMat9x9<45>(mat, t80);
    setMat9x9<46>(mat, t302);
    setMat9x9<47>(mat, t295);
    setMat9x9<48>(mat, t245);
    setMat9x9<49>(mat, t35);
    setMat9x9<50>(mat, ((t240 * -2.0f + -t279) + t283) + t102 * (t41 + t42));
    setMat9x9<51>(mat, t26);
    setMat9x9<52>(mat, t225);
    setMat9x9<53>(mat, t104);
    setMat9x9<54>(mat, t293);
    setMat9x9<55>(mat, t215);
    setMat9x9<56>(mat, t75);
    setMat9x9<57>(mat, t217);
    setMat9x9<58>(mat, t103);
    setMat9x9<59>(mat, t26);
    setMat9x9<60>(mat, ((t229 * 2.0f + -t279) + t281) + t102 * (t39 + t40));
    setMat9x9<61>(mat, t213);
    setMat9x9<62>(mat, t311);
    setMat9x9<63>(mat, t303);
    setMat9x9<64>(mat, t216);
    setMat9x9<65>(mat, t227);
    setMat9x9<66>(mat, t27);
    setMat9x9<67>(mat, t218);
    setMat9x9<68>(mat, t225);
    setMat9x9<69>(mat, t213);
    setMat9x9<70>(mat, ((t248 * -2.0f + -t279) + t282) + t102 * (t38 + t40));
    setMat9x9<71>(mat, t36);
    setMat9x9<72>(mat, t304);
    setMat9x9<73>(mat, t214);
    setMat9x9<74>(mat, t294);
    setMat9x9<75>(mat, t28);
    setMat9x9<76>(mat, t226);
    setMat9x9<77>(mat, t104);
    setMat9x9<78>(mat, t311);
    setMat9x9<79>(mat, t36);
    setMat9x9<80>(mat, ((t234 * -2.0f + -t279) + t283) + t102 * (t38 + t39));
}
template <class T>
inline void H_PT(
    T v01, T v02, T v03, T v11, T v12, T v13, T v21, T v22, T v23, T v31, T v32, T v33, Float12x12& H)
{
    T t11;
    T t12;
    T t13;
    T t18;
    T t20;
    T t22;
    T t23;
    T t24;
    T t25;
    T t26;
    T t27;
    T t28;
    T t65;
    T t66;
    T t67;
    T t68;
    T t69;
    T t70;
    T t71;
    T t77;
    T t85;
    T t86;
    T t90;
    T t94;
    T t95;
    T t99;
    T t103;
    T t38;
    T t39;
    T t40;
    T t41;
    T t42;
    T t43;
    T t44;
    T t45;
    T t46;
    T t72;
    T t73;
    T t75;
    T t78;
    T t80;
    T t82;
    T t125;
    T t126;
    T t127;
    T t128;
    T t129;
    T t130;
    T t131;
    T t133;
    T t135;
    T t149;
    T t150;
    T t151;
    T t189;
    T t190;
    T t191;
    T t192;
    T t193;
    T t194;
    T t195;
    T t196;
    T t197;
    T t198;
    T t199;
    T t200;
    T t202;
    T t205;
    T t203;
    T t204;
    T t206;
    T t241;
    T t309;
    T t310;
    T t312;
    T t313;
    T t314;
    T t315;
    T t316;
    T t317;
    T t318;
    T t319;
    T t321;
    T t322;
    T t323;
    T t324;
    T t325;
    T t261;
    T t262;
    T t599;
    T t600;
    T t602;
    T t605;
    T t609;
    T t610;
    T t611;
    T t613;
    T t615;
    T t616;
    T t621;
    T t622;
    T t623;
    T t625;
    T t645;
    T t646_tmp;
    T t646;
    T t601;
    T t603;
    T t604;
    T t606;
    T t607;
    T t608;
    T t612;
    T t614;
    T t617;
    T t618;
    T t619;
    T t620;
    T t624;
    T t626;
    T t627;
    T t628;
    T t629;
    T t630;
    T t631;
    T t632;
    T t633;
    T t634;
    T t635;
    T t636;
    T t637;
    T t638;

    /* H_PT */
    /*     H = H_PT(V01,V02,V03,V11,V12,V13,V21,V22,V23,V31,V32,V33) */
    /*     This function was generated by the Symbolic Math Toolbox version 8.3. */
    /*     10-Jun-2019 17:42:25 */
    t11  = -v11 + v01;
    t12  = -v12 + v02;
    t13  = -v13 + v03;
    t18  = -v21 + v11;
    t20  = -v22 + v12;
    t22  = -v23 + v13;
    t23  = -v31 + v11;
    t24  = -v32 + v12;
    t25  = -v33 + v13;
    t26  = -v31 + v21;
    t27  = -v32 + v22;
    t28  = -v33 + v23;
    t65  = t18 * t24;
    t66  = t20 * t23;
    t67  = t18 * t25;
    t68  = t22 * t23;
    t69  = t20 * t25;
    t70  = t22 * t24;
    t71  = t18 * t23 * 2.0f;
    t77  = t20 * t24 * 2.0f;
    t85  = t22 * t25 * 2.0f;
    t86  = t18 * t26 * 2.0f;
    t90  = t20 * t27 * 2.0f;
    t94  = t22 * t28 * 2.0f;
    t95  = t23 * t26 * 2.0f;
    t99  = t24 * t27 * 2.0f;
    t103 = t25 * t28 * 2.0f;
    t38  = t18 * t18 * 2.0f;
    t39  = t20 * t20 * 2.0f;
    t40  = t22 * t22 * 2.0f;
    t41  = t23 * t23 * 2.0f;
    t42  = t24 * t24 * 2.0f;
    t43  = t25 * t25 * 2.0f;
    t44  = t26 * t26 * 2.0f;
    t45  = t27 * t27 * 2.0f;
    t46  = t28 * t28 * 2.0f;
    t72  = t65 * 2.0f;
    t73  = t66 * 2.0f;
    t75  = t67 * 2.0f;
    t78  = t68 * 2.0f;
    t80  = t69 * 2.0f;
    t82  = t70 * 2.0f;
    t125 = t11 * t20 + -(t12 * t18);
    t126 = t11 * t22 + -(t13 * t18);
    t127 = t12 * t22 + -(t13 * t20);
    t128 = t11 * t24 + -(t12 * t23);
    t129 = t11 * t25 + -(t13 * t23);
    t130 = t12 * t25 + -(t13 * t24);
    t131 = t65 + -t66;
    t133 = t67 + -t68;
    t135 = t69 + -t70;
    t149 = t131 * t131;
    t150 = t133 * t133;
    t151 = t135 * t135;
    t189 = (t11 * t27 + -(t12 * t26)) + t131;
    t190 = (t11 * t28 + -(t13 * t26)) + t133;
    t191 = (t12 * t28 + -(t13 * t27)) + t135;
    t192 = t20 * t131 * 2.0f + t22 * t133 * 2.0f;
    t193 = t18 * t133 * 2.0f + t20 * t135 * 2.0f;
    t194 = t24 * t131 * 2.0f + t25 * t133 * 2.0f;
    t195 = t23 * t133 * 2.0f + t24 * t135 * 2.0f;
    t196 = t27 * t131 * 2.0f + t28 * t133 * 2.0f;
    t197 = t26 * t133 * 2.0f + t27 * t135 * 2.0f;
    t198 = t18 * t131 * 2.0f + -(t22 * t135 * 2.0f);
    t199 = t23 * t131 * 2.0f + -(t25 * t135 * 2.0f);
    t200 = t26 * t131 * 2.0f + -(t28 * t135 * 2.0f);
    t202 = 1.0f / ((t149 + t150) + t151);
    t205 = (t13 * t131 + t11 * t135) + -(t12 * t133);
    t203 = t202 * t202;
    t204 = pow(t202, 3.0f);
    t206 = t205 * t205;
    t241 = t131 * t135 * t202 * 2.0f;
    t309 = t11 * t202 * t205 * 2.0f;
    t310 = t12 * t202 * t205 * 2.0f;
    t13  = t13 * t202 * t205 * 2.0f;
    t312 = (-v21 + v01) * t202 * t205 * 2.0f;
    t313 = (-v22 + v02) * t202 * t205 * 2.0f;
    t314 = (-v23 + v03) * t202 * t205 * 2.0f;
    t315 = (-v31 + v01) * t202 * t205 * 2.0f;
    t316 = t18 * t202 * t205 * 2.0f;
    t317 = (-v32 + v02) * t202 * t205 * 2.0f;
    t318 = t20 * t202 * t205 * 2.0f;
    t319 = (-v33 + v03) * t202 * t205 * 2.0f;
    t11  = t22 * t202 * t205 * 2.0f;
    t321 = t23 * t202 * t205 * 2.0f;
    t322 = t24 * t202 * t205 * 2.0f;
    t323 = t25 * t202 * t205 * 2.0f;
    t324 = t26 * t202 * t205 * 2.0f;
    t325 = t27 * t202 * t205 * 2.0f;
    t12  = t28 * t202 * t205 * 2.0f;
    t261 = -(t131 * t133 * t202 * 2.0f);
    t262 = -(t133 * t135 * t202 * 2.0f);
    t599 =   t130 * t135 * t202 * 2.0f + t135 * t194 * t203 * t205 * 2.0f;
    t600 = -(t125 * t131 * t202 * 2.0f) + t131 * t193 * t203 * t205 * 2.0f;
    t602 =   t129 * t133 * t202 * 2.0f + t133 * t199 * t203 * t205 * 2.0f;
    t605 = -(t131 * t189 * t202 * 2.0f) + t131 * t197 * t203 * t205 * 2.0f;
    t609 = (  t127 * t133 * t202 * 2.0f + -t11) +   t133 * t192 * t203 * t205 * 2.0f;
    t610 = (  t126 * t135 * t202 * 2.0f + t11) +    t135 * t198 * t203 * t205 * 2.0f;
    t611 = (  t130 * t131 * t202 * 2.0f + -t322) +  t131 * t194 * t203 * t205 * 2.0f;
    t613 = (  t126 * t131 * t202 * 2.0f + -t316) +  t131 * t198 * t203 * t205 * 2.0f;
    t615 = (-(t125 * t135 * t202 * 2.0f) + -t318) + t135 * t193 * t203 * t205 * 2.0f;
    t616 = (-(t128 * t133 * t202 * 2.0f) + -t321) + t133 * t195 * t203 * t205 * 2.0f;
    t621 = (   t133 * t191 * t202 * 2.0f + -t12) +   t133 * t196 * t203 * t205 * 2.0f;
    t622 = (   t135 * t190 * t202 * 2.0f + t12) +    t135 * t200 * t203 * t205 * 2.0f;
    t623 = (   t131 * t190 * t202 * 2.0f + -t324) +  t131 * t200 * t203 * t205 * 2.0f;
    t625 = (-( t135 * t189 * t202 * 2.0f) + -t325) + t135 * t197 * t203 * t205 * 2.0f;
    t645 = ((((t127 * t129 * t202 * 2.0f + -t13) + (t72 + -(t66 * 4.0f)) * t203 * t206)
                + t129 * t192 * t203 * t205 * 2.0f)
            + t127 * t199 * t203 * t205 * 2.0f)
            + t192 * t199 * t204 * t206 * 2.0f;
    t646_tmp = t203 * t206;
    t646 = ((((t126 * t130 * t202 * 2.0f + t13) + t646_tmp * (t73 - t65 * 4.0f))
                + t126 * t194 * t203 * t205 * 2.0f)
            + t130 * t198 * t203 * t205 * 2.0f)
            + t194 * t198 * t204 * t206 * 2.0f;
    t601 =  t128 * t131 * t202 * 2.0f + -(t131 * t195 * t203 * t205 * 2.0f);
    t603 = -(t127 * t135 * t202 * 2.0f) + -(t135 * t192 * t203 * t205 * 2.0f);
    t604 = -(t126 * t133 * t202 * 2.0f) + -(t133 * t198 * t203 * t205 * 2.0f);
    t606 = -(t135 * t191 * t202 * 2.0f) + -(t135 * t196 * t203 * t205 * 2.0f);
    t607 = -(t133 * t190 * t202 * 2.0f) + -(t133 * t200 * t203 * t205 * 2.0f);
    t608 = (t125 * t133 * t202 * 2.0f + t316) + -(t133 * t193 * t203 * t205 * 2.0f);
    t612 = (t128 * t135 * t202 * 2.0f + t322) + -(t135 * t195 * t203 * t205 * 2.0f);
    t614 = (-(t127 * t131 * t202 * 2.0f) + t318) + -(t131 * t192 * t203 * t205 * 2.0f);
    t617 = (-(t130 * t133 * t202 * 2.0f) + t323) + -(t133 * t194 * t203 * t205 * 2.0f);
    t618 = (-(t129 * t131 * t202 * 2.0f) + t321) + -(t131 * t199 * t203 * t205 * 2.0f);
    t619 = (-(t129 * t135 * t202 * 2.0f) + -t323) + -(t135 * t199 * t203 * t205 * 2.0f);
    t620 = (t133 * t189 * t202 * 2.0f + t324) + -(t133 * t197 * t203 * t205 * 2.0f);
    t624 = (-(t131 * t191 * t202 * 2.0f) + t325) + -(t131 * t196 * t203 * t205 * 2.0f);
    t626 = (((t125 * t127 * t202 * 2.0f + t18 * t22 * t203 * t206 * 2.0f)
                + t125 * t192 * t203 * t205 * 2.0f)
            + -(t127 * t193 * t203 * t205 * 2.0f))
            + -(t192 * t193 * t204 * t206 * 2.0f);
    t627 = (((t128 * t130 * t202 * 2.0f + t23 * t25 * t203 * t206 * 2.0f)
                + t128 * t194 * t203 * t205 * 2.0f)
            + -(t130 * t195 * t203 * t205 * 2.0f))
            + -(t194 * t195 * t204 * t206 * 2.0f);
    t628 = (((-(t125 * t126 * t202 * 2.0f) + t20 * t22 * t203 * t206 * 2.0f)
                + t126 * t193 * t203 * t205 * 2.0f)
            + -(t125 * t198 * t203 * t205 * 2.0f))
            + t193 * t198 * t204 * t206 * 2.0f;
    t629 = (((-(t128 * t129 * t202 * 2.0f) + t24 * t25 * t203 * t206 * 2.0f)
                + t129 * t195 * t203 * t205 * 2.0f)
            + -(t128 * t199 * t203 * t205 * 2.0f))
            + t195 * t199 * t204 * t206 * 2.0f;
    t630 = (((-(t126 * t127 * t202 * 2.0f) + t18 * t20 * t203 * t206 * 2.0f)
                + -(t126 * t192 * t203 * t205 * 2.0f))
            + -(t127 * t198 * t203 * t205 * 2.0f))
            + -(t192 * t198 * t204 * t206 * 2.0f);
    t631 = (((-(t129 * t130 * t202 * 2.0f) + t23 * t24 * t203 * t206 * 2.0f)
                + -(t129 * t194 * t203 * t205 * 2.0f))
            + -(t130 * t199 * t203 * t205 * 2.0f))
            + -(t194 * t199 * t204 * t206 * 2.0f);
    t632 = (((-(t125 * t128 * t202 * 2.0f) + (t71 + t77) * t203 * t206)
                + t128 * t193 * t203 * t205 * 2.0f)
            + t125 * t195 * t203 * t205 * 2.0f)
            + -(t193 * t195 * t204 * t206 * 2.0f);
    t633 = (((-(t127 * t130 * t202 * 2.0f) + (t77 + t85) * t203 * t206)
                + -(t130 * t192 * t203 * t205 * 2.0f))
            + -(t127 * t194 * t203 * t205 * 2.0f))
            + -(t192 * t194 * t204 * t206 * 2.0f);
    t634 = (((-(t126 * t129 * t202 * 2.0f) + (t71 + t85) * t203 * t206)
                + -(t129 * t198 * t203 * t205 * 2.0f))
            + -(t126 * t199 * t203 * t205 * 2.0f))
            + -(t198 * t199 * t204 * t206 * 2.0f);
    t635 = (((t127 * t191 * t202 * 2.0f + -((t90 + t94) * t203 * t206))
                + t127 * t196 * t203 * t205 * 2.0f)
            + t191 * t192 * t203 * t205 * 2.0f)
            + t192 * t196 * t204 * t206 * 2.0f;
    t636 = (((-(t128 * t189 * t202 * 2.0f) + (t95 + t99) * t203 * t206)
                + t128 * t197 * t203 * t205 * 2.0f)
            + t189 * t195 * t203 * t205 * 2.0f)
            + -(t195 * t197 * t204 * t206 * 2.0f);
    t637 = (((t125 * t189 * t202 * 2.0f + -((t86 + t90) * t203 * t206))
                + -(t125 * t197 * t203 * t205 * 2.0f))
            + -(t189 * t193 * t203 * t205 * 2.0f))
            + t193 * t197 * t204 * t206 * 2.0f;
    t638 = (((-(t130 * t191 * t202 * 2.0f) + (t99 + t103) * t203 * t206)
                + -(t130 * t196 * t203 * t205 * 2.0f))
            + -(t191 * t194 * t203 * t205 * 2.0f))
            + -(t194 * t196 * t204 * t206 * 2.0f);
    t86 = (((t126 * t190 * t202 * 2.0f + -((t86 + t94) * t203 * t206))
            + t126 * t200 * t203 * t205 * 2.0f)
            + t190 * t198 * t203 * t205 * 2.0f)
            + t198 * t200 * t204 * t206 * 2.0f;
    t71 = (((-(t129 * t190 * t202 * 2.0f) + (t95 + t103) * t203 * t206)
            + -(t129 * t200 * t203 * t205 * 2.0f))
            + -(t190 * t199 * t203 * t205 * 2.0f))
            + -(t199 * t200 * t204 * t206 * 2.0f);
    t85 = (((t189 * t191 * t202 * 2.0f + t26 * t28 * t203 * t206 * 2.0f)
            + t189 * t196 * t203 * t205 * 2.0f)
            + -(t191 * t197 * t203 * t205 * 2.0f))
            + -(t196 * t197 * t204 * t206 * 2.0f);
    t90 = (((-(t189 * t190 * t202 * 2.0f) + t27 * t28 * t203 * t206 * 2.0f)
            + t190 * t197 * t203 * t205 * 2.0f)
            + -(t189 * t200 * t203 * t205 * 2.0f))
            + t197 * t200 * t204 * t206 * 2.0f;
    t99 = (((-(t190 * t191 * t202 * 2.0f) + t26 * t27 * t203 * t206 * 2.0f)
            + -(t190 * t196 * t203 * t205 * 2.0f))
            + -(t191 * t200 * t203 * t205 * 2.0f))
            + -(t196 * t200 * t204 * t206 * 2.0f);
    t77 = ((((-(t127 * t128 * t202 * 2.0f) + t310) + (t75 + -(t68 * 4.0f)) * t203 * t206)
            + t127 * t195 * t203 * t205 * 2.0f)
            + -(t128 * t192 * t203 * t205 * 2.0f))
            + t192 * t195 * t204 * t206 * 2.0f;
    t131 = ((((t126 * t128 * t202 * 2.0f + -t309) + (t80 + -(t70 * 4.0f)) * t203 * t206)
                + t128 * t198 * t203 * t205 * 2.0f)
            + -(t126 * t195 * t203 * t205 * 2.0f))
            + -(t195 * t198 * t204 * t206 * 2.0f);
    t133 = ((((-(t125 * t130 * t202 * 2.0f) + -t310) + t646_tmp * (t78 - t67 * 4.0f))
                + t130 * t193 * t203 * t205 * 2.0f)
            + -(t125 * t194 * t203 * t205 * 2.0f))
            + t193 * t194 * t204 * t206 * 2.0f;
    t325 = ((((t125 * t129 * t202 * 2.0f + t309) + t646_tmp * (t82 - t69 * 4.0f))
                + t125 * t199 * t203 * t205 * 2.0f)
            + -(t129 * t193 * t203 * t205 * 2.0f))
            + -(t193 * t199 * t204 * t206 * 2.0f);
    t135 = ((((t125 * t191 * t202 * 2.0f + t313) + ((t75 + t18 * t28 * 2.0f) + -t78) * t203 * t206)
                + t125 * t196 * t203 * t205 * 2.0f)
            + -(t191 * t193 * t203 * t205 * 2.0f))
            + -(t193 * t196 * t204 * t206 * 2.0f);
    t324 = ((((t127 * t189 * t202 * 2.0f + -t313) + ((t78 + t22 * t26 * 2.0f) + -t75) * t203 * t206)
                + -(t127 * t197 * t203 * t205 * 2.0f))
            + t189 * t192 * t203 * t205 * 2.0f)
            + -(t192 * t197 * t204 * t206 * 2.0f);
    t318 = ((((-(t126 * t189 * t202 * 2.0f) + t312)
                + ((t82 + t22 * t27 * 2.0f) + -t80) * t203 * t206)
                + t126 * t197 * t203 * t205 * 2.0f)
            + -(t189 * t198 * t203 * t205 * 2.0f))
            + t197 * t198 * t204 * t206 * 2.0f;
    t321 = ((((-(t130 * t189 * t202 * 2.0f) + t317)
                + -(((t78 + t25 * t26 * 2.0f) + -t75) * t203 * t206))
                + t130 * t197 * t203 * t205 * 2.0f)
            + -(t189 * t194 * t203 * t205 * 2.0f))
            + t194 * t197 * t204 * t206 * 2.0f;
    t323 = ((((t129 * t191 * t202 * 2.0f + t319)
                + -(((t72 + t23 * t27 * 2.0f) + -t73) * t203 * t206))
                + t129 * t196 * t203 * t205 * 2.0f)
            + t191 * t199 * t203 * t205 * 2.0f)
            + t196 * t199 * t204 * t206 * 2.0f;
    t322 = ((((-(t125 * t190 * t202 * 2.0f) + -t312)
                + ((t80 + t20 * t28 * 2.0f) + -t82) * t203 * t206)
                + -(t125 * t200 * t203 * t205 * 2.0f))
            + t190 * t193 * t203 * t205 * 2.0f)
            + t193 * t200 * t204 * t206 * 2.0f;
    t316 = ((((t130 * t190 * t202 * 2.0f + -t319)
                + -(((t73 + t24 * t26 * 2.0f) + -t72) * t203 * t206))
                + t130 * t200 * t203 * t205 * 2.0f)
            + t190 * t194 * t203 * t205 * 2.0f)
            + t194 * t200 * t204 * t206 * 2.0f;
    t65 = ((((-(t128 * t191 * t202 * 2.0f) + -t317)
                + -(((t75 + t23 * t28 * 2.0f) + -t78) * t203 * t206))
            + -(t128 * t196 * t203 * t205 * 2.0f))
            + t191 * t195 * t203 * t205 * 2.0f)
            + t195 * t196 * t204 * t206 * 2.0f;
    t66 = ((((-(t127 * t190 * t202 * 2.0f) + t314) + ((t73 + t20 * t26 * 2.0f) + -t72) * t203 * t206)
            + -(t127 * t200 * t203 * t205 * 2.0f))
            + -(t190 * t192 * t203 * t205 * 2.0f))
            + -(t192 * t200 * t204 * t206 * 2.0f);
    t13 = ((((t128 * t190 * t202 * 2.0f + t315)
                + -(((t80 + t24 * t28 * 2.0f) + -t82) * t203 * t206))
            + t128 * t200 * t203 * t205 * 2.0f)
            + -(t190 * t195 * t203 * t205 * 2.0f))
            + -(t195 * t200 * t204 * t206 * 2.0f);
    t12 = ((((-(t126 * t191 * t202 * 2.0f) + -t314)
                + ((t72 + t18 * t27 * 2.0f) + -t73) * t203 * t206)
            + -(t126 * t196 * t203 * t205 * 2.0f))
            + -(t191 * t198 * t203 * t205 * 2.0f))
            + -(t196 * t198 * t204 * t206 * 2.0f);
    t11 = ((((t129 * t189 * t202 * 2.0f + -t315)
                + -(((t82 + t25 * t27 * 2.0f) + -t80) * t203 * t206))
            + -(t129 * t197 * t203 * t205 * 2.0f))
            + t189 * t199 * t203 * t205 * 2.0f)
            + -(t197 * t199 * t204 * t206 * 2.0f);
    setMat12x12<0>(H, t151 * t202 * 2.0f);
    setMat12x12<1>(H, t262);
    setMat12x12<2>(H, t241);
    setMat12x12<3>(H, t606);
    setMat12x12<4>(H, t622);
    setMat12x12<5>(H, t625);
    setMat12x12<6>(H, t599);
    setMat12x12<7>(H, t619);
    setMat12x12<8>(H, t612);
    setMat12x12<9>(H, t603);
    setMat12x12<10>(H, t610);
    setMat12x12<11>(H, t615);
    setMat12x12<12>(H, t262);
    setMat12x12<13>(H, t150 * t202 * 2.0f);
    setMat12x12<14>(H, t261);
    setMat12x12<15>(H, t621);
    setMat12x12<16>(H, t607);
    setMat12x12<17>(H, t620);
    setMat12x12<18>(H, t617);
    setMat12x12<19>(H, t602);
    setMat12x12<20>(H, t616);
    setMat12x12<21>(H, t609);
    setMat12x12<22>(H, t604);
    setMat12x12<23>(H, t608);
    setMat12x12<24>(H, t241);
    setMat12x12<25>(H, t261);
    setMat12x12<26>(H, t149 * t202 * 2.0f);
    setMat12x12<27>(H, t624);
    setMat12x12<28>(H, t623);
    setMat12x12<29>(H, t605);
    setMat12x12<30>(H, t611);
    setMat12x12<31>(H, t618);
    setMat12x12<32>(H, t601);
    setMat12x12<33>(H, t614);
    setMat12x12<34>(H, t613);
    setMat12x12<35>(H, t600);
    setMat12x12<36>(H, t606);
    setMat12x12<37>(H, t621);
    setMat12x12<38>(H, t624);
    setMat12x12<39>(H, ((t191 * t191 * t202 * 2.0f + t196 * t196 * t204 * t206 * 2.0f) - t646_tmp * (t45 + t46)) + t191 * t196 * t203 * t205 * 4.0f);
    setMat12x12<40>(H, t99);
    setMat12x12<41>(H, t85);
    setMat12x12<42>(H, t638);
    setMat12x12<43>(H, t323);
    setMat12x12<44>(H, t65);
    setMat12x12<45>(H, t635);
    setMat12x12<46>(H, t12);
    setMat12x12<47>(H, t135);
    setMat12x12<48>(H, t622);
    setMat12x12<49>(H, t607);
    setMat12x12<50>(H, t623);
    setMat12x12<51>(H, t99);
    setMat12x12<52>(H, ((t190 * t190 * t202 * 2.0f + t200 * t200 * t204 * t206 * 2.0f) - t646_tmp * (t44 + t46)) + t190 * t200 * t203 * t205 * 4.0f);
    setMat12x12<53>(H, t90);
    setMat12x12<54>(H, t316);
    setMat12x12<55>(H, t71);
    setMat12x12<56>(H, t13);
    setMat12x12<57>(H, t66);
    setMat12x12<58>(H, t86);
    setMat12x12<59>(H, t322);
    setMat12x12<60>(H, t625);
    setMat12x12<61>(H, t620);
    setMat12x12<62>(H, t605);
    setMat12x12<63>(H, t85);
    setMat12x12<64>(H, t90);
    setMat12x12<65>(H, ((t189 * t189 * t202 * 2.0f + t197 * t197 * t204 * t206 * 2.0f) - t646_tmp * (t44 + t45)) - t189 * t197 * t203 * t205 * 4.0f);
    setMat12x12<66>(H, t321);
    setMat12x12<67>(H, t11);
    setMat12x12<68>(H, t636);
    setMat12x12<69>(H, t324);
    setMat12x12<70>(H, t318);
    setMat12x12<71>(H, t637);
    setMat12x12<72>(H, t599);
    setMat12x12<73>(H, t617);
    setMat12x12<74>(H, t611);
    setMat12x12<75>(H, t638);
    setMat12x12<76>(H, t316);
    setMat12x12<77>(H, t321);
    setMat12x12<78>(H, ((t130 * t130 * t202 * 2.0f + t194 * t194 * t204 * t206 * 2.0f) - t646_tmp * (t42 + t43)) + t130 * t194 * t203 * t205 * 4.0f);
    setMat12x12<79>(H, t631);
    setMat12x12<80>(H, t627);
    setMat12x12<81>(H, t633);
    setMat12x12<82>(H, t646);
    setMat12x12<83>(H, t133);
    setMat12x12<84>(H, t619);
    setMat12x12<85>(H, t602);
    setMat12x12<86>(H, t618);
    setMat12x12<87>(H, t323);
    setMat12x12<88>(H, t71);
    setMat12x12<89>(H, t11);
    setMat12x12<90>(H, t631);
    setMat12x12<91>(H, ((t129 * t129 * t202 * 2.0f + t199 * t199 * t204 * t206 * 2.0f) - t646_tmp * (t41 + t43)) + t129 * t199 * t203 * t205 * 4.0f);
    setMat12x12<92>(H, t629);
    setMat12x12<93>(H, t645);
    setMat12x12<94>(H, t634);
    setMat12x12<95>(H, t325);
    setMat12x12<96>(H, t612);
    setMat12x12<97>(H, t616);
    setMat12x12<98>(H, t601);
    setMat12x12<99>(H, t65);
    setMat12x12<100>(H, t13);
    setMat12x12<101>(H, t636);
    setMat12x12<102>(H, t627);
    setMat12x12<103>(H, t629);
    setMat12x12<104>(H, ((t128 * t128 * t202 * 2.0f + t195 * t195 * t204 * t206 * 2.0f) - t646_tmp * (t41 + t42)) - t128 * t195 * t203 * t205 * 4.0f);
    setMat12x12<105>(H, t77);
    setMat12x12<106>(H, t131);
    setMat12x12<107>(H, t632);
    setMat12x12<108>(H, t603);
    setMat12x12<109>(H, t609);
    setMat12x12<110>(H, t614);
    setMat12x12<111>(H, t635);
    setMat12x12<112>(H, t66);
    setMat12x12<113>(H, t324);
    setMat12x12<114>(H, t633);
    setMat12x12<115>(H, t645);
    setMat12x12<116>(H, t77);
    setMat12x12<117>(H, ((t127 * t127 * t202 * 2.0f + t192 * t192 * t204 * t206 * 2.0f) - t646_tmp * (t39 + t40)) + t127 * t192 * t203 * t205 * 4.0f);
    setMat12x12<118>(H, t630);
    setMat12x12<119>(H, t626);
    setMat12x12<120>(H, t610);
    setMat12x12<121>(H, t604);
    setMat12x12<122>(H, t613);
    setMat12x12<123>(H, t12);
    setMat12x12<124>(H, t86);
    setMat12x12<125>(H, t318);
    setMat12x12<126>(H, t646);
    setMat12x12<127>(H, t634);
    setMat12x12<128>(H, t131);
    setMat12x12<129>(H, t630);
    setMat12x12<130>(H, ((t126 * t126 * t202 * 2.0f + t198 * t198 * t204 * t206 * 2.0f) - t646_tmp * (t38 + t40)) + t126 * t198 * t203 * t205 * 4.0f);
    setMat12x12<131>(H, t628);
    setMat12x12<132>(H, t615);
    setMat12x12<133>(H, t608);
    setMat12x12<134>(H, t600);
    setMat12x12<135>(H, t135);
    setMat12x12<136>(H, t322);
    setMat12x12<137>(H, t637);
    setMat12x12<138>(H, t133);
    setMat12x12<139>(H, t325);
    setMat12x12<140>(H, t632);
    setMat12x12<141>(H, t626);
    setMat12x12<142>(H, t628);
    setMat12x12<143>(H, ((t125 * t125 * t202 * 2.0f + t193 * t193 * t204 * t206 * 2.0f) - t646_tmp * (t38 + t39)) - t125 * t193 * t203 * t205 * 4.0f);
}
template <class T>
inline void H_EE(
    T v01, T v02, T v03, T v11, T v12, T v13, T v21, T v22, T v23, T v31, T v32, T v33, Float12x12& H)
{
    T t11;
    T t12;
    T t13;
    T t14;
    T t15;
    T t16;
    T t26;
    T t27;
    T t28;
    T t47;
    T t48;
    T t49;
    T t50;
    T t51;
    T t52;
    T t53;
    T t54;
    T t55;
    T t56;
    T t57;
    T t58;
    T t59;
    T t65;
    T t73;
    T t35;
    T t36;
    T t37;
    T t38;
    T t39;
    T t40;
    T t98;
    T t99;
    T t100;
    T t101;
    T t103;
    T t105;
    T t107;
    T t108;
    T t109;
    T t137;
    T t138;
    T t139;
    T t140;
    T t141;
    T t142;
    T t143;
    T t144;
    T t145;
    T t146;
    T t147;
    T t148;
    T t156;
    T t159;
    T t157;
    T t262;
    T t263;
    T t264;
    T t265;
    T t266;
    T t267;
    T t268;
    T t269;
    T t270;
    T t271;
    T t272;
    T t273;
    T t274;
    T t275;
    T t276;
    T t277;
    T t278;
    T t279;
    T t298;
    T t299;
    T t300;
    T t301;
    T t302;
    T t303;
    T t310;
    T t311;
    T t312;
    T t313;
    T t314;
    T t315;
    T t322;
    T t323;
    T t325;
    T t326;
    T t327;
    T t328;
    T t329;
    T t330;
    T t335;
    T t337;
    T t339;
    T t340;
    T t341;
    T t342;
    T t343;
    T t345;
    T t348;
    T t353;
    T t356;
    T t358;
    T t359;
    T t360;
    T t362;
    T t367;
    T t368;
    T t369;
    T t371;
    T t374;
    T t377;
    T t382;
    T t386;
    T t387;
    T t398;
    T t399;
    T t403;
    T t408;
    T t423;
    T t424;
    T t427;
    T t428;
    T t431;
    T t432;
    T t433;
    T t434;
    T t437;
    T t438;
    T t441;
    T t442;
    T t446;
    T t451;
    T t455;
    T t456;
    T t467;
    T t468;
    T t472;
    T t477;
    T t491;
    T t492;
    T t495;
    T t497;
    T t499;
    T t500;
    T t503;
    T t504;
    T t506;
    T t508;
    T t550;
    T t568;
    T t519_tmp;
    T b_t519_tmp;
    T t519;
    T t520_tmp;
    T b_t520_tmp;
    T t520;
    T t521_tmp;
    T b_t521_tmp;
    T t521;
    T t522_tmp;
    T b_t522_tmp;
    T t522;
    T t523_tmp;
    T b_t523_tmp;
    T t523;
    T t524_tmp;
    T b_t524_tmp;
    T t524;
    T t525;
    T t526;
    T t527;
    T t528;
    T t529;
    T t530;
    T t531;
    T t532;
    T t533;
    T t534;
    T t535;
    T t536;
    T t537;
    T t538;
    T t539;
    T t540;
    T t542;
    T t543;
    T t544;

    /* H_EE */
    /*     H = H_EE(V01,V02,V03,V11,V12,V13,V21,V22,V23,V31,V32,V33) */
    /*     This function was generated by the Symbolic Math Toolbox version 8.3. */
    /*     14-Jun-2019 13:58:38 */
    t11  = -v11 + v01;
    t12  = -v12 + v02;
    t13  = -v13 + v03;
    t14  = -v21 + v01;
    t15  = -v22 + v02;
    t16  = -v23 + v03;
    t26  = -v31 + v21;
    t27  = -v32 + v22;
    t28  = -v33 + v23;
    t47  = t11 * t27;
    t48  = t12 * t26;
    t49  = t11 * t28;
    t50  = t13 * t26;
    t51  = t12 * t28;
    t52  = t13 * t27;
    t53  = t14 * t27;
    t54  = t15 * t26;
    t55  = t14 * t28;
    t56  = t16 * t26;
    t57  = t15 * t28;
    t58  = t16 * t27;
    t59  = t11 * t26 * 2.0f;
    t65  = t12 * t27 * 2.0f;
    t73  = t13 * t28 * 2.0f;
    t35  = t11 * t11 * 2.0f;
    t36  = t12 * t12 * 2.0f;
    t37  = t13 * t13 * 2.0f;
    t38  = t26 * t26 * 2.0f;
    t39  = t27 * t27 * 2.0f;
    t40  = t28 * t28 * 2.0f;
    t98  = t11 * t15 + -(t12 * t14);
    t99  = t11 * t16 + -(t13 * t14);
    t100 = t12 * t16 + -(t13 * t15);
    t101 = t47 + -t48;
    t103 = t49 + -t50;
    t105 = t51 + -t52;
    t107 = t53 + -t54;
    t108 = t55 + -t56;
    t109 = t57 + -t58;
    t137 = t98 + t101;
    t138 = t99 + t103;
    t139 = t100 + t105;
    t140 = (t54 + -t53) + t101;
    t141 = (t56 + -t55) + t103;
    t142 = (t58 + -t57) + t105;
    t143 = t12 * t101 * 2.0f + t13 * t103 * 2.0f;
    t144 = t11 * t103 * 2.0f + t12 * t105 * 2.0f;
    t145 = t27 * t101 * 2.0f + t28 * t103 * 2.0f;
    t146 = t26 * t103 * 2.0f + t27 * t105 * 2.0f;
    t147 = t11 * t101 * 2.0f + -(t13 * t105 * 2.0f);
    t148 = t26 * t101 * 2.0f + -(t28 * t105 * 2.0f);
    t156 = 1.0f / ((t101 * t101 + t103 * t103) + t105 * t105);
    t159 = (t16 * t101 + t14 * t105) + -(t15 * t103);
    t157 = t156 * t156;
    t57  = pow(t156, 3.0f);
    t58  = t159 * t159;
    t262 = t11 * t156 * t159 * 2.0f;
    t263 = t12 * t156 * t159 * 2.0f;
    t264 = t13 * t156 * t159 * 2.0f;
    t265 = t14 * t156 * t159 * 2.0f;
    t266 = t15 * t156 * t159 * 2.0f;
    t267 = t16 * t156 * t159 * 2.0f;
    t268 = (-v31 + v01) * t156 * t159 * 2.0f;
    t269 = (-v21 + v11) * t156 * t159 * 2.0f;
    t270 = (-v32 + v02) * t156 * t159 * 2.0f;
    t271 = (-v22 + v12) * t156 * t159 * 2.0f;
    t272 = (-v33 + v03) * t156 * t159 * 2.0f;
    t273 = (-v23 + v13) * t156 * t159 * 2.0f;
    t274 = (-v31 + v11) * t156 * t159 * 2.0f;
    t275 = (-v32 + v12) * t156 * t159 * 2.0f;
    t276 = (-v33 + v13) * t156 * t159 * 2.0f;
    t277 = t26 * t156 * t159 * 2.0f;
    t278 = t27 * t156 * t159 * 2.0f;
    t279 = t28 * t156 * t159 * 2.0f;
    t298 = t11 * t12 * t157 * t58 * 2.0f;
    t299 = t11 * t13 * t157 * t58 * 2.0f;
    t300 = t12 * t13 * t157 * t58 * 2.0f;
    t301 = t26 * t27 * t157 * t58 * 2.0f;
    t302 = t26 * t28 * t157 * t58 * 2.0f;
    t303 = t27 * t28 * t157 * t58 * 2.0f;
    t310 = (t35 + t36) * t157 * t58;
    t311 = (t35 + t37) * t157 * t58;
    t312 = (t36 + t37) * t157 * t58;
    t313 = (t38 + t39) * t157 * t58;
    t314 = (t38 + t40) * t157 * t58;
    t315 = (t39 + t40) * t157 * t58;
    t322 = (t59 + t65) * t157 * t58;
    t323 = (t59 + t73) * t157 * t58;
    t59  = (t65 + t73) * t157 * t58;
    t325 = (t47 * 2.0f + -(t48 * 4.0f)) * t157 * t58;
    t53  = -t157 * t58;
    t56  = t48 * 2.0f - t47 * 4.0f;
    t326 = t53 * t56;
    t327 = (t49 * 2.0f + -(t50 * 4.0f)) * t157 * t58;
    t55  = t50 * 2.0f - t49 * 4.0f;
    t328 = t53 * t55;
    t329 = (t51 * 2.0f + -(t52 * 4.0f)) * t157 * t58;
    t54  = t52 * 2.0f - t51 * 4.0f;
    t330 = t53 * t54;
    t53  = t157 * t58;
    t335 = t53 * t56;
    t337 = t53 * t55;
    t339 = t53 * t54;
    t340 = t143 * t143 * t57 * t58 * 2.0f;
    t341 = t144 * t144 * t57 * t58 * 2.0f;
    t342 = t145 * t145 * t57 * t58 * 2.0f;
    t343 = t146 * t146 * t57 * t58 * 2.0f;
    t345 = t147 * t147 * t57 * t58 * 2.0f;
    t348 = t148 * t148 * t57 * t58 * 2.0f;
    t36  = t98 * t143 * t157 * t159 * 2.0f;
    t353 = t99 * t143 * t157 * t159 * 2.0f;
    t356 = t99 * t144 * t157 * t159 * 2.0f;
    t65  = t100 * t144 * t157 * t159 * 2.0f;
    t358 = t107 * t143 * t157 * t159 * 2.0f;
    t359 = t98 * t145 * t157 * t159 * 2.0f;
    t360 = t108 * t143 * t157 * t159 * 2.0f;
    t54  = t107 * t144 * t157 * t159 * 2.0f;
    t362 = t99 * t145 * t157 * t159 * 2.0f;
    t53  = t98 * t146 * t157 * t159 * 2.0f;
    t56  = t109 * t143 * t157 * t159 * 2.0f;
    t27  = t108 * t144 * t157 * t159 * 2.0f;
    t55  = t100 * t145 * t157 * t159 * 2.0f;
    t367 = t99 * t146 * t157 * t159 * 2.0f;
    t368 = t109 * t144 * t157 * t159 * 2.0f;
    t369 = t100 * t146 * t157 * t159 * 2.0f;
    t38  = t107 * t145 * t157 * t159 * 2.0f;
    t371 = t108 * t145 * t157 * t159 * 2.0f;
    t374 = t108 * t146 * t157 * t159 * 2.0f;
    t28  = t109 * t146 * t157 * t159 * 2.0f;
    t377 = t98 * t147 * t157 * t159 * 2.0f;
    t382 = t100 * t147 * t157 * t159 * 2.0f;
    t386 = t107 * t147 * t157 * t159 * 2.0f;
    t387 = t98 * t148 * t157 * t159 * 2.0f;
    t103 = t108 * t147 * t157 * t159 * 2.0f;
    t101 = t99 * t148 * t157 * t159 * 2.0f;
    t398 = t109 * t147 * t157 * t159 * 2.0f;
    t399 = t100 * t148 * t157 * t159 * 2.0f;
    t403 = t107 * t148 * t157 * t159 * 2.0f;
    t408 = t109 * t148 * t157 * t159 * 2.0f;
    t73  = t137 * t143 * t157 * t159 * 2.0f;
    t423 = t138 * t143 * t157 * t159 * 2.0f;
    t424 = t138 * t144 * t157 * t159 * 2.0f;
    t37  = t139 * t144 * t157 * t159 * 2.0f;
    t427 = t140 * t143 * t157 * t159 * 2.0f;
    t428 = t137 * t145 * t157 * t159 * 2.0f;
    t16  = t140 * t144 * t157 * t159 * 2.0f;
    t11  = t137 * t146 * t157 * t159 * 2.0f;
    t431 = t141 * t143 * t157 * t159 * 2.0f;
    t432 = t138 * t145 * t157 * t159 * 2.0f;
    t433 = t141 * t144 * t157 * t159 * 2.0f;
    t434 = t138 * t146 * t157 * t159 * 2.0f;
    t105 = t142 * t143 * t157 * t159 * 2.0f;
    t14  = t139 * t145 * t157 * t159 * 2.0f;
    t437 = t142 * t144 * t157 * t159 * 2.0f;
    t438 = t139 * t146 * t157 * t159 * 2.0f;
    t35  = t140 * t145 * t157 * t159 * 2.0f;
    t441 = t141 * t145 * t157 * t159 * 2.0f;
    t442 = t141 * t146 * t157 * t159 * 2.0f;
    t39  = t142 * t146 * t157 * t159 * 2.0f;
    t446 = t137 * t147 * t157 * t159 * 2.0f;
    t451 = t139 * t147 * t157 * t159 * 2.0f;
    t455 = t140 * t147 * t157 * t159 * 2.0f;
    t456 = t137 * t148 * t157 * t159 * 2.0f;
    t13  = t141 * t147 * t157 * t159 * 2.0f;
    t26  = t138 * t148 * t157 * t159 * 2.0f;
    t467 = t142 * t147 * t157 * t159 * 2.0f;
    t468 = t139 * t148 * t157 * t159 * 2.0f;
    t472 = t140 * t148 * t157 * t159 * 2.0f;
    t477 = t142 * t148 * t157 * t159 * 2.0f;
    t47  = t143 * t144 * t57 * t58 * 2.0f;
    t15  = t143 * t145 * t57 * t58 * 2.0f;
    t491 = t143 * t146 * t57 * t58 * 2.0f;
    t492 = t144 * t145 * t57 * t58 * 2.0f;
    t12  = t144 * t146 * t57 * t58 * 2.0f;
    t40  = t145 * t146 * t57 * t58 * 2.0f;
    t495 = t143 * t147 * t57 * t58 * 2.0f;
    t497 = t144 * t147 * t57 * t58 * 2.0f;
    t499 = t143 * t148 * t57 * t58 * 2.0f;
    t500 = t145 * t147 * t57 * t58 * 2.0f;
    t503 = t146 * t147 * t57 * t58 * 2.0f;
    t504 = t144 * t148 * t57 * t58 * 2.0f;
    t506 = t145 * t148 * t57 * t58 * 2.0f;
    t508 = t146 * t148 * t57 * t58 * 2.0f;
    t57  = t147 * t148 * t57 * t58 * 2.0f;
    t550 = ((((t98 * t109 * t156 * 2.0f + -t266) + t337) + t359) + t368) + t492;
    t568 = ((((t108 * t137 * t156 * 2.0f + -t268) + t330) + t27) + t456) + t504;
    t519_tmp   = t139 * t143 * t157 * t159;
    b_t519_tmp = t100 * t143 * t157 * t159;
    t519 = (((-(t100 * t139 * t156 * 2.0f) + t312) + -t340) + b_t519_tmp * 2.0f)
           + t519_tmp * 2.0f;
    t520_tmp   = t140 * t146 * t157 * t159;
    b_t520_tmp = t107 * t146 * t157 * t159;
    t520 = (((t107 * t140 * t156 * 2.0f + t313) + -t343) + b_t520_tmp * 2.0f)
           + -(t520_tmp * 2.0f);
    t521_tmp   = t142 * t145 * t157 * t159;
    b_t521_tmp = t109 * t145 * t157 * t159;
    t521 = (((t109 * t142 * t156 * 2.0f + t315) + -t342) + -(b_t521_tmp * 2.0f))
           + t521_tmp * 2.0f;
    t522_tmp   = t137 * t144 * t157 * t159;
    b_t522_tmp = t98 * t144 * t157 * t159;
    t522 = (((-(t98 * t137 * t156 * 2.0f) + t310) + -t341) + -(b_t522_tmp * 2.0f))
           + -(t522_tmp * 2.0f);
    t523_tmp   = t138 * t147 * t157 * t159;
    b_t523_tmp = t99 * t147 * t157 * t159;
    t523 = (((-(t99 * t138 * t156 * 2.0f) + t311) + -t345) + b_t523_tmp * 2.0f) + t523_tmp * 2.0f;
    t524_tmp   = t141 * t148 * t157 * t159;
    b_t524_tmp = t108 * t148 * t157 * t159;
    t524 = (((t108 * t141 * t156 * 2.0f + t314) + -t348) + -(b_t524_tmp * 2.0f))
           + t524_tmp * 2.0f;
    t525 = (((t98 * t100 * t156 * 2.0f + t299) + t65) + -t36) + -t47;
    t526 = (((t107 * t109 * t156 * 2.0f + t302) + t38) + -t28) + -t40;
    t527 = (((-(t98 * t99 * t156 * 2.0f) + t300) + t377) + -t356) + t497;
    t528 = (((-(t99 * t100 * t156 * 2.0f) + t298) + t353) + t382) + -t495;
    t529 = (((-(t107 * t108 * t156 * 2.0f) + t303) + t374) + -t403) + t508;
    t530 = (((-(t108 * t109 * t156 * 2.0f) + t301) + -t371) + -t408) + -t506;
    t531 = (((t98 * t107 * t156 * 2.0f + t322) + t54) + -t53) + -t12;
    t532 = (((t100 * t109 * t156 * 2.0f + t59) + t55) + -t56) + -t15;
    t533 = (((t99 * t108 * t156 * 2.0f + t323) + t101) + -t103) + -t57;
    t534 = (((t98 * t140 * t156 * 2.0f + -t322) + t53) + t16) + t12;
    t535 = (((-(t107 * t137 * t156 * 2.0f) + -t322) + -t54) + t11) + t12;
    t536 = (((t100 * t142 * t156 * 2.0f + -t59) + -t55) + -t105) + t15;
    t537 = (((-(t109 * t139 * t156 * 2.0f) + -t59) + t56) + -t14) + t15;
    t538 = (((t99 * t141 * t156 * 2.0f + -t323) + -t101) + -t13) + t57;
    t539 = (((-(t108 * t138 * t156 * 2.0f) + -t323) + t103) + -t26) + t57;
    t540 = (((t137 * t139 * t156 * 2.0f + t299) + t37) + -t73) + -t47;
    t148 = (((t140 * t142 * t156 * 2.0f + t302) + t39) + -t35) + -t40;
    t542 = (((-(t137 * t138 * t156 * 2.0f) + t300) + t446) + -t424) + t497;
    t543 = (((-(t138 * t139 * t156 * 2.0f) + t298) + t423) + t451) + -t495;
    t544 = (((-(t140 * t141 * t156 * 2.0f) + t303) + t472) + -t442) + t508;
    t53  = (((-(t141 * t142 * t156 * 2.0f) + t301) + t441) + t477) + -t506;
    t157 = (((-(t139 * t142 * t156 * 2.0f) + t59) + t105) + t14) + -t15;
    t159 = (((-(t137 * t140 * t156 * 2.0f) + t322) + -t16) + -t11) + -t12;
    t147 = (((-(t138 * t141 * t156 * 2.0f) + t323) + t13) + t26) + -t57;
    t146 = ((((t100 * t107 * t156 * 2.0f + t266) + t327) + -t358) + -t369) + t491;
    t145 = ((((-(t99 * t107 * t156 * 2.0f) + -t265) + t329) + t367) + t386) + -t503;
    t144 = ((((-(t100 * t108 * t156 * 2.0f) + -t267) + t325) + t360) + -t399) + t499;
    t143 = ((((-(t99 * t109 * t156 * 2.0f) + t267) + t335) + -t362) + t398) + t500;
    t52 = ((((-(t98 * t108 * t156 * 2.0f) + t265) + t339) + -t27) + -t387) + -t504;
    t51 = ((((t109 * t140 * t156 * 2.0f + -t278) + -t302) + t28) + t35) + t40;
    t50 = ((((-(t98 * t139 * t156 * 2.0f) + t263) + -t299) + t36) + -t37) + t47;
    t49 = ((((t107 * t142 * t156 * 2.0f + t278) + -t302) + -t38) + -t39) + t40;
    t48 = ((((-(t100 * t137 * t156 * 2.0f) + -t263) + -t299) + -t65) + t73) + t47;
    t47 = ((((t99 * t137 * t156 * 2.0f + t262) + -t300) + t356) + -t446) + -t497;
    t73 = ((((t100 * t138 * t156 * 2.0f + t264) + -t298) + -t382) + -t423) + t495;
    t65 = ((((-(t109 * t141 * t156 * 2.0f) + t279) + -t301) + t408) + -t441) + t506;
    t59 = ((((t98 * t138 * t156 * 2.0f + -t262) + -t300) + -t377) + t424) + -t497;
    t40 = ((((t99 * t139 * t156 * 2.0f + -t264) + -t298) + -t353) + -t451) + t495;
    t39 = ((((-(t107 * t141 * t156 * 2.0f) + -t277) + -t303) + t403) + t442) + -t508;
    t38 = ((((-(t108 * t142 * t156 * 2.0f) + -t279) + -t301) + t371) + -t477) + t506;
    t37 = ((((-(t108 * t140 * t156 * 2.0f) + t277) + -t303) + -t374) + -t472) + -t508;
    t36 = ((((t98 * t142 * t156 * 2.0f + t271) + t328) + -t359) + t437) + -t492;
    t35 = ((((-(t109 * t137 * t156 * 2.0f) + t270) + t328) + -t368) + -t428) + -t492;
    t28 = ((((t100 * t140 * t156 * 2.0f + -t271) + -t327) + t369) + -t427) + -t491;
    t27 = ((((-(t98 * t141 * t156 * 2.0f) + -t269) + t330) + t387) + -t433) + t504;
    t26 = ((((t109 * t138 * t156 * 2.0f + -t272) + t326) + -t398) + t432) + -t500;
    t13 = ((((-(t107 * t139 * t156 * 2.0f) + -t270) + -t327) + t358) + t438) + -t491;
    t12 = ((((-(t99 * t142 * t156 * 2.0f) + -t273) + t326) + t362) + t467) + -t500;
    t11 = ((((-(t99 * t140 * t156 * 2.0f) + t269) + -t329) + -t367) + t455) + t503;
    t16 = ((((t107 * t138 * t156 * 2.0f + t268) + -t329) + -t386) + -t434) + t503;
    t15 = ((((-(t100 * t141 * t156 * 2.0f) + t273) + -t325) + t399) + t431) + -t499;
    t14 = ((((t108 * t139 * t156 * 2.0f + t272) + -t325) + -t360) + t468) + -t499;
    t105 = ((((-(t139 * t140 * t156 * 2.0f) + t275) + t327) + t427) + -t438) + t491;
    t103 = ((((t138 * t140 * t156 * 2.0f + -t274) + t329) + t434) + -t455) + -t503;
    t101 = ((((-(t137 * t142 * t156 * 2.0f) + -t275) + t337) + t428) + -t437) + t492;
    t58 = ((((t139 * t141 * t156 * 2.0f + -t276) + t325) + -t431) + -t468) + t499;
    t57 = ((((t137 * t141 * t156 * 2.0f + t274) + t339) + t433) + -t456) + -t504;
    t56 = ((((t138 * t142 * t156 * 2.0f + t276) + t335) + -t432) + -t467) + t500;
    t55 = -t315 + t342;
    setMat12x12<0>(H,  (t55 + t142 * t142 * t156 * 2.0f) - t521_tmp * 4.0f);
    setMat12x12<1>(H,  t53);
    setMat12x12<2>(H,  t148);
    setMat12x12<3>(H,  t521);
    setMat12x12<4>(H,  t38);
    setMat12x12<5>(H,  t49);
    setMat12x12<6>(H,  t157);
    setMat12x12<7>(H,  t56);
    setMat12x12<8>(H,  t101);
    setMat12x12<9>(H,  t536);
    setMat12x12<10>(H, t12);
    setMat12x12<11>(H, t36);
    setMat12x12<12>(H, t53);
    t54    = -t314 + t348;
    setMat12x12<13>(H, (t54 + t141 * t141 * t156 * 2.0f) - t524_tmp * 4.0f);
    setMat12x12<14>(H, t544);
    setMat12x12<15>(H, t65);
    setMat12x12<16>(H, t524);
    setMat12x12<17>(H, t39);
    setMat12x12<18>(H, t58);
    setMat12x12<19>(H, t147);
    setMat12x12<20>(H, t57);
    setMat12x12<21>(H, t15);
    setMat12x12<22>(H, t538);
    setMat12x12<23>(H, t27);
    setMat12x12<24>(H, t148);
    setMat12x12<25>(H, t544);
    t53    = -t313 + t343;
    setMat12x12<26>(H, (t53 + t140 * t140 * t156 * 2.0f) + t520_tmp * 4.0f);
    setMat12x12<27>(H, t51);
    setMat12x12<28>(H, t37);
    setMat12x12<29>(H, t520);
    setMat12x12<30>(H, t105);
    setMat12x12<31>(H, t103);
    setMat12x12<32>(H, t159);
    setMat12x12<33>(H, t28);
    setMat12x12<34>(H, t11);
    setMat12x12<35>(H, t534);
    setMat12x12<36>(H, t521);
    setMat12x12<37>(H, t65);
    setMat12x12<38>(H, t51);
    setMat12x12<39>(H, (t55 + t109 * t109 * t156 * 2.0f) + b_t521_tmp * 4.0f);
    setMat12x12<40>(H, t530);
    setMat12x12<41>(H, t526);
    setMat12x12<42>(H, t537);
    setMat12x12<43>(H, t26);
    setMat12x12<44>(H, t35);
    setMat12x12<45>(H, t532);
    setMat12x12<46>(H, t143);
    setMat12x12<47>(H, t550);
    setMat12x12<48>(H, t38);
    setMat12x12<49>(H, t524);
    setMat12x12<50>(H, t37);
    setMat12x12<51>(H, t530);
    setMat12x12<52>(H, (t54 + t108 * t108 * t156 * 2.0f) + b_t524_tmp * 4.0f);
    setMat12x12<53>(H, t529);
    setMat12x12<54>(H, t14);
    setMat12x12<55>(H, t539);
    setMat12x12<56>(H, t568);
    setMat12x12<57>(H, t144);
    setMat12x12<58>(H, t533);
    setMat12x12<59>(H, t52);
    setMat12x12<60>(H, t49);
    setMat12x12<61>(H, t39);
    setMat12x12<62>(H, t520);
    setMat12x12<63>(H, t526);
    setMat12x12<64>(H, t529);
    setMat12x12<65>(H, (t53 + t107 * t107 * t156 * 2.0f) - b_t520_tmp * 4.0f);
    setMat12x12<66>(H, t13);
    setMat12x12<67>(H, t16);
    setMat12x12<68>(H, t535);
    setMat12x12<69>(H, t146);
    setMat12x12<70>(H, t145);
    setMat12x12<71>(H, t531);
    setMat12x12<72>(H, t157);
    setMat12x12<73>(H, t58);
    setMat12x12<74>(H, t105);
    setMat12x12<75>(H, t537);
    setMat12x12<76>(H, t14);
    setMat12x12<77>(H, t13);
    t55    = -t312 + t340;
    setMat12x12<78>(H, (t55 + t139 * t139 * t156 * 2.0f) - t519_tmp * 4.0f);
    setMat12x12<79>(H, t543);
    setMat12x12<80>(H, t540);
    setMat12x12<81>(H, t519);
    setMat12x12<82>(H, t40);
    setMat12x12<83>(H, t50);
    setMat12x12<84>(H, t56);
    setMat12x12<85>(H, t147);
    setMat12x12<86>(H, t103);
    setMat12x12<87>(H, t26);
    setMat12x12<88>(H, t539);
    setMat12x12<89>(H, t16);
    setMat12x12<90>(H, t543);
    t54    = -t311 + t345;
    setMat12x12<91>(H, (t54 + t138 * t138 * t156 * 2.0f) - t523_tmp * 4.0f);
    setMat12x12<92>(H, t542);
    setMat12x12<93>(H, t73);
    setMat12x12<94>(H, t523);
    setMat12x12<95>(H, t59);
    setMat12x12<96>(H, t101);
    setMat12x12<97>(H, t57);
    setMat12x12<98>(H, t159);
    setMat12x12<99>(H, t35);
    setMat12x12<100>(H, t568);
    setMat12x12<101>(H, t535);
    setMat12x12<102>(H, t540);
    setMat12x12<103>(H, t542);
    t53    = -t310 + t341;
    setMat12x12<104>(H, (t53 + t137 * t137 * t156 * 2.0f) + t522_tmp * 4.0f);
    setMat12x12<105>(H, t48);
    setMat12x12<106>(H, t47);
    setMat12x12<107>(H, t522);
    setMat12x12<108>(H, t536);
    setMat12x12<109>(H, t15);
    setMat12x12<110>(H, t28);
    setMat12x12<111>(H, t532);
    setMat12x12<112>(H, t144);
    setMat12x12<113>(H, t146);
    setMat12x12<114>(H, t519);
    setMat12x12<115>(H, t73);
    setMat12x12<116>(H, t48);
    setMat12x12<117>(H, (t55 + t100 * t100 * t156 * 2.0f) - b_t519_tmp * 4.0f);
    setMat12x12<118>(H, t528);
    setMat12x12<119>(H, t525);
    setMat12x12<120>(H, t12);
    setMat12x12<121>(H, t538);
    setMat12x12<122>(H, t11);
    setMat12x12<123>(H, t143);
    setMat12x12<124>(H, t533);
    setMat12x12<125>(H, t145);
    setMat12x12<126>(H, t40);
    setMat12x12<127>(H, t523);
    setMat12x12<128>(H, t47);
    setMat12x12<129>(H, t528);
    setMat12x12<130>(H, (t54 + t99 * t99 * t156 * 2.0f) - b_t523_tmp * 4.0f);
    setMat12x12<131>(H, t527);
    setMat12x12<132>(H, t36);
    setMat12x12<133>(H, t27);
    setMat12x12<134>(H, t534);
    setMat12x12<135>(H, t550);
    setMat12x12<136>(H, t52);
    setMat12x12<137>(H, t531);
    setMat12x12<138>(H, t50);
    setMat12x12<139>(H, t59);
    setMat12x12<140>(H, t522);
    setMat12x12<141>(H, t525);
    setMat12x12<142>(H, t527);
    setMat12x12<143>(H, (t53 + t98 * t98 * t156 * 2.0f) + b_t522_tmp * 4.0f);
}

// TODO: Symmetric matrix

// Mollified EE energy
template <typename T>
inline void EEM(T input, T eps_x, T& e)
{
    T input_div_eps_x = input / eps_x;
    e                 = (-input_div_eps_x + 2.0f) * input_div_eps_x;
}
template <typename T>
inline void g_EEM(T input, T eps_x, T& g)
{
    T one_div_eps_x = 1.0f / eps_x;
    g               = 2.0f * one_div_eps_x * (-one_div_eps_x * input + 1.0f);
}
template <typename T>
inline void H_EEM(T input, T eps_x, T& H)
{
    H = -2.0f / (eps_x * eps_x);
}
inline void edge_edge_cross_norm2(const Float3& ea0,
                                  const Float3& ea1,
                                  const Float3& eb0,
                                  const Float3& eb1, Float& result)
{
    result = length_squared_vec(cross_vec((ea1 - ea0), (eb1 - eb0)));
}

template <class T>
inline void g_EECN2(
    T v01, T v02, T v03, T v11, T v12, T v13, T v21, T v22, T v23, T v31, T v32, T v33, Float12& g)
{
    T t8;
    T t9;
    T t10;
    T t11;
    T t12;
    T t13;
    T t23;
    T t24;
    T t25;
    T t26;
    T t27;
    T t28;
    T t29;
    T t30;
    T t31;
    T t32;
    T t33;

    /* COMPUTEEECROSSSQNORMGRADIENT */
    /*     G = COMPUTEEECROSSSQNORMGRADIENT(V01,V02,V03,V11,V12,V13,V21,V22,V23,V31,V32,V33) */
    /*     This function was generated by the Symbolic Math Toolbox version 8.3. */
    /*     01-Nov-2019 16:54:23 */
    t8    = -v11 + v01;
    t9    = -v12 + v02;
    t10   = -v13 + v03;
    t11   = -v31 + v21;
    t12   = -v32 + v22;
    t13   = -v33 + v23;
    t23   = t8 * t12 + -(t9 * t11);
    t24   = t8 * t13 + -(t10 * t11);
    t25   = t9 * t13 + -(t10 * t12);
    t26   = t8 * t23 * 2.0f;
    t27   = t9 * t23 * 2.0f;
    t28   = t8 * t24 * 2.0f;
    t29   = t10 * t24 * 2.0f;
    t30   = t9  * t25 * 2.0f;
    t31   = t10 * t25 * 2.0f;
    t32   = t11 * t23 * 2.0f;
    t33   = t12 * t23 * 2.0f;
    t23   = t11 * t24 * 2.0f;
    t10   = t13 * t24 * 2.0f;
    t9    = t12 * t25 * 2.0f;
    t8    = t13 * t25 * 2.0f;
    setVec<0>(g, t33 + t10);
    setVec<1>(g, -t32 + t8);
    setVec<2>(g, -t23 - t9);
    setVec<3>(g, -t33 - t10);
    setVec<4>(g, t32 - t8);
    setVec<5>(g, t23 + t9);
    setVec<6>(g, -t27 - t29);
    setVec<7>(g, t26 - t31);
    setVec<8>(g, t28 + t30);
    setVec<9>(g, t27 + t29);
    setVec<10>(g, -t26 + t31);
    setVec<11>(g, -t28 - t30);
}
template <class T>
inline void H_EECN2(
    T v01, T v02, T v03, T v11, T v12, T v13, T v21, T v22, T v23, T v31, T v32, T v33, Float12x12& H)
{
    T t8;
    T t9;
    T t10;
    T t11;
    T t12;
    T t13;
    T t32;
    T t33;
    T t34;
    T t35;
    T t48;
    T t36;
    T t49;
    T t37;
    T t38;
    T t39;
    T t40;
    T t41;
    T t42;
    T t43;
    T t44;
    T t45;
    T t46;
    T t47;
    T t50;
    T t51;
    T t52;
    T t20;
    T t23;
    T t24;
    T t25;
    T t86;
    T t87;
    T t88;
    T t74;
    T t75;
    T t76;
    T t77;
    T t78;
    T t79;
    T t89;
    T t90;
    T t91;
    T t92;
    T t93;
    T t94;
    T t95;

    /* COMPUTEEECROSSSQNORMHESSIAN */
    /*     H = COMPUTEEECROSSSQNORMHESSIAN(V01,V02,V03,V11,V12,V13,V21,V22,V23,V31,V32,V33) */
    /*     This function was generated by the Symbolic Math Toolbox version 8.3. */
    /*     01-Nov-2019 16:54:23 */
    t8     = -v11 + v01;
    t9     = -v12 + v02;
    t10    = -v13 + v03;
    t11    = -v31 + v21;
    t12    = -v32 + v22;
    t13    = -v33 + v23;
    t32    = t8 * t9 * 2.0f;
    t33    = t8 * t10 * 2.0f;
    t34    = t9 * t10 * 2.0f;
    t35    = t8 * t11 * 2.0f;
    t48    = t8 * t12;
    t36    = t48 * 2.0f;
    t49    = t9 * t11;
    t37    = t49 * 2.0f;
    t38    = t48 * 4.0f;
    t48    = t8 * t13;
    t39    = t48 * 2.0f;
    t40    = t49 * 4.0f;
    t41    = t9 * t12 * 2.0f;
    t49    = t10 * t11;
    t42    = t49 * 2.0f;
    t43    = t48 * 4.0f;
    t48    = t9 * t13;
    t44    = t48 * 2.0f;
    t45    = t49 * 4.0f;
    t49    = t10 * t12;
    t46    = t49 * 2.0f;
    t47    = t48 * 4.0f;
    t48    = t49 * 4.0f;
    t49    = t10 * t13 * 2.0f;
    t50    = t11 * t12 * 2.0f;
    t51    = t11 * t13 * 2.0f;
    t52    = t12 * t13 * 2.0f;
    t20    = t8 * t8 * 2.0f;
    t9     = t9 * t9 * 2.0f;
    t8     = t10 * t10 * 2.0f;
    t23    = t11 * t11 * 2.0f;
    t24    = t12 * t12 * 2.0f;
    t25    = t13 * t13 * 2.0f;
    t86    = t35 + t41;
    t87    = t35 + t49;
    t88    = t41 + t49;
    t74    = t20 + t9;
    t75    = t20 + t8;
    t76    = t9 + t8;
    t77    = t23 + t24;
    t78    = t23 + t25;
    t79    = t24 + t25;
    t89    = t40 + -t36;
    t90    = t36 + -t40;
    t91    = t37 + -t38;
    t92    = t38 + -t37;
    t93    = t45 + -t39;
    t94    = t39 + -t45;
    t95    = t42 + -t43;
    t37    = t43 + -t42;
    t39    = t48 + -t44;
    t45    = t44 + -t48;
    t38    = t46 + -t47;
    t40    = t47 + -t46;
    t36    = -t35 + -t41;
    t13    = -t35 + -t49;
    t11    = -t41 + -t49;
    t12    = -t20 + -t9;
    t10    = -t20 + -t8;
    t8     = -t9 + -t8;
    t9     = -t23 + -t24;
    t49    = -t23 + -t25;
    t48    = -t24 + -t25;
    setMat12x12<0>(H, t79);
    setMat12x12<1>(H, -t50);
    setMat12x12<2>(H, -t51);
    setMat12x12<3>(H, t48);
    setMat12x12<4>(H, t50);
    setMat12x12<5>(H, t51);
    setMat12x12<6>(H, t11);
    setMat12x12<7>(H, t92);
    setMat12x12<8>(H, t37);
    setMat12x12<9>(H, t88);
    setMat12x12<10>(H, t91);
    setMat12x12<11>(H, t95);
    setMat12x12<12>(H, -t50);
    setMat12x12<13>(H, t78);
    setMat12x12<14>(H, -t52);
    setMat12x12<15>(H, t50);
    setMat12x12<16>(H, t49);
    setMat12x12<17>(H, t52);
    setMat12x12<18>(H, t89);
    setMat12x12<19>(H, t13);
    setMat12x12<20>(H, t40);
    setMat12x12<21>(H, t90);
    setMat12x12<22>(H, t87);
    setMat12x12<23>(H, t38);
    setMat12x12<24>(H, -t51);
    setMat12x12<25>(H, -t52);
    setMat12x12<26>(H, t77);
    setMat12x12<27>(H, t51);
    setMat12x12<28>(H, t52);
    setMat12x12<29>(H, t9);
    setMat12x12<30>(H, t93);
    setMat12x12<31>(H, t39);
    setMat12x12<32>(H, t36);
    setMat12x12<33>(H, t94);
    setMat12x12<34>(H, t45);
    setMat12x12<35>(H, t86);
    setMat12x12<36>(H, t48);
    setMat12x12<37>(H, t50);
    setMat12x12<38>(H, t51);
    setMat12x12<39>(H, t79);
    setMat12x12<40>(H, -t50);
    setMat12x12<41>(H, -t51);
    setMat12x12<42>(H, t88);
    setMat12x12<43>(H, t91);
    setMat12x12<44>(H, t95);
    setMat12x12<45>(H, t11);
    setMat12x12<46>(H, t92);
    setMat12x12<47>(H, t37);
    setMat12x12<48>(H, t50);
    setMat12x12<49>(H, t49);
    setMat12x12<50>(H, t52);
    setMat12x12<51>(H, -t50);
    setMat12x12<52>(H, t78);
    setMat12x12<53>(H, -t52);
    setMat12x12<54>(H, t90);
    setMat12x12<55>(H, t87);
    setMat12x12<56>(H, t38);
    setMat12x12<57>(H, t89);
    setMat12x12<58>(H, t13);
    setMat12x12<59>(H, t40);
    setMat12x12<60>(H, t51);
    setMat12x12<61>(H, t52);
    setMat12x12<62>(H, t9);
    setMat12x12<63>(H, -t51);
    setMat12x12<64>(H, -t52);
    setMat12x12<65>(H, t77);
    setMat12x12<66>(H, t94);
    setMat12x12<67>(H, t45);
    setMat12x12<68>(H, t86);
    setMat12x12<69>(H, t93);
    setMat12x12<70>(H, t39);
    setMat12x12<71>(H, t36);
    setMat12x12<72>(H, t11);
    setMat12x12<73>(H, t89);
    setMat12x12<74>(H, t93);
    setMat12x12<75>(H, t88);
    setMat12x12<76>(H, t90);
    setMat12x12<77>(H, t94);
    setMat12x12<78>(H, t76);
    setMat12x12<79>(H, -t32);
    setMat12x12<80>(H, -t33);
    setMat12x12<81>(H, t8);
    setMat12x12<82>(H, t32);
    setMat12x12<83>(H, t33);
    setMat12x12<84>(H, t92);
    setMat12x12<85>(H, t13);
    setMat12x12<86>(H, t39);
    setMat12x12<87>(H, t91);
    setMat12x12<88>(H, t87);
    setMat12x12<89>(H, t45);
    setMat12x12<90>(H, -t32);
    setMat12x12<91>(H, t75);
    setMat12x12<92>(H, -t34);
    setMat12x12<93>(H, t32);
    setMat12x12<94>(H, t10);
    setMat12x12<95>(H, t34);
    setMat12x12<96>(H, t37);
    setMat12x12<97>(H, t40);
    setMat12x12<98>(H, t36);
    setMat12x12<99>(H, t95);
    setMat12x12<100>(H,t38);
    setMat12x12<101>(H,t86);
    setMat12x12<102>(H,-t33);
    setMat12x12<103>(H,-t34);
    setMat12x12<104>(H,t74);
    setMat12x12<105>(H,t33);
    setMat12x12<106>(H,t34);
    setMat12x12<107>(H,t12);
    setMat12x12<108>(H,t88);
    setMat12x12<109>(H,t90);
    setMat12x12<110>(H,t94);
    setMat12x12<111>(H,t11);
    setMat12x12<112>(H,t89);
    setMat12x12<113>(H,t93);
    setMat12x12<114>(H,t8);
    setMat12x12<115>(H,t32);
    setMat12x12<116>(H,t33);
    setMat12x12<117>(H,t76);
    setMat12x12<118>(H,-t32);
    setMat12x12<119>(H,-t33);
    setMat12x12<120>(H,t91);
    setMat12x12<121>(H,t87);
    setMat12x12<122>(H,t45);
    setMat12x12<123>(H,t92);
    setMat12x12<124>(H,t13);
    setMat12x12<125>(H,t39);
    setMat12x12<126>(H,t32);
    setMat12x12<127>(H,t10);
    setMat12x12<128>(H,t34);
    setMat12x12<129>(H,-t32);
    setMat12x12<130>(H,t75);
    setMat12x12<131>(H,-t34);
    setMat12x12<132>(H,t95);
    setMat12x12<133>(H,t38);
    setMat12x12<134>(H,t86);
    setMat12x12<135>(H,t37);
    setMat12x12<136>(H,t40);
    setMat12x12<137>(H,t36);
    setMat12x12<138>(H,t33);
    setMat12x12<139>(H,t34);
    setMat12x12<140>(H,t12);
    setMat12x12<141>(H,-t33);
    setMat12x12<142>(H,-t34);
    setMat12x12<143>(H,t74);
}
inline void edge_edge_cross_norm2_gradient(const Float3& ea0,
                                           const Float3& ea1,
                                           const Float3& eb0,
                                           const Float3& eb1,
                                           Float12& grad)
{
    details::g_EECN2(ea0[0],
                     ea0[1],
                     ea0[2],
                     ea1[0],
                     ea1[1],
                     ea1[2],
                     eb0[0],
                     eb0[1],
                     eb0[2],
                     eb1[0],
                     eb1[1],
                     eb1[2],
                     grad);
}
inline void edge_edge_cross_norm2_hessian(const Float3& ea0,
                                          const Float3& ea1,
                                          const Float3& eb0,
                                          const Float3& eb1,
                                          Float12x12& Hessian)
{
    details::H_EECN2(ea0[0],
                     ea0[1],
                     ea0[2],
                     ea1[0],
                     ea1[1],
                     ea1[2],
                     eb0[0],
                     eb0[1],
                     eb0[2],
                     eb1[0],
                     eb1[1],
                     eb1[2],
                     Hessian);
}

} // namespace details



inline void point_point_distance2_gradient(Float3& a, Float3& b, Float6& G)
{
    G.vec[0] = 2.0f * (a - b);
    G.vec[1] = -G.vec[0];
}
inline void point_edge_distance2_gradient(Float3& p, Float3& e0, Float3& e1, Float9& G)
{
    details::g_PE3D(p[0], p[1], p[2], e0[0], e0[1], e0[2], e1[0], e1[1], e1[2], G);
}
inline void point_triangle_distance2_gradient(Float3& p, Float3& t0, Float3& t1, Float3& t2, Float12& G)
{
    details::g_PT(
            p[0], p[1], p[2], t0[0], t0[1], t0[2], t1[0], t1[1], t1[2], t2[0], t2[1], t2[2], G);
}
inline void edge_edge_distance2_gradient(Float3& ea0, Float3& ea1, Float3& eb0, Float3& eb1, Float12& G)
{
    details::g_EE(ea0[0],
                  ea0[1],
                  ea0[2],
                  ea1[0],
                  ea1[1],
                  ea1[2],
                  eb0[0],
                  eb0[1],
                  eb0[2],
                  eb1[0],
                  eb1[1],
                  eb1[2],
                  G);
}

inline void point_point_distance2_hessian(Float3& a, Float3& b, Float6x6 H)
{
    H.mat[0][0] = makeFloat3x3(Float(1.0f));
    H.mat[1][1] = makeFloat3x3(Float(1.0f));
    H.mat[0][1] = makeFloat3x3(Float(-1.0f));
    H.mat[1][0] = makeFloat3x3(Float(-1.0f));
}
inline void point_edge_distance2_hessian(Float3& p, Float3& e0, Float3& e1, Float9x9& H)
{
    details::H_PE3D(p[0], p[1], p[2], e0[0], e0[1], e0[2], e1[0], e1[1], e1[2], H);
}
inline void point_triangle_distance2_hessian(Float3& p, Float3& t0, Float3& t1, Float3& t2, Float12x12& H)
{
    details::H_PT(
            p[0], p[1], p[2], t0[0], t0[1], t0[2], t1[0], t1[1], t1[2], t2[0], t2[1], t2[2], H);
}
inline void edge_edge_distance2_hessian(Float3& ea0, Float3& ea1, Float3& eb0, Float3& eb1, Float12x12& H)
{
    details::H_EE(ea0[0],
                  ea0[1],
                  ea0[2],
                  ea1[0],
                  ea1[1],
                  ea1[2],
                  eb0[0],
                  eb0[1],
                  eb0[2],
                  eb1[0],
                  eb1[1],
                  eb1[2],
                  H);
}
inline void edge_edge_mollifier_hessian(const Float3& ea0,
                                       const Float3& ea1,
                                       const Float3& eb0,
                                       const Float3& eb1,
                                       Float         eps_x,
                                       Float12x12& H)
{
    Float EECrossSqNorm;
    details::edge_edge_cross_norm2(ea0, ea1, eb0, eb1, EECrossSqNorm);
    $if (EECrossSqNorm < eps_x)
    {
        Float q_g, q_H;
        details::g_EEM(EECrossSqNorm, eps_x, q_g);
        details::H_EEM(EECrossSqNorm, eps_x, q_H);

        Var<LargeVector<12>> g;
        details::edge_edge_cross_norm2_gradient(ea0, ea1, eb0, eb1, g);
        details::edge_edge_cross_norm2_hessian(ea0, ea1, eb0, eb1, H);

        // H *= q_g;
        // H += (q_H * g) * g.transpose();
        lcsv::mult_largemat(H, H, q_g);
        H = lcsv::add_largemat(H, lcsv::outerProduct(lcsv::mult_largevec(g, q_g), g));

    // #pragma unroll
    //     for (uint i = 0; i < 4; i++)
    //     {
    //     #pragma unroll
    //         for (uint j = 0; j < 4; j++)
    //         {
    //             H[i][j] = q_g * H[i][j] + q_H * outer_product(g[i], g[j]); // ???? might need transpose ???
    //         }
    //     }
    }
    $else
    {
        // H.setZero();
        lcsv::set_largemat_zero(H);
    };
}

} // namespace DistanceGradient

void NarrowPhasesDetector::compile(luisa::compute::Device& device)
{
    using namespace luisa::compute;

    fn_reset_toi = device.compile<1>([](Var<BufferView<float>> sa_toi)
    {
        sa_toi->write(dispatch_x(), accd::line_search_max_t);
    });

    const uint offset_vv = collision_data->get_vv_count_offset();
    const uint offset_ve = collision_data->get_ve_count_offset();
    const uint offset_vf = collision_data->get_vf_count_offset();
    const uint offset_ee = collision_data->get_ee_count_offset();
    
    fn_narrow_phase_vf_ccd_query = device.compile<1>(
    [
        broadphase_count = collision_data->broad_phase_collision_count.view(offset_vf, 1),
        broadphase_list = collision_data->broad_phase_list_vf.view()
    ](
        Var<BufferView<float>> sa_toi,
        Var<BufferView<float3>> sa_x_begin_left, 
        Var<BufferView<float3>> sa_x_begin_right, 
        Var<BufferView<float3>> sa_x_end_left,
        Var<BufferView<float3>> sa_x_end_right,
        Var<BufferView<uint3>> sa_faces_right,
        Float thickness
    )
    {
        const Uint pair_idx = dispatch_x();
        const Uint vid =  broadphase_list->read(2 * pair_idx + 0);
        const Uint fid = broadphase_list->read(2 * pair_idx + 1);

        const Uint3 face = sa_faces_right.read(fid);

        Float toi = accd::line_search_max_t;
        $if (
            vid == face[0] | 
            vid == face[1] | 
            vid == face[2]) 
        {
            toi = accd::line_search_max_t;
        }
        $else
        {
            Float3 t0_p =  sa_x_begin_left->read(vid);
            Float3 t1_p =  sa_x_end_left->read(vid);
            Float3 t0_f0 = sa_x_begin_right->read(face[0]);
            Float3 t0_f1 = sa_x_begin_right->read(face[1]);
            Float3 t0_f2 = sa_x_begin_right->read(face[2]);
            Float3 t1_f0 = sa_x_end_right->read(face[0]);
            Float3 t1_f1 = sa_x_end_right->read(face[1]);
            Float3 t1_f2 = sa_x_end_right->read(face[2]);
    
            toi = accd::point_triangle_ccd(
                t0_p,  
                t1_p,              
                t0_f0, 
                t0_f1,                    
                t0_f2, 
                t1_f0,                 
                t1_f1, 
                t1_f2,              
                thickness);
            
            // $if (toi != accd::line_search_max_t)
            // {
            //     device_log("VF Pair {} : toi = {}, vid {} & fid {} (face {})", 
            //         pair_idx, toi, vid, fid, face
            //     );
            // };
        };

        toi = ParallelIntrinsic::block_intrinsic_reduce(pair_idx, toi, ParallelIntrinsic::warp_reduce_op_min<float>);

        $if (pair_idx % 256 == 0)
        {
            sa_toi.atomic(0).fetch_min(toi);
        };
    });

    fn_narrow_phase_ee_ccd_query = device.compile<1>(
    [
        broadphase_count = collision_data->broad_phase_collision_count.view(offset_ee, 1),
        broadphase_list = collision_data->broad_phase_list_ee.view()
    ](
        Var<BufferView<float>> sa_toi,
        Var<BufferView<float3>> sa_x_begin_a, 
        Var<BufferView<float3>> sa_x_begin_b, 
        Var<BufferView<float3>> sa_x_end_a,
        Var<BufferView<float3>> sa_x_end_b,
        Var<BufferView<uint2>> sa_edges_left,
        Var<BufferView<uint2>> sa_edges_right,
        Float thickness
    )
    {
        const Uint pair_idx = dispatch_x();
        const Uint left =  broadphase_list->read(2 * pair_idx + 0);
        const Uint right = broadphase_list->read(2 * pair_idx + 1);
        const Uint2 left_edge  = sa_edges_left.read(left);
        const Uint2 right_edge = sa_edges_right.read(right);

        Float toi = accd::line_search_max_t;
        $if (
            left_edge[0] == right_edge[0] |
            left_edge[0] == right_edge[1] |
            left_edge[1] == right_edge[0] |
            left_edge[1] == right_edge[1]) 
        {
            toi = accd::line_search_max_t;
        }
        $else
        {
            Float3 ea_t0_p0 = (sa_x_begin_a->read(left_edge[0]));
            Float3 ea_t0_p1 = (sa_x_begin_a->read(left_edge[1]));
            Float3 eb_t0_p0 = (sa_x_begin_b->read(right_edge[0]));
            Float3 eb_t0_p1 = (sa_x_begin_b->read(right_edge[1]));
            Float3 ea_t1_p0 = (sa_x_end_a->read(left_edge[0]));
            Float3 ea_t1_p1 = (sa_x_end_a->read(left_edge[1]));
            Float3 eb_t1_p0 = (sa_x_end_b->read(right_edge[0]));
            Float3 eb_t1_p1 = (sa_x_end_b->read(right_edge[1]));
    
            toi = accd::edge_edge_ccd(
                ea_t0_p0, 
                ea_t0_p1, 
                eb_t0_p0, 
                eb_t0_p1, 
                ea_t1_p0, 
                ea_t1_p1, 
                eb_t1_p0, 
                eb_t1_p1, 
                thickness);
        };
        
        // $if (toi != host_accd::line_search_max_t) 
        // {
        //     device_log("EE Pair {} : toi = {}, edge1 {} ({}) & edge2 {} ({})", pair_idx, toi, left, left_edge, right, right_edge);
        // };

        toi = ParallelIntrinsic::block_intrinsic_reduce(pair_idx, toi, ParallelIntrinsic::warp_reduce_op_min<float>);

        $if (pair_idx % 256 == 0)
        {
            sa_toi.atomic(0).fetch_min(toi);
        };
    });


    auto fn_narrow_phase_vf_dcd_query = device.compile<1>(
    [
        broadphase_count = collision_data->broad_phase_collision_count.view(offset_vf, 1),
        broadphase_list = collision_data->broad_phase_list_vf.view(),
        narrowphase_count_vv = collision_data->narrow_phase_collision_count.view(offset_vv, 1),
        narrowphase_count_ve = collision_data->narrow_phase_collision_count.view(offset_ve, 1),
        narrowphase_count_vf = collision_data->narrow_phase_collision_count.view(offset_vf, 1),
        narrowphase_list_vv = collision_data->narrow_phase_list_vv.view(),
        narrowphase_list_ve = collision_data->narrow_phase_list_ve.view(),
        narrowphase_list_vf = collision_data->narrow_phase_list_vf.view()
    ](
        Var<BufferView<float3>> sa_x_left, 
        Var<BufferView<float3>> sa_x_right,
        Var<BufferView<uint3>> sa_faces_right,
        Float thickness
    )
    {
        const Uint pair_idx = dispatch_x();
        const Uint vid =  broadphase_list->read(2 * pair_idx + 0);
        const Uint fid = broadphase_list->read(2 * pair_idx + 1);
        const Uint3 face = sa_faces_right.read(fid);

        $if (
            vid == face[0] | 
            vid == face[1] | 
            vid == face[2]) 
        {

        }
        $else
        {
            Float3 p =  sa_x_left->read(vid);
            Float3 t0 = sa_x_right->read(face[0]);
            Float3 t1 = sa_x_right->read(face[1]);
            Float3 t2 = sa_x_right->read(face[2]);

            Float3 bary = distance::point_triangle_distance_coeff(p, t0, t1, t2);
            Uint3 valid_face = face;
            Uint valid_count = distance::point_triangle_type(bary, face, valid_face);
            
            Float3 x = bary[0] * (t0 - p) +
                       bary[1] * (t1 - p) +
                       bary[2] * (t2 - p);
            Float d2 = length_squared_vec(x);
            $if (d2 < thickness * thickness)
            {
                $if (valid_count == 3)
                {
                    Uint idx = narrowphase_count_vf->atomic(0).fetch_add(1u);
                    // narrowphase_list_vf->write(idx, makeUint4(vid, valid_face[0], valid_face[1], valid_face[2]));
                }
                $elif (valid_count == 2)
                {
                    Uint idx = narrowphase_count_ve->atomic(0).fetch_add(1u);
                    // narrowphase_list_ve->write(idx, makeUint3(vid, valid_face[0], valid_face[1]));
                }
                $else // valid_count == 1
                {
                    Uint idx = narrowphase_count_vv->atomic(0).fetch_add(1u);
                    // narrowphase_list_ve->write(idx, makeUint2(vid, valid_face[0]));
                };
            };
        };
    });

    auto fn_narrow_phase_ee_dcd_query = device.compile<1>(
    [
        broadphase_count = collision_data->broad_phase_collision_count.view(offset_ee, 1),
        broadphase_list = collision_data->broad_phase_list_ee.view(),
        narrowphase_count_ee = collision_data->narrow_phase_collision_count.view(offset_ee, 1),
        narrowphase_list_ee = collision_data->narrow_phase_list_ee.view()
    ](
        Var<BufferView<float3>> sa_x_a, 
        Var<BufferView<float3>> sa_x_b,
        Var<BufferView<uint2>> sa_edges_left,
        Var<BufferView<uint2>> sa_edges_right,
        Float d_hat,
        Float thickness
    )
    {
        const Uint pair_idx = dispatch_x();
        const Uint left =  broadphase_list->read(2 * pair_idx + 0);
        const Uint right = broadphase_list->read(2 * pair_idx + 1);
        const Uint2 left_edge  = sa_edges_left.read(left);
        const Uint2 right_edge = sa_edges_right.read(right);
        $if (
            left_edge[0] == right_edge[0] |
            left_edge[0] == right_edge[1] |
            left_edge[1] == right_edge[0] |
            left_edge[1] == right_edge[1]) 
        {
        }
        $else
        {
            Float3 ea_p0 = (sa_x_a->read(left_edge[0]));
            Float3 ea_p1 = (sa_x_a->read(left_edge[1]));
            Float3 eb_p0 = (sa_x_b->read(right_edge[0]));
            Float3 eb_p1 = (sa_x_b->read(right_edge[1]));
    
            Float4 bary = distance::edge_edge_distance_coeff(ea_p0, ea_p1, eb_p0, eb_p1);
            Bool is_ee = all_vec(bary != 0.0f);
            // Uint2 valid_edge1 = left_edge;
            // Uint2 valid_edge2 = right_edge;
            // Uint2 valid_counts = distance::edge_edge_type(bary, left_edge, right_edge, valid_edge1, valid_edge2);

            Float3 x0 = bary[0] * ea_p0 + bary[1] * ea_p1;
            Float3 x1 = bary[2] * eb_p0 + bary[3] * eb_p1;
            Float d2 = length_squared_vec(x1 - x0);

            $if (d2 < thickness * thickness & is_ee)
            {
                Uint idx = narrowphase_count_ee->atomic(0).fetch_add(1u);
                // narrowphase_list_ee->write(idx, makeUint4(left_edge[0], left_edge[1], right_edge[0], right_edge[1]));
            };
            // Corner case (VV, VE) will only be considered in VF detection
        };
    });

    // const float d_hat = 1e-3;
    
    auto fn_narrow_phase_vf_dcd_for_barrier_energy = device.compile<1>(
    [
        contact_energy = collision_data->contact_energy.view(),
        broadphase_count = collision_data->broad_phase_collision_count.view(offset_vf, 1),
        broadphase_list = collision_data->broad_phase_list_vf.view(),
        narrowphase_count_vv = collision_data->narrow_phase_collision_count.view(offset_vv, 1),
        narrowphase_count_ve = collision_data->narrow_phase_collision_count.view(offset_ve, 1),
        narrowphase_count_vf = collision_data->narrow_phase_collision_count.view(offset_vf, 1),
        narrowphase_list_vv = collision_data->narrow_phase_list_vv.view(),
        narrowphase_list_ve = collision_data->narrow_phase_list_ve.view(),
        narrowphase_list_vf = collision_data->narrow_phase_list_vf.view()
    ]( 
        Var<BufferView<float3>> sa_x_left, 
        Var<BufferView<float3>> sa_x_right,
        Var<BufferView<uint3>> sa_faces_right,
        Float d_hat,
        Float thickness
    )
    {
        const Uint pair_idx = dispatch_x();
        const Uint vid =  broadphase_list->read(2 * pair_idx + 0);
        const Uint fid = broadphase_list->read(2 * pair_idx + 1);
        const Uint3 face = sa_faces_right.read(fid);

        Float energy = 0.0f;
        $if (
            vid == face[0] | 
            vid == face[1] | 
            vid == face[2]) 
        {

        }
        $else
        {
            Float3 p =  sa_x_left->read(vid);
            Float3 t0 = sa_x_right->read(face[0]);
            Float3 t1 = sa_x_right->read(face[1]);
            Float3 t2 = sa_x_right->read(face[2]);

            Float3 bary = distance::point_triangle_distance_coeff(p, t0, t1, t2);
            Uint3 valid_face = face;
            Uint valid_count = distance::point_triangle_type(bary, face, valid_face);
            
            Float3 x = bary[0] * (t0 - p) +
                       bary[1] * (t1 - p) +
                       bary[2] * (t2 - p);
            Float d2 = length_squared_vec(x);
            $if (d2 < square_scalar(thickness + d_hat))
            {
                cipc::NoKappa_Barrier(energy, sqrt_scalar(d2), d_hat, thickness);
            };
        };
        
        energy = ParallelIntrinsic::block_intrinsic_reduce(pair_idx, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);

        $if (pair_idx % 256 == 0)
        {
            // $if (energy != 0.0f) 
            {
                contact_energy->atomic(0).fetch_add(energy);
            };
        };
    });

    auto fn_narrow_phase_ee_dcd_for_barrier_energy = device.compile<1>(
    [
        contact_energy = collision_data->contact_energy.view(),
        broadphase_count = collision_data->broad_phase_collision_count.view(offset_ee, 1),
        broadphase_list = collision_data->broad_phase_list_ee.view(),
        narrowphase_count_ee = collision_data->narrow_phase_collision_count.view(offset_ee, 1),
        narrowphase_list_ee = collision_data->narrow_phase_list_ee.view()
    ](
        Var<BufferView<float3>> sa_x_a, 
        Var<BufferView<float3>> sa_x_b,
        Var<BufferView<uint2>> sa_edges_left,
        Var<BufferView<uint2>> sa_edges_right,
        Float d_hat,
        Float thickness
    )
    {
        const Uint pair_idx = dispatch_x();
        const Uint left =  broadphase_list->read(2 * pair_idx + 0);
        const Uint right = broadphase_list->read(2 * pair_idx + 1);
        const Uint2 left_edge  = sa_edges_left.read(left);
        const Uint2 right_edge = sa_edges_right.read(right);

        Float energy = 0.0f;
        $if (
            left_edge[0] == right_edge[0] |
            left_edge[0] == right_edge[1] |
            left_edge[1] == right_edge[0] |
            left_edge[1] == right_edge[1]) 
        {
        }
        $else
        {
            Float3 ea_p0 = (sa_x_a->read(left_edge[0]));
            Float3 ea_p1 = (sa_x_a->read(left_edge[1]));
            Float3 eb_p0 = (sa_x_b->read(right_edge[0]));
            Float3 eb_p1 = (sa_x_b->read(right_edge[1]));
    
            Float4 bary = distance::edge_edge_distance_coeff(ea_p0, ea_p1, eb_p0, eb_p1);
            Bool is_ee = all_vec(bary != 0.0f);
            Float3 x0 = bary[0] * ea_p0 + bary[1] * ea_p1;
            Float3 x1 = bary[2] * eb_p0 + bary[3] * eb_p1;
            Float d2 = length_squared_vec(x1 - x0);

            $if (d2 < square_scalar(thickness + d_hat))
            {
                cipc::NoKappa_Barrier(energy, sqrt_scalar(d2), d_hat, thickness);
            };
        };

        energy = ParallelIntrinsic::block_intrinsic_reduce(pair_idx, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
        
        $if (pair_idx % 256 == 0)
        {
            // $if (energy != 0.0f) 
            {
                contact_energy->atomic(0).fetch_add(energy);
            };
        };
    });
   

}

void NarrowPhasesDetector::reset_toi(Stream& stream)
{
    auto& sa_toi = collision_data->toi_per_vert;
    stream << fn_reset_toi(sa_toi).dispatch(sa_toi.size());
}
void NarrowPhasesDetector::host_reset_toi(Stream& stream)
{
    auto& sa_toi = host_collision_data->toi_per_vert;
    CpuParallel::parallel_set(sa_toi, 0.0f);
}
void NarrowPhasesDetector::download_collision_count(Stream& stream)
{
    auto broadphase_count = collision_data->broad_phase_collision_count.view();
    auto& host_count = host_collision_data->broad_phase_collision_count;

    stream 
        << broadphase_count.copy_to(host_count.data()) 
        << luisa::compute::synchronize();

    const uint num_vf_broadphase = host_count[collision_data->get_vf_count_offset()];
    const uint num_ee_broadphase = host_count[collision_data->get_ee_count_offset()];

    // luisa::log_info("num_vf_broadphase = {}", num_vf_broadphase); // TODO: Indirect Dispatch
    // luisa::log_info("num_ee_broadphase = {}", num_ee_broadphase); // TODO: Indirect Dispatch
}

float NarrowPhasesDetector::get_global_toi(Stream& stream)
{
    stream << luisa::compute::synchronize();

    auto& host_toi = host_collision_data->toi_per_vert[0];
    if (host_toi != host_accd::line_search_max_t) luisa::log_info("             CCD linesearch toi = {}", host_toi);
    host_toi /= host_accd::line_search_max_t;
    if (host_toi < 1e-5)
    {
        luisa::log_error("  small toi : {}", host_toi);
    }
    return host_toi;
    // 
}

// Device CCD
void NarrowPhasesDetector::narrow_phase_ccd_query_from_vf_pair(Stream& stream, 
    const Buffer<float3>& sa_x_begin_left, 
    const Buffer<float3>& sa_x_begin_right, 
    const Buffer<float3>& sa_x_end_left,
    const Buffer<float3>& sa_x_end_right,
    const Buffer<uint3>& sa_faces_right,
    const float thickness)
{
    auto& sa_toi = collision_data->toi_per_vert;
    auto broadphase_count = collision_data->broad_phase_collision_count.view();
    auto& host_toi = host_collision_data->toi_per_vert;
    auto& host_count = host_collision_data->broad_phase_collision_count;

    const uint num_vf_broadphase = host_count[collision_data->get_vf_count_offset()];
    const uint num_ee_broadphase = host_count[collision_data->get_ee_count_offset()];

    // std::vector<float3> host_x_begin(sa_x_begin_left.size());
    // std::vector<float3> host_x_end(sa_x_end_left.size());
    // std::vector<uint3> host_faces(sa_faces_right.size());
    // stream 
    //         << sa_x_begin_left.copy_to(host_x_begin.data())
    //         << sa_x_end_left.copy_to(host_x_end.data())
    //         << sa_faces_right.copy_to(host_faces.data())
    //         << luisa::compute::synchronize();

    // host_narrow_phase_ccd_query_from_vf_pair(stream, 
    //         host_x_begin, 
    //         host_x_begin, 
    //         host_x_end, 
    //         host_x_end, 
    //         host_faces, 
    //         1e-3);

    stream << fn_narrow_phase_vf_ccd_query(
        sa_toi,
        sa_x_begin_left,
        sa_x_begin_right, // sa_x_begin_right
        sa_x_end_left,
        sa_x_end_right, // sa_x_end_right
        sa_faces_right, thickness
    ).dispatch(num_vf_broadphase) 
        << sa_toi.view(0, 1).copy_to(host_toi.data())
    ;

    
}

void NarrowPhasesDetector::narrow_phase_ccd_query_from_ee_pair(Stream& stream, 
    const Buffer<float3>& sa_x_begin_a, 
    const Buffer<float3>& sa_x_begin_b, 
    const Buffer<float3>& sa_x_end_a,
    const Buffer<float3>& sa_x_end_b,
    const Buffer<uint2>& sa_edges_left,
    const Buffer<uint2>& sa_edges_right,
    const float thickness)
{
    auto broadphase_count = collision_data->broad_phase_collision_count.view();
    auto& sa_toi = collision_data->toi_per_vert;
    auto& host_count = host_collision_data->broad_phase_collision_count;
    auto& host_toi = host_collision_data->toi_per_vert;

    const uint num_vf_broadphase = host_count[collision_data->get_vf_count_offset()]; 
    const uint num_ee_broadphase = host_count[collision_data->get_ee_count_offset()];

    // luisa::log_info("curr toi = {} from VF", host_toi[0]);

    // std::vector<float3> host_x_begin(sa_x_begin_a.size());
    // std::vector<float3> host_x_end(sa_x_end_a.size());
    // std::vector<uint2> host_edges(sa_edges_left.size());
    // stream 
    //         << sa_x_begin_a.copy_to(host_x_begin.data())
    //         << sa_x_end_a.copy_to(host_x_end.data())
    //         << sa_edges_left.copy_to(host_edges.data())
    //         << luisa::compute::synchronize();

    // host_narrow_phase_ccd_query_from_ee_pair(stream, 
    //         host_x_begin, 
    //         host_x_begin, 
    //         host_x_end, 
    //         host_x_end, 
    //         host_edges, 
    //         host_edges, 
    //         1e-3);

    stream << fn_narrow_phase_ee_ccd_query(sa_toi,
        sa_x_begin_a,
        sa_x_begin_b,
        sa_x_end_a,
        sa_x_end_b,
        sa_edges_left,
        sa_edges_left, thickness
    ).dispatch(num_ee_broadphase) 
        << sa_toi.view(0, 1).copy_to(host_toi.data())
    ;
}

// Host CCD
void NarrowPhasesDetector::host_narrow_phase_ccd_query_from_vf_pair(Stream& stream, 
    const std::vector<float3>& sa_x_begin_left, 
    const std::vector<float3>& sa_x_begin_right, 
    const std::vector<float3>& sa_x_end_left,
    const std::vector<float3>& sa_x_end_right,
    const std::vector<uint3>& sa_faces_right,
    const float thickness)
{
    auto& sa_toi = host_collision_data->toi_per_vert;
    auto& host_count = host_collision_data->broad_phase_collision_count;
    auto& host_list = host_collision_data->broad_phase_list_vf;

    const uint num_vf_broadphase = host_count[collision_data->get_vf_count_offset()]; 
    const uint num_ee_broadphase = host_count[collision_data->get_ee_count_offset()];
    stream 
        << collision_data->broad_phase_list_vf.view(0, num_vf_broadphase * 2).copy_to(host_list.data()) 
        << luisa::compute::synchronize();

    // luisa::log_info("num_vf_broadphase = {}", num_vf_broadphase);
    // luisa::log_info("num_ee_broadphase = {}", num_ee_broadphase);

    uint2* pair_view = (lcsv::uint2*)host_list.data();
    // CpuParallel::parallel_sort(pair_view, pair_view + num_vf_broadphase, [](const uint2& left, const uint2& right)
    // {
    //     if (left[0] == right[0]) { return left[1] < right[1]; }
    //     return left[0] < right[0];
    // });

    float min_toi = host_accd::line_search_max_t;
    min_toi = CpuParallel::parallel_for_and_reduce(0, num_vf_broadphase, [&](const uint pair_idx)
    {
        const auto pair = pair_view[pair_idx];
        const uint left = pair[0];
        const uint right = pair[1];
        const uint3 right_face = sa_faces_right[right];

        if (left == right_face[0] || left == right_face[1] || left == right_face[2]) return host_accd::line_search_max_t;
        
        EigenFloat3 t0_p =  float3_to_eigen3(sa_x_begin_left[left]);
        EigenFloat3 t1_p =  float3_to_eigen3(sa_x_end_left[left]);
        EigenFloat3 t0_f0 = float3_to_eigen3(sa_x_begin_right[right_face[0]]);
        EigenFloat3 t0_f1 = float3_to_eigen3(sa_x_begin_right[right_face[1]]);
        EigenFloat3 t0_f2 = float3_to_eigen3(sa_x_begin_right[right_face[2]]);
        EigenFloat3 t1_f0 = float3_to_eigen3(sa_x_end_right[right_face[0]]);
        EigenFloat3 t1_f1 = float3_to_eigen3(sa_x_end_right[right_face[1]]);
        EigenFloat3 t1_f2 = float3_to_eigen3(sa_x_end_right[right_face[2]]);

        float toi = host_accd::point_triangle_ccd(t0_p,  t1_p,
                                      t0_f0, t0_f1,
                                      t0_f2, t1_f0,
                                      t1_f1, t1_f2,
                                      thickness);

        if (toi != host_accd::line_search_max_t) 
        {
            // luisa::log_info("BroadPhase Pair {} : toi = {}, vid {} & fid {} (face {})", 
            //     pair_idx, toi, left, right, right_face,
            // );
            luisa::log_info("VF Pair {} : toi = {}, vid {} & fid {} (face {}), dist = {} -> {}", 
                pair_idx, toi, left, right, right_face, 
                host_distance::point_triangle_distance_squared_unclassified(t0_p, t0_f0, t0_f1, t0_f2),
                host_distance::point_triangle_distance_squared_unclassified(t1_p, t1_f0, t1_f1, t1_f2)
            );
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_begin_left[left]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_end_left[left]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_begin_right[right_face[0]]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_begin_right[right_face[1]]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_begin_right[right_face[2]]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_end_right[right_face[0]]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_end_right[right_face[1]]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_end_right[right_face[2]]);
        }
        return toi;
    }, [](const float left, const float right) { return min_scalar(left, right); }, host_accd::line_search_max_t);

    sa_toi[0] = min_scalar(min_toi, sa_toi[0]);

    // min_toi /= host_accd::line_search_max_t;
    // if (min_toi < 1e-5)
    // {
    //     luisa::log_error("toi is too small : {}", min_toi);
    // }
    // luisa::log_info("toi = {}", min_toi);
    // sa_toi[0] = min_toi;

}

void NarrowPhasesDetector::host_narrow_phase_ccd_query_from_ee_pair(Stream& stream, 
    const std::vector<float3>& sa_x_begin_a, 
    const std::vector<float3>& sa_x_begin_b, 
    const std::vector<float3>& sa_x_end_a,
    const std::vector<float3>& sa_x_end_b,
    const std::vector<uint2>& sa_edges_left,
    const std::vector<uint2>& sa_edges_right,
    const float thickness)
{
    auto& sa_toi = host_collision_data->toi_per_vert;
    auto& host_count = host_collision_data->broad_phase_collision_count;
    auto& host_list = host_collision_data->broad_phase_list_ee;

    const uint num_vf_broadphase = host_count[collision_data->get_vf_count_offset()]; 
    const uint num_ee_broadphase = host_count[collision_data->get_ee_count_offset()];

    stream 
        << collision_data->broad_phase_list_ee.view(0, num_ee_broadphase * 2).copy_to(host_list.data()) 
        << luisa::compute::synchronize();

    // luisa::log_info("num_ee_broadphase = {}", num_ee_broadphase);

    uint2* pair_view = (lcsv::uint2*)host_list.data();
    // CpuParallel::parallel_sort(pair_view, pair_view + num_ee_broadphase, [](const uint2& left, const uint2& right)
    // {
    //     if (left[0] == right[0]) { return left[1] < right[1]; }
    //     return left[0] < right[0];
    // });

    float min_toi = 1.25f;
    min_toi = CpuParallel::parallel_for_and_reduce(0, num_ee_broadphase, [&](const uint pair_idx)
    {
        const auto pair = pair_view[pair_idx];
        const uint left = pair[0];
        const uint right = pair[1];
        const uint2 left_edge  = sa_edges_left[left];
        const uint2 right_edge = sa_edges_right[right];

        if (
            left_edge[0] == right_edge[0] || 
            left_edge[0] == right_edge[1] || 
            left_edge[1] == right_edge[0] || 
            left_edge[1] == right_edge[1]) return host_accd::line_search_max_t;
        
        EigenFloat3 ea_t0_p0 = float3_to_eigen3(sa_x_begin_a[left_edge[0]]);
        EigenFloat3 ea_t0_p1 = float3_to_eigen3(sa_x_begin_a[left_edge[1]]);
        EigenFloat3 eb_t0_p0 = float3_to_eigen3(sa_x_begin_b[right_edge[0]]);
        EigenFloat3 eb_t0_p1 = float3_to_eigen3(sa_x_begin_b[right_edge[1]]);
        EigenFloat3 ea_t1_p0 = float3_to_eigen3(sa_x_end_a[left_edge[0]]);
        EigenFloat3 ea_t1_p1 = float3_to_eigen3(sa_x_end_a[left_edge[1]]);
        EigenFloat3 eb_t1_p0 = float3_to_eigen3(sa_x_end_b[right_edge[0]]);
        EigenFloat3 eb_t1_p1 = float3_to_eigen3(sa_x_end_b[right_edge[1]]);

        float toi = host_accd::edge_edge_ccd(
            ea_t0_p0, 
            ea_t0_p1, 
            eb_t0_p0, 
            eb_t0_p1, 
            ea_t1_p0, 
            ea_t1_p1, 
            eb_t1_p0, 
            eb_t1_p1, 
            thickness);

        if (toi != host_accd::line_search_max_t) 
        {
            luisa::log_info("EE Pair {} : toi = {}, edge1 {} ({}) & edge2 {} ({})", pair_idx, toi, left, left_edge, right, right_edge);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_begin_a[left_edge[0]]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_begin_a[left_edge[1]]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_begin_b[right_edge[0]]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_begin_b[right_edge[1]]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_end_a[left_edge[0]]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_end_a[left_edge[1]]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_end_b[right_edge[0]]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_end_b[right_edge[1]]);
        }
        return toi;
    }, [](const float left, const float right) { return min_scalar(left, right); }, host_accd::line_search_max_t);

    sa_toi[0] = min_scalar(min_toi, sa_toi[0]);

    // min_toi /= host_accd::line_search_max_t;
    // if (min_toi < 1e-5)
    // {
    //     luisa::log_error("toi is too small : {}", min_toi);
    // }
    // luisa::log_info("toi = {}", min_toi);
    // sa_toi[0] = min_scalar(min_toi, sa_toi[0]);
}

// Device DCD
void NarrowPhasesDetector::narrow_phase_dcd_query_from_vf_pair(Stream& stream, 
    const Buffer<float3>& sa_x_left, 
    const Buffer<float3>& sa_x_right, 
    const Buffer<uint3>& sa_faces_right,
    const float thickness)
{

}

void NarrowPhasesDetector::narrow_phase_dcd_query_from_ee_pair(Stream& stream, 
    const Buffer<float3>& sa_x_left, 
    const Buffer<float3>& sa_x_right, 
    const Buffer<uint2>& sa_edges_left,
    const Buffer<uint2>& sa_edges_right,
    const float thickness)
{

}









void NarrowPhasesDetector::unit_test(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    using namespace luisa::compute;

    // VF CCD Test
    {
        const float desire_toi = 0.6930697;
        luisa::log_info("VF Test, desire for toi {}", desire_toi);

        const uint vid = 1;
        const uint fid = 2;
        const uint3 face = uint3(4, 7, 5);
        float3 case_t0_p  = makeFloat3(0.48159984, -0.26639974, -0.48159984);
        float3 case_t1_p  = makeFloat3(0.47421163, -0.3129394, -0.47421163);
        float3 case_t0_f0 = makeFloat3(-0.4, -0.3, -0.5);
        float3 case_t0_f1 = makeFloat3(0.6, -0.3, 0.5);
        float3 case_t0_f2 = makeFloat3(0.6, -0.3, -0.5);
        float3 case_t1_f0 = makeFloat3(-0.4, -0.3, -0.5);
        float3 case_t1_f1 = makeFloat3(0.6, -0.3, 0.5);
        float3 case_t1_f2 = makeFloat3(0.6, -0.3, -0.5);

        {
            const auto t0_p  = float3_to_eigen3(case_t0_p ); 
            const auto t1_p  = float3_to_eigen3(case_t1_p );
            const auto t0_f0 = float3_to_eigen3(case_t0_f0); 
            const auto t0_f1 = float3_to_eigen3(case_t0_f1);
            const auto t0_f2 = float3_to_eigen3(case_t0_f2); 
            const auto t1_f0 = float3_to_eigen3(case_t1_f0);
            const auto t1_f1 = float3_to_eigen3(case_t1_f1); 
            const auto t1_f2 = float3_to_eigen3(case_t1_f2);

            float toi = host_accd::point_triangle_ccd(t0_p,  t1_p,
                                      t0_f0, t0_f1,
                                      t0_f2, t1_f0,
                                      t1_f1, t1_f2,
                                      1e-3);
            luisa::log_info("BroadPhase Pair {} : toi = {}, vid {} & fid {} (face {})", 0, toi, vid, fid, face);
        }
        {
            auto fn_test_ccd_vf = device.compile<1>([&](Float thickness)
            {
                Uint pair_idx = 0;
                Float toi = accd::line_search_max_t;
                
                {
                    Float3 t0_p  = case_t0_p ;
                    Float3 t1_p  = case_t1_p ;
                    Float3 t0_f0 = case_t0_f0;
                    Float3 t0_f1 = case_t0_f1;
                    Float3 t0_f2 = case_t0_f2;
                    Float3 t1_f0 = case_t1_f0;
                    Float3 t1_f1 = case_t1_f1;
                    Float3 t1_f2 = case_t1_f2;
        
                   Float toi = accd::point_triangle_ccd(t0_p,  t1_p,
                                      t0_f0, t0_f1,
                                      t0_f2, t1_f0,
                                      t1_f1, t1_f2,
                                      thickness);  
                    device_log("BroadPhase Pair {} : toi = {}, vid {} & fid {} (face {})", pair_idx, toi, vid, fid, face);
                };

                // toi = ParallelIntrinsic::block_intrinsic_reduce(pair_idx, toi, ParallelIntrinsic::warp_reduce_op_min<float>);
            });
        
            stream << fn_test_ccd_vf(1e-3).dispatch(1) << synchronize();
        }
    }

    // EE CCD Test
    {
        float desire_toi = 0.91535777;
        luisa::log_info("EE Test, desire for toi {}", desire_toi);

        const uint left = 4;
        const uint right = 6;
        const uint2 left_edge = uint2(2, 3);
        const uint2 right_edge = uint2(4, 6);
        
        float3 case_ea_t0_p0 = makeFloat3(-0.499492, -0.279657, 0.460444);
        float3 case_ea_t0_p1 = makeFloat3(0.499997, -0.248673, 0.468853);
        float3 case_eb_t0_p0 = makeFloat3(-0.4, -0.3, -0.5);
        float3 case_eb_t0_p1 = makeFloat3(-0.4, -0.3, 0.5);
        float3 case_ea_t1_p0 = makeFloat3(-0.49939114, -0.30410385, 0.4529846);
        float3 case_ea_t1_p1 = makeFloat3(0.4999971, -0.27044764, 0.4630015);
        float3 case_eb_t1_p0 = makeFloat3(-0.4, -0.3, -0.5);
        float3 case_eb_t1_p1 = makeFloat3(-0.4, -0.3, 0.5);
    
        {
            const auto ea00 = float3_to_eigen3(case_ea_t0_p0); 
            const auto ea01 = float3_to_eigen3(case_ea_t0_p1);
            const auto eb00 = float3_to_eigen3(case_eb_t0_p0); 
            const auto eb01 = float3_to_eigen3(case_eb_t0_p1);
            const auto ea10 = float3_to_eigen3(case_ea_t1_p0); 
            const auto ea11 = float3_to_eigen3(case_ea_t1_p1);
            const auto eb10 = float3_to_eigen3(case_eb_t1_p0); 
            const auto eb11 = float3_to_eigen3(case_eb_t1_p1);
    
            float toi = host_accd::edge_edge_ccd(
                ea00, 
                ea01, 
                eb00, 
                eb01, 
                ea10, 
                ea11, 
                eb10, 
                eb11, 1e-3);
            luisa::log_info("BroadPhase Pair {} : toi = {}, edge1 {} ({}) & edge2 {} ({})", 0, toi, left, left_edge, right, right_edge);
        }
    
        auto fn_test_ccd_ee = device.compile<1>([&](Float thickness)
        {
            Uint pair_idx = 0;
            Float toi = accd::line_search_max_t;
            
            {
                Float3 ea_t0_p0 = case_ea_t0_p0;
                Float3 ea_t0_p1 = case_ea_t0_p1;
                Float3 eb_t0_p0 = case_eb_t0_p0;
                Float3 eb_t0_p1 = case_eb_t0_p1;
                Float3 ea_t1_p0 = case_ea_t1_p0;
                Float3 ea_t1_p1 = case_ea_t1_p1;
                Float3 eb_t1_p0 = case_eb_t1_p0;
                Float3 eb_t1_p1 = case_eb_t1_p1;
    
                toi = accd::edge_edge_ccd(
                    ea_t0_p0, 
                    ea_t0_p1, 
                    eb_t0_p0, 
                    eb_t0_p1, 
                    ea_t1_p0, 
                    ea_t1_p1, 
                    eb_t1_p0, 
                    eb_t1_p1, 
                    thickness);  
            };
            
            // $if (toi != host_accd::line_search_max_t) 
            {
                device_log("BroadPhase Pair {} : toi = {}, edge1 {} ({}) & edge2 {} ({})", pair_idx, toi, left, left_edge, right, right_edge);
            };
    
            // toi = ParallelIntrinsic::block_intrinsic_reduce(pair_idx, toi, ParallelIntrinsic::warp_reduce_op_min<float>);
        });
    
        stream << fn_test_ccd_ee(1e-3).dispatch(1) << synchronize();
    }
}


}
