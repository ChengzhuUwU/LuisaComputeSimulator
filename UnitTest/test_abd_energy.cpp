
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include "Core/float_nxn.h"
#include "Core/lc_to_eigen.h"
#include "Energy/stretch_energy.h"
#include "luisa/core/logging.h"
#include <luisa/dsl/sugar.h>
#include <vector>

auto compute_inertia_gradient_hessian(const std::vector<luisa::float3>& abd_q,
                                      const std::vector<luisa::float3>& abd_q_tilde,
                                      const float                       h_2_inv,
                                      lcs::EigenFloat12x12&             M)
{
    using namespace lcs;

    EigenFloat12 delta = EigenFloat12::Zero();

    delta.block<3, 1>(0, 0) = float3_to_eigen3(abd_q[0] - abd_q_tilde[0]);
    delta.block<3, 1>(3, 0) = float3_to_eigen3(abd_q[1] - abd_q_tilde[1]);
    delta.block<3, 1>(6, 0) = float3_to_eigen3(abd_q[2] - abd_q_tilde[2]);
    delta.block<3, 1>(9, 0) = float3_to_eigen3(abd_q[3] - abd_q_tilde[3]);

    EigenFloat12    gradient = h_2_inv * M * delta;
    EigenFloat12x12 hessian  = h_2_inv * M;


    std::cout << "Inertia gradient: " << std::endl << gradient << std::endl;
    std::cout << "Inertia hessian: " << std::endl << hessian << std::endl;
    return std::make_pair(gradient, hessian);
}

auto compute_ortho_gradient_hessian(const std::vector<luisa::float3>& abd_q, const float stiffness_ortho)
{
    using namespace lcs;

    float3x3 A = luisa::make_float3x3(abd_q[1], abd_q[2], abd_q[3]);
    // A          = luisa::transpose(A);

    EigenFloat12x12 cgA = EigenFloat12x12::Zero();
    EigenFloat12    cgB = EigenFloat12::Zero();

    const float kappa = 1e5f;

    float stiff = kappa;  //* V;
    for (uint ii = 0; ii < 3; ii++)
    {
        float3 grad = (-1.0f) * A[ii];
        for (uint jj = 0; jj < 3; jj++)
        {
            grad += dot_vec(A[ii], A[jj]) * A[jj];
        }
        LUISA_INFO("grad of col {} = {}", ii, grad);
        cgB.block<3, 1>(3 + 3 * ii, 0) -= 4 * stiff * float3_to_eigen3(grad);
    }

    // Curr q = [[ 0.0071148   0.66069971  0.01422961  0.8271494  -0.55152995 -0.07775201  0.49132671  0.6148592   0.61662802 -0.28366761 -0.53456093  0.79869019]]
    // Output ortho B = [[    0.        ]
    //  [    0.        ]
    //  [    0.        ]
    //  [-2167.57202995]
    //  [-6399.44981761]
    //  [-4335.1440599 ]
    //  [-3564.46779844]
    //  [ 9571.30183649]
    //  [-7128.93559687]
    //  [-3703.46747295]
    //  [-5549.55463064]
    //  [-7406.9349459 ]], H = [[      0.               0.               0.               0.               0.               0.               0.               0.               0.               0.               0.               0.        ]
    //  [      0.               0.               0.               0.               0.               0.               0.               0.               0.               0.               0.               0.        ]
    //  [      0.               0.               0.               0.               0.               0.               0.               0.               0.               0.               0.               0.        ]
    //  [      0.               0.               0.          673851.31646209 -183464.38545501  -20888.71157841  170297.5398687  -108392.55871789  -15280.65602889  -94617.75549172   62580.47295392    8822.29087167]
    //  [      0.               0.               0.         -183464.38545501  506633.83697376   15182.38818141  203432.16843377 -127908.00420563  -19122.61599552 -176864.700285    117166.98648457   16625.27501305]
    //  [      0.               0.               0.          -20888.71157841   15182.38818141  409853.46628949  204017.39732932 -136035.52817096  -11440.32476452  264254.44281388 -176200.62391876  -25603.46626114]
    //  [      0.               0.               0.          170297.5398687   203432.16843377  204017.39732932  498852.48069411  119853.38167308  126022.62639349  -45972.76304252  -69766.25609615  -69966.958027  ]
    //  [      0.               0.               0.         -108392.55871789 -127908.00420563 -136035.52817096  119853.38167308  538291.29901294  149685.12786802 -105057.62515555 -121695.25694203 -131850.09834133]
    //  [      0.               0.               0.          -15280.65602889  -19122.61599552  -11440.32476452  126022.62639349  149685.12786802  561638.20115053  156967.128851    196432.80519259  206774.52465045]
    //  [      0.               0.               0.          -94617.75549172 -176864.700285    264254.44281388  -45972.76304252 -105057.62515555  156967.128851    436256.56028733   59669.72962466  -85788.71385636]
    //  [      0.               0.               0.           62580.47295392  117166.98648457 -176200.62391876  -69766.25609615 -121695.25694203  196432.80519259   59669.72962466  503150.64643735 -172750.06436395]
    //  [      0.               0.               0.            8822.29087167   16625.27501305  -25603.46626114  -69966.958027   -131850.09834133  206774.52465045  -85788.71385636 -172750.06436395  666486.49144159]]

    for (uint ii = 0; ii < 3; ii++)
    {
        for (uint jj = 0; jj < 3; jj++)
        {
            float3x3 hessian = Zero3x3;
            if (ii == jj)
            {
                float3x3 qiqiT = outer_product(A[ii], A[ii]);
                float    qiTqi = dot_vec(A[ii], A[ii]) - 1.0f;
                float3x3 term2 = qiTqi * Identity3x3;
                for (uint kk = 0; kk < 3; kk++)
                {
                    hessian = hessian + outer_product(A[kk], A[kk]);
                }
                hessian = hessian + qiqiT + term2;
            }
            else
            {
                hessian = outer_product(A[jj], A[ii]) + dot_vec(A[ii], A[jj]) * Identity3x3;
            }
            LUISA_INFO("hess of {} adj {} = {}", ii, jj, hessian);
            cgA.block<3, 3>(3 + 3 * ii, 3 + 3 * jj) += 4.0f * stiff * float3x3_to_eigen3x3(hessian);
        }
    }
    return std::make_pair(cgB, cgA);
}

typedef Eigen::Matrix<float, 2, 1> Vector2;
typedef Eigen::Matrix<float, 3, 1> Vector3;
typedef Eigen::Matrix<float, 6, 1> Vector6;
typedef Eigen::Matrix<float, 3, 2> Matrix3x2;
typedef Eigen::Matrix<float, 6, 6> Matrix6x6;

Vector6 flatten(const Matrix3x2& A)
{
    Vector6      column;
    unsigned int index = 0;
    for (unsigned int j = 0; j < 2; j++)
        for (unsigned int i = 0; i < 3; i++, index++)
            column[index] = A(i, j);
    return column;
}

void stretchHessian(const Matrix3x2& F, Matrix6x6& H, const float _mu)
{
    H.setZero();
    const Vector2 u(1.0, 0.0);
    const Vector2 v(0.0, 1.0);
    const double  I5u        = (F * u).transpose() * (F * u);
    const double  I5v        = (F * v).transpose() * (F * v);
    const double  invSqrtI5u = 1.0 / sqrt(I5u);
    const double  invSqrtI5v = 1.0 / sqrt(I5v);
    // set the block diagonals, build the rank-three
    // subspace with all-(1 / invSqrtI5) eigenvalues
    H(0, 0) = H(1, 1) = H(2, 2) = std::max((1.0 - invSqrtI5u), 0.0);
    H(3, 3) = H(4, 4) = H(5, 5) = std::max((1.0 - invSqrtI5v), 0.0);
    // modify the upper block diagonal, bump the single
    // outer-product eigenvalue back to just 1, unless it
    // was clamped, then just set it directly to 1
    const Vector3 fu     = F.col(0).normalized();
    const double  uCoeff = (1.0 - invSqrtI5u >= 0.0) ? invSqrtI5u : 1.0;
    H.block<3, 3>(0, 0) += uCoeff * (fu * fu.transpose());
    // modify the lower block diagonal similarly
    const Vector3 fv     = F.col(1).normalized();
    const double  vCoeff = (1.0 - invSqrtI5v >= 0.0) ? invSqrtI5v : 1.0;
    H.block<3, 3>(3, 3) += vCoeff * (fv * fv.transpose());
    // the leading 2 is absorbed by the mu / 2 coefficient
    H *= _mu;
}

using namespace lcs;
inline float6x6 shear_hessian(const float2x3& F, float mu)
{
    float6x6 H = float6x6::zero();

    const float3& Fu = F.cols[0];
    const float3& Fv = F.cols[1];

    const float I6     = luisa::dot(Fu, Fv);
    const float signI6 = luisa::sign(I6);

    H.scalar<3, 0>() = H.scalar<4, 1>() = H.scalar<5, 2>() = H.scalar<0, 3>() = H.scalar<1, 4>() =
        H.scalar<2, 5>()                                                      = 1.0f;

    const float6 g = StretchEnergy::detail::flatten(F * luisa::make_float2x2(0, 1, 1, 0));  // F * (a b^T + b a^T)

    const float I2      = luisa::dot(Fu, Fu) + luisa::dot(Fv, Fv);  // F.squaredNorm();
    const float lambda0 = 0.5f * (I2 + luisa::sqrt(I2 * I2 + 12.0f * I6 * I6));

    const float6 q0     = (I6 * H * g + lambda0 * g).normalize();
    float6x6     T      = float6x6::identity();
    T                   = 0.5f * (T + signI6 * H);
    const float6 Tq     = T * q0;
    const float  normTq = Tq.squared_norm();

    H = luisa::abs(I6) * (T - float6x6::outer_product(Tq, Tq) / normTq) + lambda0 * float6x6::outer_product(q0, q0);

    return mu * H;
}
void shearHessian(const Matrix3x2& F, Matrix6x6& pPpF, const float _mu)
{
    const Vector2 u(1.0, 0.0);
    const Vector2 v(0.0, 1.0);
    const double  I6     = (F * u).transpose() * (F * v);
    const double  signI6 = (I6 >= 0) ? 1.0 : -1.0;
    Matrix6x6     H      = Matrix6x6::Zero();
    H(3, 0) = H(4, 1) = H(5, 2) = H(0, 3) = H(1, 4) = H(2, 5) = 1.0;
    const Vector6 g = flatten(F * (u * v.transpose() + v * u.transpose()));

    // get the novel eigenvalue
    const double I2      = F.squaredNorm();
    const double lambda0 = 0.5 * (I2 + sqrt(I2 * I2 + 12.0 * I6 * I6));

    // get the novel eigenvector
    // the H multiply is a column swap; could be optimized more
    const Vector6 q0 = (I6 * H * g + lambda0 * g).normalized();
    Matrix6x6     T  = Matrix6x6::Identity();

    T                    = 0.5 * (T + signI6 * H);
    const Vector6 Tq     = T * q0;
    const double  normTq = Tq.squaredNorm();

    // std::cout << "q0 \n" << (q0) << std::endl;
    // std::cout << "q0 q0T = \n" << (q0 * q0.transpose()) << std::endl;
    pPpF = fabs(I6) * (T - (Tq * Tq.transpose()) / normTq) + lambda0 * (q0 * q0.transpose());
    // half from mu and leading 2 on Hessian cancel
    pPpF *= _mu;
}
void test_FEM_BW98()
{
    using namespace lcs;

    std::vector<luisa::float3> sa_x = {
        luisa::make_float3(0.0f, 0.0f, 0.0f),
        luisa::make_float3(2.0f, 0.0f, 0.0f),
        luisa::make_float3(1.0f, 2.0f, 0.0f),
    };
    std::vector<luisa::float3> sa_rest_x = {
        luisa::make_float3(0.0f, 0.0f, 0.0f),
        luisa::make_float3(1.0f, 0.0f, 0.0f),
        luisa::make_float3(0.0f, 1.0f, 0.0f),
    };

    const auto& x0 = sa_x[0];
    const auto& x1 = sa_x[1];
    const auto& x2 = sa_x[2];
    const auto& X0 = sa_rest_x[0];
    const auto& X1 = sa_rest_x[1];
    const auto& X2 = sa_rest_x[2];

    uint3       face = uint3(0, 1, 2);
    const float area = 0.5f;

    float3   r_1    = X1 - X0;
    float3   r_2    = X2 - X0;
    float3   cross  = cross_vec(r_1, r_2);
    float3   axis_1 = normalize_vec(r_1);
    float3   axis_2 = normalize_vec(cross_vec(cross, axis_1));
    float2   uv0    = float2(dot_vec(axis_1, X0), dot_vec(axis_2, X0));
    float2   uv1    = float2(dot_vec(axis_1, X1), dot_vec(axis_2, X1));
    float2   uv2    = float2(dot_vec(axis_1, X2), dot_vec(axis_2, X2));
    float2   duv0   = uv1 - uv0;
    float2   duv1   = uv2 - uv0;
    float2x2 Dm1    = float2x2(duv0, duv1);

    // auto Dm2 = StretchEnergy::libuipc::Dm2x2(float3_to_eigen3(X0), float3_to_eigen3(X1), float3_to_eigen3(X2));
    // std::cout << "Dm1 = " << luisa::format("{}", Dm1) << std::endl;
    // std::cout << "Dm2 = " << Dm2 << std::endl;

    float3   vert_pos[3]  = {sa_x[face[0]], sa_x[face[1]], sa_x[face[2]]};
    float3   x_bars[3]    = {sa_rest_x[face[0]], sa_rest_x[face[1]], sa_rest_x[face[2]]};
    float3   gradients[3] = {Zero3, Zero3, Zero3};
    float3x3 hessians[3][3];
    for (auto& tmp : hessians)
    {
        for (auto& hess : tmp)
            hess = Zero3x3;
    }

    Eigen::Matrix<float, 9, 1> G;
    Eigen::Matrix<float, 9, 9> H;

    const float                      mass = 0.01f;
    const float                      dt   = 0.01f;
    const Eigen::Matrix<float, 9, 9> M    = mass / dt / dt * Eigen::Matrix<float, 9, 9>::Identity();

    const float youngs        = 1e4f;
    const float possion_rate  = 0.2f;
    auto [mu_tmp, lambda_tmp] = StretchEnergy::convert_prop(youngs, possion_rate);
    const float mu            = mu_tmp;
    const float lambda        = lambda_tmp;

    {
        float2x2 Dm_inv = luisa::inverse(Dm1);
        {
            auto dfdx = StretchEnergy::detail::get_dFdx(Dm_inv);
            std::cout << "dFdx1 = \n" << dfdx.to_eigen_matrix() << std::endl;
        }
        {
            lcs::LargeMatrix<6, 9> dfdx_T = lcs::LargeMatrix<6, 9>::zero();

            const auto& InverseDm = Dm_inv;
            const float d0        = InverseDm[0][0];
            const float d1        = InverseDm[0][1];
            const float d2        = InverseDm[1][0];
            const float d3        = InverseDm[1][1];
            const float s0        = d0 + d1;
            const float s1        = d2 + d3;
            dfdx_T.scalar<0, 0>() = -s0;
            dfdx_T.scalar<3, 0>() = -s1;
            dfdx_T.scalar<1, 1>() = -s0;
            dfdx_T.scalar<4, 1>() = -s1;
            dfdx_T.scalar<2, 2>() = -s0;
            dfdx_T.scalar<5, 2>() = -s1;
            dfdx_T.scalar<0, 3>() = d0;
            dfdx_T.scalar<3, 3>() = d2;
            dfdx_T.scalar<1, 4>() = d0;
            dfdx_T.scalar<4, 4>() = d2;
            dfdx_T.scalar<2, 5>() = d0;
            dfdx_T.scalar<5, 5>() = d2;
            dfdx_T.scalar<0, 6>() = d1;
            dfdx_T.scalar<3, 6>() = d3;
            dfdx_T.scalar<1, 7>() = d1;
            dfdx_T.scalar<4, 7>() = d3;
            dfdx_T.scalar<2, 8>() = d1;
            dfdx_T.scalar<5, 8>() = d3;
            std::cout << "dFdx2 = \n" << lcs::transpose(dfdx_T).to_eigen_matrix() << std::endl;
        }
        {
            // lcs::LargeMatrix<9, 6> result;
            // const float2x3         diff_mat = StretchEnergy::detail::make_diff_mat3x2();
            // for (unsigned i = 0; i < 3; ++i)
            // {
            //     for (unsigned j = 0; j < 2; ++j)
            //     {
            //         for (unsigned dim = 0; dim < 3; ++dim)
            //         {
            //             result.scalar(i, 3 * j + dim)     = diff_mat[j][dim] * Dm_inv[0][i];
            //             result.scalar(i + 3, 3 * j + dim) = diff_mat[j][dim] * Dm_inv[1][i];
            //         }
            //     }
            // }
            // std::cout << "dFdx3 = \n" << (result).to_eigen_matrix() << std::endl;
        }
    }

    return;

    {
        float2x3 F = makeFloat2x3(x1 - x0, x2 - x0) * luisa::inverse(Dm1);

        float6x6 H_stretch = StretchEnergy::detail::stretch_hessian(F, mu);
        // float6x6 H_shear   = StretchEnergy::detail::shear_hessian(F, lambda);
        float6x6 H_shear = shear_hessian(F, lambda);
        // StretchEnergy::compute_gradient_hessian(
        //     vert_pos[0], vert_pos[1], vert_pos[2], Dm_inv, mu, lambda, area, gradients, hessians);

        for (uint ii = 0; ii < 3; ii++)
        {
            G.block<3, 1>(ii * 3, 0) = float3_to_eigen3(gradients[ii]);
            for (uint jj = 0; jj < 3; jj++)
            {
                H.block<3, 3>(ii * 3, jj * 3) = float3x3_to_eigen3x3(hessians[ii][jj]);
            }
        }
        std::cout << "Impl : \n";
        // std::cout << "Gradient : \n" << G << std::endl;
        // std::cout << "Result : \n" << (M + H).inverse() * G << std::endl;
        // std::cout << "Hessian : \n" << H << std::endl;
        // std::cout << "F : \n" << F.to_eigen_matrix() << std::endl;
        std::cout << "Hessian_stretch : \n" << H_stretch.to_eigen_matrix() << std::endl;
        std::cout << "Hessian_shear : \n" << H_shear.to_eigen_matrix() << std::endl;
    }
    {
        auto F = (makeFloat2x3(x1 - x0, x2 - x0) * Dm1).to_eigen_matrix();
        // auto F = StretchEnergy::libuipc::Ds3x2(float3_to_eigen3(x0), float3_to_eigen3(x1), float3_to_eigen3(x2))
        //          * Dm2.inverse();

        // float2x3 de0dF   = detail::stretch_gradient(F, mu);
        // float2x3 de1dF   = detail::shear_gradient(F, lambda);

        Matrix6x6 H_stretch;
        Matrix6x6 H_shear;
        Matrix3x2 F_eigen = F;
        stretchHessian(F_eigen, H_stretch, mu);
        shearHessian(F_eigen, H_shear, lambda);

        std::cout << "F : \n" << F << std::endl;
        std::cout << "Hessian_stretch : \n" << H_stretch << std::endl;
        std::cout << "Hessian_shear : \n" << H_shear << std::endl;
    }
}

int main()
{
    using namespace lcs;

    EigenFloat12x12 cgA = EigenFloat12x12::Zero();
    EigenFloat12    cgB = EigenFloat12::Zero();


    std::vector<luisa::float3> abd_q = {
        luisa::make_float3(0.0071148, 0.66069971, 0.0142296),
        luisa::make_float3(0.8271494, -0.55152995, -0.07775201),
        luisa::make_float3(0.49132671, 0.6148592, 0.61662802),
        luisa::make_float3(-0.28366761, -0.53456093, 0.79869019),
    };
    std::vector<luisa::float3> abd_q_tilde = {
        luisa::make_float3(0.0f, 0.0f, 0.0f),
        luisa::make_float3(1.0f, 0.0f, 0.0f),
        luisa::make_float3(0.0f, 1.0f, 0.0f),
        luisa::make_float3(0.0f, 0.0f, 1.0f),
    };

    // auto result = compute_ortho_gradient_hessian(abd_q, 1e5f);
    // std::cout << "Ortho gradient: " << std::endl << result.first << std::endl;
    // std::cout << "Ortho hessian: " << std::endl << result.second << std::endl;

    test_FEM_BW98();


    // float6 vec1;
    // vec1.vec[0] = luisa::make_float3(0, 1, 2);
    // vec1.vec[1] = luisa::make_float3(3, 4, 5);

    // float6 vec2;
    // vec2.vec[0] = luisa::make_float3(0, 1, 100);
    // vec2.vec[1] = luisa::make_float3(1000, 10000, 100000);
    // std::cout << "Val 1 = \n" << float6x6::outer_product(vec1, vec2).to_eigen_matrix() << std::endl;
    // std::cout << "Val 2 = \n"
    //           << vec1.to_eigen_matrix() * vec2.to_eigen_matrix().transpose() << std::endl;
    // std::cout << "Val 1 = \n" << float6x6::outer_product(vec2, vec1).to_eigen_matrix() << std::endl;
    // std::cout << "Val 2 = \n"
    //           << vec2.to_eigen_matrix() * vec1.to_eigen_matrix().transpose() << std::endl;
}