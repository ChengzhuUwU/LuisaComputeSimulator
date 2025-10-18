
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include "Core/float_nxn.h"
#include "Core/lc_to_eigen.h"
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

    auto result = compute_ortho_gradient_hessian(abd_q, 1e5f);
    std::cout << "Ortho gradient: " << std::endl << result.first << std::endl;
    std::cout << "Ortho hessian: " << std::endl << result.second << std::endl;
}