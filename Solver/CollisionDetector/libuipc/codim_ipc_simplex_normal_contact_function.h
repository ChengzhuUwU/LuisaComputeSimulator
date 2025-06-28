#pragma once

#include <Eigen/Geometry>
#include "CollisionDetector/libuipc/distance/distance_flagged.h"
#include "CollisionDetector/libuipc/codim_ipc_contact_function.h"
#include "CollisionDetector/libuipc/distance/edge_edge_mollifier.h"
// #include <type_define.h>
// #include <contact_system/contact_coeff.h>
// #include <contact_system/contact_models/codim_ipc_contact_function.h>

namespace uipc::backend::cuda
{
namespace sym::codim_ipc_simplex_contact
{
    // inline  float PT_kappa(const muda::CDense2D<ContactCoeff>& table,
    //                                  const Vector4i&                     cids)
    // {
    //     float kappa = 0.0;
    //     for(int j = 1; j < 4; ++j)
    //     {
    //         ContactCoeff coeff = table(cids[0], cids[j]);
    //         kappa += coeff.kappa;
    //     }
    //     return kappa / 3.0;
    // }

    // inline  float EE_kappa(const muda::CDense2D<ContactCoeff>& table,
    //                                  const Vector4i&                     cids)
    // {
    //     float kappa = 0.0;
    //     for(int j = 0; j < 2; ++j)
    //     {
    //         for(int k = 2; k < 4; ++k)
    //         {
    //             ContactCoeff coeff = table(cids[j], cids[k]);
    //             kappa += coeff.kappa;
    //         }
    //     }
    //     return kappa / 4.0;
    // }

    // inline  float PE_kappa(const muda::CDense2D<ContactCoeff>& table,
    //                                  const Vector3i&                     cids)
    // {
    //     float kappa = 0.0;
    //     for(int j = 1; j < 3; ++j)
    //     {
    //         ContactCoeff coeff = table(cids[0], cids[j]);
    //         kappa += coeff.kappa;
    //     }
    //     return kappa / 2.0;
    // }

    // inline  float PP_kappa(const muda::CDense2D<ContactCoeff>& table,
    //                                  const Vector2i&                     cids)
    // {
    //     ContactCoeff coeff = table(cids[0], cids[1]);
    //     return coeff.kappa;
    // }

    
    using Vector3 = Eigen::Vector3f;
    using Vector6 = Eigen::Matrix<float, 6, 1>;
    using Vector9 = Eigen::Matrix<float, 9, 1>;
    using Vector12 = Eigen::Matrix<float, 12, 1>;
    using Matrix3x3 = Eigen::Matrix<float, 3, 3>;
    using Matrix6x6 = Eigen::Matrix<float, 6, 6>;
    using Matrix9x9 = Eigen::Matrix<float, 9, 9>;
    using Matrix12x12 = Eigen::Matrix<float, 12, 12>;
    using Vector2i = Eigen::Vector2i;
    using Vector3i = Eigen::Vector3i;
    using Vector4i = Eigen::Vector4i;

    inline  float PT_barrier_energy(float          kappa,
                                              float          d_hat,
                                              float          thickness,
                                              const Vector3& P,
                                              const Vector3& T0,
                                              const Vector3& T1,
                                              const Vector3& T2)
    {
        using namespace codim_ipc_contact;
        using namespace distance;
        float D;
        point_triangle_distance2(P, T0, T1, T2, D);
        float B;
        KappaBarrier(B, kappa, D, d_hat, thickness);
        return B;
    }

    inline  float PT_barrier_energy(const Vector4i& flag,
                                              float           kappa,
                                              float           d_hat,
                                              float           thickness,
                                              const Vector3&  P,
                                              const Vector3&  T0,
                                              const Vector3&  T1,
                                              const Vector3&  T2)
    {
        using namespace codim_ipc_contact;
        using namespace distance;
        float D;
        point_triangle_distance2(flag, P, T0, T1, T2, D);
        float B;
        KappaBarrier(B, kappa, D, d_hat, thickness);
        return B;
    }

    inline  void PT_barrier_gradient_hessian(Vector12&      G,
                                                       Matrix12x12&   H,
                                                       float          kappa,
                                                       float          d_hat,
                                                       float          thickness,
                                                       const Vector3& P,
                                                       const Vector3& T0,
                                                       const Vector3& T1,
                                                       const Vector3& T2)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        float D;
        point_triangle_distance2(P, T0, T1, T2, D);

        Vector12 GradD;
        point_triangle_distance2_gradient(P, T0, T1, T2, GradD);

        float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        //tex:
        //$$
        // G = \frac{\partial B}{\partial D} \frac{\partial D}{\partial x}
        //$$
        G = dBdD * GradD;

        float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        Matrix12x12 HessD;
        point_triangle_distance2_hessian(P, T0, T1, T2, HessD);

        //tex:
        //$$
        // H = \frac{\partial^2 B}{\partial D^2} \frac{\partial D}{\partial x} \frac{\partial D}{\partial x}^T + \frac{\partial B}{\partial D} \frac{\partial^2 D}{\partial x^2}
        //$$
        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
    }

    inline  void PT_barrier_gradient_hessian(Vector12&       G,
                                                       Matrix12x12&    H,
                                                       const Vector4i& flag,
                                                       float           kappa,
                                                       float           d_hat,
                                                       float          thickness,
                                                       const Vector3& P,
                                                       const Vector3& T0,
                                                       const Vector3& T1,
                                                       const Vector3& T2)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        float D;
        point_triangle_distance2(flag, P, T0, T1, T2, D);

        Vector12 GradD;
        point_triangle_distance2_gradient(flag, P, T0, T1, T2, GradD);

        float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        //tex:
        //$$
        // G = \frac{\partial B}{\partial D} \frac{\partial D}{\partial x}
        //$$
        G = dBdD * GradD;

        float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        Matrix12x12 HessD;
        point_triangle_distance2_hessian(flag, P, T0, T1, T2, HessD);

        //tex:
        //$$
        // H = \frac{\partial^2 B}{\partial D^2} \frac{\partial D}{\partial x} \frac{\partial D}{\partial x}^T + \frac{\partial B}{\partial D} \frac{\partial^2 D}{\partial x^2}
        //$$
        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
    }


    inline  float mollified_EE_barrier_energy(const Vector4i& flag,
                                                        float           kappa,
                                                        float           d_hat,
                                                        float thickness,
                                                        const Vector3& t0_Ea0,
                                                        const Vector3& t0_Ea1,
                                                        const Vector3& t0_Eb0,
                                                        const Vector3& t0_Eb1,
                                                        const Vector3& Ea0,
                                                        const Vector3& Ea1,
                                                        const Vector3& Eb0,
                                                        const Vector3& Eb1)
    {
        // using mollifier to improve the smoothness of the edge-edge barrier
        using namespace codim_ipc_contact;
        using namespace distance;
        float D;
        edge_edge_distance2(flag, Ea0, Ea1, Eb0, Eb1, D);
        float B;
        KappaBarrier(B, kappa, D, d_hat, thickness);

        float eps_x;
        edge_edge_mollifier_threshold(t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, eps_x);

        float ek;
        edge_edge_mollifier(Ea0, Ea1, Eb0, Eb1, eps_x, ek);

        return ek * B;
    }

    inline  void mollified_EE_barrier_gradient_hessian(Vector12&    G,
                                                                 Matrix12x12& H,
                                                                 const Vector4i& flag,
                                                                 float kappa,
                                                                 float d_hat,
                                                                 float thickness,
                                                                 const Vector3& t0_Ea0,
                                                                 const Vector3& t0_Ea1,
                                                                 const Vector3& t0_Eb0,
                                                                 const Vector3& t0_Eb1,
                                                                 const Vector3& Ea0,
                                                                 const Vector3& Ea1,
                                                                 const Vector3& Eb0,
                                                                 const Vector3& Eb1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        float D;
        edge_edge_distance2(flag, Ea0, Ea1, Eb0, Eb1, D);

        //tex: $$ \nabla D$$
        Vector12 GradD;
        edge_edge_distance2_gradient(flag, Ea0, Ea1, Eb0, Eb1, GradD);

        //tex: $$ \nabla^2 D$$
        Matrix12x12 HessD;
        edge_edge_distance2_hessian(flag, Ea0, Ea1, Eb0, Eb1, HessD);

        float B;
        KappaBarrier(B, kappa, D, d_hat, thickness);

        //tex: $$ \frac{\partial B}{\partial D} $$
        float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        //tex: $$ \frac{\partial^2 B}{\partial D^2} $$
        float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        //tex: $$ \nabla B = \frac{\partial B}{\partial D} \nabla D$$
        Vector12 GradB = dBdD * GradD;

        //tex:
        //$$
        // \nabla^2 B = \frac{\partial^2 B}{\partial D^2} \nabla D \nabla D^T + \frac{\partial B}{\partial D} \nabla^2 D
        //$$
        Matrix12x12 HessB = ddBddD * GradD * GradD.transpose() + dBdD * HessD;

        //tex: $$ \epsilon_x $$
        float eps_x;
        edge_edge_mollifier_threshold(t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, eps_x);

        //tex: $$ e_k $$
        float ek;
        edge_edge_mollifier(Ea0, Ea1, Eb0, Eb1, eps_x, ek);

        //tex: $$\nabla e_k$$
        Vector12 Gradek;
        edge_edge_mollifier_gradient(Ea0, Ea1, Eb0, Eb1, eps_x, Gradek);


        //tex: $$ \nabla^2 e_k$$
        Matrix12x12 Hessek;
        edge_edge_mollifier_hessian(Ea0, Ea1, Eb0, Eb1, eps_x, Hessek);

        //tex:
        //$$
        // G = \nabla e_k B + e_k \nabla B
        //$$
        G = Gradek * B + ek * GradB;

        //tex: $$ \nabla^2 e_k B + \nabla e_k \nabla B^T + \nabla B \nabla e_k^T + e_k \nabla^2 B$$
        H = Hessek * B + Gradek * GradB.transpose() + GradB * Gradek.transpose() + ek * HessB;
    }

    inline  float PE_barrier_energy(const Vector3i& flag,
                                              float           kappa,
                                              float           d_hat,
                                              float           thickness,
                                              const Vector3&  P,
                                              const Vector3&  E0,
                                              const Vector3&  E1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;
        float D = 0.0;
        point_edge_distance2(flag, P, E0, E1, D);
        float E = 0.0;
        KappaBarrier(E, kappa, D, d_hat, thickness);
        return E;
    }

    inline  void PE_barrier_gradient_hessian(Vector9&        G,
                                                       Matrix9x9&      H,
                                                       const Vector3i& flag,
                                                       float           kappa,
                                                       float           d_hat,
                                                       float          thickness,
                                                       const Vector3& P,
                                                       const Vector3& E0,
                                                       const Vector3& E1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        float D = 0.0;
        point_edge_distance2(flag, P, E0, E1, D);

        Vector9 GradD;
        point_edge_distance2_gradient(flag, P, E0, E1, GradD);

        Matrix9x9 HessD;
        point_edge_distance2_hessian(flag, P, E0, E1, HessD);

        float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        //tex:
        //$$
        // G = \frac{\partial B}{\partial D} \frac{\partial D}{\partial x}
        //$$
        G = dBdD * GradD;

        float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        //tex:
        //$$
        // H = \frac{\partial^2 B}{\partial D^2} \frac{\partial D}{\partial x} \frac{\partial D}{\partial x}^T + \frac{\partial B}{\partial D} \frac{\partial^2 D}{\partial x^2}
        //$$
        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
    }

    inline  float PP_barrier_energy(const Vector2i& flag,
                                              float           kappa,
                                              float           d_hat,
                                              float           thickness,
                                              const Vector3&  P0,
                                              const Vector3&  P1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;
        float D = 0.0;
        point_point_distance2(flag, P0, P1, D);
        float E = 0.0;
        KappaBarrier(E, kappa, D, d_hat, thickness);
        return E;
    }

    inline  void PP_barrier_gradient_hessian(Vector6&        G,
                                                       Matrix6x6&      H,
                                                       const Vector2i& flag,
                                                       float           kappa,
                                                       float           d_hat,
                                                       float          thickness,
                                                       const Vector3& P0,
                                                       const Vector3& P1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        float D = 0.0;
        point_point_distance2(flag, P0, P1, D);

        Vector6 GradD;
        point_point_distance2_gradient(flag, P0, P1, GradD);

        Matrix6x6 HessD;
        point_point_distance2_hessian(flag, P0, P1, HessD);

        float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        //tex:
        //$$
        // G = \frac{\partial B}{\partial D} \frac{\partial D}{\partial x}
        //$$
        G = dBdD * GradD;

        float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        //tex:
        //$$
        // H = \frac{\partial^2 B}{\partial D^2} \frac{\partial D}{\partial x} \frac{\partial D}{\partial x}^T + \frac{\partial B}{\partial D} \frac{\partial^2 D}{\partial x^2}
        //$$
        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
    }
}  // namespace sym::codim_ipc_simplex_contact
}  // namespace uipc::backend::cuda
