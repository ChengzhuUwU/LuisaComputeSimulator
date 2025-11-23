#pragma once

#include <variant>

namespace lcs
{

namespace Initializer
{

    enum class ConstitutiveStretchModelCloth
    {
        // None     = 0,
        Spring   = 0,  // Impl
        FEM_BW98 = 1,  // Impl
    };
    enum class ConstitutiveBendingModelCloth
    {
        None             = 0,
        QuadraticBending = 1,  // Impl
        DihedralAngle    = 2,  // Impl
    };
    enum class ConstitutiveModelTet
    {
        // None             = 0,
        Spring           = 0,  // Impl
        StVK             = 1,
        StableNeoHookean = 2,
        Corotated        = 3,
        ARAP             = 4,
    };
    enum class ConstitutiveModelRigid
    {
        // None          = 0,
        Spring           = 0,
        Orthogonality    = 1,  // Impl
        ARAP             = 2,
        StableNeoHookean = 3,  // Full space simulation
    };
    enum class ConstitutiveModelRod
    {
        Spring = 0,
    };

    struct MaterialBase
    {
        float mass        = 0.0f;
        float density     = 1e3f;
        float d_hat       = 1e-3f;
        float friction_mu = 0.5f;
        bool  is_shell    = true;
    };

    struct ClothMaterial : MaterialBase
    {
        ConstitutiveStretchModelCloth stretch_model  = ConstitutiveStretchModelCloth::FEM_BW98;
        ConstitutiveBendingModelCloth bending_model  = ConstitutiveBendingModelCloth::DihedralAngle;
        float                         thickness      = 1e-3f;
        float                         youngs_modulus = 1e5f;
        float                         poisson_ratio  = 0.25f;
        float                         area_bending_stiffness = 5e-3f;
        // float                         area_youngs_modulus = 1e3f;
    };
    struct TetMaterial : MaterialBase
    {
        ConstitutiveModelTet model          = ConstitutiveModelTet::Spring;
        float                youngs_modulus = 1e6f;
        float                poisson_ratio  = 0.35f;
    };
    struct RigidMaterial : MaterialBase
    {
        ConstitutiveModelRigid model     = ConstitutiveModelRigid::Orthogonality;
        bool                   is_solid  = true;
        float                  thickness = 1e-3f;
        float                  stiffness = 1e6f;
        // float                  youngs_modulus  = 1e9f;
        // float                  poisson_ratio   = 0.35f;
    };
    struct RodMaterial : MaterialBase
    {
        ConstitutiveModelRod model              = ConstitutiveModelRod::Spring;
        float                radius             = 1e-3f;
        float                bending_stiffness  = 1e4f;
        float                twisting_stiffness = 1e4f;
    };

    using MaterialVariant = std::variant<ClothMaterial, TetMaterial, RigidMaterial, RodMaterial>;
}  // namespace Initializer


}  // namespace lcs