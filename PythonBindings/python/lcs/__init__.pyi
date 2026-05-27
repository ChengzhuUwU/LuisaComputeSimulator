"""Type stub for the lcs package.

This file provides hand-maintained type annotations that supplement the
auto-generated lcs_py.pyi stub. When pybind11-stubgen cannot infer a return
type (generating ``...``), add the correct type here by overriding the
corresponding method or attribute signature on the subclass.

Real classes use ``class X(_X): ...`` (creates a cached local class
definition so Pyright/Pylance resolves members without re-traversing the
full auto-generated stub on every completion request).  MaterialType enum
values (Cloth, Particle, etc.) are not classes and use plain aliases.

``from lcs_py import X as _X`` is used instead of ``import lcs_py`` so
Pyright resolves each type individually from the .pyi stub rather than
accidentally resolving ``lcs_py`` to the .so C extension (which carries
no type information).

To regenerate lcs_py.pyi after C++ binding changes:
    cmake --build build --target stubs
Then review this file — any overrides that no longer match reality should be
removed or updated.
"""

from lcs_py import (
    ConstWorldData as _ConstWorldData,
    FixedPointsType as _FixedPointsType,
    Float3 as _Float3,
    MakeFixedPointsInterface as _MakeFixedPointsInterface,
    MaterialType as _MaterialType,
    NewtonSolver as _NewtonSolver,
    SceneParams as _SceneParams,
    WorldData as _WorldData,
    # MaterialType enum values (not classes — imported as aliases)
    Cloth,
    Particle,
    Rigid,
    Rod,
    Tetrahedral,
)

# ── Real classes: subclass for cached local type resolution ────────────

class ConstWorldData(_ConstWorldData): ...
class FixedPointsType(_FixedPointsType): ...
class Float3(_Float3): ...
class MakeFixedPointsInterface(_MakeFixedPointsInterface): ...
class MaterialType(_MaterialType): ...
class NewtonSolver(_NewtonSolver): ...
class SceneParams(_SceneParams): ...
class WorldData(_WorldData): ...


__all__ = [
    "Cloth",
    "ConstWorldData",
    "FixedPointsType",
    "Float3",
    "MakeFixedPointsInterface",
    "MaterialType",
    "NewtonSolver",
    "Particle",
    "Rigid",
    "Rod",
    "SceneParams",
    "Tetrahedral",
    "WorldData",
]
