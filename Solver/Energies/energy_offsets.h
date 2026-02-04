#pragma once

#include <cstdint>

namespace lcs
{
constexpr uint32_t offset_inertia          = 0;
constexpr uint32_t offset_ground_collision = 1;
constexpr uint32_t offset_ground_friction  = 7;
constexpr uint32_t offset_stretch_spring   = 2;
constexpr uint32_t offset_stretch_face     = 3;
constexpr uint32_t offset_bending          = 4;
constexpr uint32_t offset_abd_inertia      = 5;
constexpr uint32_t offset_abd_ortho        = 6;

}  // namespace lcs
