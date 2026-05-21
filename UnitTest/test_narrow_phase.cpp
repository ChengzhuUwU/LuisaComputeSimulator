/**
 * @file test_narrow_phase.cpp
 * @brief Unit tests for NarrowPhase collision detection
 *
 * Test Coverage:
 * - Ground truth distance computation
 * - Contact pair data structures
 */

#include "test_base_solver.h"
#include "test_framework.h"
#include <cmath>

using namespace lcs;
using namespace lcs::test;

// =============================================================================
// Ground Truth Helpers
// =============================================================================

namespace
{

	/**
	 * Signed distance from point p to triangle (v0, v1, v2)
	 */
	float signed_distance_point_triangle(float3 p, float3 v0, float3 v1, float3 v2)
	{
		float3 n = luisa::cross(v1 - v0, v2 - v0);
		float  area = luisa::length(n);
		if (area < 1e-8f)
			return 0.0f;
		n = n / area;

		float dist = luisa::dot(p - v0, n);
		return dist;
	}

	/**
	 * Closest point on line segment to another point
	 */
	float3 closest_point_on_segment(float3 p, float3 a, float3 b)
	{
		float3 ab = b - a;
		float  t = luisa::dot(p - a, ab) / luisa::dot(ab, ab);
		t = std::max(0.0f, std::min(1.0f, t));
		return a + t * ab;
	}

	/**
	 * Distance between two line segments
	 */
	float distance_segment_segment(float3 p1, float3 q1, float3 p2, float3 q2)
	{
		float3 u = q1 - p1;
		float3 v = q2 - p2;
		float3 w = p1 - p2;

		float a = luisa::dot(u, u);
		float b = luisa::dot(u, v);
		float c = luisa::dot(v, v);
		float d = luisa::dot(u, w);
		float e = luisa::dot(v, w);

		float D = a * c - b * b;
		float sc, sN, sD = D;
		float tc, tN, tD = D;

		if (D < 1e-8f)
		{
			sN = 0.0;
			sD = 1.0;
			tN = e;
			tD = c;
		}
		else
		{
			sN = (b * e - c * d);
			tN = (a * e - b * d);
			if (sN < 0.0)
			{
				sN = 0.0;
				tN = e;
				tD = c;
			}
			else if (sN > sD)
			{
				sN = sD;
				tN = e + b;
				tD = c;
			}
		}

		if (tN < 0.0)
		{
			tN = 0.0;
			if (-d < 0.0)
				sN = 0.0;
			else if (-d > a)
				sN = sD;
			else
			{
				sN = -d;
				sD = a;
			}
		}
		else if (tN > tD)
		{
			tN = tD;
			if ((a - d) < 0.0)
				sN = 0.0;
			else if ((a - d) > a)
				sN = sD;
			else
			{
				sN = a - d;
				sD = a;
			}
		}

		sc = (std::abs(sN) < 1e-8f ? 0.0 : sN / sD);
		tc = (std::abs(tN) < 1e-8f ? 0.0 : tN / tD);

		float3 dP = w + (sc * u) - (tc * v);
		return luisa::sqrt(luisa::dot(dP, dP));
	}

} // anonymous namespace

// =============================================================================
// Test Cases
// =============================================================================

class TestNarrowPhase : public TestNewtonSolverBase
{
public:
	// -------------------------------------------------------------------------
	// Test 1: Point-triangle distance ground truth
	// -------------------------------------------------------------------------
	bool test_point_triangle_distance()
	{
		std::cout << "\n  [Test] Point-triangle distance ground truth...\n";

		// Triangle vertices
		float3 v0 = luisa::make_float3(0.0f, 0.0f, 0.0f);
		float3 v1 = luisa::make_float3(1.0f, 0.0f, 0.0f);
		float3 v2 = luisa::make_float3(0.0f, 1.0f, 0.0f);

		// Test point above the triangle
		float3 p1 = luisa::make_float3(0.25f, 0.25f, 0.1f);
		float  dist1 = signed_distance_point_triangle(p1, v0, v1, v2);

		// Test point below the triangle
		float3 p2 = luisa::make_float3(0.25f, 0.25f, -0.1f);
		float  dist2 = signed_distance_point_triangle(p2, v0, v1, v2);

		std::cout << "    Point above triangle, distance: " << dist1 << " (expected: 0.1)\n";
		std::cout << "    Point below triangle, distance: " << dist2 << " (expected: -0.1)\n";

		TEST_ASSERT_NEAR(dist1, 0.1f, 1e-5f, "Distance above triangle mismatch");
		TEST_ASSERT_NEAR(dist2, -0.1f, 1e-5f, "Distance below triangle mismatch");

		std::cout << "    PASSED\n";
		return true;
	}

	// -------------------------------------------------------------------------
	// Test 2: Segment-segment distance ground truth
	// -------------------------------------------------------------------------
	bool test_segment_segment_distance()
	{
		std::cout << "\n  [Test] Segment-segment distance ground truth...\n";

		// Two parallel segments
		float3 p1 = luisa::make_float3(0.0f, 0.0f, 0.0f);
		float3 q1 = luisa::make_float3(1.0f, 0.0f, 0.0f);
		float3 p2 = luisa::make_float3(0.0f, 0.5f, 0.0f);
		float3 q2 = luisa::make_float3(1.0f, 0.5f, 0.0f);

		float dist = distance_segment_segment(p1, q1, p2, q2);

		std::cout << "    Parallel segments distance: " << dist << " (expected: 0.5)\n";

		TEST_ASSERT_NEAR(dist, 0.5f, 1e-5f, "Parallel segment distance mismatch");

		// Intersecting segments
		float3 r1 = luisa::make_float3(0.0f, 0.0f, 0.0f);
		float3 s1 = luisa::make_float3(1.0f, 1.0f, 0.0f);
		float3 r2 = luisa::make_float3(0.0f, 1.0f, 0.0f);
		float3 s2 = luisa::make_float3(1.0f, 0.0f, 0.0f);

		float dist_intersect = distance_segment_segment(r1, s1, r2, s2);

		std::cout << "    Intersecting segments distance: " << dist_intersect << " (expected: 0)\n";

		TEST_ASSERT_NEAR(dist_intersect, 0.0f, 1e-5f, "Intersecting segment distance mismatch");

		std::cout << "    PASSED\n";
		return true;
	}

	// -------------------------------------------------------------------------
	// Test 3: Collision pair structure
	// -------------------------------------------------------------------------
	bool test_collision_pair_structure()
	{
		std::cout << "\n  [Test] Collision pair structure...\n";

		CollisionPair::CollisionPairTemplate pair;

		// Create a VF pair
		uint4  indices = luisa::make_uint4(0, 1, 2, 3);
		float3 normal = luisa::make_float3(0.0f, 0.0f, 1.0f);
		float  k1 = 1e6f, k2 = 1e6f;
		float  area = 0.5f;
		float3 bary = luisa::make_float3(0.33f, 0.33f, 0.34f);

		pair.make_vf_pair(indices, normal, k1, k2, area, bary);

		TEST_ASSERT(pair.get_collision_type() == CollisionPair::type_vf(),
			"Pair type should be VF");

		float retrieved_area = pair.get_area();
		TEST_ASSERT_NEAR(retrieved_area, area, 1e-5f, "Area retrieval mismatch");

		float3 retrieved_normal = pair.get_normal();
		TEST_ASSERT_VEC3_NEAR(retrieved_normal, normal, 1e-5f, "Normal retrieval mismatch");

		std::cout << "    Pair type: VF\n";
		std::cout << "    Area: " << retrieved_area << "\n";
		std::cout << "    PASSED\n";
		return true;
	}

	// -------------------------------------------------------------------------
	// Test 4: EE pair structure
	// -------------------------------------------------------------------------
	bool test_ee_pair_structure()
	{
		std::cout << "\n  [Test] EE pair structure...\n";

		CollisionPair::CollisionPairTemplate pair;

		// Create an EE pair
		uint4  indices = luisa::make_uint4(0, 1, 2, 3);
		float3 normal = luisa::make_float3(0.0f, 0.0f, 1.0f);
		float  k1 = 1e6f, k2 = 1e6f;
		float  area = 0.0f;
		float2 edge1_bary = luisa::make_float2(0.5f, 0.0f);
		float2 edge2_bary = luisa::make_float2(0.5f, 0.0f);

		pair.make_ee_pair(indices, normal, k1, k2, area, edge1_bary, edge2_bary);

		TEST_ASSERT(pair.get_collision_type() == CollisionPair::type_ee(),
			"Pair type should be EE");

		auto stiff = pair.get_stiff();
		TEST_ASSERT_NEAR(stiff.x, k1, 1e-5f, "Stiffness k1 mismatch");
		TEST_ASSERT_NEAR(stiff.y, k2, 1e-5f, "Stiffness k2 mismatch");

		std::cout << "    Pair type: EE\n";
		std::cout << "    Stiffness: [" << stiff.x << ", " << stiff.y << "]\n";
		std::cout << "    PASSED\n";
		return true;
	}

	// -------------------------------------------------------------------------
	// Test 5: Weight computation
	// -------------------------------------------------------------------------
	bool test_weight_computation()
	{
		std::cout << "\n  [Test] Collision pair weight computation...\n";

		CollisionPair::CollisionPairTemplate pair;

		// VF pair with barycentric coordinates
		uint4  indices = luisa::make_uint4(0, 1, 2, 3);
		float3 normal = luisa::make_float3(0.0f, 0.0f, 1.0f);
		pair.make_vf_pair(indices, normal, 1e6f, 1e6f, 0.5f,
			luisa::make_float3(0.33f, 0.33f, 0.34f));

		float4 weight = pair.get_vf_weight();
		float  sum = weight.x + weight.y + weight.z + weight.w;

		std::cout << "    VF weight: [" << weight.x << ", " << weight.y << ", " << weight.z << ", " << weight.w << "]\n";
		std::cout << "    Weight sum: " << sum << " (expected: 0.0)\n";

		TEST_ASSERT_NEAR(weight.x, 1.0f, 1e-5f, "Query vertex weight should be 1.0");
		TEST_ASSERT_NEAR(sum, 0.0f, 1e-5f,
			"VF constraint weights should sum to 0.0 for relative displacement invariance");

		std::cout << "    PASSED\n";
		return true;
	}
};

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv)
{
	luisa::log_level_info();
	std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
	std::cout << "║  NarrowPhase Tests                                         ║\n";
	std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";

	TestSuiteResult suite;
	suite.name = "NarrowPhase";

	int total = 0, passed = 0;

	TestNarrowPhase test;

	total++;
	if (test.test_point_triangle_distance())
	{
		passed++;
		suite.add(TestResult(true), false);
	}
	else
	{
		suite.add(TestResult(false), false);
	}

	total++;
	if (test.test_segment_segment_distance())
	{
		passed++;
		suite.add(TestResult(true), false);
	}
	else
	{
		suite.add(TestResult(false), false);
	}

	total++;
	if (test.test_collision_pair_structure())
	{
		passed++;
		suite.add(TestResult(true), false);
	}
	else
	{
		suite.add(TestResult(false), false);
	}

	total++;
	if (test.test_ee_pair_structure())
	{
		passed++;
		suite.add(TestResult(true), false);
	}
	else
	{
		suite.add(TestResult(false), false);
	}

	total++;
	if (test.test_weight_computation())
	{
		passed++;
		suite.add(TestResult(true), false);
	}
	else
	{
		suite.add(TestResult(false), false);
	}

	print_suite_summary(suite);

	std::cout << "\n";
	std::cout << "═══════════════════════════════════════════════════════════════════════\n";
	auto pass_str = std::to_string(passed);
	auto total_str = std::to_string(total);
	int	 padding = std::max(0, 33 - static_cast<int>(pass_str.size()) - static_cast<int>(total_str.size()));
	std::cout << "  NarrowPhase Tests: " << passed << "/" << total << " passed"
			  << std::string(padding, ' ') << "\n";
	std::cout << "═══════════════════════════════════════════════════════════════════════\n";

	return (passed == total) ? 0 : 1;
}
