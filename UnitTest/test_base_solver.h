/**
 * @file test_base_solver.h
 * @brief TestNewtonSolverBase: base class for all NewtonSolver module tests
 *
 * Philosophy:
 * - Each module test (LBVH, NarrowPhase, Energy, PCG, CCD) inherits from this class
 * - Protected members of NewtonSolver are accessible through inheritance
 * - Each subclass calls setUp() once, then runs its module-specific tests
 * - No cross-test pollution: each test method gets a fresh solver instance
 */
#pragma once

#include "SimulationSolver/newton_solver.h"
#include "SimulationCore/scene_params.h"
#include "Initializer/init_mesh_data.h"
#include "Core/float_n.h"
#include "test_framework.h"

namespace lcs::test
{

class TestNewtonSolverBase : public NewtonSolver
{
public:
	TestNewtonSolverBase() = default;
	~TestNewtonSolverBase() = default;

	// =============================================================================
	// Device Setup
	// =============================================================================

	void init_device(const std::string& backend = "")
	{
#if defined(__APPLE__)
		std::string use_backend = backend.empty() ? "metal" : backend;
#else
		std::string use_backend = backend.empty() ? "cuda" : backend;
#endif
		create_device("", use_backend);
	}

	// =============================================================================
	// Scene Setup Utilities
	// =============================================================================

protected:
	// Setup a simple cloth scene (single object, free fall)
	void setup_cloth_scene(int grid_size = 3, float spacing = 0.1f)
	{
		Initializer::WorldData world;
		world.set_name("test_cloth");

		std::vector<std::array<float, 3>> vertices;
		std::vector<std::array<uint, 3>> faces;
		generate_cloth_grid(grid_size, vertices, faces, spacing);
		world.load_mesh_from_array(vertices, faces);
		world.set_material_type(Material::MaterialType::Cloth);
		Material::ClothMaterial cloth_mat;
		cloth_mat.thickness = 0.001f;
		cloth_mat.youngs_modulus = 1e6f;
		cloth_mat.area_bending_stiffness = 5e-3f;
		world.set_physics_material(cloth_mat);
		register_world_data(world);

		auto& params = get_config();
		params.implicit_dt = 1.0f / 60.0f;
		params.num_substep = 1;
		params.nonlinear_iter_count = 3;
		params.pcg_iter_count = 50;
		params.gravity = luisa::make_float3(0.0f, -9.8f, 0.0f);
		params.damping_rate = 0.0f;
		params.stiffness_collision = 1e6f;
		params.d_hat = 1e-3f;
		params.use_floor = false;
		params.use_self_collision = false;
		params.use_ccd_linesearch = false;
		params.use_energy_linesearch = false;
		params.simulate_cloth = true;
	}

	// Setup a tetrahedral soft body scene
	void setup_soft_body_scene()
	{
		Initializer::WorldData world;
		world.set_name("test_tet");

		std::vector<std::array<float, 3>> vertices;
		std::vector<std::array<uint, 4>> tets;
		generate_tetrahedron(vertices, tets);

		world.load_tet_mesh_from_array(vertices, tets);
		world.set_physics_material_tet(1e5f, 0.3f, Material::ConstitutiveModelTet::StableNeoHookean);
		register_world_data(world);

		auto& params = get_config();
		params.implicit_dt = 1.0f / 60.0f;
		params.num_substep = 1;
		params.nonlinear_iter_count = 3;
		params.pcg_iter_count = 50;
		params.gravity = luisa::make_float3(0.0f, -9.8f, 0.0f);
		params.damping_rate = 0.0f;
		params.use_floor = false;
		params.use_self_collision = false;
		params.simulate_tet = true;
	}

	// Setup two objects separated by a gap (for collision detection)
	void setup_collision_gap_scene(float gap = 0.05f)
	{
		{
			Initializer::WorldData world;
			world.set_name("object_bottom");
			world.load_mesh_from_array(
				{{0.0f, 0.0f, 0.0f}, {0.1f, 0.0f, 0.0f}, {0.05f, 0.1f, 0.0f}},
				{{0u, 1u, 2u}});
			world.set_material_type(Material::MaterialType::Cloth);
			Material::ClothMaterial m;
			m.thickness = 0.001f;
			m.youngs_modulus = 1e6f;
			world.set_physics_material(m);
			register_world_data(world);
		}
		{
			Initializer::WorldData world;
			world.set_name("object_top");
			world.load_mesh_from_array(
				{{0.0f, gap, 0.0f}, {0.1f, gap, 0.0f}, {0.05f, gap + 0.1f, 0.0f}},
				{{0u, 1u, 2u}});
			world.set_material_type(Material::MaterialType::Cloth);
			Material::ClothMaterial m;
			m.thickness = 0.001f;
			m.youngs_modulus = 1e6f;
			world.set_physics_material(m);
			register_world_data(world);
		}

		auto& params = get_config();
		params.implicit_dt = 1.0f / 60.0f;
		params.num_substep = 1;
		params.nonlinear_iter_count = 3;
		params.pcg_iter_count = 50;
		params.gravity = luisa::make_float3(0.0f, 0.0f, 0.0f);
		params.damping_rate = 0.0f;
		params.use_floor = false;
		params.use_self_collision = true;
		params.use_ccd_linesearch = false;
		params.use_energy_linesearch = false;
		params.simulate_cloth = true;
	}

	// Setup two objects that will definitely collide
	void setup_collision_penetration_scene()
	{
		{
			Initializer::WorldData world;
			world.set_name("object_a");
			world.load_mesh_from_array(
				{{0.0f, 0.0f, 0.0f}, {0.1f, 0.0f, 0.0f}, {0.05f, 0.1f, 0.0f}},
				{{0u, 1u, 2u}});
			world.set_material_type(Material::MaterialType::Cloth);
			Material::ClothMaterial m;
			m.thickness = 0.001f;
			m.youngs_modulus = 1e6f;
			world.set_physics_material(m);
			register_world_data(world);
		}
		{
			Initializer::WorldData world;
			world.set_name("object_b");
			// Overlapping in Y by 0.05
			world.load_mesh_from_array(
				{{0.0f, 0.03f, 0.0f}, {0.1f, 0.03f, 0.0f}, {0.05f, 0.08f, 0.0f}},
				{{0u, 1u, 2u}});
			world.set_material_type(Material::MaterialType::Cloth);
			Material::ClothMaterial m;
			m.thickness = 0.001f;
			m.youngs_modulus = 1e6f;
			world.set_physics_material(m);
			register_world_data(world);
		}

		auto& params = get_config();
		params.implicit_dt = 1.0f / 60.0f;
		params.num_substep = 1;
		params.nonlinear_iter_count = 3;
		params.pcg_iter_count = 50;
		params.gravity = luisa::make_float3(0.0f, 0.0f, 0.0f);
		params.damping_rate = 0.0f;
		params.use_floor = false;
		params.use_self_collision = true;
		params.use_ccd_linesearch = false;
		params.use_energy_linesearch = false;
		params.simulate_cloth = true;
	}

	// Setup a fixed-point stretch test
	void setup_stretch_scene()
	{
		Initializer::WorldData world;
		world.set_name("stretch_cloth");
		std::vector<std::array<float, 3>> vertices;
		std::vector<std::array<uint, 3>> faces;
		generate_cloth_grid(2, vertices, faces, 0.1f);
		world.load_mesh_from_array(vertices, faces);
		world.set_material_type(Material::MaterialType::Cloth);
		Material::ClothMaterial m;
		m.thickness = 0.001f;
		m.youngs_modulus = 1e5f;
		m.area_bending_stiffness = 0.0f;
		world.set_physics_material(m);
		// Fix left edge
		world.add_fixed_point_from_indices({0, 2});
		register_world_data(world);

		auto& params = get_config();
		params.implicit_dt = 1.0f / 60.0f;
		params.num_substep = 1;
		params.nonlinear_iter_count = 5;
		params.pcg_iter_count = 50;
		params.gravity = luisa::make_float3(0.0f, 0.0f, 0.0f);
		params.damping_rate = 0.0f;
		params.use_floor = false;
		params.use_self_collision = false;
		params.simulate_cloth = true;
	}

	void upload_host_positions_to_device()
	{
		auto& s = stream();
		s << sim_data->sa_x_step_start.copy_from(host_sim_data->sa_x_step_start.data())
		  << sim_data->sa_x.copy_from(host_sim_data->sa_x.data())
		  << luisa::compute::synchronize();
	}

	void refresh_face_lbvh_host_state()
	{
		auto& s = stream();
		s << lbvh_data_face->sa_parrent.copy_to(lbvh_data_face->host_parrent.data())
		  << lbvh_data_face->sa_children.copy_to(lbvh_data_face->host_children.data())
		  << lbvh_data_face->sa_node_aabb_v2.copy_to(lbvh_data_face->host_node_aabb_v2.data())
		  << lbvh_data_face->sa_is_healthy.copy_to(lbvh_data_face->host_is_healthy.data())
		  << luisa::compute::synchronize();
	}

	void construct_face_lbvh_at_current_state()
	{
		host_sim_data->sa_x_step_start = host_sim_data->sa_x_outer;
		host_sim_data->sa_x = host_sim_data->sa_x_outer;
		upload_host_positions_to_device();
		lbvh_face->reduce_face_tree_aabb(stream(), sim_data->sa_x_step_start, sim_data->sa_contact_active_faces);
		lbvh_face->construct_tree(stream());
		lbvh_face->update_face_tree_leave_aabb(stream(),
			sim_data->sa_contact_active_verts_offset,
			sim_data->sa_x_step_start,
			sim_data->sa_x,
			sim_data->sa_contact_active_faces);
		lbvh_face->refit(stream());
		refresh_face_lbvh_host_state();
	}

	void refit_face_lbvh_at_current_state()
	{
		upload_host_positions_to_device();
		lbvh_face->update_face_tree_leave_aabb(stream(),
			sim_data->sa_contact_active_verts_offset,
			sim_data->sa_x_step_start,
			sim_data->sa_x,
			sim_data->sa_contact_active_faces);
		lbvh_face->refit(stream());
		refresh_face_lbvh_host_state();
	}

	// =============================================================================
	// Accessors (mirrors existing TestNewtonSolver in test_newton_solver_integration.cpp)
	// =============================================================================

public:
	SimulationData<std::vector>* get_host_sim_data() { return host_sim_data; }
	MeshData<std::vector>*		 get_host_mesh_data() { return host_mesh_data; }
	CollisionData<std::vector>* get_host_collision_data() { return host_collision_data; }
	CollisionData<luisa::compute::Buffer>* get_device_collision_data() { return collision_data; }

	LbvhData<luisa::compute::Buffer>* get_lbvh_data_face() { return lbvh_data_face; }
	LbvhData<luisa::compute::Buffer>* get_lbvh_data_edge() { return lbvh_data_edge; }

	LBVH*					   get_lbvh_face() { return lbvh_face; }
	LBVH*					   get_lbvh_edge() { return lbvh_edge; }
	NarrowPhasesDetector*	   get_narrow_phase() { return narrow_phase_detector; }
	ConjugateGradientSolver*   get_pcg_solver() { return pcg_solver; }

	// =============================================================================
	// Device / Stream Accessors
	// =============================================================================

	luisa::compute::Device&  device() { return *device_state.device; }
	luisa::compute::Stream&  stream() { return *device_state.stream; }

	// =============================================================================
	// Convenience: Data Buffer Accessors
	// =============================================================================

	std::vector<luisa::float3> get_current_positions()
	{
		auto* sim = get_host_sim_data();
		std::vector<luisa::float3> result;
		if (sim)
			result = sim->sa_q;
		return result;
	}

	std::vector<luisa::float3> get_rest_positions()
	{
		auto* sim = get_host_sim_data();
		std::vector<luisa::float3> result;
		if (sim)
			result = sim->sa_rest_x;
		return result;
	}

	// =============================================================================
	// Diagnostic Utilities
	// =============================================================================

	std::string vec3_str(luisa::float3 v) const
	{
		return std::format("[{:.4f}, {:.4f}, {:.4f}]", v.x, v.y, v.z);
	}

	void print_sim_summary()
	{
		auto* sim = get_host_sim_data();
		auto* mesh = get_host_mesh_data();
		if (!sim || !mesh)
			return;

		std::cout << "    Vertices: " << mesh->num_verts << "\n";
		std::cout << "    Faces: " << mesh->num_faces << "\n";
		std::cout << "    DOF: " << sim->num_dof << "\n";
		std::cout << "    Gravity: " << vec3_str(get_config().gravity) << "\n";
	}

	void print_collision_summary()
	{
		auto* cd = get_host_collision_data();
		if (!cd)
			return;

		std::cout << "    Broadphase VF list size: " << cd->broad_phase_list_vf.size() << "\n";
		std::cout << "    Broadphase EE list size: " << cd->broad_phase_list_ee.size() << "\n";
		std::cout << "    Narrowphase pairs: " << cd->narrow_phase_list.size() << "\n";
		std::cout << "    Contact triplets: " << cd->triplet_data.sa_triplet_info.size() << "\n";
	}
};

// =============================================================================
// Convenience macros for subclass test registration
// =============================================================================

#define REGISTER_TEST(class_name, test_name)          \
	[TestCase{#test_name, [] { return class_name{}.test_name(); }}]

} // namespace lcs::test
