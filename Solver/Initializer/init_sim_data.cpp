#include "Initializer/init_sim_data.h"
#include "CollisionDetector/aabb.h"
#include "Core/affine_position.h"
#include "Core/float_n.h"
#include "Core/float_nxn.h"
#include "Core/lc_to_eigen.h"
#include "Energies/detail/bending_energy.hpp"
#include "Energies/detail/fem_utils.h"
#include "Initializer/init_mesh_data.h"
#include "MeshOperation/mesh_reader.h"
#include "Initializer/initializer_utils.h"
#include "luisa/core/logging.h"
#include "luisa/core/mathematics.h"
#include <cmath>
#include <limits>
#include <span>
#include <type_traits>

namespace lcs::Initializer
{
	using namespace Material;

	template <size_t N, typename TypeOffset, size_t NumOffDiag = N * (N - 1)>
	std::array<TypeOffset, N*(N - 1)> get_offsets_in_adjlist_from_adjacent_list(
		const std::vector<std::vector<uint>>& vert_adj_verts,
		const std::vector<uint>&			  vert_adj_verts_csr,
		const auto&							  element)
	{
		std::array<TypeOffset, N*(N - 1)> offsets = { 0 };
		uint							  idx = 0;
		for (uint ii = 0; ii < N; ii++)
		{
			const uint vid = element[ii];
			for (uint jj = 0; jj < N; jj++)
			{
				if (ii != jj)
				{
					const uint adj_vid = element[jj];
					// const uint				 row = min_scalar(vid, adj_vid);
					// const uint				 col = max_scalar(vid, adj_vid);
					const std::vector<uint>& adj_list = vert_adj_verts[vid];
					auto					 it = std::find(adj_list.begin(), adj_list.end(), adj_vid);
					if (it == adj_list.end())
					{
						LUISA_ERROR("Offset in adjlist not found! pair = ({}, {}), adj_list of {} = {}",
							vid,
							adj_vid,
							vid,
							adj_list);
					}
					const uint local_offset = static_cast<uint>(std::distance(adj_list.begin(), it));
					offsets[idx] = TypeOffset(vert_adj_verts_csr[vid] + local_offset);
					idx += 1;
				}
			}
		}
		return offsets;
	}

	static void compute_trimesh_dyadic_mass(const std::vector<float3>& pos_view,
		const std::vector<uint3>&									   tri_view,
		const uint													   prefix_face_start,
		const uint													   prefix_face_end,
		float														   rho,
		float&														   m,
		float3&														   m_x_bar,
		float3x3&													   m_x_bar_x_bar)
	{
		m = 0.0;
		m_x_bar = float3(0.0f);
		m_x_bar_x_bar = float3x3::fill(0.0f);

		// Using Divergence theorem to compute the dyadic mass
		// by integrating on the surface of the trimesh
		// for(auto&& [i, F] : (tri_view))

		for (size_t fid = prefix_face_start; fid < prefix_face_end; fid++)
		{
			const auto& F = tri_view[fid];

			const auto& p0 = pos_view[F[0]];
			const auto& p1 = pos_view[F[1]];
			const auto& p2 = pos_view[F[2]];

			auto e1 = p1 - p0;
			auto e2 = p2 - p0;

			auto N = luisa::cross(e1, e2); // e1.cross(e2);

			m += rho * luisa::dot(p0, N) / 6.0;

			{
				auto Q = [](const uint a, const float rho, const float3& N, const float3& p0, const float3& p1, const float3& p2)
				{
					float V = 0.0;

					V += p0[a] * p0[a] / 12;
					V += p0[a] * p1[a] / 12;
					V += p0[a] * p2[a] / 12;

					V += p1[a] * p1[a] / 12;
					V += p1[a] * p2[a] / 12;
					V += p2[a] * p2[a] / 12;

					return rho / 2 * N[a] * V;
				};

				for (uint a = 0; a < 3; a++)
				{
					m_x_bar[a] += Q(a, rho, N, p0, p1, p2);
				}
			}

			{
				auto Q = [](const uint a, float rho, const float3& N, const float3& p0, const float3& p1, const float3& p2)
				{
					float V = 0.0;

					float p0a_2 = p0[a] * p0[a];
					float p1a_2 = p1[a] * p1[a];
					float p2a_2 = p2[a] * p2[a];

					float p0a_3 = p0a_2 * p0[a];
					float p1a_3 = p1a_2 * p1[a];
					float p2a_3 = p2a_2 * p2[a];

					V += p0a_3 / 20;
					V += p0a_2 * p1[a] / 20;
					V += p0a_2 * p2[a] / 20;

					V += p0[a] * p1a_2 / 20;
					V += p0[a] * p1[a] * p2[a] / 20;
					V += p0[a] * p2a_2 / 20;

					V += p1a_3 / 20;
					V += p1a_2 * p2[a] / 20;
					V += p1[a] * p2a_2 / 20;

					V += p2a_3 / 20;

					return rho / 3 * N[a] * V;
				};

				for (uint j = 0; j < 3; j++) // diagonal
					m_x_bar_x_bar[j][j] += Q(j, rho, N, p0, p1, p2);
			}

			{
				auto Q = [](const uint a, const uint b, float rho, const float3& N, const float3& p0, const float3& p1, const float3& p2)
				{
					float V = 0.0;

					float p0a_2 = p0[a] * p0[a];
					float p1a_2 = p1[a] * p1[a];
					float p2a_2 = p2[a] * p2[a];

					V += p0a_2 * p0[b] / 20;
					V += p0a_2 * p1[b] / 60;
					V += p0a_2 * p2[b] / 60;
					V += p0[a] * p0[b] * p1[a] / 30;

					V += p0[a] * p0[b] * p2[a] / 30;
					V += p0[a] * p1[a] * p1[b] / 30;
					V += p0[a] * p1[a] * p2[b] / 60;
					V += p0[a] * p1[b] * p2[a] / 60;

					V += p0[a] * p2[a] * p2[b] / 30;
					V += p0[b] * p1a_2 / 60;
					V += p0[b] * p1[a] * p2[a] / 60;
					V += p0[b] * p2a_2 / 60;

					V += p1a_2 * p1[b] / 20;
					V += p1a_2 * p2[b] / 60;
					V += p1[a] * p1[b] * p2[a] / 30;
					V += p1[a] * p2[a] * p2[b] / 30;

					V += p1[b] * p2a_2 / 60;
					V += p2a_2 * p2[b] / 20;

					return rho / 2 * N[a] * V;
				};

				m_x_bar_x_bar[0][1] += Q(0, 1, rho, N, p0, p1, p2);
				m_x_bar_x_bar[0][2] += Q(0, 2, rho, N, p0, p1, p2);
				m_x_bar_x_bar[1][2] += Q(1, 2, rho, N, p0, p1, p2);
			}

			// symmetric
			m_x_bar_x_bar[1][0] = m_x_bar_x_bar[0][1];
			m_x_bar_x_bar[2][0] = m_x_bar_x_bar[0][2];
			m_x_bar_x_bar[2][1] = m_x_bar_x_bar[1][2];
		}

		// m_x_bar_x_bar = luisa::transpose(m_x_bar_x_bar);
	}

	static void insert_adj_vert(std::vector<std::vector<uint>>& adj_map, const uint& vid1, const uint& vid2)
	{
		if (vid1 == vid2)
			std::cerr << "Try to build connection with self vertex";
		// if (vid1 < vid2) // Only store upper-triangular part of the adjacency to avoid duplication, since the adjacency is symmetric
		{
			auto& inner_list = adj_map[vid1];
			auto  find_result = std::find(inner_list.begin(), inner_list.end(), vid2);
			if (find_result == inner_list.end())
			{
				inner_list.push_back(vid2);
			}
		}
	};

	template <typename Func>
	void traverse_indices_func_uint(const uint& idx, Func func)
	{
		func(0u, idx);
	}

	template <typename Element, size_t N, typename Func>
	void traverse_indices_func_array(const Element& indices, Func func)
	{
		for (size_t i = 0; i < N; i++)
		{
			func(static_cast<uint>(i), indices[i]);
		}
	}

	template <typename Derived, typename FuncTraverse>
	static void build_adj_list_and_init_grad_hess(std::vector<std::vector<uint>>& adj_map,
		Constitutions::ConstitutionInterface<std::vector, Derived>&				  constitution_template,
		FuncTraverse															  traverse_func)
	{
		constexpr size_t N = Derived::get_num_verts_per_constaint();
		constexpr size_t grad_size = N;
		constexpr size_t hess_size = N * N;

		if (!constitution_template.is_valid())
		{
			// LUISA_INFO("Init constraint {:12}: numElement = {}, stride = {}", Derived::get_constitution_name(), 0, N);
		}
		else
		{
			const uint num_dof = adj_map.size();
			const uint num_constraint = constitution_template.get_num_indices();

			LUISA_INFO("     Init energy {:12}: stride = {}, numElement = {}", Derived::get_constitution_name(), N, num_constraint);

			const auto& sa_constitution_elements = constitution_template.get_indices();
			auto&		gradient = constitution_template.get_constraint_gradients();
			auto&		hessian = constitution_template.get_constraint_hessians();
			auto&		vert_adj_constraints = constitution_template.get_vert_adj_constraints();
			auto&		vert_adj_constraints_csr = constitution_template.get_vert_adj_constraints_csr();

			gradient.resize(num_constraint * grad_size);
			hessian.resize(num_constraint * hess_size);

			vert_adj_constraints.resize(num_dof);
			for (uint eid = 0; eid < num_constraint; eid++)
			{
				auto element = sa_constitution_elements[eid];
				traverse_func(element,
					[&](const uint ii, const uint vid)
					{ vert_adj_constraints[vid].push_back(eid); });
				if constexpr (N != 1)
				{
					traverse_func(element,
						[&](const uint ii, const uint vid)
						{
							traverse_func(element,
								[&](const uint jj, const uint adj_vid)
								{
									if (ii != jj)
									{
										insert_adj_vert(adj_map, vid, adj_vid);
									}
								});
						});
				}
			}
			upload_2d_csr_from(vert_adj_constraints_csr, vert_adj_constraints);
		}
	};

	// Overload without explicit traverse function: choose appropriate traverser automatically.
	template <typename Derived>
	static void build_adj_list_and_init_grad_hess(std::vector<std::vector<uint>>& adj_map,
		Constitutions::ConstitutionInterface<std::vector, Derived>&				  constitution_template)
	{
		constexpr size_t N = Derived::get_num_verts_per_constaint();

		auto auto_traverse = [](auto const& element, auto func)
		{
			using ElemT = std::decay_t<decltype(element)>;
			if constexpr (std::is_integral_v<ElemT>)
			{
				traverse_indices_func_uint(element, func);
			}
			else
			{
				traverse_indices_func_array<ElemT, N>(element, func);
			}
		};

		build_adj_list_and_init_grad_hess(adj_map, constitution_template, auto_traverse);
	}

	template <typename Derived>
	static void init_constitution_offsets_in_adjlist(const std::vector<std::vector<uint>>& adj_map,
		const std::vector<uint>&														   adj_map_csr,
		Constitutions::ConstitutionInterface<std::vector, Derived>&						   constitution_template)
	{
		constexpr size_t N = Derived::get_num_verts_per_constaint();
		constexpr size_t num_offdiag = N * (N - 1);

		const auto& sa_constitution_elements = constitution_template.get_indices();
		auto&		sa_constitution_offsets_in_adjlist = constitution_template.constraint_offsets_in_adjlist;

		if constexpr (N != 1)
		{
			sa_constitution_offsets_in_adjlist.resize(sa_constitution_elements.size() * num_offdiag);
			CpuParallel::parallel_for(0,
				sa_constitution_elements.size(),
				[&](const uint eid)
				{
					auto element = sa_constitution_elements[eid];
					auto mask =
						get_offsets_in_adjlist_from_adjacent_list<N, uint>(adj_map, adj_map_csr, element); // size = N*(N-1)

					if constexpr (true)
					{
						// Validate that each stored slot maps to the expected absolute triplet index.
						uint slot_idx = 0;
						for (uint ii = 0; ii < N; ii++)
						{
							for (uint jj = 0; jj < N; jj++)
							{
								if (ii == jj)
								{
									continue;
								}

								const uint vid = element[ii];
								const uint adj_vid = element[jj];
								// const uint row = min_scalar(vid, adj_vid);
								// const uint col = max_scalar(vid, adj_vid);

								const auto& row_adj = adj_map[vid];
								auto		row_it = std::find(row_adj.begin(), row_adj.end(), adj_vid);
								if (row_it == row_adj.end())
								{
									LUISA_ERROR("Error in init offset map: row {} cannot find adjacent col {} (eid = {}, ii = {}, jj = {})",
										vid,
										adj_vid,
										eid,
										ii,
										jj);
								}
								const uint expected_idx =
									adj_map_csr[vid] + static_cast<uint>(std::distance(row_adj.begin(), row_it));
								const uint actual_idx = mask[slot_idx];
								if (actual_idx != expected_idx)
								{
									LUISA_ERROR("Error in init offset map: mismatch at eid = {}, slot = {}, pair ({}, {}) expected triplet_idx {}, got {}",
										eid,
										slot_idx,
										vid,
										adj_vid,
										expected_idx,
										actual_idx);
								}
								slot_idx += 1;
							}
						}
					}

					std::memcpy(sa_constitution_offsets_in_adjlist.data() + eid * num_offdiag,
						mask.data(),
						sizeof(uint) * num_offdiag);
				});
		}
		else
		{
			sa_constitution_offsets_in_adjlist.resize(1, 0);
		}
	}

	void init_sim_data(const std::vector<lcs::Initializer::WorldData>& world_data,
		lcs::MeshData<std::vector>*									   mesh_data,
		lcs::SimulationData<std::vector>*							   sim_data,
		const std::vector<lcs::FixedJointConstraintDesc>&			   fixed_joint_descs,
		const std::vector<lcs::PrismaticJointConstraintDesc>&		   prismatic_joint_descs,
		const std::vector<lcs::RevoluteJointConstraintDesc>&		   revolute_joint_descs)
	{
		// Calculate number of energy element
		constexpr bool cull_unused_constraints = true;

		std::vector<uint> stretch_spring_indices = fn_get_active_indices(
			[&](const uint eid)
			{
				const uint	mesh_idx = mesh_data->sa_edge_mesh_id[eid];
				const auto& mesh_info = world_data[mesh_idx];
				bool		use_spring = false;
				if (mesh_info.holds<ClothMaterial>())
				{
					use_spring = mesh_info.get_material<ClothMaterial>().stretch_model
						== ConstitutiveStretchModelCloth::Spring;
				}
				else if (mesh_info.holds<TetMaterial>())
				{
					use_spring = mesh_info.get_material<TetMaterial>().model == ConstitutiveModelTet::Spring;
				}
				// else if (shell_info.holds<RigidMaterial>())
				// {
				//     use_spring = shell_info.get<RigidMaterial>().model == ConstitutiveModelRigid::Spring;
				// }
				else if (mesh_info.holds<RodMaterial>())
				{
					use_spring = mesh_info.get_material<RodMaterial>().model == ConstitutiveModelRod::Spring;
				}
				// bool  is_cloth   = mesh_data->sa_vert_mesh_type[edge[0]] == uint(ShellTypeCloth);
				uint2 edge = mesh_data->sa_edges[eid];
				bool  is_dynamic = cull_unused_constraints ? !mesh_data->sa_is_fixed[edge[0]] || !mesh_data->sa_is_fixed[edge[1]] : true;
				return (use_spring && is_dynamic) ? 1 : 0;
			},
			mesh_data->num_edges);

		std::vector<uint> stretch_face_indices = fn_get_active_indices(
			[&](const uint fid)
			{
				const uint	mesh_idx = mesh_data->sa_face_mesh_id[fid];
				const auto& mesh_info = world_data[mesh_idx];
				bool		use_stretch_face = mesh_info.holds<ClothMaterial>()
					&& mesh_info.get_material<ClothMaterial>().stretch_model
						== ConstitutiveStretchModelCloth::FEM_BW98;
				uint3 face = mesh_data->sa_faces[fid];
				bool  is_dynamic = cull_unused_constraints ? !mesh_data->sa_is_fixed[face[0]] || !mesh_data->sa_is_fixed[face[1]]
						 || !mesh_data->sa_is_fixed[face[2]]
														   : true;
				return (use_stretch_face && is_dynamic) ? 1 : 0;
			},
			mesh_data->num_faces);

		std::vector<uint> bending_edge_indices = fn_get_active_indices(
			[&](const uint eid)
			{
				const uint	mesh_idx = mesh_data->sa_dihedral_edge_mesh_id[eid];
				const auto& mesh_info = world_data[mesh_idx];
				bool		use_bending = mesh_info.holds<ClothMaterial>()
					&& mesh_info.get_material<ClothMaterial>().bending_model
						!= ConstitutiveBendingModelCloth::Empty;
				uint4 edge = mesh_data->sa_dihedral_edges[eid];
				bool  is_dynamic = cull_unused_constraints ? !mesh_data->sa_is_fixed[edge[0]] || !mesh_data->sa_is_fixed[edge[1]]
						 || !mesh_data->sa_is_fixed[edge[2]] || !mesh_data->sa_is_fixed[edge[3]]
														   : true;
				return (use_bending && is_dynamic) ? 1 : 0;
			},
			mesh_data->num_dihedral_edges);

		std::vector<uint> stress_tet_indices = fn_get_active_indices(
			[&](const uint tid)
			{
				const uint	mesh_idx = mesh_data->sa_tet_mesh_id[tid];
				const auto& mesh_info = world_data[mesh_idx];
				bool		use_stress = false;
				if (mesh_info.holds<TetMaterial>())
				{
					auto model = mesh_info.get_material<TetMaterial>().model;
					use_stress = model == ConstitutiveModelTet::StableNeoHookean
						|| model == ConstitutiveModelTet::ARAP
						|| model == ConstitutiveModelTet::Corotated
						|| model == ConstitutiveModelTet::StVK;
				}
				uint4 tet = mesh_data->sa_tetrahedrons[tid];
				bool  is_dynamic = cull_unused_constraints ? !mesh_data->sa_is_fixed[tet[0]] || !mesh_data->sa_is_fixed[tet[1]]
						 || !mesh_data->sa_is_fixed[tet[2]] || !mesh_data->sa_is_fixed[tet[3]]
														   : true;
				return (use_stress && is_dynamic) ? 1 : 0;
			},
			mesh_data->num_tets);

		std::vector<uint> affine_body_indices = fn_get_active_indices(
			[&](const uint meshIdx)
			{
				const uint curr_prefix = mesh_data->prefix_num_verts[meshIdx];
				const uint first_vid = curr_prefix;
				const bool has_boundary_edge = // Unclosed
					(mesh_data->prefix_num_edges[meshIdx + 1] - mesh_data->prefix_num_edges[meshIdx])
					!= (mesh_data->prefix_num_dihedral_edges[meshIdx + 1] - mesh_data->prefix_num_dihedral_edges[meshIdx]);
				// bool has_dynamic_vert = mesh_data->sa_is_fixed[first_vid];
				// bool is_rigid = (mesh_data->sa_vert_mesh_type[first_vid] == uint(ShellTypeRigid));  // ;&& !has_boundary_edge;
				bool is_rigid = world_data[meshIdx].holds<RigidMaterial>();
				// LUISA_INFO("Mesh {} is rigid = {}, has_boundary_edge = {}", meshIdx, is_rigid, has_boundary_edge);
				return (is_rigid) ? 1 : 0; // has_dynamic_vert
			},
			mesh_data->num_meshes);

		std::vector<uint> soft_vert_indices =
			fn_get_active_indices([&](const uint vid)
				{ return mesh_data->sa_vert_mesh_type[vid] == uint(MaterialType::Rigid) ? 0 : 1; },
				mesh_data->num_verts);

		const uint num_stretch_springs = static_cast<uint>(stretch_spring_indices.size());
		const uint num_stretch_faces = static_cast<uint>(stretch_face_indices.size());
		const uint num_bending_edges = static_cast<uint>(bending_edge_indices.size());
		const uint num_stress_tets = static_cast<uint>(stress_tet_indices.size());
		const uint num_affine_bodies = static_cast<uint>(affine_body_indices.size());
		const uint num_verts_soft = static_cast<uint>(soft_vert_indices.size());
		const uint num_verts_rigid = mesh_data->num_verts - num_verts_soft;
		const uint num_verts_total = mesh_data->num_verts;
		const uint num_dof = num_verts_soft + num_affine_bodies * 4; // 12 DOF per affine body (4 * xyz)

		LUISA_INFO("Initialized energy element counts:");
		LUISA_INFO("      num Stretch Spring = {} (<{})", num_stretch_springs, mesh_data->num_edges);
		LUISA_INFO("      num Stretch Face   = {} (<{})", num_stretch_faces, mesh_data->num_faces);
		LUISA_INFO("      num Bending Edge   = {} (<{})", num_bending_edges, mesh_data->num_dihedral_edges);
		LUISA_INFO("      num Stress Tet     = {} (<{})", num_stress_tets, mesh_data->num_tets);
		LUISA_INFO("      num Affine Body    = {} (<{})", num_affine_bodies, mesh_data->num_meshes);
		LUISA_INFO("      Total DOF   = {}, NumVertSoft = {}, NumAffineBodies = {}", num_dof, num_verts_soft, num_affine_bodies);
		LUISA_INFO("      Total Verts = {}, NumVertSoft = {}, NumVertRigid    = {}", num_dof, num_verts_soft, num_verts_rigid);

		sim_data->num_verts_total = num_verts_total;
		sim_data->num_verts_soft = num_verts_soft;
		sim_data->num_verts_rigid = mesh_data->num_verts - num_verts_soft;
		sim_data->num_affine_bodies = num_affine_bodies;
		sim_data->num_dof = num_dof;
		sim_data->sa_num_dof.resize(1);
		sim_data->sa_num_dof[0] = num_dof;

		// Resize state buffers
		{
			sim_data->sa_rest_q.resize(num_dof);	   // Constant
			sim_data->sa_rest_q_v.resize(num_dof);	   // Constant
			sim_data->sa_q.resize(num_dof);			   // Re-calculate every frame
			sim_data->sa_dq.resize(num_dof);		   // Re-calculate every frame
			sim_data->sa_q_v.resize(num_dof);		   // Re-calculate every frame
			sim_data->sa_q_iter_start.resize(num_dof); // Re-calculate every frame
			sim_data->sa_q_step_start.resize(num_dof); // Re-calculate every frame
			sim_data->sa_q_outer.resize(num_dof);	   // Input from outer, or reset by rest state
			sim_data->sa_q_v_outer.resize(num_dof);	   // Input from outer, or reset by rest state
			sim_data->sa_q_is_fixed.resize(num_dof);
			sim_data->sa_q_property.resize(num_dof);
			sim_data->sa_q_tilde.resize(num_dof);
		}

		// Resize position/velocity buffer
		{
			sim_data->sa_scaled_model_x.resize(num_verts_total); // Constant
			sim_data->sa_rest_x.resize(num_verts_total);		 // Constant
			sim_data->sa_rest_v.resize(num_verts_total);		 // Constant
			sim_data->sa_x.resize(num_verts_total);				 // Re-calculate every frame
			sim_data->sa_dx.resize(num_verts_total);			 // Re-calculate every frame
			sim_data->sa_v.resize(num_verts_total);				 // Re-calculate every frame
			sim_data->sa_x_step_start.resize(num_verts_total);	 // Re-calculate every frame
			sim_data->sa_x_iter_start.resize(num_verts_total);	 // Re-calculate every frame
			sim_data->sa_x_outer.resize(num_verts_total);
			sim_data->sa_v_outer.resize(num_verts_total);
			sim_data->sa_x_to_dof_map.resize(num_verts_total);
			sim_data->sa_x_property.resize(num_verts_total);
			sim_data->sa_vert_affine_bodies_id.resize(num_verts_total, -1u);
		}

		// Init state buffers
		{
			// Soft body rest q
			{
				const uint prefix_vid_soft = 0;
				const uint num_dofs_soft = num_verts_soft;
				auto	   soft_rest_q = std::span(sim_data->sa_rest_q).subspan(prefix_vid_soft, num_dofs_soft);
				auto	   soft_rest_q_v = std::span(sim_data->sa_rest_q_v).subspan(prefix_vid_soft, num_dofs_soft);
				CpuParallel::parallel_for(0,
					num_dofs_soft,
					[&](const uint vid)
					{
						const uint dof_vid = soft_vert_indices[vid];
						soft_rest_q[vid] = mesh_data->sa_rest_x[dof_vid];
						soft_rest_q_v[vid] = mesh_data->sa_rest_v[dof_vid];
					});
			}

			// Rigid body rest q
			{
				const uint prefix_vid_rigid = num_verts_soft;
				const uint num_dofs_rigid = num_affine_bodies * 4;
				auto	   rigid_rest_q = std::span(sim_data->sa_rest_q).subspan(prefix_vid_rigid, num_dofs_rigid);
				auto	   rigid_rest_q_v = std::span(sim_data->sa_rest_q_v).subspan(prefix_vid_rigid, num_dofs_rigid);

				for (uint body_idx = 0; body_idx < num_affine_bodies; body_idx++)
				{
					const uint meshIdx = affine_body_indices[body_idx];

					float3 init_translation = mesh_data->sa_rest_translate[meshIdx];
					float3 init_rotation = mesh_data->sa_rest_rotation[meshIdx];
					// float3 init_scale = mesh_data->sa_rest_scale[meshIdx];
					float3	 init_scale = luisa::make_float3(1.0f); // Since we use |AAT-I|
					float4x4 init_transform_matrix = lcs::make_model_matrix(init_translation, init_rotation, init_scale);
					float4x3 rest_q = AffineBodyDynamics::extract_q_from_affine_matrix(init_transform_matrix);
					rigid_rest_q[4 * body_idx + 0] = rest_q[0]; // = init_transform_matrix[0].xyz()
					rigid_rest_q[4 * body_idx + 1] = rest_q[1]; // = init_transform_matrix[1].xyz()
					rigid_rest_q[4 * body_idx + 2] = rest_q[2]; // = init_transform_matrix[2].xyz()
					rigid_rest_q[4 * body_idx + 3] = rest_q[3]; // = init_transform_matrix[3].xyz()
					// LUISA_INFO("Affine Body {} Rest q = \n{},\n{},\n{},\n{}",
					//            body_idx,
					//            rest_q[0],
					//            rest_q[1],
					//            rest_q[2],
					//            rest_q[3]);
					rigid_rest_q_v[4 * body_idx + 0] = Zero3;
					rigid_rest_q_v[4 * body_idx + 1] = Zero3;
					rigid_rest_q_v[4 * body_idx + 2] = Zero3;
					rigid_rest_q_v[4 * body_idx + 3] = Zero3;
					// LUISA_INFO("Affine Body {} Rest q = {}", body_idx, rest_q);
				}
			}

			// Init position/velocity
			{
				CpuParallel::parallel_copy(mesh_data->sa_rest_x, sim_data->sa_rest_x);
				CpuParallel::parallel_copy(mesh_data->sa_rest_v, sim_data->sa_rest_v);
				CpuParallel::parallel_copy(mesh_data->sa_scaled_model_x, sim_data->sa_scaled_model_x);
				CpuParallel::parallel_set(sim_data->sa_x_property, VertexProperty());
				CpuParallel::parallel_set(sim_data->sa_q_property, DofProperty());

				// uint attribute_info = 0;
				// bool is_fixed() const { return (attribute_info & 0x1) != 0; }
				// bool is_rigid_body() const { return (attribute_info & 0x8) != 0; }
				// bool is_self_collision_disabled() const { return (attribute_info & 0x200) != 0; }
				// bool is_ccd_disabled() const { return (attribute_info & 0x400) != 0; }
				// bool is_friction_disabled() const { return (attribute_info & 0x800) != 0; }
				// bool is_gravity_disabled() const { return (attribute_info & 0x1000) != 0; }
				// uint get_object_id() const { return (attribute_info >> 16) & 0x7FFF; }

				// Soft body vertices map to dof
				CpuParallel::parallel_for(0,
					num_verts_soft,
					[&](const uint vid)
					{
						const uint mesh_idx = mesh_data->sa_vert_mesh_id[vid];
						sim_data->sa_x_property[vid].set_object_id(mesh_idx);
						sim_data->sa_x_to_dof_map[vid].set_as_soft_body(vid);
						sim_data->sa_q_property[vid].set_is_soft();
						const bool is_fixed = mesh_data->sa_is_fixed[vid];
						sim_data->sa_q_is_fixed[vid] = is_fixed;
						if (is_fixed)
						{
							sim_data->sa_x_property[vid].set_is_fixed();
							sim_data->sa_q_property[vid].set_is_fixed();
						}
					});

				// Rigid body vertices map to dof
				for (uint body_idx = 0; body_idx < num_affine_bodies; body_idx++)
				{
					const uint dof_idx = num_verts_soft + body_idx * 4;
					const uint meshIdx = affine_body_indices[body_idx];
					const uint prefix_vid = mesh_data->prefix_num_verts[meshIdx];
					const uint suffix_vid = mesh_data->prefix_num_verts[meshIdx + 1];

					bool has_fixed_vert = std::any_of(mesh_data->sa_is_fixed.begin() + prefix_vid,
						mesh_data->sa_is_fixed.begin() + suffix_vid,
						[](const uint is_fixed)
						{ return is_fixed; });
					for (uint vid = prefix_vid; vid < suffix_vid; vid++)
					{
						sim_data->sa_x_property[vid].set_is_rigid_body();
						sim_data->sa_x_property[vid].set_object_id(meshIdx);
						sim_data->sa_x_to_dof_map[vid].set_as_rigid_body(dof_idx);
						sim_data->sa_vert_affine_bodies_id[vid] = body_idx;
						if (has_fixed_vert)
						{
							sim_data->sa_x_property[vid].set_is_fixed();
						}
					}
					sim_data->sa_q_is_fixed[dof_idx + 0] = has_fixed_vert;
					sim_data->sa_q_is_fixed[dof_idx + 1] = has_fixed_vert;
					sim_data->sa_q_is_fixed[dof_idx + 2] = has_fixed_vert;
					sim_data->sa_q_is_fixed[dof_idx + 3] = has_fixed_vert;
					sim_data->sa_q_property[dof_idx + 0].set_is_translation_dof();
					sim_data->sa_q_property[dof_idx + 0].set_is_rigid();
					sim_data->sa_q_property[dof_idx + 1].set_is_rigid();
					sim_data->sa_q_property[dof_idx + 2].set_is_rigid();
					sim_data->sa_q_property[dof_idx + 3].set_is_rigid();
					if (has_fixed_vert)
					{
						sim_data->sa_q_property[dof_idx + 0].set_is_fixed();
						sim_data->sa_q_property[dof_idx + 1].set_is_fixed();
						sim_data->sa_q_property[dof_idx + 2].set_is_fixed();
						sim_data->sa_q_property[dof_idx + 3].set_is_fixed();
					}
				}
			}
		}

		// Init energy
		{
			sim_data->sa_system_energy.resize(10240);

			// Rest spring length
			auto& stretch_spring_data = sim_data->get_stretch_spring_data();
			stretch_spring_data.constraint_indices.resize(num_stretch_springs);
			stretch_spring_data.sa_stretch_spring_rest_state_length.resize(num_stretch_springs);
			stretch_spring_data.sa_stretch_spring_stiffness.resize(num_stretch_springs);

			CpuParallel::parallel_for(0,
				num_stretch_springs,
				[&](const uint eid)
				{
					const uint orig_eid = stretch_spring_indices[eid];
					uint2	   edge = mesh_data->sa_edges[orig_eid];
					float3	   x1 = mesh_data->sa_rest_x[edge[0]];
					float3	   x2 = mesh_data->sa_rest_x[edge[1]];
					stretch_spring_data.constraint_indices[eid] = edge;
					stretch_spring_data.sa_stretch_spring_rest_state_length[eid] =
						lcs::length_vec(x1 - x2);

					const auto& mesh_info = world_data[mesh_data->sa_edge_mesh_id[orig_eid]];
					float		stiffness = 1.0f;
					if (mesh_info.holds<ClothMaterial>())
					{
						const auto& material = mesh_info.get_material<ClothMaterial>();
						const float E = material.youngs_modulus;
						const float nu = material.poisson_ratio;
						auto [mu, lambda] = FemUtils::convert_lame_params_2d(E, nu);
						stiffness = mu * material.thickness; // scale by thickness
					}
					else if (mesh_info.holds<TetMaterial>())
					{
						const auto& material = mesh_info.get_material<TetMaterial>();
						const float E = material.youngs_modulus;
						const float nu = material.poisson_ratio;
						auto [mu, lambda] = FemUtils::convert_lame_params_3d(E, nu);
						stiffness = mu;
					}
					else if (mesh_info.holds<RodMaterial>())
					{
						stiffness = mesh_info.get_material<RodMaterial>().bending_stiffness;
					}
					stretch_spring_data.sa_stretch_spring_stiffness[eid] = stiffness;
				});

			// Rest stretch face length
			auto& stretch_face_data = sim_data->get_stretch_face_data();
			stretch_face_data.constraint_indices.resize(num_stretch_faces);
			stretch_face_data.sa_stretch_faces_mu_lambda.resize(num_stretch_faces);
			stretch_face_data.sa_stretch_faces_rest_area.resize(num_stretch_faces);
			stretch_face_data.sa_stretch_faces_Dm_inv.resize(num_stretch_faces);
			CpuParallel::parallel_for(0,
				num_stretch_faces,
				[&](const uint fid)
				{
					const uint	  orig_fid = stretch_face_indices[fid];
					uint3		  face = mesh_data->sa_faces[orig_fid];
					const float3  vert_pos[3] = { mesh_data->sa_rest_x[face[0]],
						 mesh_data->sa_rest_x[face[1]],
						 mesh_data->sa_rest_x[face[2]] };
					const float3& x_0 = vert_pos[0];
					const float3& x_1 = vert_pos[1];
					const float3& x_2 = vert_pos[2];

					const float2x2 inv_duv = FemUtils::get_Dm_inv(x_0, x_1, x_2);
					const float	   area = compute_face_area(x_0, x_1, x_2);

					const auto& mesh_info = world_data[mesh_data->sa_face_mesh_id[orig_fid]];
					const auto& material = mesh_info.get_material<ClothMaterial>();

					const float E = material.youngs_modulus;
					const float nu = material.poisson_ratio;

					auto [mu, lambda] = FemUtils::convert_lame_params_2d(E, nu);
					mu = material.thickness * mu; // scale by thickness
					lambda = material.thickness * lambda;
					stretch_face_data.sa_stretch_faces_mu_lambda[fid] =
						luisa::make_float2(mu, lambda);
					stretch_face_data.constraint_indices[fid] = face;
					stretch_face_data.sa_stretch_faces_rest_area[fid] = area;
					stretch_face_data.sa_stretch_faces_Dm_inv[fid] = inv_duv;
				});

			// Rest bending info
			auto& bending_edge_data = sim_data->get_bending_edge_data();
			bending_edge_data.constraint_indices.resize(num_bending_edges);
			bending_edge_data.sa_bending_edges_rest_area.resize(num_bending_edges);
			bending_edge_data.sa_bending_edges_rest_angle.resize(num_bending_edges);
			bending_edge_data.sa_bending_edges_stiffness.resize(num_bending_edges);
			bending_edge_data.sa_bending_edges_Q.resize(num_bending_edges);
			CpuParallel::parallel_for(
				0,
				num_bending_edges,
				[&](const uint eid)
				{
					const uint orig_eid = bending_edge_indices[eid];
					uint4	   edge = mesh_data->sa_dihedral_edges[orig_eid];

					edge = luisa::make_uint4(edge[0], edge[1], edge[2], edge[3]);
					const float3 vert_pos[4] = { mesh_data->sa_rest_x[edge[0]],
						mesh_data->sa_rest_x[edge[1]],
						mesh_data->sa_rest_x[edge[2]],
						mesh_data->sa_rest_x[edge[3]] };

					// Rest state angle
					{
						const float3& x0 = vert_pos[0];
						const float3& x1 = vert_pos[1];
						const float3& x2 = vert_pos[2];
						const float3& x3 = vert_pos[3];

						const float angle = lcs::detail::bending_energy::compute_theta(x0, x1, x2, x3);

						const float A1 = compute_face_area(x0, x1, x2);
						const float A2 = compute_face_area(x0, x1, x3);
						const float L0 = luisa::length(x0 - x1);
						const float h_bar = (A1 + A2) / (3.0f * L0);

						if (luisa::isnan(angle))
							LUISA_ERROR("is nan rest angle {}", eid);

						bending_edge_data.constraint_indices[eid] = edge;
						bending_edge_data.sa_bending_edges_rest_area[eid] = h_bar;
						bending_edge_data.sa_bending_edges_rest_angle[eid] = angle;
						bending_edge_data.sa_bending_edges_stiffness[eid] =
							world_data[mesh_data->sa_dihedral_edge_mesh_id[orig_eid]]
								.get_material<ClothMaterial>()
								.area_bending_stiffness;
					}

					// Rest state Q
					{
						auto calculateCotTheta = [](const float3& x, const float3& y)
						{
							// const float scaled_cos_theta = dot_vec(x, y);
							// const float scaled_sin_theta = (sqrt_scalar(1.0f - square_scalar(scaled_cos_theta)));
							const float scaled_cos_theta = luisa::dot(x, y);
							const float scaled_sin_theta = luisa::length(luisa::cross(x, y));
							return scaled_cos_theta / scaled_sin_theta;
						};

						float3		 e0 = vert_pos[1] - vert_pos[0];
						float3		 e1 = vert_pos[2] - vert_pos[0];
						float3		 e2 = vert_pos[3] - vert_pos[0];
						float3		 e3 = vert_pos[2] - vert_pos[1];
						float3		 e4 = vert_pos[3] - vert_pos[1];
						const float	 cot_01 = calculateCotTheta(e0, -e1);
						const float	 cot_02 = calculateCotTheta(e0, -e2);
						const float	 cot_03 = calculateCotTheta(e0, e3);
						const float	 cot_04 = calculateCotTheta(e0, e4);
						const float4 K =
							luisa::make_float4(cot_03 + cot_04, cot_01 + cot_02, -cot_01 - cot_03, -cot_02 - cot_04);
						const float A_0 = 0.5f * luisa::length(luisa::cross(e0, e1));
						const float A_1 = 0.5f * luisa::length(luisa::cross(e0, e2));
						// if (is_nan_vec<float4>(K) || is_inf_vec<float4>(K)) fast_print_err("Q of Bending is Illigal");
						const float4x4 m_Q = (3.f / (A_0 + A_1)) * lcs::outer_product(K, K); // Q = 3 qq^T / (A0+A1) ==> Q is symmetric
						bending_edge_data.sa_bending_edges_Q[eid] = m_Q;					 // See : A quadratic bending model for inextensible surfaces.
					}
				});

			// Rest tetrahedron info
			auto& stress_tet_data = sim_data->get_stress_tet_data();
			stress_tet_data.constraint_indices.resize(num_stress_tets);
			stress_tet_data.sa_stress_tets_model.resize(num_stress_tets);
			stress_tet_data.sa_stress_tets_rest_volume.resize(num_stress_tets);
			stress_tet_data.sa_stress_tets_mu_lambda.resize(num_stress_tets);
			stress_tet_data.sa_stress_tets_Dm_inv.resize(num_stress_tets);
			CpuParallel::parallel_for(0,
				num_stress_tets,
				[&](const uint tid)
				{
					const uint	 orig_tid = stress_tet_indices[tid];
					uint4		 tet = mesh_data->sa_tetrahedrons[orig_tid];
					const float3 vert_pos[4] = {
						mesh_data->sa_rest_x[tet[0]],
						mesh_data->sa_rest_x[tet[1]],
						mesh_data->sa_rest_x[tet[2]],
						mesh_data->sa_rest_x[tet[3]]
					};
					const float3& x0 = vert_pos[0];
					const float3& x1 = vert_pos[1];
					const float3& x2 = vert_pos[2];
					const float3& x3 = vert_pos[3];
					const float	  volume = compute_tet_volume(x0, x1, x2, x3);

					const float3x3 Dm = luisa::make_float3x3(x1 - x0, x2 - x0, x3 - x0);
					const float3x3 Dm_inv = luisa::inverse(Dm);
					{
						if (luisa::determinant(Dm) <= 0.0f)
						{
							LUISA_ERROR("Tet {} has non-positive volume: {}, rest positions = {}, {}, {}, {}",
								orig_tid,
								volume,
								x0,
								x1,
								x2,
								x3);
						}
					}
					const auto& mesh_info = world_data[mesh_data->sa_tet_mesh_id[orig_tid]];
					const auto& material = mesh_info.get_material<TetMaterial>();
					const auto	model = material.model;
					const float E = material.youngs_modulus;
					const float nu = material.poisson_ratio;
					auto [mu, lambda] = FemUtils::convert_lame_params_3d(E, nu);
					// LUISA_INFO("Tet {}: volume = {},  mu = {}, lambda = {}", orig_tid, volume, mu, lambda);
					stress_tet_data.constraint_indices[tid] = tet;
					stress_tet_data.sa_stress_tets_model[tid] = static_cast<uint>(model);
					stress_tet_data.sa_stress_tets_rest_volume[tid] = volume;
					stress_tet_data.sa_stress_tets_Dm_inv[tid] = Dm_inv;
					stress_tet_data.sa_stress_tets_mu_lambda[tid] =
						luisa::make_float2(mu, lambda);
				});

			const float default_stiffness_dirichlet = 1e5f;

			// Init soft inertia info
			auto& soft_inertia_data = sim_data->get_soft_inertia_data();
			soft_inertia_data.constraint_indices.resize(num_verts_soft);
			soft_inertia_data.sa_soft_vert_mass.resize(num_verts_soft);
			soft_inertia_data.sa_stiffness_dirichlet.resize(num_verts_soft);
			CpuParallel::parallel_for(0,
				num_verts_soft,
				[&](const uint vid)
				{
					const uint orig_vid = soft_vert_indices[vid];
					const bool is_fixed = mesh_data->sa_is_fixed[orig_vid];
					soft_inertia_data.constraint_indices[vid] = vid;
					soft_inertia_data.sa_soft_vert_mass[vid] = mesh_data->sa_vert_mass[orig_vid];
					soft_inertia_data.sa_stiffness_dirichlet[vid] = is_fixed ? default_stiffness_dirichlet : 1.0f;
				});

			// Rest affine body info
			sim_data->sa_affine_bodies_mesh_id.resize(num_affine_bodies);

			auto& abd_ortho_data = sim_data->get_abd_orthogonality_data();
			abd_ortho_data.abd_kappa.resize(num_affine_bodies);
			abd_ortho_data.abd_volume.resize(num_affine_bodies);
			abd_ortho_data.constraint_indices.resize(num_affine_bodies);

			auto& abd_inertia_data = sim_data->get_abd_inertia_data();
			abd_inertia_data.constraint_indices.resize(num_affine_bodies);
			abd_inertia_data.sa_stiffness_dirichlet.resize(num_affine_bodies);
			abd_inertia_data.sa_affine_bodies_mass_matrix.resize(num_affine_bodies);
			abd_inertia_data.sa_affine_bodies_mass_matrix_full.resize(num_affine_bodies);

			for (uint body_idx = 0; body_idx < num_affine_bodies; body_idx++)
			{
				const uint	meshIdx = affine_body_indices[body_idx];
				const auto& mesh_info = world_data[meshIdx];

				sim_data->sa_affine_bodies_mesh_id[body_idx] = meshIdx;

				const uint prefix_dof_abd = num_verts_soft;
				abd_inertia_data.constraint_indices[body_idx] =
					luisa::make_uint4(prefix_dof_abd + 4 * body_idx + 0,
						prefix_dof_abd + 4 * body_idx + 1,
						prefix_dof_abd + 4 * body_idx + 2,
						prefix_dof_abd + 4 * body_idx + 3);
				abd_ortho_data.constraint_indices[body_idx] =
					luisa::make_uint3(prefix_dof_abd + 4 * body_idx + 1, // Only affect rotation & scaling part
						prefix_dof_abd + 4 * body_idx + 2,				 //
						prefix_dof_abd + 4 * body_idx + 3);

				const uint prefix_vid = mesh_data->prefix_num_verts[meshIdx];
				const uint suffix_vid = mesh_data->prefix_num_verts[meshIdx + 1];
				const uint prefix_fid = mesh_data->prefix_num_faces[meshIdx];
				const uint suffix_fid = mesh_data->prefix_num_faces[meshIdx + 1];
				const uint prefix_eid = mesh_data->prefix_num_edges[meshIdx];
				const uint suffix_eid = mesh_data->prefix_num_edges[meshIdx + 1];

				// Init ABD mass matrix
				{
					EigenFloat12x12 body_mass = EigenFloat12x12::Zero();
					float4x4		compressed_mass_matrix;

					float	 M_body = 0.0f;
					float3	 MI_body = luisa::make_float3(0.0f);
					float3x3 I_body = luisa::make_float3x3(0.0f);
					if (mesh_info.get_is_shell())
					{
						// std::vector<float3> virtual_solid_verts((next_prefix_verts - curr_prefix_verts) * 2);
						// std::vector<uint3>  virtual_solid_faces(
						//     (mesh_data->prefix_num_faces[meshIdx + 1] - mesh_data->prefix_num_faces[meshIdx]) * 2);
						// for (uint vid = curr_prefix_verts; vid < next_prefix_verts; vid++)
						// {
						//     float3 vert_pos       = mesh_data->sa_scaled_model_x[vid];
						//     float  half_thickness = 0.5f * mesh_info.get_thickness();
						//     float3 normal         = luisa::make_float3(0, 0, 0);
						//     for (const uint adj_fid : mesh_data->vert_adj_faces[vid])
						//     {
						//         uint3  face = mesh_data->sa_faces[adj_fid];
						//         float3 p0   = mesh_data->sa_scaled_model_x[face.x];
						//         float3 p1   = mesh_data->sa_scaled_model_x[face.y];
						//         float3 p2   = mesh_data->sa_scaled_model_x[face.z];
						//         float  area = compute_face_area(p0, p1, p2);
						//         normal += area * luisa::normalize(luisa::cross(p1 - p0, p2 - p0));
						//     }
						//     normal = luisa::normalize(normal);
						//     virtual_solid_verts[2 * (vid - curr_prefix_verts) + 0] = vert_pos + half_thickness * normal;
						//     virtual_solid_verts[2 * (vid - curr_prefix_verts) + 1] = vert_pos - half_thickness * normal;
						// }
						// for (uint fid = curr_prefix_faces; fid < next_prefix_faces; fid++)
						// {
						//     uint3 face = mesh_data->sa_faces[fid];
						//     virtual_solid_faces[2 * (fid - curr_prefix_faces) + 0] =
						//         luisa::make_uint3(2 * (face.x - curr_prefix_verts) + 0,
						//                           2 * (face.y - curr_prefix_verts) + 0,
						//                           2 * (face.z - curr_prefix_verts) + 0);
						//     virtual_solid_faces[2 * (fid - curr_prefix_faces) + 1] =
						//         luisa::make_uint3(2 * (face.z - curr_prefix_verts) + 1,
						//                           2 * (face.y - curr_prefix_verts) + 1,
						//                           2 * (face.x - curr_prefix_verts) + 1);
						// }
						// compute_trimesh_dyadic_mass(virtual_solid_verts,
						//                             virtual_solid_faces,
						//                             0,
						//                             static_cast<uint>(virtual_solid_faces.size()),
						//                             mesh_info.get_density(),
						//                             M_body,
						//                             MI_body,
						//                             I_body);

						for (uint vid = prefix_vid; vid < suffix_vid; vid++)
						{
							float  vert_mass = mesh_data->sa_vert_mass[vid];
							float3 vert_pos = sim_data->sa_scaled_model_x[vid];

							M_body += vert_mass;
							MI_body += vert_mass * vert_pos;
							I_body = I_body + vert_mass * outer_product(vert_pos, vert_pos);
						}
					}
					else // Solid body
					{
						// If provided tetrahedron mesh for solid part
						if ((mesh_data->prefix_num_tets[meshIdx + 1] - mesh_data->prefix_num_tets[meshIdx]) > 0)
						{
							for (uint vid = prefix_vid; vid < suffix_vid; vid++)
							{
								float  vert_mass = mesh_data->sa_vert_mass[vid];
								float3 vert_pos = sim_data->sa_scaled_model_x[vid];

								M_body += vert_mass;
								MI_body += vert_mass * vert_pos;
								I_body = I_body + vert_mass * outer_product(vert_pos, vert_pos);
							}
						}
						else // If we only have surface mesh: integrate from surface triangles
						{
							compute_trimesh_dyadic_mass(sim_data->sa_scaled_model_x,
								mesh_data->sa_faces,
								mesh_data->prefix_num_faces[meshIdx],
								mesh_data->prefix_num_faces[meshIdx + 1],
								mesh_info.get_density(),
								M_body,
								MI_body,
								I_body);
						}
					}

					body_mass.block<3, 3>(0, 0) = M_body * EigenFloat3x3::Identity();

					for (uint i = 0; i < 3; i++)
						body_mass.block<3, 3>(3 + i * 3, 0) = MI_body[i] * EigenFloat3x3::Identity();

					for (uint i = 0; i < 3; i++)
						body_mass.block<3, 3>(0, 3 + i * 3) = MI_body[i] * EigenFloat3x3::Identity();

					for (uint i = 0; i < 3; i++)
						for (uint j = 0; j < 3; j++)
							body_mass.block<3, 3>(3 + i * 3, 3 + j * 3) = I_body[i][j] * EigenFloat3x3::Identity();

					body_mass.diagonal() = body_mass.diagonal().cwiseMax(Epsilon);

					for (uint i = 0; i < 4; i++)
					{
						for (uint j = 0; j < 4; j++)
						{
							compressed_mass_matrix[j][i] = body_mass(i * 3 + 0, j * 3 + 0);
						}
					}

					abd_inertia_data.sa_affine_bodies_mass_matrix[body_idx] = compressed_mass_matrix;
					abd_inertia_data.sa_affine_bodies_mass_matrix_full[body_idx] = body_mass;

					// if (num_affine_bodies < 20)
					// {
					// 	std::cout << "Mass Matrix = \n" << body_mass << std::endl;
					// 	LUISA_INFO("Affine Body {} Mass Matrix : ", body_idx);
					// 	LUISA_INFO("Affine Body {} Mass Matrix : {}", body_idx, compressed_mass_matrix[0]);
					// 	LUISA_INFO("Affine Body {} Mass Matrix : {}", body_idx, compressed_mass_matrix[1]);
					// 	LUISA_INFO("Affine Body {} Mass Matrix : {}", body_idx, compressed_mass_matrix[2]);
					// 	LUISA_INFO("Affine Body {} Mass Matrix : {}", body_idx, compressed_mass_matrix[3]);
					// }
				}

				const bool has_fixed_vert = sim_data->sa_q_is_fixed[prefix_dof_abd + 4 * body_idx];
				abd_inertia_data.sa_stiffness_dirichlet[body_idx] = has_fixed_vert ? default_stiffness_dirichlet : 1.0f;

				float area = std::reduce(mesh_data->sa_rest_vert_area.begin() + prefix_vid,
					mesh_data->sa_rest_vert_area.begin() + suffix_vid,
					0.0f);

				abd_ortho_data.abd_volume[body_idx] = mesh_data->sa_rest_body_volume[meshIdx];
				abd_ortho_data.abd_kappa[body_idx] = mesh_info.get_material<RigidMaterial>().stiffness;

				// EigenFloat12 gravity_sum = EigenFloat12::Zero();
				// CpuParallel::single_thread_for(curr_prefix_verts,
				//                                next_prefix_verts,
				//                                [&](const uint vid)
				//                                {
				//                                    sim_data->sa_vert_affine_bodies_id[vid] = body_idx;
				//                                    float  mass   = mesh_data->sa_vert_mass[vid];
				//                                    float3 rest_x = mesh_data->sa_model_x[vid];
				//                                    auto J = AffineBodyDynamics::get_jacobian_dxdq(rest_x);
				//                                    gravity_sum +=
				//                                        mass * J.transpose()
				//                                        * float3_to_eigen3(luisa::make_float3(0, -9.8, 0));
				//                                });  // / area_mass[1];

				// EigenFloat12 body_gravity = body_mass.inverse() * gravity_sum;
				// sim_data->sa_affine_bodies_gravity[4 * body_idx + 0] =
				//     eigen3_to_float3(body_gravity.block<3, 1>(0, 0));
				// sim_data->sa_affine_bodies_gravity[4 * body_idx + 1] =
				//     eigen3_to_float3(body_gravity.block<3, 1>(3, 0));
				// sim_data->sa_affine_bodies_gravity[4 * body_idx + 2] =
				//     eigen3_to_float3(body_gravity.block<3, 1>(6, 0));
				// sim_data->sa_affine_bodies_gravity[4 * body_idx + 3] =
				//     eigen3_to_float3(body_gravity.block<3, 1>(9, 0));
				// LUISA_INFO("Affine body {} : Area = {}, Gravity = {}", body_idx, area, body_gravity);
			};
		}

		// Init joint constraints
		{
			auto normalize_axis = [](const float3& axis) -> float3
			{
				const float n2 = axis.x * axis.x + axis.y * axis.y + axis.z * axis.z;
				if (n2 < 1.0e-12f)
				{
					return luisa::make_float3(1.0f, 0.0f, 0.0f);
				}
				const float inv_n = 1.0f / std::sqrt(n2);
				return inv_n * axis;
			};

			auto try_get_rigid_q_indices = [&](uint registration_id, uint4& out_indices) -> bool
			{
				if (registration_id >= mesh_data->input_to_sorted_mesh_id.size())
				{
					LUISA_WARNING("Joint uses invalid registration id {}", registration_id);
					return false;
				}

				const uint sorted_idx = mesh_data->input_to_sorted_mesh_id[registration_id];
				if (sorted_idx >= world_data.size())
				{
					LUISA_WARNING("Joint registration id {} maps to invalid sorted index {}", registration_id, sorted_idx);
					return false;
				}
				if (!world_data[sorted_idx].holds<lcs::Material::RigidMaterial>())
				{
					LUISA_WARNING("Joint registration id {} is not rigid.", registration_id);
					return false;
				}

				const uint prefix_vid = mesh_data->prefix_num_verts[sorted_idx];
				if (prefix_vid >= sim_data->sa_x_to_dof_map.size())
				{
					LUISA_WARNING("Joint registration id {} has invalid prefix_vid {}", registration_id, prefix_vid);
					return false;
				}

				const uint dof_start = sim_data->sa_x_to_dof_map[prefix_vid].get_dof_idx();
				if (dof_start + 3u >= sim_data->sa_q.size())
				{
					LUISA_WARNING("Joint registration id {} has invalid dof_start {}", registration_id, dof_start);
					return false;
				}

				out_indices = luisa::make_uint4(dof_start + 0u, dof_start + 1u, dof_start + 2u, dof_start + 3u);
				return true;
			};

			auto&				joint_data = sim_data->get_joint_constraint_data();
			const luisa::float3 default_axis{ 1.f, 0.f, 0.f };
			const auto&			rest_q = sim_data->sa_rest_q;

			auto make_rest_A = [&](const uint4& idx_a)
			{
				return luisa::make_float3x3(rest_q[idx_a.y], rest_q[idx_a.z], rest_q[idx_a.w]);
			};
			auto make_rest_B = [&](const uint4& idx_b)
			{
				return luisa::make_float3x3(rest_q[idx_b.y], rest_q[idx_b.z], rest_q[idx_b.w]);
			};
			auto compute_rest_position_delta_local_a = [&](const uint4& idx_a, const uint4& idx_b, const float3& anchor_a, const float3& anchor_b)
			{
				const float3 p_a = rest_q[idx_a.x] + rest_q[idx_a.y] * anchor_a.x + rest_q[idx_a.z] * anchor_a.y + rest_q[idx_a.w] * anchor_a.z;
				const float3 p_b = rest_q[idx_b.x] + rest_q[idx_b.y] * anchor_b.x + rest_q[idx_b.z] * anchor_b.y + rest_q[idx_b.w] * anchor_b.z;
				const float3 d_world = p_b - p_a;
				const auto	 A_inv = luisa::inverse(make_rest_A(idx_a));
				return A_inv * d_world;
			};
			auto compute_rest_rotation_a_to_b = [&](const uint4& idx_a, const uint4& idx_b)
			{
				const auto A_inv = luisa::inverse(make_rest_A(idx_a));
				const auto B = make_rest_B(idx_b);
				const auto R_ab = A_inv * B;
				return std::array<float3, 3>{ R_ab[0], R_ab[1], R_ab[2] };
			};

			using uint8 = std::array<uint, 8>;

			for (const auto& desc : fixed_joint_descs)
			{
				uint4 idx_a{};
				uint4 idx_b{};
				if (!try_get_rigid_q_indices(desc.body_a_registration, idx_a)
					|| !try_get_rigid_q_indices(desc.body_b_registration, idx_b))
				{
					continue;
				}
				const auto rest_pos_local_a = compute_rest_position_delta_local_a(idx_a, idx_b, desc.anchor_a_local, desc.anchor_b_local);
				const auto rest_rot_a_to_b = compute_rest_rotation_a_to_b(idx_a, idx_b);
				uint8	   idx_ext = { idx_a[0], idx_a[1], idx_a[2], idx_a[3], idx_b[0], idx_b[1], idx_b[2], idx_b[3] };
				joint_data.constraint_indices.push_back(idx_ext);
				joint_data.anchor_a_local.push_back(desc.anchor_a_local);
				joint_data.anchor_b_local.push_back(desc.anchor_b_local);
				joint_data.rest_position_delta.push_back(rest_pos_local_a);
				joint_data.rest_rot_col0_a_to_b.push_back(rest_rot_a_to_b[0]);
				joint_data.rest_rot_col1_a_to_b.push_back(rest_rot_a_to_b[1]);
				joint_data.rest_rot_col2_a_to_b.push_back(rest_rot_a_to_b[2]);
				joint_data.axis_world.push_back(default_axis);
				joint_data.axis_a_local.push_back(default_axis);
				joint_data.axis_b_local.push_back(default_axis);
				joint_data.stiffness.push_back(luisa::make_float2(desc.stiffness_pos, desc.stiffness_rot));
				joint_data.joint_type.push_back(static_cast<uint>(JointConstraintType::Fixed));
				joint_data.slide_limits.push_back(luisa::make_float2(-std::numeric_limits<float>::max(), std::numeric_limits<float>::max()));
			}

			for (const auto& desc : prismatic_joint_descs)
			{
				uint4 idx_a{};
				uint4 idx_b{};
				if (!try_get_rigid_q_indices(desc.body_a_registration, idx_a)
					|| !try_get_rigid_q_indices(desc.body_b_registration, idx_b))
				{
					continue;
				}
				const auto rest_rot_a_to_b = compute_rest_rotation_a_to_b(idx_a, idx_b);
				// For a prismatic joint, only the rest offset perpendicular to the sliding
				// axis should be constrained. The component along the axis is the slide
				// coordinate and is governed only by slide_limits.
				const auto	rest_pos_local_a = compute_rest_position_delta_local_a(idx_a, idx_b, desc.anchor_a_local, desc.anchor_b_local);
				const auto	A0_inv_prismatic = luisa::inverse(make_rest_A(idx_a));
				const auto	axis_a_local_prismatic = normalize_axis(A0_inv_prismatic * normalize_axis(desc.axis_world));
				const float rest_slide = luisa::dot(rest_pos_local_a, axis_a_local_prismatic);
				const auto	rest_pos_perp_local_a = rest_pos_local_a - rest_slide * axis_a_local_prismatic;
				uint8		idx_ext = { idx_a[0], idx_a[1], idx_a[2], idx_a[3], idx_b[0], idx_b[1], idx_b[2], idx_b[3] };
				joint_data.constraint_indices.push_back(idx_ext);
				joint_data.anchor_a_local.push_back(desc.anchor_a_local);
				joint_data.anchor_b_local.push_back(desc.anchor_b_local);
				joint_data.rest_position_delta.push_back(rest_pos_perp_local_a);
				joint_data.rest_rot_col0_a_to_b.push_back(rest_rot_a_to_b[0]);
				joint_data.rest_rot_col1_a_to_b.push_back(rest_rot_a_to_b[1]);
				joint_data.rest_rot_col2_a_to_b.push_back(rest_rot_a_to_b[2]);
				// Compute body-local sliding axis: axis_a_local = normalize(A0^{-1} * axis_world).
				// This ensures the sliding direction co-rotates with body A at runtime.
				joint_data.axis_world.push_back(normalize_axis(desc.axis_world));
				joint_data.axis_a_local.push_back(axis_a_local_prismatic);
				joint_data.axis_b_local.push_back(default_axis);
				joint_data.stiffness.push_back(luisa::make_float2(desc.stiffness_pos, desc.stiffness_rot));
				joint_data.joint_type.push_back(static_cast<uint>(JointConstraintType::Prismatic));
				joint_data.slide_limits.push_back(luisa::make_float2(desc.slide_min, desc.slide_max));
			}

			for (const auto& desc : revolute_joint_descs)
			{
				uint4 idx_a{};
				uint4 idx_b{};
				if (!try_get_rigid_q_indices(desc.body_a_registration, idx_a)
					|| !try_get_rigid_q_indices(desc.body_b_registration, idx_b))
				{
					continue;
				}
				const auto rest_pos_local_a = compute_rest_position_delta_local_a(idx_a, idx_b, desc.anchor_a_local, desc.anchor_b_local);
				const auto rest_rot_a_to_b = compute_rest_rotation_a_to_b(idx_a, idx_b);

				uint8 idx_ext = { idx_a[0], idx_a[1], idx_a[2], idx_a[3], idx_b[0], idx_b[1], idx_b[2], idx_b[3] };
				joint_data.constraint_indices.push_back(idx_ext);
				joint_data.anchor_a_local.push_back(desc.anchor_a_local);
				joint_data.anchor_b_local.push_back(desc.anchor_b_local);
				joint_data.rest_position_delta.push_back(rest_pos_local_a);
				joint_data.rest_rot_col0_a_to_b.push_back(rest_rot_a_to_b[0]);
				joint_data.rest_rot_col1_a_to_b.push_back(rest_rot_a_to_b[1]);
				joint_data.rest_rot_col2_a_to_b.push_back(rest_rot_a_to_b[2]);
				joint_data.axis_world.push_back(normalize_axis(desc.axis_world));
				joint_data.axis_a_local.push_back(normalize_axis(desc.axis_a_local));
				joint_data.axis_b_local.push_back(normalize_axis(desc.axis_b_local));
				joint_data.stiffness.push_back(luisa::make_float2(desc.stiffness_pos, desc.stiffness_axis));
				joint_data.joint_type.push_back(static_cast<uint>(JointConstraintType::Revolute));
			}

			// Pre-allocate gradient/hessian buffers (filled by eval shader at runtime)
			const size_t num_joints = joint_data.constraint_indices.size();
			joint_data.constraint_gradients.resize(num_joints * 8u, luisa::make_float3(0.f));
			joint_data.constraint_hessians.resize(num_joints * 64u, luisa::make_float3x3(0.f));
		}

		// Init Energy Adjacent List
		{
			// num_variables_in_system
			sim_data->vert_adj_material_force_verts.resize(num_dof);

			auto& adj_map = sim_data->vert_adj_material_force_verts;

			LUISA_INFO("Building adjacent list for material forces...");

			// Vert adj soft-body fixed constraints
			auto& soft_inertia_data = sim_data->get_soft_inertia_data();
			build_adj_list_and_init_grad_hess(adj_map, soft_inertia_data);

			// Vert adj stretch springs
			auto& stretch_spring_data = sim_data->get_stretch_spring_data();
			build_adj_list_and_init_grad_hess(adj_map, stretch_spring_data);

			// Vert adj stretch faces
			auto& stretch_face_data = sim_data->get_stretch_face_data();
			build_adj_list_and_init_grad_hess(adj_map, stretch_face_data);

			// Vert adj bending edges
			auto& bending_edge_data = sim_data->get_bending_edge_data();
			build_adj_list_and_init_grad_hess(adj_map, bending_edge_data);

			// Vert adj stress tets
			auto& stress_tet_data = sim_data->get_stress_tet_data();
			build_adj_list_and_init_grad_hess(adj_map, stress_tet_data);

			// Vert adj affine-body inertia
			auto& abd_inertia_data = sim_data->get_abd_inertia_data();
			build_adj_list_and_init_grad_hess(adj_map, abd_inertia_data);

			// Vert adj affine-body orthogonality
			auto& abd_ortho_data = sim_data->get_abd_orthogonality_data();
			build_adj_list_and_init_grad_hess(adj_map, abd_ortho_data);

			// Vert adj joint constraints
			auto& joint_data = sim_data->get_joint_constraint_data();
			build_adj_list_and_init_grad_hess(adj_map, joint_data);

			// Sort adjacents
			CpuParallel::parallel_for(0,
				num_dof,
				[&](const uint vid)
				{
					std::vector<uint>& adj_list = sim_data->vert_adj_material_force_verts[vid];
					std::sort(adj_list.begin(), adj_list.end());
					if (adj_list.size() > 255)
						LUISA_ERROR("Adjacent count out of range {}");
				});
			upload_2d_csr_from(sim_data->sa_vert_adj_material_force_verts_csr, sim_data->vert_adj_material_force_verts);

			// Final off-diag hessian
			const uint orig_hessian_nnz = sim_data->sa_vert_adj_material_force_verts_csr.size(); //  - mesh_data->num_verts - 1

			constexpr bool use_block_scan = true;
			constexpr bool use_warp_scan = false;
			constexpr uint segment_size = use_block_scan ? 256 : use_warp_scan ? 32
																			   : 1;
			const uint	   alinged_nnz = segment_size == 1 ? orig_hessian_nnz : (orig_hessian_nnz + segment_size - 1) / segment_size * segment_size;
			LUISA_INFO("Original Hessian NNZ = {}, Aligned Hessian NNZ = {}", orig_hessian_nnz, alinged_nnz);
			// const uint alinged_nnz = orig_hessian_nnz;
			sim_data->sa_cgA_fixtopo_offdiag_triplet.resize(alinged_nnz);
			sim_data->sa_cgA_fixtopo_offdiag_triplet_info.resize(alinged_nnz, make_matrix_triplet_info(-1u, -1u, 0));

			CpuParallel::parallel_for(
				0,
				num_dof,
				[&](const uint vid)
				{
					const uint curr_prefix = sim_data->sa_vert_adj_material_force_verts_csr[vid];
					const uint next_prefix = sim_data->sa_vert_adj_material_force_verts_csr[vid + 1];
					for (uint idx = curr_prefix; idx < next_prefix; idx++) // Outer-of-range part will set to zero
					{
						const uint adj_vid = sim_data->sa_vert_adj_material_force_verts_csr[idx];
						const uint triplet_property =
							MatrixTriplet::make_triplet_property_in_block(idx, curr_prefix, next_prefix - 1);

						// LUISA_INFO("Hessian Triplet ({}, {}) at idx = {}, property = {:16b}", vid, adj_vid, idx, triplet_property);
						sim_data->sa_cgA_fixtopo_offdiag_triplet_info[idx] =
							make_matrix_triplet_info(vid, adj_vid, triplet_property);
					}
				});

			// Check
			if constexpr (use_block_scan)
			{
				std::vector<ushort> is_tirplet_accessed(alinged_nnz, false);
				CpuParallel::single_thread_for(
					0,
					alinged_nnz / 256,
					[&](const uint blockIdx)
					{
						const uint blockPrefix = blockIdx * 256;
						const uint blockEnd = blockPrefix + 256;

						uint			  last_start = -1u;
						uint			  last_end = -1u;
						std::vector<bool> cache_used(32, false);
						for (uint idx = blockPrefix; idx < blockEnd; idx++)
						{
							auto& info = sim_data->sa_cgA_fixtopo_offdiag_triplet_info[idx];
							auto  property = info[2];
							if (MatrixTriplet::is_first_col_in_row(property))
							{
								if (last_end != -1u || last_start != -1u)
								{
									LUISA_ERROR("Last segment not closed");
								}
								last_start = idx;
							}
							if (MatrixTriplet::is_last_col_in_row(property))
							{
								if (last_start == -1u || last_end != -1u)
								{
									LUISA_ERROR("Last segment not started");
								}
								last_end = idx;
							}
							if (last_start != -1u && last_end != -1u)
							{
								// if (info[0] < 10)
								// {
								//     LUISA_INFO("Vert {}'s triplet (threadIdx from {:3} to {:3}), from {} to {} , same warp ? {}, try to find {} ({})",
								//                info[0],
								//                last_start % 256,
								//                last_end % 256,
								//                last_start,
								//                last_end,
								//                MatrixTriplet::is_first_and_last_col_in_same_warp(info[2]),
								//                blockPrefix + MatrixTriplet::read_first_col_threadIdx(info[2]),
								//                MatrixTriplet::read_first_col_threadIdx(info[2]));
								// }
								for (uint i = last_start; i <= last_end; i++)
								{
									if (is_tirplet_accessed[i])
									{
										LUISA_ERROR("Triplet {} has been accessed", i);
									}
									is_tirplet_accessed[i] = true;
								}
								const uint start_threadIdx = last_start % 256;
								const uint start_warpIdx = start_threadIdx / 32;
								const uint end_threadIdx = last_end % 256;
								const uint end_warpIdx = end_threadIdx / 32;

								uint3 info_start = sim_data->sa_cgA_fixtopo_offdiag_triplet_info[last_start];
								uint3 info_end = sim_data->sa_cgA_fixtopo_offdiag_triplet_info[last_end];

								if (info_start[0] != info_end[0])
								{
									LUISA_ERROR("RowIdx not match");
								}
								const uint vert_prefix =
									sim_data->sa_vert_adj_material_force_verts_csr[info_start[0]];
								const uint vert_suffix =
									sim_data->sa_vert_adj_material_force_verts_csr[info_start[0] + 1];
								if (MatrixTriplet::is_first_and_last_col_in_same_warp(info_start[2])
									!= MatrixTriplet::is_first_and_last_col_in_same_warp(info_end[2]))
								{
									LUISA_ERROR("Reduce mode not match : {} - {} : (Triplet Range {} - {}) startTriplet = {} (threadIdx = {}, warpIdx = {}), endTriplet = {} (threadIdx = {}, warpIdx = {})",
										MatrixTriplet::is_first_and_last_col_in_same_warp(info_start[2]),
										MatrixTriplet::is_first_and_last_col_in_same_warp(info_end[2]),
										vert_prefix,
										vert_suffix - 1,
										last_start,
										start_threadIdx,
										start_warpIdx,
										last_end,
										end_threadIdx,
										end_warpIdx);
								}
								bool in_same_warp = MatrixTriplet::is_first_and_last_col_in_same_warp(info_start[2]);
								if (in_same_warp)
								{
									if (last_start / 32 != last_end / 32)
									{
										LUISA_ERROR("Not in the same warp");
									}
									const uint start_laneIdx = last_start % 32;
									const uint desire_laneIdx = MatrixTriplet::read_first_col_info(info_end[2]);
									if (start_laneIdx != desire_laneIdx)
									{
										LUISA_ERROR("LaneIdx not match!!");
									}
								}
								else
								{
									if (last_start / 32 == last_end / 32)
									{
										LUISA_ERROR("In the same warp");
									}
									if (cache_used[start_warpIdx])
									{
										LUISA_ERROR("Cache has been used");
									}
									const uint desire_warpIdx = MatrixTriplet::read_first_col_info(info_end[2]);
									if (start_warpIdx != desire_warpIdx)
									{
										LUISA_ERROR("WarpIdx not match: startTriplet = {} (threadIdx = {}, warpIdx = {}), endTriplet = {} (threadIdx = {}, Desire for {})",
											last_start,
											start_threadIdx,
											start_warpIdx,
											last_end,
											last_end % 256,
											desire_warpIdx);
									}
									cache_used[start_warpIdx] = true;
								}
								last_start = -1u;
								last_end = -1u;
							}
						}
					});
				for (uint triplet_idx = 0; triplet_idx < orig_hessian_nnz; triplet_idx++)
				{
					auto triplet_info = sim_data->sa_cgA_fixtopo_offdiag_triplet_info[triplet_idx];
					if (MatrixTriplet::is_valid(triplet_info[2]))
					{
						if (!is_tirplet_accessed[triplet_idx])
						{
							LUISA_ERROR("Triplet {} has not been accessed", triplet_idx);
						}
					}
				}
			}
		}

		// Find material-force-offset
		{
			const std::vector<std::vector<uint>>& adj_list = sim_data->vert_adj_material_force_verts;
			const std::vector<uint>&			  csr = sim_data->sa_vert_adj_material_force_verts_csr;

			// Spring energy
			auto& stretch_spring_data = sim_data->get_stretch_spring_data();
			init_constitution_offsets_in_adjlist(adj_list, csr, stretch_spring_data);

			// Stretch face energy
			auto& stretch_face_data = sim_data->get_stretch_face_data();
			init_constitution_offsets_in_adjlist(adj_list, csr, stretch_face_data);

			// Bending angle energy
			auto& bending_edge_data = sim_data->get_bending_edge_data();
			init_constitution_offsets_in_adjlist(adj_list, csr, bending_edge_data);

			// Stress tetrahedron energy
			auto& stress_tet_data = sim_data->get_stress_tet_data();
			init_constitution_offsets_in_adjlist(adj_list, csr, stress_tet_data);

			// Affine body inertia & ground collision data
			auto& abd_inertia_data = sim_data->get_abd_inertia_data();
			init_constitution_offsets_in_adjlist(adj_list, csr, abd_inertia_data);

			// Affine body orthogonality
			auto& abd_ortho_data = sim_data->get_abd_orthogonality_data();
			init_constitution_offsets_in_adjlist(adj_list, csr, abd_ortho_data);

			// Soft body inertia & ground collision data
			auto& soft_inertia_data = sim_data->get_soft_inertia_data();
			init_constitution_offsets_in_adjlist(adj_list, csr, soft_inertia_data);

			// Joint constraint data
			auto& joint_data = sim_data->get_joint_constraint_data();
			init_constitution_offsets_in_adjlist(adj_list, csr, joint_data);
		}
	}

	void upload_sim_buffers(luisa::compute::Device&	 device,
		luisa::compute::Stream&						 stream,
		lcs::SimulationData<std::vector>*			 input_data,
		lcs::SimulationData<luisa::compute::Buffer>* output_data)
	{
		output_data->num_dof = input_data->num_dof;
		output_data->num_affine_bodies = input_data->num_affine_bodies;
		output_data->num_verts_rigid = input_data->num_verts_rigid;
		output_data->num_verts_soft = input_data->num_verts_soft;
		output_data->num_verts_total = input_data->num_verts_total;

		// State buffers
		{
			const uint num_dof = output_data->num_dof;
			stream << upload_buffer(device, output_data->sa_num_dof, input_data->sa_num_dof)
				   << upload_buffer(device, output_data->sa_rest_q, input_data->sa_rest_q)
				   << upload_buffer(device, output_data->sa_rest_q_v, input_data->sa_rest_q_v)
				   << upload_buffer(device, output_data->sa_q_is_fixed, input_data->sa_q_is_fixed)
				   << upload_buffer(device, output_data->sa_q_property, input_data->sa_q_property);
			resize_buffer(device, output_data->sa_q, num_dof);
			resize_buffer(device, output_data->sa_q_v, num_dof);
			resize_buffer(device, output_data->sa_q_iter_start, num_dof);
			resize_buffer(device, output_data->sa_q_step_start, num_dof);
			resize_buffer(device, output_data->sa_q_tilde, num_dof);
			resize_buffer(device, output_data->sa_dq, num_dof);
		}

		// Position / Velocity buffers
		{
			const uint num_verts_total = output_data->num_verts_rigid + output_data->num_verts_soft;
			stream << upload_buffer(device, output_data->sa_rest_x, input_data->sa_rest_x)
				   << upload_buffer(device, output_data->sa_rest_v, input_data->sa_rest_v)
				   << upload_buffer(device, output_data->sa_scaled_model_x, input_data->sa_scaled_model_x)
				   << upload_buffer(device, output_data->sa_x_to_dof_map, input_data->sa_x_to_dof_map)
				   << upload_buffer(device, output_data->sa_x_property, input_data->sa_x_property);
			resize_buffer(device, output_data->sa_x, num_verts_total);
			resize_buffer(device, output_data->sa_v, num_verts_total);
			resize_buffer(device, output_data->sa_x_step_start, num_verts_total);
			resize_buffer(device, output_data->sa_x_iter_start, num_verts_total);
			resize_buffer(device, output_data->sa_dx, num_verts_total);
		}

		resize_buffer(device, output_data->sa_system_energy, input_data->sa_system_energy.size());

		stream << upload_buffer(device, output_data->sa_cgA_fixtopo_offdiag_triplet, input_data->sa_cgA_fixtopo_offdiag_triplet)
			   << upload_buffer(device, output_data->sa_cgA_fixtopo_offdiag_triplet_info, input_data->sa_cgA_fixtopo_offdiag_triplet_info)
			   << upload_buffer(device, output_data->sa_vert_adj_material_force_verts_csr, input_data->sa_vert_adj_material_force_verts_csr);

		//
		// Constitution Data
		//
		auto& stretch_spring_I = input_data->get_stretch_spring_data();
		auto& stretch_spring_O = output_data->get_stretch_spring_data();
		if (stretch_spring_I.is_valid())
		{
			stream
				<< upload_buffer(device, stretch_spring_O.constraint_indices, stretch_spring_I.constraint_indices)
				<< upload_buffer(device,
					   stretch_spring_O.sa_stretch_spring_rest_state_length,
					   stretch_spring_I.sa_stretch_spring_rest_state_length)
				<< upload_buffer(device, stretch_spring_O.sa_stretch_spring_stiffness, stretch_spring_I.sa_stretch_spring_stiffness)
				<< upload_buffer(device, stretch_spring_O.constraint_offsets_in_adjlist, stretch_spring_I.constraint_offsets_in_adjlist)
				<< upload_buffer(device, stretch_spring_O.constraint_gradients, stretch_spring_I.constraint_gradients)
				<< upload_buffer(device, stretch_spring_O.constraint_hessians, stretch_spring_I.constraint_hessians)
				<< upload_buffer(device, stretch_spring_O.vert_adj_constraints_csr, stretch_spring_I.vert_adj_constraints_csr);
		}

		auto& stretch_face_I = input_data->get_stretch_face_data();
		auto& stretch_face_O = output_data->get_stretch_face_data();
		if (stretch_face_I.is_valid())
		{
			stream
				<< upload_buffer(device, stretch_face_O.constraint_indices, stretch_face_I.constraint_indices)
				<< upload_buffer(device, stretch_face_O.sa_stretch_faces_mu_lambda, stretch_face_I.sa_stretch_faces_mu_lambda)
				<< upload_buffer(device, stretch_face_O.sa_stretch_faces_rest_area, stretch_face_I.sa_stretch_faces_rest_area)
				<< upload_buffer(device, stretch_face_O.sa_stretch_faces_Dm_inv, stretch_face_I.sa_stretch_faces_Dm_inv)
				<< upload_buffer(device, stretch_face_O.constraint_offsets_in_adjlist, stretch_face_I.constraint_offsets_in_adjlist)
				<< upload_buffer(device, stretch_face_O.constraint_gradients, stretch_face_I.constraint_gradients)
				<< upload_buffer(device, stretch_face_O.constraint_hessians, stretch_face_I.constraint_hessians)
				<< upload_buffer(device, stretch_face_O.vert_adj_constraints_csr, stretch_face_I.vert_adj_constraints_csr);
		}

		auto& bending_edge_I = input_data->get_bending_edge_data();
		auto& bending_edge_O = output_data->get_bending_edge_data();
		if (bending_edge_I.is_valid())
		{
			stream
				<< upload_buffer(device, bending_edge_O.constraint_indices, bending_edge_I.constraint_indices)
				<< upload_buffer(device, bending_edge_O.sa_bending_edges_rest_area, bending_edge_I.sa_bending_edges_rest_area)
				<< upload_buffer(device, bending_edge_O.sa_bending_edges_rest_angle, bending_edge_I.sa_bending_edges_rest_angle)
				<< upload_buffer(device, bending_edge_O.sa_bending_edges_stiffness, bending_edge_I.sa_bending_edges_stiffness)
				<< upload_buffer(device, bending_edge_O.sa_bending_edges_Q, bending_edge_I.sa_bending_edges_Q)
				<< upload_buffer(device, bending_edge_O.constraint_offsets_in_adjlist, bending_edge_I.constraint_offsets_in_adjlist)
				<< upload_buffer(device, bending_edge_O.constraint_gradients, bending_edge_I.constraint_gradients)
				<< upload_buffer(device, bending_edge_O.constraint_hessians, bending_edge_I.constraint_hessians)
				<< upload_buffer(device, bending_edge_O.vert_adj_constraints_csr, bending_edge_I.vert_adj_constraints_csr);
		}

		auto& stress_tet_I = input_data->get_stress_tet_data();
		auto& stress_tet_O = output_data->get_stress_tet_data();
		if (stress_tet_I.is_valid())
		{
			stream
				<< upload_buffer(device, stress_tet_O.constraint_indices, stress_tet_I.constraint_indices)
				<< upload_buffer(device, stress_tet_O.sa_stress_tets_model, stress_tet_I.sa_stress_tets_model)
				<< upload_buffer(device, stress_tet_O.sa_stress_tets_mu_lambda, stress_tet_I.sa_stress_tets_mu_lambda)
				<< upload_buffer(device, stress_tet_O.sa_stress_tets_rest_volume, stress_tet_I.sa_stress_tets_rest_volume)
				<< upload_buffer(device, stress_tet_O.sa_stress_tets_Dm_inv, stress_tet_I.sa_stress_tets_Dm_inv)
				<< upload_buffer(device, stress_tet_O.constraint_offsets_in_adjlist, stress_tet_I.constraint_offsets_in_adjlist)
				<< upload_buffer(device, stress_tet_O.constraint_gradients, stress_tet_I.constraint_gradients)
				<< upload_buffer(device, stress_tet_O.constraint_hessians, stress_tet_I.constraint_hessians)
				<< upload_buffer(device, stress_tet_O.vert_adj_constraints_csr, stress_tet_I.vert_adj_constraints_csr);
		}

		auto& soft_inertia_I = input_data->get_soft_inertia_data();
		auto& soft_inertia_O = output_data->get_soft_inertia_data();
		if (soft_inertia_I.is_valid())
		{
			stream
				<< upload_buffer(device, soft_inertia_O.constraint_indices, soft_inertia_I.constraint_indices)
				<< upload_buffer(device, soft_inertia_O.sa_soft_vert_mass, soft_inertia_I.sa_soft_vert_mass)
				<< upload_buffer(device, soft_inertia_O.sa_stiffness_dirichlet, soft_inertia_I.sa_stiffness_dirichlet)
				<< upload_buffer(device, soft_inertia_O.constraint_offsets_in_adjlist, soft_inertia_I.constraint_offsets_in_adjlist)
				<< upload_buffer(device, soft_inertia_O.constraint_gradients, soft_inertia_I.constraint_gradients)
				<< upload_buffer(device, soft_inertia_O.constraint_hessians, soft_inertia_I.constraint_hessians)
				<< upload_buffer(device, soft_inertia_O.vert_adj_constraints_csr, soft_inertia_I.vert_adj_constraints_csr);
		}

		auto& abd_inertia_I = input_data->get_abd_inertia_data();
		auto& abd_inertia_O = output_data->get_abd_inertia_data();

		if (abd_inertia_I.is_valid())
		{
			stream
				<< upload_buffer(device, output_data->sa_affine_bodies_mesh_id, input_data->sa_affine_bodies_mesh_id)
				// << upload_buffer(device, output_data->sa_affine_bodies_is_fixed, input_data->sa_affine_bodies_is_fixed)

				<< upload_buffer(device, abd_inertia_O.constraint_indices, abd_inertia_I.constraint_indices)
				<< upload_buffer(device, abd_inertia_O.sa_affine_bodies_mass_matrix, abd_inertia_I.sa_affine_bodies_mass_matrix)
				<< upload_buffer(device, abd_inertia_O.sa_stiffness_dirichlet, abd_inertia_I.sa_stiffness_dirichlet)
				<< upload_buffer(device, abd_inertia_O.constraint_offsets_in_adjlist, abd_inertia_I.constraint_offsets_in_adjlist)
				<< upload_buffer(device, abd_inertia_O.constraint_gradients, abd_inertia_I.constraint_gradients)
				<< upload_buffer(device, abd_inertia_O.constraint_hessians, abd_inertia_I.constraint_hessians)
				<< upload_buffer(device, abd_inertia_O.vert_adj_constraints_csr, abd_inertia_I.vert_adj_constraints_csr);
		}

		auto& abd_ortho_I = input_data->get_abd_orthogonality_data();
		auto& abd_ortho_O = output_data->get_abd_orthogonality_data();
		if (abd_ortho_I.is_valid())
		{
			stream << upload_buffer(device, abd_ortho_O.abd_volume, abd_ortho_I.abd_volume)
				   << upload_buffer(device, abd_ortho_O.abd_kappa, abd_ortho_I.abd_kappa)
				   << upload_buffer(device, abd_ortho_O.constraint_indices, abd_ortho_I.constraint_indices)
				   << upload_buffer(device, abd_ortho_O.constraint_offsets_in_adjlist, abd_ortho_I.constraint_offsets_in_adjlist)
				   << upload_buffer(device, abd_ortho_O.constraint_gradients, abd_ortho_I.constraint_gradients)
				   << upload_buffer(device, abd_ortho_O.constraint_hessians, abd_ortho_I.constraint_hessians)
				   << upload_buffer(device, abd_ortho_O.vert_adj_constraints_csr, abd_ortho_I.vert_adj_constraints_csr);
		}

		auto& joint_I = input_data->get_joint_constraint_data();
		auto& joint_O = output_data->get_joint_constraint_data();
		if (joint_I.is_valid())
		{
			stream
				<< upload_buffer(device, joint_O.constraint_indices, joint_I.constraint_indices)
				<< upload_buffer(device, joint_O.anchor_a_local, joint_I.anchor_a_local)
				<< upload_buffer(device, joint_O.anchor_b_local, joint_I.anchor_b_local)
				<< upload_buffer(device, joint_O.rest_position_delta, joint_I.rest_position_delta)
				<< upload_buffer(device, joint_O.rest_rot_col0_a_to_b, joint_I.rest_rot_col0_a_to_b)
				<< upload_buffer(device, joint_O.rest_rot_col1_a_to_b, joint_I.rest_rot_col1_a_to_b)
				<< upload_buffer(device, joint_O.rest_rot_col2_a_to_b, joint_I.rest_rot_col2_a_to_b)
				<< upload_buffer(device, joint_O.axis_world, joint_I.axis_world)
				<< upload_buffer(device, joint_O.axis_a_local, joint_I.axis_a_local)
				<< upload_buffer(device, joint_O.axis_b_local, joint_I.axis_b_local)
				<< upload_buffer(device, joint_O.stiffness, joint_I.stiffness)
				<< upload_buffer(device, joint_O.joint_type, joint_I.joint_type)
				<< upload_buffer(device, joint_O.slide_limits, joint_I.slide_limits)
				<< upload_buffer(device, joint_O.constraint_gradients, joint_I.constraint_gradients)
				<< upload_buffer(device, joint_O.constraint_hessians, joint_I.constraint_hessians)
				<< upload_buffer(device, joint_O.constraint_offsets_in_adjlist, joint_I.constraint_offsets_in_adjlist)
				<< upload_buffer(device, joint_O.vert_adj_constraints_csr, joint_I.vert_adj_constraints_csr);
		}

		stream << upload_buffer(device,
			output_data->sa_vert_affine_bodies_id,
			input_data->sa_vert_affine_bodies_id); // Basic information

		stream << luisa::compute::synchronize();
	}

	void init_colored_data(lcs::SimulationData<std::vector>* sim_data)
	{
		// Constraint Graph Coloring
		std::vector<std::vector<uint>> tmp_clusterd_constraint_stretch_mass_spring;
		std::vector<std::vector<uint>> tmp_clusterd_constraint_bending;
		auto*						   colored_data = &sim_data->colored_data;
		{
			auto& stretch_spring_data = sim_data->get_stretch_spring_data();
			fn_graph_coloring_per_constraint("Distance  Spring Constraint",
				tmp_clusterd_constraint_stretch_mass_spring,
				stretch_spring_data.vert_adj_constraints,
				stretch_spring_data.constraint_indices,
				2);

			auto& bending_edge_data = sim_data->get_bending_edge_data();
			fn_graph_coloring_per_constraint("Bending   Angle  Constraint",
				tmp_clusterd_constraint_bending,
				bending_edge_data.vert_adj_constraints,
				bending_edge_data.constraint_indices,
				4);

			colored_data->num_clusters_springs = tmp_clusterd_constraint_stretch_mass_spring.size();
			colored_data->num_clusters_bending_edges = tmp_clusterd_constraint_bending.size();

			fn_get_prefix(colored_data->sa_prefix_merged_springs, tmp_clusterd_constraint_stretch_mass_spring);
			fn_get_prefix(colored_data->sa_prefix_merged_bending_edges, tmp_clusterd_constraint_bending);

			upload_2d_csr_from(colored_data->sa_clusterd_springs, tmp_clusterd_constraint_stretch_mass_spring);
			upload_2d_csr_from(colored_data->sa_clusterd_bending_edges, tmp_clusterd_constraint_bending);
		}

		// Vertex Block Descent Coloring
		{
			// Graph Coloring
			const uint num_dof = sim_data->num_dof;
			colored_data->sa_Hf.resize(num_dof * 12);
			colored_data->sa_Hf1.resize(num_dof);

			const std::vector<std::vector<uint>>& vert_adj_verts = sim_data->vert_adj_material_force_verts;
			std::vector<std::vector<uint>>		  clusterd_vertices_bending;
			std::vector<uint>					  prefix_vertices_bending;

			fn_graph_coloring_per_vertex(vert_adj_verts, clusterd_vertices_bending, prefix_vertices_bending);
			colored_data->num_clusters_per_vertex_with_material_constraints = clusterd_vertices_bending.size();
			upload_from(colored_data->prefix_per_vertex_with_material_constraints, prefix_vertices_bending);
			upload_2d_csr_from(colored_data->clusterd_per_vertex_with_material_constraints, clusterd_vertices_bending);

			// Reverse map
			colored_data->per_vertex_bending_cluster_id.resize(num_dof);
			for (uint cluster = 0; cluster < colored_data->num_clusters_per_vertex_with_material_constraints; cluster++)
			{
				const uint next_prefix = colored_data->clusterd_per_vertex_with_material_constraints[cluster + 1];
				const uint curr_prefix = colored_data->clusterd_per_vertex_with_material_constraints[cluster];
				const uint num_verts_cluster = next_prefix - curr_prefix;
				CpuParallel::parallel_for(0,
					num_verts_cluster,
					[&](const uint i)
					{
						const uint vid =
							colored_data->clusterd_per_vertex_with_material_constraints[curr_prefix + i];
						colored_data->per_vertex_bending_cluster_id[vid] = cluster;
					});
			}
		}

		// Colored contraint precomputation
		{
			// Spring Constraint
			{
				auto&	   stretch_spring_data = sim_data->get_stretch_spring_data();
				const uint num_stretch_springs = stretch_spring_data.get_num_indices();

				colored_data->sa_merged_stretch_springs.resize(num_stretch_springs);
				colored_data->sa_merged_stretch_spring_rest_length.resize(num_stretch_springs);
				colored_data->sa_lambda_stretch_mass_spring.resize(num_stretch_springs);

				uint prefix = 0;
				for (uint cluster = 0; cluster < tmp_clusterd_constraint_stretch_mass_spring.size(); cluster++)
				{
					const auto& curr_cluster = tmp_clusterd_constraint_stretch_mass_spring[cluster];
					CpuParallel::parallel_for(0,
						curr_cluster.size(),
						[&](const uint i)
						{
							const uint eid = curr_cluster[i];
							{
								colored_data->sa_merged_stretch_springs[prefix + i] =
									stretch_spring_data.constraint_indices[eid];
								colored_data->sa_merged_stretch_spring_rest_length[prefix + i] =
									stretch_spring_data.sa_stretch_spring_rest_state_length[eid];
							}
						});
					prefix += curr_cluster.size();
				}
				if (prefix != stretch_spring_data.constraint_indices.size())
					LUISA_ERROR("Sum of Mass Spring Cluster Is Not Equal  Than Orig");
			}

			// Bending Constraint
			{
				auto&	   bending_edge_data = sim_data->get_bending_edge_data();
				const uint num_bending_edges = bending_edge_data.get_num_indices();

				colored_data->sa_merged_bending_edges.resize(num_bending_edges);
				colored_data->sa_merged_bending_edges_angle.resize(num_bending_edges);
				colored_data->sa_merged_bending_edges_Q.resize(num_bending_edges);
				colored_data->sa_lambda_bending.resize(num_bending_edges);

				uint prefix = 0;
				for (uint cluster = 0; cluster < tmp_clusterd_constraint_bending.size(); cluster++)
				{
					const auto& curr_cluster = tmp_clusterd_constraint_bending[cluster];
					CpuParallel::parallel_for(0,
						curr_cluster.size(),
						[&](const uint i)
						{
							const uint eid = curr_cluster[i];
							{
								colored_data->sa_merged_bending_edges[prefix + i] =
									bending_edge_data.constraint_indices[eid];
								colored_data->sa_merged_bending_edges_angle[prefix + i] =
									bending_edge_data.sa_bending_edges_rest_angle[eid];
								colored_data->sa_merged_bending_edges_Q[prefix + i] =
									bending_edge_data.sa_bending_edges_Q[eid];
							}
						});
					prefix += curr_cluster.size();
				}
				if (prefix != bending_edge_data.constraint_indices.size())
					LUISA_ERROR("Sum of Bending Cluster Is Not Equal Than Orig");
			}
		}
	}

	void upload_colored_data(luisa::compute::Device& device,
		luisa::compute::Stream&						 stream,
		lcs::SimulationData<std::vector>*			 input_data,
		lcs::SimulationData<luisa::compute::Buffer>* output_data)
	{
		output_data->colored_data.num_clusters_springs = input_data->colored_data.num_clusters_springs;
		output_data->colored_data.num_clusters_bending_edges = input_data->colored_data.num_clusters_bending_edges;
		output_data->colored_data.num_clusters_per_vertex_with_material_constraints =
			input_data->colored_data.num_clusters_per_vertex_with_material_constraints;

		auto& colored_data_I = input_data->colored_data;
		auto& colored_data_O = output_data->colored_data;
		if (!colored_data_I.sa_merged_stretch_springs.empty())
		{
			stream
				<< upload_buffer(device, colored_data_O.sa_merged_stretch_springs, colored_data_I.sa_merged_stretch_springs)
				<< upload_buffer(device, colored_data_O.sa_merged_stretch_spring_rest_length, colored_data_I.sa_merged_stretch_spring_rest_length)
				<< upload_buffer(device, colored_data_O.sa_clusterd_springs, colored_data_I.sa_clusterd_springs)
				<< upload_buffer(device, colored_data_O.sa_prefix_merged_springs, colored_data_I.sa_prefix_merged_springs)
				<< upload_buffer(device, colored_data_O.sa_lambda_stretch_mass_spring, colored_data_I.sa_lambda_stretch_mass_spring);
		}
		if (!colored_data_I.sa_merged_bending_edges.empty())
		{
			stream
				<< upload_buffer(device, colored_data_O.sa_merged_bending_edges, colored_data_I.sa_merged_bending_edges)
				<< upload_buffer(device, colored_data_O.sa_merged_bending_edges_angle, colored_data_I.sa_merged_bending_edges_angle)
				<< upload_buffer(device, colored_data_O.sa_merged_bending_edges_Q, colored_data_I.sa_merged_bending_edges_Q)
				<< upload_buffer(device, colored_data_O.sa_clusterd_bending_edges, colored_data_I.sa_clusterd_bending_edges)
				<< upload_buffer(device, colored_data_O.sa_prefix_merged_bending_edges, colored_data_I.sa_prefix_merged_bending_edges)
				<< upload_buffer(device, colored_data_O.sa_lambda_bending, colored_data_I.sa_lambda_bending);
		}
		if (!colored_data_I.prefix_per_vertex_with_material_constraints.empty())
		{
			stream << upload_buffer(device,
				colored_data_O.prefix_per_vertex_with_material_constraints,
				colored_data_I.prefix_per_vertex_with_material_constraints)
				   << upload_buffer(device,
						  colored_data_O.clusterd_per_vertex_with_material_constraints,
						  colored_data_I.clusterd_per_vertex_with_material_constraints)
				   << upload_buffer(device, colored_data_O.per_vertex_bending_cluster_id, colored_data_I.per_vertex_bending_cluster_id)
				   << upload_buffer(device, colored_data_O.sa_Hf, colored_data_I.sa_Hf)
				   << upload_buffer(device, colored_data_O.sa_Hf1, colored_data_I.sa_Hf1);
		}
		stream << luisa::compute::synchronize();
	}

	void resize_pcg_data(luisa::compute::Device&	 device,
		luisa::compute::Stream&						 stream,
		lcs::MeshData<std::vector>*					 mesh_data,
		lcs::SimulationData<std::vector>*			 host_data,
		lcs::SimulationData<luisa::compute::Buffer>* device_data)
	{
		const uint num_verts = host_data->num_dof;

		resize_buffer(host_data->sa_cgX, num_verts);
		resize_buffer(host_data->sa_cgB, num_verts);
		resize_buffer(host_data->sa_cgA_diag, num_verts);

		resize_buffer(host_data->sa_cgMutex, num_verts);
		resize_buffer(host_data->sa_cgMinv, num_verts);
		resize_buffer(host_data->sa_cgP, num_verts);
		resize_buffer(host_data->sa_cgQ, num_verts);
		resize_buffer(host_data->sa_cgR, num_verts);
		resize_buffer(host_data->sa_cgZ, num_verts);
		resize_buffer(host_data->sa_block_result, num_verts);
		resize_buffer(host_data->sa_convergence, 10240);

		resize_buffer(device, device_data->sa_cgX, num_verts);
		resize_buffer(device, device_data->sa_cgB, num_verts);
		resize_buffer(device, device_data->sa_cgA_diag, num_verts);
		resize_buffer(device, device_data->sa_cgMinv, num_verts);
		resize_buffer(device, device_data->sa_cgP, num_verts);
		resize_buffer(device, device_data->sa_cgQ, num_verts);
		resize_buffer(device, device_data->sa_cgR, num_verts);
		resize_buffer(device, device_data->sa_cgZ, num_verts);
		resize_buffer(device, device_data->sa_block_result, num_verts);
		resize_buffer(device, device_data->sa_convergence, 10240);
	}

} // namespace lcs::Initializer
