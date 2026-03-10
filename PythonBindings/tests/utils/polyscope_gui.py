"""
Polyscope-based GUI for LuisaComputeSimulator.

Encapsulates all polyscope initialisation, mesh registration, ImGui panels,
and the continuous-simulation callback.  Usage:

    from polyscope_gui import SimulationGUI
    gui = SimulationGUI(solver, config_ref, output_dir)
    gui.show()   # blocking – opens the polyscope window
"""

import os
import numpy as np
import polyscope as ps
import polyscope.imgui as psim


class SimulationGUI:
    """Thin wrapper around polyscope that mirrors the C++ Polyscope GUI."""

    MERGE_RENDER_THRESHOLD = 20

    def __init__(self, solver, config_ref, output_dir: str = "."):
        """
        Parameters
        ----------
        solver : lcs_py.NewtonSolver
            An already-initialised solver (``init_solver()`` must have been called).
        config_ref : lcs_py.SceneParams
            Reference returned by ``lcs.get_scene_params()``.
        output_dir : str
            Directory used when saving OBJ files from the GUI.
        """
        self._solver = solver
        self._config = config_ref
        self._output_dir = output_dir

        # UI state
        self._is_simulating = False

        # Polyscope handles
        self._surface_meshes: list = []
        self._mesh_names: list = []

        # Rendering mode state
        self._use_merged_render = False

        # Global-vertex highlight state
        self._global_to_local_vid = None
        self._global_to_world_data_idx = None
        self._highlight_global_vid = 0
        self._highlight_mesh_idx = None
        self._highlight_local_vid = None
        self._highlight_cloud = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(self):
        """Initialise polyscope, register meshes, set the callback and open the window."""
        ps.init()
        self._register_meshes()
        ps.set_user_callback(self._ui_callback)
        ps.show()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_meshes(self):
        verts_list, faces_list = self._solver.get_sim_result()
        self._mesh_names = self._solver.get_mesh_names()
        self._surface_meshes = []

        mesh_count = len(verts_list)
        self._use_merged_render = mesh_count > self.MERGE_RENDER_THRESHOLD

        if self._use_merged_render:
            self._register_merged_mesh(verts_list, faces_list)
            return

        for idx, name in enumerate(self._mesh_names):
            ps_mesh = ps.register_surface_mesh(
                f"{name}{idx}", verts_list[idx], faces_list[idx]
            )
            ps_mesh.set_enabled(True)
            self._surface_meshes.append(ps_mesh)

    def _register_merged_mesh(self, verts_list, faces_list):
        """Merge all objects into one render mesh and mark each object's faces by color."""
        merged_verts = []
        merged_faces = []
        merged_face_colors = []

        vertex_offset = 0

        for mesh_idx, (verts, faces) in enumerate(zip(verts_list, faces_list)):
            verts_np = np.asarray(verts)
            faces_np = np.asarray(faces, dtype=np.int32)

            merged_verts.append(verts_np)
            merged_faces.append(faces_np + vertex_offset)

            # Stable pseudo-random color per object id.
            obj_color = self._face_color_from_index(mesh_idx)
            merged_face_colors.append(np.tile(obj_color, (faces_np.shape[0], 1)))

            vertex_offset += verts_np.shape[0]

        all_verts = np.concatenate(merged_verts, axis=0)
        all_faces = np.concatenate(merged_faces, axis=0)
        all_face_colors = np.concatenate(merged_face_colors, axis=0)

        merged_mesh = ps.register_surface_mesh("merged_scene", all_verts, all_faces)
        merged_mesh.add_color_quantity(
            "object_face_color", all_face_colors, defined_on="faces", enabled=True
        )
        merged_mesh.set_enabled(True)
        self._surface_meshes = [merged_mesh]

    def _face_color_from_index(self, idx: int):
        """Generate a deterministic, visually-separated RGB color."""
        # Golden-ratio hue stepping gives well-spaced colors for many objects.
        hue = (0.61803398875 * (idx + 1)) % 1.0
        sat = 0.65
        val = 0.95
        return np.asarray(self._hsv_to_rgb(hue, sat, val), dtype=np.float32)

    @staticmethod
    def _hsv_to_rgb(h, s, v):
        i = int(h * 6.0)
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i %= 6
        if i == 0:
            return [v, t, p]
        if i == 1:
            return [q, v, p]
        if i == 2:
            return [p, v, t]
        if i == 3:
            return [p, q, v]
        if i == 4:
            return [t, p, v]
        return [v, p, q]

    def _update_gui_vertices(self):
        """Fetch latest simulation vertices and push them to polyscope."""
        v_list, _ = self._solver.get_sim_result()

        if self._use_merged_render:
            merged_v = np.concatenate([np.asarray(v) for v in v_list], axis=0)
            self._surface_meshes[0].update_vertex_positions(merged_v)
        else:
            for idx in range(len(self._surface_meshes)):
                self._surface_meshes[idx].update_vertex_positions(v_list[idx])

        self._update_highlight_marker(v_list)

    def _resolve_global_vid(self, global_vid: int):
        world_data_idx = int(self._solver.query_registration_vid_from_global_vid(global_vid))
        local_vid = int(self._solver.query_local_vid_from_global_vid(global_vid))
        return world_data_idx, local_vid

    def _set_highlight_global_vid(self, global_vid: int):
        world_data_idx, local_vid = self._resolve_global_vid(global_vid)
        self._highlight_global_vid = int(global_vid)
        self._highlight_mesh_idx = world_data_idx
        self._highlight_local_vid = local_vid

    def _update_highlight_marker(self, v_list):
        if self._highlight_mesh_idx is None or self._highlight_local_vid is None:
            return

        if self._highlight_mesh_idx < 0 or self._highlight_mesh_idx >= len(v_list):
            return

        verts = np.asarray(v_list[self._highlight_mesh_idx])
        if self._highlight_local_vid < 0 or self._highlight_local_vid >= verts.shape[0]:
            return

        point = np.asarray([verts[self._highlight_local_vid]], dtype=np.float32)
        if self._highlight_cloud is None:
            self._highlight_cloud = ps.register_point_cloud("highlight_vertex", point)
            color = np.asarray([[1.0, 0.15, 0.1]], dtype=np.float32)
            self._highlight_cloud.add_color_quantity(
                "highlight_color", color, enabled=True
            )
        else:
            self._highlight_cloud.update_point_positions(point)

    def _physics_step(self):
        """Run one simulation step (GPU or CPU depending on config)."""
        if self._config.use_gpu:
            self._solver.physics_step_gpu()
        else:
            self._solver.physics_step_cpu()

    # ------------------------------------------------------------------
    # ImGui callback (called every frame by polyscope)
    # ------------------------------------------------------------------

    def _ui_callback(self):
        self._handle_keyboard()
        self._panel_parameters()
        self._panel_simulation()
        self._panel_highlight()
        self._panel_collision()
        self._panel_data_io()
        self._continuous_simulation_loop()

    # ---- Keyboard shortcuts ----------------------------------------------

    def _handle_keyboard(self):
        # Escape -> close window  (ImGuiKey_Escape = 526)
        if psim.IsKeyPressed(psim.ImGuiKey_Escape):
            ps.unshow()
        # Space -> advance single frame  (ImGuiKey_Space = 525)
        if psim.IsKeyPressed(psim.ImGuiKey_Space):
            self._physics_step()
            self._update_gui_vertices()

    # ---- Parameters panel ------------------------------------------------

    def _panel_parameters(self):
        cfg = self._config
        if psim.TreeNode("Parameters"):
            _, cfg.nonlinear_iter_count = psim.InputInt(
                "Num Nonlinear-Iteration", cfg.nonlinear_iter_count
            )
            _, cfg.pcg_iter_count = psim.InputInt(
                "Num PCG-Iteration", cfg.pcg_iter_count
            )
            _, cfg.implicit_dt = psim.SliderFloat(
                "Implicit Timestep", cfg.implicit_dt, v_min=0.0001, v_max=0.2
            )
            _, cfg.use_energy_linesearch = psim.Checkbox(
                "Use Energy LineSearch", cfg.use_energy_linesearch
            )
            _, cfg.use_ccd_linesearch = psim.Checkbox(
                "Use CCD LineSearch", cfg.use_ccd_linesearch
            )
            _, cfg.print_system_energy = psim.Checkbox(
                "Print Energy", cfg.print_system_energy
            )
            _, cfg.use_gpu = psim.Checkbox("Use GPU Solver", cfg.use_gpu)
            _, cfg.use_self_collision = psim.Checkbox(
                "Use Self-Collision", cfg.use_self_collision
            )
            psim.TreePop()
        if cfg.print_system_energy:
            cfg.use_energy_linesearch = True 
            # energy = self._solver.get_system_energy()
            # psim.TextUnformatted(f"System Energy: {energy:.6e}")

    # ---- Simulation panel ------------------------------------------------

    def _panel_simulation(self):
        cfg = self._config
        # Prefer TreeNodeEx default-open; gracefully fallback for older bindings.
        tree_node_ex = getattr(psim, "TreeNodeEx", None)
        if tree_node_ex is not None:
            default_open_flag = getattr(psim, "ImGuiTreeNodeFlags_DefaultOpen", 0)
            is_open = tree_node_ex("Simulation", default_open_flag)
        else:
            set_next_item_open = getattr(psim, "SetNextItemOpen", None)
            if set_next_item_open is not None:
                set_next_item_open(True, getattr(psim, "ImGuiCond_Once", 0))
            is_open = psim.TreeNode("Simulation")

        if is_open:
            psim.TextUnformatted(f"Frame {cfg.current_frame}")

            if psim.Button("Reset"):
                cfg.current_frame = 0
                self._solver.restart_system()
                self._update_gui_vertices()

            if psim.Button("Advance Single Frame"):
                self._physics_step()
                self._update_gui_vertices()

            if psim.Button("Start Simulation"):
                self._is_simulating = True

            if psim.Button("End Simulation"):
                self._is_simulating = False

            psim.TreePop()

    # ---- Highlight panel ------------------------------------------------
    def _panel_highlight(self):
        if psim.TreeNode("Highlight Vertex"):
            changed, global_vid = psim.InputInt(
                "Highlight Global Vertex ID", self._highlight_global_vid
            )
            if changed:
                self._highlight_global_vid = max(0, int(global_vid))

            if psim.Button("Highlight Vertex"):
                try:
                    self._set_highlight_global_vid(self._highlight_global_vid)
                    self._update_gui_vertices()
                except Exception as exc:
                    print(f"[SimulationGUI] failed to highlight vertex: {exc}")

            if (
                self._highlight_mesh_idx is not None
                and self._highlight_local_vid is not None
            ):
                psim.TextUnformatted(
                    f"global={self._highlight_global_vid}, mesh_idx={self._highlight_mesh_idx}, local_vid={self._highlight_local_vid}"
                )

            psim.TreePop()

    # ---- Collision panel -------------------------------------------------

    def _panel_collision(self):
        cfg = self._config
        if cfg.use_floor:
            ps.set_ground_plane_mode("tile_reflection")
            ps.set_ground_plane_height_mode("manual")
            ps.set_ground_plane_height(cfg.floor.y)
        else:
            ps.set_ground_plane_mode("none")

        if psim.TreeNode("Collision"):
            _, cfg.use_floor = psim.Checkbox("Use Ground Collision", cfg.use_floor)
            if cfg.use_floor:
                changed, new_floor_y = psim.SliderFloat(
                    "Floor Y", cfg.floor.y, v_min=-1.0, v_max=1.0
                )
                if changed:
                    floor_vec = cfg.floor
                    floor_vec.y = new_floor_y
                    cfg.floor = floor_vec
            psim.TreePop()

    # ---- Data IO panel ---------------------------------------------------

    def _panel_data_io(self):
        cfg = self._config
        if psim.TreeNode("Data IO"):
            if psim.Button("Save mesh combined"):
                obj_path = os.path.join(self._output_dir, f"frame_{cfg.current_frame}_combined.obj")
                self._solver.save_sim_result(obj_path=obj_path)
                print(f"Saved combined mesh to {obj_path}")
            if psim.Button("Save mesh separate"):
                from utils.mesh_proc import write_obj
                v_list, f_list = self._solver.get_sim_result()
                for idx in range(len(v_list)):
                    verts = np.asarray(v_list[idx])
                    faces = np.asarray(f_list[idx], dtype=np.int32)
                    obj_path = os.path.join(self._output_dir, f"frame_{cfg.current_frame}_mesh{idx}.obj")
                    write_obj(
                        obj_path,
                        verts,
                        faces,
                    )
                    print(f"Saved mesh {idx} to {obj_path}")
            _, cfg.output_per_frame = psim.Checkbox(
                "Output Each Frame", cfg.output_per_frame
            )
            psim.TreePop()

    # ---- Continuous simulation loop --------------------------------------

    def _continuous_simulation_loop(self):
        if not self._is_simulating:
            return

        cfg = self._config

        if cfg.output_per_frame and cfg.current_frame == 0:
            obj_path = os.path.join(self._output_dir, f"frame_0_init.obj")
            self._solver.save_sim_result(obj_path=obj_path)

        self._physics_step()
        self._update_gui_vertices()

        if cfg.output_per_frame:
            animation_fps = 60.0
            output_freq = max(1, int((1.0 / animation_fps) / cfg.implicit_dt))
            if cfg.current_frame % output_freq == 0:
                obj_path = os.path.join(self._output_dir, f"frame_{cfg.current_frame}.obj")
                self._solver.save_sim_result(obj_path=obj_path)
