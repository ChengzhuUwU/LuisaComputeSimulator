"""
Polyscope-based GUI for LuisaComputeSimulator.

Encapsulates all polyscope initialisation, mesh registration, ImGui panels,
and the continuous-simulation callback.  Usage:

    from polyscope_gui import SimulationGUI
    gui = SimulationGUI(solver, config_ref, output_dir)
    gui.show()   # blocking – opens the polyscope window
"""

import os
import polyscope as ps
import polyscope.imgui as psim


class SimulationGUI:
    """Thin wrapper around polyscope that mirrors the C++ Polyscope GUI."""

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
        for idx, name in enumerate(self._mesh_names):
            ps_mesh = ps.register_surface_mesh(
                f"{name}{idx}", verts_list[idx], faces_list[idx]
            )
            ps_mesh.set_enabled(True)
            self._surface_meshes.append(ps_mesh)

    def _update_gui_vertices(self):
        """Fetch latest simulation vertices and push them to polyscope."""
        v_list, _ = self._solver.get_sim_result()
        for idx in range(len(self._surface_meshes)):
            self._surface_meshes[idx].update_vertex_positions(v_list[idx])

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
            _, cfg.stiffness_bending_ui = psim.SliderFloat(
                "Bending Stiffness", cfg.stiffness_bending_ui, v_min=0.0, v_max=10.0
            )
            _, cfg.print_system_energy = psim.Checkbox(
                "Print Energy", cfg.print_system_energy
            )
            _, cfg.use_gpu = psim.Checkbox("Use GPU Solver", cfg.use_gpu)
            _, cfg.use_self_collision = psim.Checkbox(
                "Use Self-Collision", cfg.use_self_collision
            )
            psim.TreePop()

    # ---- Simulation panel ------------------------------------------------

    def _panel_simulation(self):
        cfg = self._config
        if psim.TreeNode("Simulation"):
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
            if psim.Button("Save mesh"):
                self._solver.save_sim_result(
                    obj_path=os.path.join(
                        self._output_dir, f"frame_{cfg.current_frame}.obj"
                    )
                )
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
            self._solver.save_sim_result(
                obj_path=os.path.join(self._output_dir, "frame_0_init.obj")
            )

        self._physics_step()
        self._update_gui_vertices()

        if cfg.output_per_frame:
            animation_fps = 60.0
            output_freq = max(1, int((1.0 / animation_fps) / cfg.implicit_dt))
            if cfg.current_frame % output_freq == 0:
                self._solver.save_sim_result(
                    obj_path=os.path.join(
                        self._output_dir, f"frame_{cfg.current_frame}.obj"
                    )
                )
