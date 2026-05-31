"""
Base class for LuisaComputeSimulator test scripts.

Encapsulates the common headless/GUI execution loop so test scripts only
override simulation-specific hooks.

Usage::

    from utils.test_runner import TestRunner

    class MyTest(TestRunner):
        def on_pre_step(self, frame_idx):
            update_animation()

        def on_post_step(self, frame_idx):
            if frame_idx % 50 == 0:
                m = compute_metrics()
                print_metrics(m)

    runner = MyTest(solver, config_ref, output_dir, headless=args.headless)
    runner.run(advance_frames=args.advance_frames)
"""

import os


class TestRunner:
    """Headless / GUI execution loop that delegates to ``on_pre_step`` / ``on_post_step``.

    Parameters
    ----------
    solver : lcs_py.NewtonSolver
        Initialised solver (``init_solver()`` must have been called).
    config_ref : lcs_py.SceneParams
        Reference from ``solver.get_config()``.
    output_dir : str
        Directory for OBJ/result exports.
    headless : bool
        If True, run the headless loop; otherwise open the polyscope GUI.
    output_frequency : int
        Frame output cadence.  ``<= 0`` disables intermediate saves,
        ``1`` saves every frame, ``N`` saves every N-th frame.
    init_output_name : str or None
        OBJ name for the initial-state save (before any step).  Set to
        ``None`` to skip.
    result_output_name : str or None
        OBJ name for the final-state save (after the last step).  Set to
        ``None`` to skip.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        solver,
        config_ref,
        output_dir,
        headless=False,
        output_frequency=-1,
        init_output_name=f"frame_{0:04}.obj",
        result_output_name="frame_last.obj",
    ):
        self.solver = solver
        self.config = config_ref
        self.output_dir = output_dir
        self.headless = headless
        self.output_frequency = output_frequency
        self.init_output_name = init_output_name
        self.result_output_name = result_output_name

    # ------------------------------------------------------------------
    # Hooks — override in subclasses
    # ------------------------------------------------------------------

    def on_pre_step(self, frame_idx: int):
        """Called **before** each physics step.

        Override to update animations, targets, or tracking state.
        """

    def on_post_step(self, frame_idx: int):
        """Called **after** each physics step.

        Override to compute metrics, print progress, etc.
        """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def physics_step(self):
        """Run one simulation step (GPU or CPU depending on config)."""
        if self.config.get_use_gpu():
            self.solver.physics_step_gpu()
        else:
            self.solver.physics_step_cpu()

    def should_output(self, frame_idx: int) -> bool:
        """Return True if *frame_idx* should produce an output file."""
        if self.output_frequency <= 0:
            return False
        return frame_idx % self.output_frequency == 0

    def _output_path(self, name: str) -> str:
        return os.path.join(self.output_dir, name)

    # ------------------------------------------------------------------
    # Execution loops
    # ------------------------------------------------------------------

    def run_headless(self, advance_frames: int):
        """Headless loop: save init, step N frames, save result."""
        if self.init_output_name:
            self.solver.save_sim_result(self._output_path(self.init_output_name))

        for frame in range(advance_frames):
            self.on_pre_step(frame)
            self.physics_step()
            self.on_post_step(frame)

            if self.should_output(frame):
                self.solver.save_sim_result(
                    self._output_path(f"frame_{frame}.obj")
                )

        if self.result_output_name:
            self.solver.save_sim_result(self._output_path(self.result_output_name))

    def run_gui(self):
        """Open the polyscope GUI window (blocking)."""
        import utils.polyscope_gui

        outer = self

        class _GUISubclass(utils.polyscope_gui.SimulationGUI):
            def _physics_step(self):
                frame = int(self._config.current_frame)
                outer.on_pre_step(frame)
                super()._physics_step()
                outer.on_post_step(frame)

                if outer.should_output(frame):
                    outer.solver.save_sim_result(
                        outer._output_path(f"frame_{frame}.obj")
                    )

        gui = _GUISubclass(self.solver, self.config, self.output_dir)
        gui.show()

    def run(self, advance_frames: int = 0):
        """Dispatch to headless or GUI mode.

        In GUI mode *advance_frames* is ignored — the loop runs until the
        user closes the window.
        """
        if self.headless:
            self.run_headless(advance_frames)
        else:
            self.run_gui()
