"""
Visualization utilities for LuisaComputeSimulator powered by polyscope.

Quick start::

    from lcs_gui import SimulationGUI
    gui = SimulationGUI(solver, config, output_dir=".")
    gui.show()  # opens the polyscope window (requires `pip install polyscope`)

Sub-modules:
    polyscope_gui   — ``SimulationGUI`` class (lazy-imports polyscope)
    mesh_proc       — ``write_obj``, ``get_sample_tet_grid``
"""

from .polyscope_gui import SimulationGUI

__all__ = ["SimulationGUI"]
