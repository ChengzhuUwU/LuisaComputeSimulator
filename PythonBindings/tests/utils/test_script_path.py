"""Project root path for test scripts.

Usage:
    from utils.test_script_path import PROJECT_ROOT
    cube_mesh_path = os.path.join(PROJECT_ROOT, "Resources", "InputMesh", "cube.obj")
    output_dir = os.path.join(PROJECT_ROOT, "Resources", "OutputMesh")
"""

import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
