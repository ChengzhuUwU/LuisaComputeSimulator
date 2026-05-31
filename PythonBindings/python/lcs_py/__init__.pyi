"""
Python bindings for basic NewtonSolver scene building (lightweight)
"""
from __future__ import annotations
import numpy
import pybind11_stubgen.typing_ext
import typing
__all__: list[str] = ['Cloth', 'ConstWorldData', 'FixedPointsType', 'Float3', 'MakeFixedPointsInterface', 'MaterialType', 'NewtonSolver', 'Particle', 'Rigid', 'Rod', 'SceneParams', 'Tetrahedral', 'WorldData']
class ConstWorldData:
    def get_fixed_point_indices(self) -> list:
        """
        Return currently registered fixed-point local vertex indices as a Python list
        """
    def get_name(self) -> str:
        """
        Return object name.
        """
    def get_registration_index(self) -> int:
        """
        Return object registration id.
        """
    def get_rest_positions(self) -> numpy.ndarray[numpy.float32]:
        """
        Return rest positions (after object transform) as an (N,3) float32 numpy array
        """
    def get_rest_rotation(self) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)]:
        """
        Return rest Euler rotation as [x, y, z].
        """
    def get_rest_scale(self) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)]:
        """
        Return rest scale as [x, y, z].
        """
    def get_rest_translation(self) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)]:
        """
        Return rest translation as [x, y, z].
        """
class FixedPointsType:
    """
    Members:
    
      None_
    
      FromIndices
    
      FromFunction
    
      Left
    
      Right
    
      Front
    
      Back
    
      Up
    
      Down
    
      LeftUp
    
      LeftDown
    
      LeftFront
    
      LeftBack
    
      RightUp
    
      RightDown
    
      RightFront
    
      RightBack
    
      FrontUp
    
      FrontDown
    
      BackUp
    
      BackDown
    
      All
    """
    All: typing.ClassVar[FixedPointsType]  # value = <FixedPointsType.All: 21>
    Back: typing.ClassVar[FixedPointsType]  # value = <FixedPointsType.Back: 6>
    BackDown: typing.ClassVar[FixedPointsType]  # value = <FixedPointsType.BackDown: 20>
    BackUp: typing.ClassVar[FixedPointsType]  # value = <FixedPointsType.BackUp: 19>
    Down: typing.ClassVar[FixedPointsType]  # value = <FixedPointsType.Down: 8>
    FromFunction: typing.ClassVar[FixedPointsType]  # value = <FixedPointsType.FromFunction: 2>
    FromIndices: typing.ClassVar[FixedPointsType]  # value = <FixedPointsType.FromIndices: 1>
    Front: typing.ClassVar[FixedPointsType]  # value = <FixedPointsType.Front: 5>
    FrontDown: typing.ClassVar[FixedPointsType]  # value = <FixedPointsType.FrontDown: 18>
    FrontUp: typing.ClassVar[FixedPointsType]  # value = <FixedPointsType.FrontUp: 17>
    Left: typing.ClassVar[FixedPointsType]  # value = <FixedPointsType.Left: 3>
    LeftBack: typing.ClassVar[FixedPointsType]  # value = <FixedPointsType.LeftBack: 12>
    LeftDown: typing.ClassVar[FixedPointsType]  # value = <FixedPointsType.LeftDown: 10>
    LeftFront: typing.ClassVar[FixedPointsType]  # value = <FixedPointsType.LeftFront: 11>
    LeftUp: typing.ClassVar[FixedPointsType]  # value = <FixedPointsType.LeftUp: 9>
    None_: typing.ClassVar[FixedPointsType]  # value = <FixedPointsType.None_: 0>
    Right: typing.ClassVar[FixedPointsType]  # value = <FixedPointsType.Right: 4>
    RightBack: typing.ClassVar[FixedPointsType]  # value = <FixedPointsType.RightBack: 16>
    RightDown: typing.ClassVar[FixedPointsType]  # value = <FixedPointsType.RightDown: 14>
    RightFront: typing.ClassVar[FixedPointsType]  # value = <FixedPointsType.RightFront: 15>
    RightUp: typing.ClassVar[FixedPointsType]  # value = <FixedPointsType.RightUp: 13>
    Up: typing.ClassVar[FixedPointsType]  # value = <FixedPointsType.Up: 7>
    __members__: typing.ClassVar[dict[str, FixedPointsType]]  # value = {'None_': <FixedPointsType.None_: 0>, 'FromIndices': <FixedPointsType.FromIndices: 1>, 'FromFunction': <FixedPointsType.FromFunction: 2>, 'Left': <FixedPointsType.Left: 3>, 'Right': <FixedPointsType.Right: 4>, 'Front': <FixedPointsType.Front: 5>, 'Back': <FixedPointsType.Back: 6>, 'Up': <FixedPointsType.Up: 7>, 'Down': <FixedPointsType.Down: 8>, 'LeftUp': <FixedPointsType.LeftUp: 9>, 'LeftDown': <FixedPointsType.LeftDown: 10>, 'LeftFront': <FixedPointsType.LeftFront: 11>, 'LeftBack': <FixedPointsType.LeftBack: 12>, 'RightUp': <FixedPointsType.RightUp: 13>, 'RightDown': <FixedPointsType.RightDown: 14>, 'RightFront': <FixedPointsType.RightFront: 15>, 'RightBack': <FixedPointsType.RightBack: 16>, 'FrontUp': <FixedPointsType.FrontUp: 17>, 'FrontDown': <FixedPointsType.FrontDown: 18>, 'BackUp': <FixedPointsType.BackUp: 19>, 'BackDown': <FixedPointsType.BackDown: 20>, 'All': <FixedPointsType.All: 21>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Float3:
    x: float
    y: float
    z: float
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, x: float, y: float, z: float) -> None:
        ...
    def __repr__(self) -> str:
        ...
class MakeFixedPointsInterface:
    def __init__(self) -> None:
        ...
    @property
    def method(self) -> FixedPointsType:
        """
        Region selector for fixed points (FixedPointsType enum).
        """
    @method.setter
    def method(self, arg0: FixedPointsType) -> None:
        ...
    @property
    def range(self) -> float:
        """
        Tolerance used by region-based selectors (e.g. Left/Right).
        """
    @range.setter
    def range(self, arg0: float) -> None:
        ...
class MaterialType:
    """
    Members:
    
      Particle
    
      Cloth
    
      Tetrahedral
    
      Rigid
    
      Rod
    """
    Cloth: typing.ClassVar[MaterialType]  # value = <MaterialType.Cloth: 1>
    Particle: typing.ClassVar[MaterialType]  # value = <MaterialType.Particle: 0>
    Rigid: typing.ClassVar[MaterialType]  # value = <MaterialType.Rigid: 3>
    Rod: typing.ClassVar[MaterialType]  # value = <MaterialType.Rod: 4>
    Tetrahedral: typing.ClassVar[MaterialType]  # value = <MaterialType.Tetrahedral: 2>
    __members__: typing.ClassVar[dict[str, MaterialType]]  # value = {'Particle': <MaterialType.Particle: 0>, 'Cloth': <MaterialType.Cloth: 1>, 'Tetrahedral': <MaterialType.Tetrahedral: 2>, 'Rigid': <MaterialType.Rigid: 3>, 'Rod': <MaterialType.Rod: 4>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class NewtonSolver:
    def __init__(self) -> None:
        ...
    def add_fixed_joint(self, body_a_registration: int, body_b_registration: int, anchor_a_local: numpy.ndarray[numpy.float32], anchor_b_local: numpy.ndarray[numpy.float32], stiffness_pos: float = 10000.0, stiffness_rot: float = 1000.0) -> None:
        """
        Add a fixed joint between two rigid bodies.
        
        The local anchors are expressed in each body's rest local frame. The joint constrains
        both anchor coincidence and relative orientation.
        """
    def add_prismatic_joint(self, body_a_registration: int, body_b_registration: int, anchor_a_local: numpy.ndarray[numpy.float32], anchor_b_local: numpy.ndarray[numpy.float32], axis_world: numpy.ndarray[numpy.float32], stiffness_pos: float = 10000.0, stiffness_rot: float = 1000.0, slide_min: float = ..., slide_max: float = ...) -> None:
        """
        Add a prismatic joint between two rigid bodies.
        
        axis_world defines the free sliding axis in the rest pose. The joint constrains the
        relative offset perpendicular to the axis and locks relative orientation; slide_min
        and slide_max bound the scalar coordinate along the axis.
        """
    def add_revolute_joint(self, body_a_registration: int, body_b_registration: int, anchor_a_local: numpy.ndarray[numpy.float32], anchor_b_local: numpy.ndarray[numpy.float32], axis_world: numpy.ndarray[numpy.float32], axis_a_local: numpy.ndarray[numpy.float32], axis_b_local: numpy.ndarray[numpy.float32], stiffness_pos: float = 10000.0, stiffness_axis: float = 1000.0) -> None:
        """
        Add a revolute/hinge joint between two rigid bodies.
        
        The anchors are kept coincident while axis_a_local and axis_b_local are aligned,
        leaving rotation around the hinge axis free.
        """
    def cleanup_device(self) -> None:
        """
        Release owned device resources (no-op for borrowed device).
        """
    def create_world_data_from_array(self, name: str, vertices: numpy.ndarray[numpy.float64], triangles: numpy.ndarray[numpy.int32]) -> WorldData:
        ...
    def create_world_data_from_file_path(self, name: str, obj_file_path: str) -> WorldData:
        ...
    def create_world_data_from_tet_array(self, name: str, vertices: numpy.ndarray[numpy.float64], tets: numpy.ndarray[numpy.int32]) -> WorldData:
        """
        Create a tetrahedral-mesh WorldData from numpy arrays.
        
        vertices: (N,3) float array of rest-pose positions
        tets:     (M,4) int array of tetrahedron vertex indices
        Surface topology (faces, edges, surface_verts) is extracted automatically.
        Call set_physics_material_tet() on the returned object to set material params,
        then register_world_data() to add it to the solver.
        """
    def get_config(self) -> SceneParams:
        """
        Return reference to solver-owned SceneParams config
        """
    def get_device_ptr(self) -> int:
        """
        Return the raw pointer (as int) to the active luisa::compute::Device.
        """
    def get_mesh_names(self) -> list:
        """
        Return registered mesh names ordered by registration id.
        """
    def get_object_by_registration_id(self, registration_id: int) -> ConstWorldData:
        ...
    def get_object_sim_result_by_registration_id(self, registration_id: int) -> tuple:
        """
        Return one object simulation result as tuple (vertices, faces) by registration id
        """
    def get_rigid_body_rotation_axis_angle(self, registration_id: int) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)]:
        """
        Return rigid body rotation vector (axis * angle_rad) as (rx, ry, rz).
        """
    def get_rigid_body_rotation_quaternion(self, registration_id: int) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(4)]:
        """
        Return rigid body rotation quaternion as (qx, qy, qz, qw).
        """
    def get_rigid_body_scaling(self, registration_id: int) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)]:
        """
        Return rigid body scaling extracted from ABD affine matrix as (sx, sy, sz).
        """
    def get_rigid_body_translation(self, registration_id: int) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)]:
        """
        Return rigid body translation as (tx, ty, tz).
        """
    def get_sim_result(self) -> tuple:
        """
        Return simulation results as a tuple (vertices_list, faces_list) of numpy arrays
        """
    def get_stream_ptr(self) -> int:
        """
        Return the raw pointer (as int) to the active luisa::compute::Stream.
        """
    def get_vert_mass(self, global_vid: int) -> float:
        """
        Return mass of a vertex by global vertex id
        """
    def init_device(self, backend_name: typing.Any = None, binary_path: typing.Any = None) -> None:
        """
        Create and own a luisa compute device/stream.
        
        backend_name: optional backend string (e.g. 'cuda','metal','dx')
        binary_path: optional binary path passed to luisa::compute::Context
        """
    def init_solver(self) -> None:
        """
        Initialize the underlying solver using the device set via init_device()/set_device()
        """
    def load_scene_from_json(self, json_path: str) -> None:
        """
        Load world_data and scene params from a JSON scene file (same format as app_simulation).
        """
    def num_meshes(self) -> int:
        """
        Return the number of registered mesh/world objects.
        """
    def physics_step_cpu(self) -> None:
        """
        Advance one simulation frame using the CPU solver path.
        """
    def physics_step_gpu(self) -> None:
        """
        Advance one simulation frame using the GPU solver path.
        """
    def print_registered_meshes_info(self) -> None:
        """
        Print registered meshes info
        """
    def query_local_vid_from_global_vid(self, global_vid: int) -> int:
        """
        Look up the local vertex index for a given global vertex id.
        """
    def query_registration_vid_from_global_vid(self, global_vid: int) -> int:
        """
        Look up the world_data (mesh) registration index for a given global vertex id.
        """
    def register_world_data(self, world_data: WorldData) -> int:
        """
        Register configured WorldData and return object registration id
        """
    def restart_system(self) -> None:
        """
        Reset positions/velocities to initial rest state
        """
    def save_sim_result(self, obj_path: str) -> None:
        """
        Save current simulation result to an OBJ file.
        """
    def set_device(self, device_ptr: int, stream_ptr: int) -> None:
        """
        Borrow an existing luisa Device/Stream (non-owning).
        
        device_ptr: integer address of a luisa::compute::Device object
        stream_ptr: integer address of a luisa::compute::Stream object
        The caller must ensure these objects outlive this solver.
        """
    def update_per_body_animation(self, mesh_idx: int, target_translation: numpy.ndarray[numpy.float32], target_rotation: numpy.ndarray[numpy.float32]) -> None:
        """
        Update animated rigid body target translation and rotation for a registered object.
        """
    def update_per_vertex_animation(self, mesh_idx: int, local_vid: int, target_pos: numpy.ndarray[numpy.float32]) -> None:
        """
        Update one animated vertex target position for a registered object.
        """
class SceneParams:
    def get_bending_stiffness_scaling(self) -> float:
        """
        Return the bending stiffness scaling factor for current settings.
        """
    def get_collision_detection_frequece(self) -> int:
        ...
    def get_contact_energy_type(self) -> int:
        ...
    def get_current_frame(self) -> int:
        ...
    def get_current_nonlinear_iter(self) -> int:
        ...
    def get_current_pcg_it(self) -> int:
        ...
    def get_current_substep(self) -> int:
        ...
    def get_d_hat(self) -> float:
        ...
    def get_damping_rate(self) -> float:
        ...
    def get_dt(self) -> float:
        ...
    def get_explicit_dt(self) -> float:
        ...
    def get_fix_scene(self) -> bool:
        ...
    def get_floor(self) -> Float3:
        ...
    def get_gravity(self) -> Float3:
        ...
    def get_implicit_dt(self) -> float:
        ...
    def get_nonlinear_iter_count(self) -> int:
        ...
    def get_num_substep(self) -> int:
        ...
    def get_output_per_frame(self) -> bool:
        ...
    def get_output_per_iteration(self) -> bool:
        ...
    def get_pcg_iter_count(self) -> int:
        ...
    def get_print_collision_info(self) -> bool:
        ...
    def get_print_pcg_info(self) -> bool:
        ...
    def get_print_system_energy(self) -> bool:
        ...
    def get_scene_id(self) -> int:
        ...
    def get_stiffness_bending_ui(self) -> float:
        ...
    def get_stiffness_collision(self) -> float:
        ...
    def get_stiffness_dirichlet(self) -> float:
        ...
    def get_substep_dt(self) -> float:
        """
        Return the current substep time step.
        """
    def get_use_ccd_linesearch(self) -> bool:
        ...
    def get_use_energy_linesearch(self) -> bool:
        ...
    def get_use_floor(self) -> bool:
        ...
    def get_use_gpu(self) -> bool:
        ...
    def get_use_self_collision(self) -> bool:
        ...
    def set_collision_detection_frequece(self, v: int) -> None:
        ...
    def set_contact_energy_type(self, v: int) -> None:
        ...
    def set_current_frame(self, v: int) -> None:
        ...
    def set_d_hat(self, v: float) -> None:
        ...
    def set_damping_rate(self, v: float) -> None:
        ...
    def set_explicit_dt(self, v: float) -> None:
        ...
    def set_fix_scene(self, v: bool) -> None:
        ...
    def set_floor(self, v: Float3) -> None:
        ...
    def set_gravity(self, v: Float3) -> None:
        ...
    def set_implicit_dt(self, v: float) -> None:
        ...
    def set_nonlinear_iter_count(self, v: int) -> None:
        ...
    def set_output_per_frame(self, v: bool) -> None:
        ...
    def set_output_per_iteration(self, v: bool) -> None:
        ...
    def set_pcg_iter_count(self, v: int) -> None:
        ...
    def set_print_collision_info(self, v: bool) -> None:
        ...
    def set_print_pcg_info(self, v: bool) -> None:
        ...
    def set_print_system_energy(self, v: bool) -> None:
        ...
    def set_scene_id(self, v: int) -> None:
        ...
    def set_stiffness_bending_ui(self, v: float) -> None:
        ...
    def set_stiffness_collision(self, v: float) -> None:
        ...
    def set_stiffness_dirichlet(self, v: float) -> None:
        ...
    def set_use_ccd_linesearch(self, v: bool) -> None:
        ...
    def set_use_energy_linesearch(self, v: bool) -> None:
        ...
    def set_use_floor(self, v: bool) -> None:
        ...
    def set_use_gpu(self, v: bool) -> None:
        ...
    def set_use_self_collision(self, v: bool) -> None:
        ...
    def update_dt(self, dt: float) -> None:
        """
        Update the frame time step and derived substep time step.
        """
class WorldData:
    def add_fixed_point_by_indices(self, indices: numpy.ndarray[numpy.int32]) -> WorldData:
        ...
    def add_fixed_point_by_method(self, method: str, range: float = 0.0010000000474974513) -> WorldData:
        ...
    def get_fixed_point_indices(self) -> list:
        """
        Return currently registered fixed-point local vertex indices as a Python list
        """
    def get_id(self) -> int:
        """
        Return object registration id.
        """
    def get_name(self) -> str:
        """
        Return object name.
        """
    def get_registration_index(self) -> int:
        """
        Return object registration id.
        """
    def get_rest_positions(self) -> numpy.ndarray[numpy.float32]:
        """
        Return rest positions (after object transform) as an (N,3) float32 numpy array
        """
    def get_rest_rotation(self) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)]:
        """
        Return rest Euler rotation as [x, y, z].
        """
    def get_rest_scale(self) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)]:
        """
        Return rest scale as [x, y, z].
        """
    def get_rest_translation(self) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)]:
        """
        Return rest translation as [x, y, z].
        """
    def set_name(self, name: str) -> WorldData:
        """
        Set the object name used in solver logs and exported results.
        """
    def set_physics_material_cloth(self, stretch_model: str = 'FEM_BW98', bending_model: str = 'DihedralAngle', thickness: float = 0.0010000000474974513, youngs_modulus: float = 1000000.0, poisson_ratio: float = 0.3499999940395355, area_bending_stiffness: float = 0.004999999888241291, d_hat: float = 0.0010000000474974513, contact_offset: float = 0.0) -> WorldData:
        ...
    def set_physics_material_rigid(self, model: str = 'Orthogonality', thickness: float = 0.0010000000474974513, stiffness: float = 1000000.0, density: float = 1000.0, mass: float = 0.0, d_hat: float = 0.0010000000474974513, contact_offset: float = 0.0) -> WorldData:
        ...
    def set_physics_material_rod(self, model: str = 'Spring', radius: float = 0.0010000000474974513, bending_stiffness: float = 10000.0, twisting_stiffness: float = 10000.0, density: float = 1000.0, mass: float = 0.0, d_hat: float = 0.0010000000474974513, contact_offset: float = 0.0) -> WorldData:
        ...
    def set_physics_material_tet(self, model: str = 'Spring', youngs_modulus: float = 1000000.0, poisson_ratio: float = 0.3499999940395355, density: float = 1000.0, mass: float = 0.0, d_hat: float = 0.0010000000474974513, contact_offset: float = 0.0) -> WorldData:
        ...
    def set_rotation(self, x: float, y: float, z: float) -> WorldData:
        """
        Set rest Euler rotation in radians.
        """
    def set_scale(self, scale: float) -> WorldData:
        """
        Set uniform rest scale.
        """
    def set_simulation_type(self, material_type: MaterialType) -> WorldData:
        """
        Set the object simulation/material category.
        """
    def set_translation(self, x: float, y: float, z: float) -> WorldData:
        """
        Set rest translation in world coordinates.
        """
Cloth: MaterialType  # value = <MaterialType.Cloth: 1>
Particle: MaterialType  # value = <MaterialType.Particle: 0>
Rigid: MaterialType  # value = <MaterialType.Rigid: 3>
Rod: MaterialType  # value = <MaterialType.Rod: 4>
Tetrahedral: MaterialType  # value = <MaterialType.Tetrahedral: 2>
