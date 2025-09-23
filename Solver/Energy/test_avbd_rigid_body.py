import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation

EPS = 1e-12
GRAVITY = np.array([0.0, -9.8, 0.0], dtype=float)

def quaternion_multiply(q1, q2):
    """Multiply two quaternions"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quaternion_conjugate(q):
    """Conjugate of quaternion"""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_rotate_vector(q, v):
    """Rotate vector v by quaternion q"""
    q_v = np.array([0, v[0], v[1], v[2]])
    q_conj = quaternion_conjugate(q)
    result = quaternion_multiply(quaternion_multiply(q, q_v), q_conj)
    return result[1:]

def compute_global_inertia_tensor(verts, mass):
    """Compute inertia tensor for rigid body"""
    com = np.average(verts, axis=0, weights=mass)
    I = np.zeros((3, 3))
    
    for i in range(len(verts)):
        r = verts[i] - com
        r_sq = np.dot(r, r)
        I += mass[i] * (r_sq * np.eye(3) - np.outer(r, r))
    
    return I

class RigidBody:
    def __init__(self, vertices, mass):
        self.X = vertices  # Model space vertices
        self.mass = mass
        self.total_mass = np.sum(mass)
        
        # Center of mass in model space
        self.com_model = np.average(self.X, axis=0, weights=mass)
        
        # Inertia tensor in model space
        self.I_model = compute_global_inertia_tensor(self.X, mass)
        
        # State variables
        self.position = np.array([0.0, 1.0, 0.0])  # World space COM
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        
        # World space vertices
        self.vertices = self.get_world_vertices()
    
    def get_world_vertices(self):
        """Transform model space vertices to world space"""
        rotated_vertices = np.array([quaternion_rotate_vector(self.orientation, v - self.com_model) 
                                   for v in self.X])
        return rotated_vertices + self.position
    
    def get_inertia_tensor_world(self):
        """Get inertia tensor in world space"""
        R = Rotation.from_quat(self.orientation[[1, 2, 3, 0]]).as_matrix()  # xyzw to wxyz
        return R @ self.I_model @ R.T
    
    def predict_step(self, dt):
        """Predict next state using gravity"""
        # Predict position
        position_tilde = self.position + self.velocity * dt + 0.5 * GRAVITY * dt**2
        
        # Predict orientation (simplified)
        # For small rotations: q_tilde ≈ q + 0.5 * dt * [0, ω] * q
        omega_quat = np.array([0, self.angular_velocity[0], 
                              self.angular_velocity[1], self.angular_velocity[2]])
        orientation_tilde = self.orientation + 0.5 * dt * quaternion_multiply(omega_quat, self.orientation)
        orientation_tilde /= np.linalg.norm(orientation_tilde)
        
        return position_tilde, orientation_tilde
    
    def apply_correction(self, delta_p, delta_q, alpha=1.0):
        """Apply correction to state"""
        self.position += alpha * delta_p
        
        # Update orientation: q_new = normalize(q + alpha * delta_q)
        new_orientation = self.orientation + alpha * delta_q
        self.orientation = new_orientation / np.linalg.norm(new_orientation)
        
        # Update world vertices
        self.vertices = self.get_world_vertices()

def assemble_rigid_body_inertia(A, b, body, position_tilde, orientation_tilde, dt):
    """Assemble inertia terms for rigid body"""
    h2_inv = 1.0 / (dt * dt)
    
    # Translation part
    delta_p = body.position - position_tilde
    b[0:3] -= body.total_mass * h2_inv * delta_p
    A[0:3, 0:3] += np.eye(3) * (body.total_mass * h2_inv)
    
    # Rotation part (simplified)
    # For small rotations, we can approximate the rotation correction
    delta_q = body.orientation - orientation_tilde
    b[3:6] -= h2_inv * delta_q  # Simplified inertia for rotation
    
    # Add some regularization for rotation
    A[3:7, 3:7] += np.eye(4) * (h2_inv * 0.1)

def assemble_ground_constraints(A, b, body, stiffness=1e5):
    """Assemble ground collision constraints"""
    for i, vertex in enumerate(body.vertices):
        if vertex[1] < 0.0:  # Below ground
            penetration = vertex[1]
            
            # Jacobian for translation
            J_p = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 4)
            
            # Jacobian for rotation (simplified)
            r = vertex - body.position
            J_r = np.array([0.0, 0.0, 0.0] + 
                          [0.0, r[2], -r[1],  # d/dω_x
                           -r[2], 0.0, r[0],  # d/dω_y  
                           r[1], -r[0], 0.0])  # d/dω_z
            
            # Combined Jacobian
            J = np.concatenate([J_p, J_r])
            
            # Force and Hessian
            force = stiffness * penetration * np.array([0, 1, 0] + [0] * 4)
            hessian = stiffness * np.outer(J, J)
            
            b[:6] += force
            A[:6, :6] += hessian

def run_simulation():
    # Model space vertices (tetrahedron)
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    
    # Vertex masses
    masses = np.array([1.0, 1.0, 1.0, 1.0])
    
    # Create rigid body
    body = RigidBody(X, masses)
    body.position = np.array([0.0, 2.0, 0.0])  # Start above ground
    body.vertices = body.get_world_vertices()
    
    # Simulation parameters
    dt = 0.02
    nsteps = 400
    stiffness_ground = 1e5
    
    # Visualization setup
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 3)
    ax.set_zlim(-2, 2)
    
    # Initial plot
    scat = ax.scatter(body.vertices[:, 0], body.vertices[:, 1], body.vertices[:, 2], c='r', s=50)
    
    # Draw edges
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    lines = []
    for i, j in edges:
        line, = ax.plot([body.vertices[i, 0], body.vertices[j, 0]], 
                       [body.vertices[i, 1], body.vertices[j, 1]], 
                       [body.vertices[i, 2], body.vertices[j, 2]], 'b-')
        lines.append(line)
    
    # Ground plane
    xx, zz = np.meshgrid(np.linspace(-2, 2, 2), np.linspace(-2, 2, 2))
    yy = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')
    
    def update(frame):
        # Prediction step
        position_tilde, orientation_tilde = body.predict_step(dt)
        
        # Solve constraints
        A = np.zeros((6, 6))  # 3 translation + 3 rotation (quaternion)
        b = np.zeros(6)
        
        assemble_rigid_body_inertia(A, b, body, position_tilde, orientation_tilde, dt)
        assemble_ground_constraints(A, b, body, stiffness_ground)
        
        # Solve for correction
        try:
            delta = np.linalg.solve(A + np.eye(3) * 1e-8, -b)
        except:
            delta = np.zeros(7)
        
        # Apply correction
        delta_p = delta[0:3]
        delta_q = delta[3:7]
        body.apply_correction(delta_p, delta_q)
        
        # Update velocities
        body.velocity = (body.position - position_tilde) / dt
        
        # Update visualization
        scat._offsets3d = (body.vertices[:, 0], body.vertices[:, 1], body.vertices[:, 2])
        
        for idx, (i, j) in enumerate(edges):
            lines[idx].set_data([body.vertices[i, 0], body.vertices[j, 0]], 
                               [body.vertices[i, 1], body.vertices[j, 1]])
            lines[idx].set_3d_properties([body.vertices[i, 2], body.vertices[j, 2]])
        
        return [scat] + lines
    
    ani = FuncAnimation(fig, update, frames=nsteps, interval=50, blit=False)
    plt.show()

if __name__ == "__main__":
    run_simulation()