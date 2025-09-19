import numpy as np
from simulator import Simulator, centerline

# Track precomputation
track_length = 100.0
s_points = np.linspace(0, track_length, 500)  # Higher resolution
centerline_points = np.array([centerline(s) for s in s_points])

# MPC Parameters
dt = 0.1
L = 1.58
N = 4          # Even shorter horizon for speed
v_ref = 8.0    # Much higher target speed

# Constraints (from problem statement)
THETA_MIN, THETA_MAX = -0.7, 0.7
THETA_DOT_MIN, THETA_DOT_MAX = -1.0, 1.0
ACCEL_MIN, ACCEL_MAX = -4.0, 10.0

# Cost weights - more aggressive tuning
w_pos = 5000.0      # Lower centerline penalty (allow more deviation)
w_speed = 200.0     # Higher speed penalty (push for speed)
w_heading = 500.0   # Lower heading penalty (allow more aggressive turns)
w_u = 0.1           # Very low control effort (allow aggressive inputs)
w_boundary = 50000.0  # Still high for safety

sim = Simulator()

class RobustMPC:
    def __init__(self, sim):
        self.sim = sim
        # Use actual cone positions from simulator
        self.left_cones = sim.left_cones
        self.right_cones = sim.right_cones
        self.all_cones = sim.cones
        
    def _get_track_info(self, pos):
        """Get closest point on centerline and track direction"""
        dists = np.linalg.norm(centerline_points - pos, axis=1)
        closest_idx = np.argmin(dists)
        
        # Get track direction (tangent)
        next_idx = (closest_idx + 1) % len(centerline_points)
        prev_idx = (closest_idx - 1) % len(centerline_points)
        
        track_dir = centerline_points[next_idx] - centerline_points[prev_idx]
        track_dir = track_dir / np.linalg.norm(track_dir)
        track_heading = np.arctan2(track_dir[1], track_dir[0])
        
        return s_points[closest_idx], centerline_points[closest_idx], track_heading
    
    def _get_cone_distances(self, pos):
        """Get distances to nearest left and right cones"""
        left_dists = np.linalg.norm(self.left_cones - pos, axis=1)
        right_dists = np.linalg.norm(self.right_cones - pos, axis=1)
        
        min_left_dist = np.min(left_dists)
        min_right_dist = np.min(right_dists)
        
        return min_left_dist, min_right_dist
    
    def _get_reference_trajectory(self, s_current, current_heading):
        """Generate reference trajectory ahead - more aggressive lookahead"""
        ref_states = []
        
        for k in range(N + 1):
            # More aggressive lookahead - look further ahead
            s_look = (s_current + k * v_ref * dt * 1.5) % track_length
            idx = np.argmin(np.abs(s_points - s_look))
            
            ref_pos = centerline_points[idx]
            
            # Calculate reference heading (track direction)
            next_idx = (idx + 1) % len(centerline_points)
            track_dir = centerline_points[next_idx] - centerline_points[idx]
            if np.linalg.norm(track_dir) > 0:
                ref_heading = np.arctan2(track_dir[1], track_dir[0])
            else:
                ref_heading = current_heading
                
            ref_states.append([ref_pos[0], ref_pos[1], ref_heading, v_ref])
            
        return np.array(ref_states)
    
    def _simulate_step(self, state, control):
        """Single step KBM simulation with constraint enforcement"""
        x, y, theta, v, phi = state
        a, phi_dot = control
        
        # Enforce control constraints
        a = np.clip(a, ACCEL_MIN, ACCEL_MAX)
        phi_dot = np.clip(phi_dot, THETA_DOT_MIN, THETA_DOT_MAX)
        
        # Kinematic bicycle model
        x_new = x + dt * v * np.cos(theta)
        y_new = y + dt * v * np.sin(theta)
        theta_new = theta + dt * (v / L) * np.tan(phi)
        v_new = v + dt * a
        phi_new = phi + dt * phi_dot
        
        # Enforce state constraints
        phi_new = np.clip(phi_new, THETA_MIN, THETA_MAX)
        v_new = max(0.5, v_new)  # Allow higher minimum speed
        
        return np.array([x_new, y_new, theta_new, v_new, phi_new])
    
    def _evaluate_trajectory(self, state, controls, ref_trajectory):
        """Evaluate cost of control sequence using actual cone positions"""
        cost = 0.0
        current_state = state.copy()
        
        for k in range(N):
            # Simulate one step
            current_state = self._simulate_step(current_state, controls[k])
            
            # Position error (centerline tracking)
            pos_error = np.linalg.norm(current_state[:2] - ref_trajectory[k, :2])
            cost += w_pos * pos_error**2
            
            # Heading error
            heading_error = current_state[2] - ref_trajectory[k, 2]
            # Normalize angle difference
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            cost += w_heading * heading_error**2
            
            # Speed tracking
            speed_error = current_state[3] - ref_trajectory[k, 3]
            cost += w_speed * speed_error**2
            
            # Control effort
            cost += w_u * (controls[k, 0]**2 + controls[k, 1]**2)
            
            # ACTUAL collision detection with cones
            if self.sim._check_collision(current_state):
                cost += w_boundary  # Huge penalty for collision
                
            # Cone avoidance using actual cone positions - less conservative
            left_dist, right_dist = self._get_cone_distances(current_state[:2])
            min_safe_distance = 0.6  # Smaller minimum safe distance for more aggressive driving
            
            if left_dist < min_safe_distance:
                cost += w_boundary * (min_safe_distance - left_dist)**2
            if right_dist < min_safe_distance:
                cost += w_boundary * (min_safe_distance - right_dist)**2
        
        return cost
    
    def solve(self, state):
        """Solve MPC optimization with grid search"""
        # Get track reference
        s_current, closest_center, track_heading = self._get_track_info(state[:2])
        ref_trajectory = self._get_reference_trajectory(s_current, state[2])
        
        # Grid search parameters - more aggressive ranges
        accel_options = np.linspace(-4.0, 8.0, 4)  # Use full acceleration range
        steer_options = np.linspace(-1.0, 1.0, 4)  # Use full steering rate range
        
        best_cost = float('inf')
        best_controls = np.zeros((N, 2))
        
        # Search over first control input
        for a in accel_options:
            for phi_dot in steer_options:
                # Simple strategy: apply same control for entire horizon
                controls = np.tile([a, phi_dot], (N, 1))
                
                # Evaluate this control sequence
                cost = self._evaluate_trajectory(state, controls, ref_trajectory)
                
                if cost < best_cost:
                    best_cost = cost
                    best_controls = controls.copy()
        
        # Return first control input (MPC receding horizon)
        optimal_control = best_controls[0]
        
        # Final safety clipping
        optimal_control[0] = np.clip(optimal_control[0], ACCEL_MIN, ACCEL_MAX)
        optimal_control[1] = np.clip(optimal_control[1], THETA_DOT_MIN, THETA_DOT_MAX)
        
        return optimal_control

# Initialize MPC controller
mpc = RobustMPC(sim)

def controller(x):
    """Main controller function"""
    return mpc.solve(x)

# Analysis function
def analyze_lap(sim):
    ts, xs, us, crash, slip = sim.get_results()
    print(f"Simulation Results:")
    print(f"  Crashes: {np.sum(crash)}")
    print(f"  Slips: {np.sum(slip)}")
    print(f"  Max speed: {np.max(xs[3]):.2f} m/s")
    print(f"  Avg speed: {np.mean(xs[3]):.2f} m/s")
    
    # Check constraint violations
    steering_violations = np.sum((xs[4] < THETA_MIN) | (xs[4] > THETA_MAX))
    print(f"  Steering violations: {steering_violations}")

# Run simulation
sim.set_controller(controller)
sim.run()
analyze_lap(sim)
sim.animate()
sim.plot()