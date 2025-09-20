import numpy as np
from scipy.optimize import minimize
from simulator import Simulator, centerline

# Track precomputation
track_length = 100.0
s_points = np.linspace(0, track_length, 500)
centerline_points = np.array([centerline(s) for s in s_points])

# MPC PARAMETERS
dt = 0.1      # Discretization time step
L = 1.58      # Vehicle wheelbase
N = 5         # Prediction horizon length
v_ref = 7.0   # Target velocity

# HARD CONSTRAINTS (These can NEVER be violated)
THETA_MIN, THETA_MAX = -0.7, 0.7           # Steering angle bounds
THETA_DOT_MIN, THETA_DOT_MAX = -1.0, 1.0   # Steering rate bounds  
ACCEL_MIN, ACCEL_MAX = -4.0, 10.0          # Acceleration bounds
MAX_COMBINED_ACCEL = 12.0                  # Friction limit

# COST FUNCTION WEIGHTS (For optimization objective)
Q_pos = 1000.0      # Position tracking weight
Q_heading = 500.0   # Heading tracking weight  
Q_speed = 100.0     # Speed tracking weight
R_control = 1.0     # Control effort weight

sim = Simulator()

class ProperMPC:
    def __init__(self, sim):
        self.sim = sim
        
    def _get_reference_trajectory(self, current_pos, current_heading):
        """
        REFERENCE GENERATION: Creates desired trajectory for vehicle to follow
        Returns: Array of reference states [x, y, heading, velocity] for horizon N
        """
        # Find closest point on track centerline
        dists = np.linalg.norm(centerline_points - current_pos, axis=1)
        closest_idx = np.argmin(dists)
        s_current = s_points[closest_idx]
        
        ref_trajectory = []
        for k in range(N):
            # Look ahead along track based on target speed
            s_ahead = (s_current + k * v_ref * dt) % track_length
            idx = np.argmin(np.abs(s_points - s_ahead))
            
            # Get reference position
            ref_pos = centerline_points[idx]
            
            # Calculate reference heading (track tangent direction)
            next_idx = (idx + 1) % len(centerline_points)
            track_tangent = centerline_points[next_idx] - centerline_points[idx]
            ref_heading = np.arctan2(track_tangent[1], track_tangent[0])
            
            ref_trajectory.append([ref_pos[0], ref_pos[1], ref_heading, v_ref])
            
        return np.array(ref_trajectory)
    
    def _kinematic_bicycle_model(self, state, control):
        """
        SYSTEM DYNAMICS: x_{k+1} = f(x_k, u_k)
        Kinematic bicycle model - predicts next state from current state and control
        """
        x, y, theta, v, phi = state
        a, phi_dot = control
        
        # KBM equations:
        x_next = x + dt * v * np.cos(theta)           # x position
        y_next = y + dt * v * np.sin(theta)           # y position  
        theta_next = theta + dt * (v/L) * np.tan(phi) # heading angle
        v_next = v + dt * a                           # velocity
        phi_next = phi + dt * phi_dot                 # steering angle
        
        return np.array([x_next, y_next, theta_next, v_next, phi_next])
    
    def _mpc_cost_function(self, u_flat, current_state, ref_trajectory):
        """
        MPC OBJECTIVE FUNCTION: J = Σ(||x_k - x_ref||²_Q + ||u_k||²_R)
        This is what we minimize - tracking error + control effort
        """
        # Reshape flat control vector into N x 2 matrix
        u_sequence = u_flat.reshape(N, 2)
        
        total_cost = 0.0
        state = current_state.copy()
        
        for k in range(N):
            # PREDICTION: Forward simulate using system dynamics
            state = self._kinematic_bicycle_model(state, u_sequence[k])
            
            # COST TERMS:
            # 1. Position tracking: ||[x,y] - [x_ref,y_ref]||²
            pos_error = np.linalg.norm(state[:2] - ref_trajectory[k, :2])
            total_cost += Q_pos * pos_error**2
            
            # 2. Heading tracking: (theta - theta_ref)²
            heading_error = state[2] - ref_trajectory[k, 2]
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            total_cost += Q_heading * heading_error**2
            
            # 3. Speed tracking: (v - v_ref)²
            speed_error = state[3] - ref_trajectory[k, 3]
            total_cost += Q_speed * speed_error**2
            
            # 4. Control effort: ||u||² = a² + φ_dot²
            total_cost += R_control * np.sum(u_sequence[k]**2)
            
        return total_cost
    
    def _constraint_functions(self, u_flat, current_state):
        """
        MPC CONSTRAINTS: g(x,u) ≤ 0
        These define the feasible region - solutions must satisfy ALL constraints
        """
        u_sequence = u_flat.reshape(N, 2)
        constraints = []
        state = current_state.copy()
        
        for k in range(N):
            # Forward simulate
            state = self._kinematic_bicycle_model(state, u_sequence[k])
            
            # CONSTRAINT 1: No collision with cones (most critical)
            if self.sim._check_collision(state):
                constraints.append(1.0)  # Violated if > 0
            else:
                constraints.append(-1.0)  # Satisfied if ≤ 0
                
            # CONSTRAINT 2: Combined acceleration limit (friction)
            combined_accel = np.sqrt(u_sequence[k, 0]**2 + 
                           ((state[3]**2/0.79) * np.sin(np.arctan(0.5*np.tan(state[4]))))**2)
            constraints.append(combined_accel - MAX_COMBINED_ACCEL)
            
            # CONSTRAINT 3: Steering angle limits  
            constraints.append(state[4] - THETA_MAX)    # phi ≤ phi_max
            constraints.append(THETA_MIN - state[4])    # phi ≥ phi_min
            
        return np.array(constraints)
    
    def solve(self, current_state):
        """
        MPC OPTIMIZATION: Solve the constrained optimization problem
        
        Standard MPC formulation:
        min  J(x,u) = Σ(||x_k-x_ref||²_Q + ||u_k||²_R)
        s.t. x_{k+1} = f(x_k, u_k)           [dynamics]
             g(x_k, u_k) ≤ 0                [inequality constraints]
             u_min ≤ u_k ≤ u_max            [bound constraints]
        """
        # Generate reference trajectory
        ref_traj = self._get_reference_trajectory(current_state[:2], current_state[2])
        
        # DECISION VARIABLES: Control sequence u = [u_0, u_1, ..., u_{N-1}]
        # Each u_k = [acceleration, steering_rate]
        u_init = np.zeros(N * 2)  # Initial guess: do nothing
        
        # BOUND CONSTRAINTS: u_min ≤ u ≤ u_max
        bounds = []
        for k in range(N):
            bounds.append((ACCEL_MIN, ACCEL_MAX))        # Acceleration bounds
            bounds.append((THETA_DOT_MIN, THETA_DOT_MAX)) # Steering rate bounds
            
        # INEQUALITY CONSTRAINTS: g(x,u) ≤ 0
        constraints = {
            'type': 'ineq',
            'fun': lambda u: -self._constraint_functions(u, current_state)  # Flip sign for ≤ 0
        }
        
        # SOLVE OPTIMIZATION PROBLEM
        try:
            result = minimize(
                fun=lambda u: self._mpc_cost_function(u, current_state, ref_traj),
                x0=u_init,
                bounds=bounds,
                constraints=constraints,
                method='SLSQP',  # Sequential Least Squares Programming
                options={'maxiter': 100, 'ftol': 1e-4}
            )
            
            if result.success:
                optimal_u = result.x.reshape(N, 2)
                return optimal_u[0]  # RECEDING HORIZON: Apply only first control
            else:
                # Fallback: emergency brake and minimal steering
                return np.array([-2.0, 0.0])
                
        except:
            # Fallback for any optimization failure
            return np.array([-2.0, 0.0])

# Initialize MPC controller
mpc = ProperMPC(sim)

def controller(x):
    """
    MAIN CONTROLLER: Called at each time step with current state
    Returns: Control input [acceleration, steering_rate]
    """
    return mpc.solve(x)

def analyze_lap(sim):
    """Analyze simulation results"""
    ts, xs, us, crash, slip = sim.get_results()
    print(f"Results:")
    print(f"  Crashes: {np.sum(crash)} (should be 0)")
    print(f"  Slips: {np.sum(slip)}")
    print(f"  Max speed: {np.max(xs[3]):.1f} m/s")
    print(f"  Avg speed: {np.mean(xs[3]):.1f} m/s")
    
    # Check hard constraint violations
    steering_viols = np.sum((xs[4] < THETA_MIN) | (xs[4] > THETA_MAX))
    print(f"  Steering violations: {steering_viols} (should be 0)")

# Run simulation
sim.set_controller(controller)
sim.run()
analyze_lap(sim)
sim.animate()
sim.plot()