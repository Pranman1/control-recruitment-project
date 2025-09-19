import numpy as np
from simulator import Simulator, centerline

# Track precomputation
track_length = 100.0
s_points = np.linspace(0, track_length, 300)
centerline_points = np.array([centerline(s) for s in s_points])

# Params
dt = 0.1
L = 1.58
N = 8          # prediction horizon
v_ref = 6.0    # target speed

# Cost weights
w_pos = 20000.0      # centerline tracking
w_speed = 100.0     # speed tracking
w_u = 0.1         # control effort
w_out = 500.0     # penalty for leaving track

sim = Simulator()

class MPCWithBoundary:
    def __init__(self, sim):
        self.sim = sim

    def _linearized_AB(self, state):
        """Linearize KBM dynamics at current state"""
        x, y, theta, v, phi = state
        A = np.eye(5)
        A[0, 2] = -dt * v * np.sin(theta)
        A[0, 3] = dt * np.cos(theta)
        A[1, 2] = dt * v * np.cos(theta)
        A[1, 3] = dt * np.sin(theta)
        A[2, 3] = dt * np.tan(phi) / L
        A[2, 4] = dt * v / (L * np.cos(phi)**2)
        A[3, 3] = 1.0
        A[4, 4] = 1.0

        B = np.zeros((5, 2))
        B[3, 0] = dt   # accel -> velocity
        B[4, 1] = dt   # steering rate -> phi
        return A, B

    def _get_reference(self, pos, s_current):
        """Reference = centerline points with target speed"""
        ref = []
        for k in range(N+1):
            s_look = (s_current + k * 1.5) % track_length
            idx = np.argmin(np.abs(s_points - s_look))
            ref.append([centerline_points[idx, 0], centerline_points[idx, 1], v_ref])
        return np.array(ref)

    def _rollout_cost(self, state, U, ref):
        """Compute rollout cost with soft penalties"""
        cost = 0.0
        x = state.copy()

        for k in range(N):
            # Apply controls
            a, phi_dot = U[k]
            x[0] += dt * x[3] * np.cos(x[2])
            x[1] += dt * x[3] * np.sin(x[2])
            x[2] += dt * (x[3] / L) * np.tan(x[4])
            x[3] += dt * a
            x[4] += dt * phi_dot

            # Track following cost
            track_err = (x[0] - ref[k, 0])**2 + (x[1] - ref[k, 1])**2
            cost += w_pos * track_err

            # Speed cost
            cost += w_speed * (x[3] - v_ref)**2

            # Control effort
            cost += w_u * (a**2 + phi_dot**2)

            # Collision penalty
            if self.sim._check_collision(x):
                cost += w_out

        return cost

    def solve(self, state):
        """Brute force search over discretized controls (fast enough for small N)"""
        accel_options = np.linspace(-2.0, 3.0, 3)      # coarse accel set
        steer_rate_options = np.linspace(-0.5, 0.5, 3) # coarse steering vel set

        pos = state[:2]
        dists = np.linalg.norm(centerline_points - pos, axis=1)
        closest_idx = np.argmin(dists)
        s_current = s_points[closest_idx]

        ref = self._get_reference(pos, s_current)

        best_cost = 1e9
        best_u0 = np.array([0.0, 0.0])

        # Try all combinations of first-step controls
        for a in accel_options:
            for sd in steer_rate_options:
                U = np.tile([a, sd], (N, 1))  # repeat same action for horizon
                cost = self._rollout_cost(state.copy(), U, ref)
                if cost < best_cost:
                    best_cost = cost
                    best_u0 = np.array([a, sd])

        return best_u0

mpc = MPCWithBoundary(sim)

def controller(x):
    return mpc.solve(x)

def analyze_lap(sim):
    ts, xs, us, crash, slip = sim.get_results()
    print(f"Crashes: {np.sum(crash)}")
    print(f"Slips: {np.sum(slip)}")
    print(f"Max speed: {np.max(xs[3]):.2f} m/s")

# Run
sim.set_controller(controller)
sim.run()
analyze_lap(sim)
sim.animate()
sim.plot()
