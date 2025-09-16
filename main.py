import numpy as np
from simulator import Simulator, centerline

# Pre-compute centerline points for efficiency
track_length = 100.0
s_points = np.linspace(0, track_length, 100)  # Reduced number of points
centerline_points = np.array([centerline(s) for s in s_points])

sim = Simulator()

# Pure Pursuit parameters
L = 1.58     # wheelbase [m] (from README)
L_d = 3.0    # lookahead distance [m] (start conservative)
v_ref = 5.0  # target speed [m/s] (start slower)

def controller(x):
    """Pure Pursuit controller with adaptive lookahead"""
    # 1. Current state
    pos_x = x[0]
    pos_y = x[1]
    heading = x[2]    # 0 = pointing left
    velocity = x[3]
    steering = x[4]
    
    # 2. Find closest point using pre-computed centerline
    pos = np.array([pos_x, pos_y])
    dists = np.linalg.norm(centerline_points - pos, axis=1)
    closest_idx = np.argmin(dists)
    s_current = s_points[closest_idx]
    
    # 3. Calculate adaptive lookahead distance
    L_d_adaptive = max(2.0, 0.5 + 0.3 * velocity)  # More conservative
    
    # 4. Find lookahead point
    # Calculate desired s-coordinate
    s_look = (s_current + L_d_adaptive) % track_length
    
    # Find closest pre-computed point
    look_idx = np.argmin(np.abs(s_points - s_look))
    target_x, target_y = centerline_points[look_idx]
    
    # 5. Transform to vehicle coordinates
    # Shift and rotate target point into car's reference frame
    dx = target_x - pos_x
    dy = target_y - pos_y
    
    # Rotate by -heading to get into car's frame
    dx_body = np.cos(-heading) * dx - np.sin(-heading) * dy
    dy_body = np.sin(-heading) * dx + np.cos(-heading) * dy
    
    # 6. Pure Pursuit steering control
    # Calculate curvature command
    kappa = 2.0 * dy_body / (dx_body * dx_body + dy_body * dy_body)
    
    # Convert curvature to steering angle
    theta_des = np.arctan2(L * kappa, 1.0)
    
    # Calculate steering rate command (P controller)
    k_steering = 2.0  # Steering gain
    theta_dot = k_steering * (theta_des - steering)
    
    # Clip steering rate to valid range
    theta_dot = np.clip(theta_dot, -1.0, 1.0)
    
    # 7. Speed control
    # Start with conservative speed control
    target_speed = min(v_ref, 3.0 + velocity)  # Gradual speed increase
    
    # Reduce speed in curves
    curve_factor = 1.0 / (1.0 + 5.0 * abs(kappa))
    target_speed *= curve_factor
    
    # Simple P controller for speed
    speed_error = target_speed - velocity
    accel = 1.0 * speed_error
    accel = np.clip(accel, -4.0, 10.0)
    
    return np.array([accel, theta_dot])

sim.set_controller(controller)
sim.run()
sim.animate()  
sim.plot()