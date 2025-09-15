import numpy as np
from simulator import Simulator, centerline

sim = Simulator()

def controller(x):
    """controller for a car

    Args:
        x (ndarray): numpy array of shape (5,) containing [x, y, heading, velocity, steering angle]

    Returns:
        ndarray: numpy array of shape (2,) containing [fwd acceleration, steering rate]
    """
    # Just return zero controls for visualization
    return np.array([0.0, 0.0])  # [acceleration, steering_rate]

sim.set_controller(controller)
sim.run()
sim.animate()  # This will save as 'sim.gif'
sim.plot()