import numpy as np

class ConstantTorqueController:
    def __init__(self, torque_val):
        self.torque_val = torque_val

    def compute_wheel_torques(self):
        return np.full(6, self.torque_val)
