import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, target, current):
        error = target - current

        p = self.kp * error

        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -2.0, 2.0)
        i = self.ki * self.integral

        derivative = (error - self.prev_error) / self.dt
        d = self.kd * derivative
        self.prev_error = error

        output = p + i + d
        return np.clip(output, -2.0, 2.0)
