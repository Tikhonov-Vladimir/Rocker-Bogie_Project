import numpy as np
from control.pid_controller import PIDController
from control.acceleration_filter import AccelerationFilter

class AdaptiveSpeedController:
    def __init__(self, model, data, a_crit=15.0):
        self.model = model
        self.data = data
        self.a_crit = a_crit
        self.v_target = 0.4
        self.v_max = 5
        self.v_min = 0.05

        self.speed_pid = PIDController(kp=100, ki=0.7, kd=0.05, dt=0.002)

        self.a_low = 6.0
        self.a_high = 11.0
        self.adapt_rate = 0.008

        # Фильтр для ускорения
        self.acc_filter = AccelerationFilter(buffer_size=10)

        self.last_print_time = 0
        self.print_counter = 0

    def compute_wheel_torques(self, sim_time):
        # Получаем фильтрованное ускорение
        current_vel_z = self.data.qvel[2]
        current_time = self.data.time
        acc_z = self.acc_filter.compute(current_vel_z, current_time)
        acc_z_abs = abs(acc_z)

        # Адаптация скорости
        if acc_z_abs > self.a_high:
            self.v_target = max(self.v_min, self.v_target - self.adapt_rate * 3)
            reason = "ТОРМОЖУ"
            reason_detail = f"(|AccZ|={acc_z_abs:.1f} > {self.a_high})"
        elif acc_z_abs < self.a_low:
            self.v_target = min(self.v_max, self.v_target + self.adapt_rate)
            reason = "РАЗГОН"
            reason_detail = f"(|AccZ|={acc_z_abs:.1f} < {self.a_low})"
        else:
            reason = "ДЕРЖУ"
            reason_detail = f"(|AccZ|={acc_z_abs:.1f} в норме)"

        current_speed = self.data.qvel[0]
        base_torque = self.speed_pid.compute(self.v_target, current_speed)
        torques = np.full(6, base_torque / 6)

        # Лог каждые 0.1 секунды
        if sim_time - self.last_print_time >= 0.1:
            self.print_counter += 1
            print(f"[{sim_time:5.2f}s] {reason:8} {reason_detail:25} | V_target={self.v_target:.2f} -> V_actual={current_speed:.2f} | Torque={base_torque:6.3f} | AccZ={acc_z:6.1f}")
            self.last_print_time = sim_time

        return torques, self.v_target, acc_z
