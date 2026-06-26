from collections import deque
import numpy as np

class AccelerationFilter:
    def __init__(self, buffer_size=10):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.prev_vel_z = 0.0
        self.prev_time = 0.0

    def compute(self, current_vel_z, current_time):
        """Вычисляет и фильтрует вертикальное ускорение"""
        if self.prev_time > 1e-6 and current_time - self.prev_time > 1e-6:
            raw_acc = (current_vel_z - self.prev_vel_z) / (current_time - self.prev_time)
        else:
            raw_acc = 0.0

        self.prev_vel_z = current_vel_z
        self.prev_time = current_time

        # Добавляем в буфер
        self.buffer.append(raw_acc)

        # Медианная фильтрация
        if len(self.buffer) >= 5:
            filtered_acc = np.median(list(self.buffer))
        else:
            filtered_acc = raw_acc

        return filtered_acc
