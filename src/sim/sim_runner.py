import mujoco
import numpy as np
import time
import matplotlib.pyplot as plt
from control.constant_torque import ConstantTorqueController
from control.adaptive_speed import AdaptiveSpeedController
from control.acceleration_filter import AccelerationFilter

def run_simulation(mode='constant', torque_value=0.2, scene_path='src/assets/scene_stairs_logs_modified.xml', nogui=False):
    # Загружаем модель
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Начальная позиция
    data.qpos[0] = -3
    data.qvel[0] = 0

    # Выбор контроллера
    if mode == 'constant':
        controller = ConstantTorqueController(torque_value)
        print(f"\n{'='*60}")
        print(f"ЭКСПЕРИМЕНТ: ПОСТОЯННЫЙ МОМЕНТ")
        print(f"{'='*60}")
        print(f"  Момент на колесо: {torque_value} Н·м")
        print(f"  Дистанция: 9 метров")
        print(f"  Задержка перед стартом: 3 секунды")
        print(f"  Фильтрация ускорения: медианная (окно 10)")
    else:
        controller = AdaptiveSpeedController(model, data)
        print(f"\n{'='*60}")
        print(f"ЭКСПЕРИМЕНТ: АДАПТИВНАЯ СКОРОСТЬ")
        print(f"{'='*60}")
        print(f"  Критическое ускорение: {controller.a_crit} м/с²")
        print(f"  Порог разгона: |AccZ| < {controller.a_low} м/с²")
        print(f"  Порог торможения: |AccZ| > {controller.a_high} м/с²")
        print(f"  Макс. скорость: {controller.v_max} м/с")
        print(f"  Начальная скорость: {controller.v_target} м/с")
        print(f"  Дистанция: 9 метров")
        print(f"  Задержка перед стартом: 3 секунды")
        print(f"  Фильтрация ускорения: медианная (окно 10)")

    # Запуск viewer
    if not nogui:
        viewer = mujoco.viewer.launch_passive(model, data)
    else:
        viewer = None

    # Данные для записи
    times = []
    speeds = []
    positions = []
    positions_y = []
    accs = []
    targets = []

    step = 0
    start_pos = -1.5
    distance = 9
    completed = False

    # Фильтр ускорения
    acc_filter = AccelerationFilter(buffer_size=10)

    print(f"\n[0.0s] Ожидание 3 секунды перед стартом...")
    wait_start = time.time()
    if not nogui and viewer is not None:
        while time.time() - wait_start < 5.0 and viewer.is_running():
            viewer.sync()
            mujoco.mj_step(model, data)
            time.sleep(0.001)
    else:
        while time.time() - wait_start < 5.0:
            mujoco.mj_step(model, data)
            time.sleep(0.001)

    print(f"[3.0s] СТАРТ! Начинаем движение...\n")
    print("Время    | Режим         | Состояние                       | Скорость           | Момент   | Ускорение")
    print("-" * 95)

    last_print_time = 0

    try:
        while (not nogui and viewer is not None and viewer.is_running()) or nogui:
            if not nogui and viewer is not None:
                viewer.sync()

            sim_time = data.time
            pos_x = data.qpos[0]
            pos_y = data.qpos[1]
            vel_x = data.qvel[0]

            if mode == 'adaptive':
                torques, v_target, acc_z = controller.compute_wheel_torques(sim_time)
                targets.append(v_target)
                accs.append(acc_z)
            else:
                torques = controller.compute_wheel_torques()
                targets.append(torque_value)

                # Фильтрация ускорения
                current_vel_z = data.qvel[2]
                current_time = data.time
                acc_z = acc_filter.compute(current_vel_z, current_time)
                accs.append(acc_z)

                # Лог для постоянного режима
                if sim_time - last_print_time >= 0.1 and not nogui:
                    print(f"[{sim_time:5.2f}s] {'ПОСТОЯННЫЙ':12} | X={pos_x:5.2f}м, V={vel_x:5.2f}м/с, T={torques[0]:6.3f}Н·м, AccZ={acc_z:6.1f}")
                    last_print_time = sim_time

            # Применение управления
            data.ctrl[0] = 0
            data.ctrl[1] = 0
            for i in range(6):
                data.ctrl[2 + i] = torques[i]

            times.append(sim_time)
            speeds.append(vel_x)
            positions.append(pos_x)
            positions_y.append(pos_y)

            if pos_x >= start_pos + distance:
                completed = True
                total_time = sim_time
                print(f"\n{'='*60}")
                print(f"ДИСТАНЦИЯ {distance} МЕТРОВ ПРОЙДЕНА!")
                print(f"  Время прохождения: {total_time:.2f} секунд")
                print(f"  Средняя скорость: {np.mean(speeds):.3f} м/с")
                print(f"  Макс. скорость: {np.max(speeds):.3f} м/с")
                print(f"{'='*60}\n")
                break

            mujoco.mj_step(model, data)
            step += 1
            time.sleep(0.001)

            if nogui and step % 1000 == 0:
                print(f"[{sim_time:.2f}s] X={pos_x:.2f}м, V={vel_x:.2f}м/с")

    except KeyboardInterrupt:
        print("\n\nПрервано пользователем")
    finally:
        if viewer is not None:
            viewer.close()

    return {
        'mode': mode,
        'time': np.array(times),
        'speed': np.array(speeds),
        'position': np.array(positions),
        'position_y': np.array(positions_y),
        'acc_z': np.array(accs),
        'target': np.array(targets),
        'total_time': total_time if completed else times[-1] if len(times) > 0 else 0
    }
