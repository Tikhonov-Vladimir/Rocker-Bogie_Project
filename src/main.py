import argparse
from sim.sim_runner import run_simulation

def main():
    parser = argparse.ArgumentParser(description='Симуляция ровера с разными режимами управления')
    parser.add_argument('--mode', type=str, default='constant',
                        choices=['constant', 'adaptive'],
                        help='Режим управления: constant (постоянный момент) или adaptive (адаптивная скорость)')
    parser.add_argument('--torque', type=float, default=0.2,
                        help='Момент на колесо (для режима constant)')
    parser.add_argument('--scene', type=str, default='src/assets/scene_stairs_logs_modified.xml',
                        help='Путь к файлу сцены')
    parser.add_argument('--nogui', action='store_true',
                        help='Запуск без GUI (только сбор данных)')

    args = parser.parse_args()

    # Запуск симуляции
    data = run_simulation(
        mode=args.mode,
        torque_value=args.torque,
        scene_path=args.scene,
        nogui=args.nogui
    )

    # Отображение результатов
    if data and not args.nogui:
        from control.adaptive_speed import print_results
        print_results(data)

if __name__ == "__main__":
    main()
