# Реализация режима полета БВС по данным оптического потока

Проект содержит два уровня реализации.

Первый уровень — исследовательский Python-контур:

1. генерация реалистичного синтетического лога БВС;
2. преобразование данных оптического потока, дальномера и IMU в оценку горизонтальной скорости;
3. EKF для оценки относительного положения и скорости без ГНСС;
4. контроль качества канала ОП и формирование защитных статусов;
5. пример MAVLink-моста для чтения/передачи данных.

Второй уровень — профессиональное RTOS-ориентированное C++17-ядро:

1. детерминированная обработка `OPTICAL_FLOW_RAD` без зависимости от Python/Linux;
2. учет ориентации датчика относительно корпуса;
3. учет рычага установки датчика относительно центра масс;
4. safety state machine для `FLOW_NAV`, `DEGRADED_HOLD`, `FAILSAFE_LAND`;
5. innovation gate для EKF;
6. CMake-сборка и автономные C++-тесты.

## Быстрый запуск Python-уровня

```bash
cd uav_oflow_navigation
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

pip install -e .[dev]

python src/synthetic_data.py --out data/synthetic_oflow_log.csv --duration 90 --seed 42
python src/run_offline_demo.py --input data/synthetic_oflow_log.csv --outdir outputs
pytest -q
```

После запуска будут созданы:

- `data/synthetic_oflow_log.csv` — синтетический лог;
- `outputs/estimated_nav.csv` — результат оценивания;
- `outputs/trajectory_xy.png` — сравнение истинной и оцененной траектории;
- `outputs/velocity.png` — сравнение истинной и оцененной скорости;
- `outputs/status_summary.json` — сводка по статусам режима.

## Быстрый запуск RTOS C++ core

```bash
cmake -S firmware/rtos_core -B build/rtos_core -DOFNAV_BUILD_TESTS=ON
cmake --build build/rtos_core
ctest --test-dir build/rtos_core --output-on-failure
```

RTOS-ядро находится в:

```text
firmware/rtos_core/
├── include/ofnav/ofnav.hpp
├── src/ofnav.cpp
├── tests/test_ofnav.cpp
└── CMakeLists.txt
```

## Принятая система координат

В Python-демонстраторе используется упрощенная горизонтальная NED/ENU-совместимая плоскость:

- `n` — продольная горизонтальная координата, м;
- `e` — поперечная горизонтальная координата, м;
- `vx_body` — скорость вперед по оси X корпуса БВС, м/с;
- `vy_body` — скорость вправо по оси Y корпуса БВС, м/с;
- `yaw` — курс БВС, рад.

Для `MAVLink OPTICAL_FLOW_RAD` используется следующее приближение:

```text
flow_x_rate = (integrated_x - integrated_xgyro) / dt
flow_y_rate = (integrated_y - integrated_ygyro) / dt

vx_body =  height * flow_y_rate
vy_body = -height * flow_x_rate
```

В RTOS-ядре модель расширена:

```text
v_sensor_x =  height * flow_y_rate * flow_scale_y
v_sensor_y = -height * flow_x_rate * flow_scale_x
v_body_at_sensor = R_body_sensor * v_sensor
v_body_at_cg = v_body_at_sensor - omega_body × r_sensor_body
v_nav = R_yaw * v_body_at_cg
```

Знак по Y важен: линейное движение датчика по положительной оси Y дает отрицательный поток вокруг X.

## Состав синтетического лога

Основные поля:

- `t_s` — время, с;
- `true_n_m`, `true_e_m` — истинное относительное положение, м;
- `true_vn_mps`, `true_ve_mps` — истинная горизонтальная скорость, м/с;
- `true_alt_m` — истинная высота над подстилающей поверхностью, м;
- `roll_rad`, `pitch_rad`, `yaw_rad` — ориентация БВС;
- `gyro_x_radps`, `gyro_y_radps`, `gyro_z_radps` — гироскопы;
- `acc_x_mps2`, `acc_y_mps2` — горизонтальные ускорения в связанной СК;
- `range_m` — дальномер, м;
- `of_integrated_x_rad`, `of_integrated_y_rad` — интегральный оптический поток, рад;
- `of_integrated_xgyro_rad`, `of_integrated_ygyro_rad`, `of_integrated_zgyro_rad` — интегральные угловые скорости, рад;
- `of_quality` — качество ОП, 0...255;
- `texture_class` — тип поверхности;
- `light_lux` — освещенность;
- `gps_available` — признак доступности ГНСС.

В генераторе заложены реалистичные деградации:

- участок слабой текстуры;
- участок вибраций;
- кратковременная потеря дальномера;
- участок плохой освещенности/дыма.

## Связь с реальным автопилотом

Для PX4 штатный путь — чтобы датчик ОП публиковал `OPTICAL_FLOW_RAD`, а дальномер — `DISTANCE_SENSOR`; EKF2 использует ОП при наличии валидного дальномера, включенном контроле ОП и достаточном качестве. Для ArduPilot аналогично требуются корректные параметры датчика ОП, ориентации, дальномера и проверка логов.

`src/mavlink_bridge.py` — диагностический модуль. Он читает поток MAVLink, вычисляет скорость по ОП и может логировать результат. Передача оценок внешней скорости в автопилот через companion computer должна включаться только после проверки параметров EKF/External Vision на конкретной прошивке.

`firmware/rtos_core` — переносимое ядро для настоящей бортовой реализации. Его следует оборачивать в PX4/NuttX-модуль, ArduPilot backend или RTOS-задачу конкретного полетного контроллера.

## Рекомендуемые параметры для первого этапа

### PX4, общий ориентир

```text
EKF2_OF_CTRL      = включить использование optical flow
EKF2_OF_QMIN      = 100...150 для первого этапа
SENS_FLOW_MINHGT  = 0.15...0.30 м
SENS_FLOW_MAXHGT  = 3.0...5.0 м для малых высот
EKF2_OF_POS_X/Y/Z = фактическое смещение датчика от центра масс
```

### ArduPilot, общий ориентир

```text
FLOW_TYPE         = тип установленного датчика
FLOW_ORIENT_YAW   = ориентация датчика
FLOW_FXSCALER     = масштаб по X
FLOW_FYSCALER     = масштаб по Y
RNGFND1_TYPE      = тип дальномера
RNGFND1_MIN_CM    = минимальная дальность
RNGFND1_MAX_CM    = максимальная дальность
LOG_DISARMED      = 1 для стендовой проверки
```

Конкретные значения зависят от датчика, прошивки, ориентации установки и версии автопилота.

## Безопасность

Перед летными испытаниями:

1. снять винты на стендовой проверке;
2. проверить знаки `OF.flowX/flowY` относительно `IMU.GyrX/GyrY`;
3. проверить дальномер на фактической высоте;
4. ограничить углы крена/тангажа;
5. ограничить горизонтальную скорость;
6. выполнять первый полет только с внешним пилотом и возможностью ручного перехвата.

К реальному полету допускается не Python-скрипт, а прошитая и проверенная интеграция RTOS-ядра со штатным автопилотом и его failsafe-логикой.
