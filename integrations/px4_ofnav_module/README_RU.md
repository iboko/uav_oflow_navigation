# PX4 uORB/NuttX-модуль `ofnav`

Это полноценный in-tree PX4-модуль, который запускает C++ RTOS-ядро `ofnav` внутри PX4/NuttX и работает с uORB-топиками, а не с Python/MAVLink-скриптом.

## Назначение

Модуль читает:

```text
sensor_optical_flow
Данные оптического потока: pixel_flow, delta_angle, distance_m, integration_timespan_us, quality.

distance_sensor
Резервная/внешняя высота по дальномеру, если sensor_optical_flow не содержит distance_m.

vehicle_imu
Интегральные углы и скорости IMU: delta_angle, delta_velocity, delta_angle_dt, delta_velocity_dt.

vehicle_attitude
Кватернион ориентации БВС.
```

Модуль публикует:

```text
debug_vect OFNAV_V
x = VN, м/с
y = VE, м/с
z = NavMode

debug_vect OFNAV_H
x = высота, м
y = качество ОП, 0...255
z = RejectReason
```

Безопасное поведение по умолчанию: модуль **не управляет моторами**, **не подменяет EKF2** и **не отправляет команды в САУ**. Сначала он используется как нативный PX4/NuttX-монитор канала ОП. После SITL/HIL можно расширять его до публикации в fusion/control path.

## Установка в PX4-Autopilot

Из корня этого репозитория:

```bash
chmod +x tools/install_px4_ofnav_module.sh
./tools/install_px4_ofnav_module.sh ../PX4-Autopilot boards/px4/sitl/default.px4board
```

Для железа пример:

```bash
./tools/install_px4_ofnav_module.sh ../PX4-Autopilot boards/px4/fmu-v6x/default.px4board
```

Если у вас другой контроллер, замените board config на свой, например `boards/px4/fmu-v5/default.px4board` или соответствующую плату.

## Сборка SITL

В корне PX4:

```bash
cd ../PX4-Autopilot
make px4_sitl gz_x500
```

В PX4 shell:

```sh
ofnav start -r 100 -q 120
ofnav status
```

Проверка топиков:

```sh
listener sensor_optical_flow 5
listener distance_sensor 5
listener vehicle_imu 5
listener vehicle_attitude 5
listener debug_vect 20
```

Частота публикации:

```sh
uorb top
```

## Запуск с менее строгой проверкой дальномера

По умолчанию модуль требует, чтобы `distance_sensor.orientation == ROTATION_DOWNWARD_FACING`. Если в SITL или драйвере ориентация не заполнена, можно временно запустить так:

```sh
ofnav start -r 100 -q 120 -n
```

`-n` нельзя считать нормой для реального полета. Это только для SITL/стендовой диагностики.

## Сборка под плату

Пример для Pixhawk 6X/FMAv6X:

```bash
cd ../PX4-Autopilot
make px4_fmu-v6x_default
```

Загрузка на плату:

```bash
make px4_fmu-v6x_default upload
```

Для другой платы замените target на свой, например:

```bash
make px4_fmu-v5_default upload
```

## Минимальный стендовый тест без винтов

1. Подключить датчик ОП и дальномер.
2. Снять винты.
3. Подключить QGroundControl.
4. Открыть MAVLink Console.
5. Выполнить:

```sh
ofnav start -r 100 -q 120
listener sensor_optical_flow 10
listener distance_sensor 10
listener debug_vect 20
ofnav status
```

6. Поднять БВС над текстурированной поверхностью на 0.5...1.5 м.
7. Медленно переместить вперед/вправо.
8. Проверить, что `OFNAV_V.x/y` меняются без скачков, а `OFNAV_H.y` держится выше порога качества.

## Коды `NavMode`

```text
0 = STANDBY
1 = FLOW_NAV
2 = DEGRADED_HOLD
3 = FAILSAFE_LAND
4 = MANUAL_REQUIRED
```

## Коды `RejectReason`

```text
0  = OK
1  = NON_FINITE_INPUT
2  = BAD_DELTA_TIME
3  = LOW_QUALITY
4  = BAD_HEIGHT
5  = HEIGHT_TOO_LOW
6  = HEIGHT_TOO_HIGH
7  = EXCESSIVE_TILT
8  = FLOW_RATE_LIMIT
9  = VELOCITY_LIMIT
10 = INNOVATION_GATE
11 = STALE_RANGE
12 = STALE_FLOW
13 = IMU_INVALID
```

## Что делать после первого запуска

1. Проверить знаки `OFNAV_V.x/y` при движении вперед/вправо.
2. Проверить качество ОП над разными поверхностями.
3. Проверить реакцию на закрытие датчика ОП.
4. Проверить реакцию на потерю дальномера.
5. Проверить ULog и `debug_vect`.
6. Только после этого обсуждать публикацию velocity в EKF2/vehicle_visual_odometry или отдельный control path.
