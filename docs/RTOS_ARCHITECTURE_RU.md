# Архитектура RTOS-реализации режима полета БВС по оптическому потоку

## 1. Разделение уровней

Проект разделен на два уровня.

Первый уровень — исследовательский и диагностический:

- Python;
- генерация синтетических логов;
- SITL/replay;
- анализ CSV/ULog/DataFlash;
- построение графиков;
- проверка знаков осей и параметров.

Второй уровень — бортовой RTOS-уровень:

- C++17 без динамической аллокации в рабочем цикле;
- фиксированные структуры сообщений;
- периодический вызов из RTOS-задачи;
- отсутствие зависимости от Linux/ROS/Python;
- safety state machine;
- интеграция с PX4/NuttX или ArduPilot/AP_HAL.

## 2. Поток данных

```text
Датчик ОП       ┐
Дальномер       ├─> timestamp alignment ─> ofnav::OfNavRuntime ─> оценка скорости/положения ─> САУ/EKF автопилота
IMU             │
Ориентация      ┘
```

Базовые сообщения RTOS-ядра:

- `OpticalFlowRadSample`;
- `RangeSample`;
- `ImuSample`;
- `AttitudeSample`.

Выход:

- `RuntimeOutput`;
- `FlowVelocityEstimate`;
- `EkfState`;
- `NavMode`;
- `HealthFlags`.

## 3. Математическая модель ОП

После компенсации вращения:

```text
flow_x_rate = (integrated_x - integrated_xgyro) / dt
flow_y_rate = (integrated_y - integrated_ygyro) / dt
```

Для направленного вниз датчика по соглашению MAVLink:

```text
v_sensor_x =  h * flow_y_rate * scale_y
v_sensor_y = -h * flow_x_rate * scale_x
```

Высота:

```text
h = range * cos(roll) * cos(pitch)
```

Ориентация датчика:

```text
v_body_at_sensor = R_body_sensor * v_sensor
```

Компенсация рычага:

```text
v_body_at_cg = v_body_at_sensor - omega_body × r_sensor_body
```

Поворот в навигационную систему:

```text
v_nav = R_yaw * v_body_at_cg
```

## 4. EKF

Компактный горизонтальный EKF использует состояние:

```text
x = [n, e, vn, ve, bax, bay]^T
```

Это не замена штатному EKF2/PX4 или EKF3/ArduPilot. Его назначение:

- независимая проверка ОП-канала;
- companion-computer режим;
- HIL/SITL;
- контроль качества оценок;
- подготовка к переносу в штатный автопилот.

Перед использованием как основного оценивателя нужно расширить состояние до высоты, вертикальной скорости, ориентации и смещений гироскопов либо передавать ОП в штатный EKF автопилота.

## 5. Safety state machine

Состояния:

- `STANDBY` — режим не используется;
- `FLOW_NAV` — ОП валиден и используется;
- `DEGRADED_HOLD` — ОП/качество/инновация деградировали;
- `FAILSAFE_LAND` — критичный отказ: IMU, дальномер, высота, чрезмерный наклон;
- `MANUAL_REQUIRED` — требуется ручной перехват, резерв для интеграции.

Минимальная логика:

```text
нет IMU              -> FAILSAFE_LAND
нет дальномера       -> FAILSAFE_LAND
недопустимая высота  -> FAILSAFE_LAND
низкое качество ОП   -> DEGRADED_HOLD
innovation reject    -> DEGRADED_HOLD
ОП валиден           -> FLOW_NAV
```

## 6. RTOS-интеграция

Рекомендуемая частота задач:

```text
IMU task       250...1000 Hz
Range task      20...100 Hz
Optical flow    20...100 Hz
OfNav task      50...200 Hz
Logger task     10...50 Hz
```

Требования:

- использовать monotonic timestamp в микросекундах;
- не блокировать flight-control task;
- не писать на SD-карту из высокоприоритетной задачи;
- не выполнять heap allocation после инициализации;
- все параметры менять только через атомарный snapshot или защищенный parameter update;
- все отказные события логировать.

## 7. PX4/NuttX

Практический вариант:

1. оставить штатный EKF2 как основной оцениватель;
2. обеспечить корректные `OPTICAL_FLOW_RAD` и `DISTANCE_SENSOR`;
3. использовать `ofnav_rtos_core` как дополнительный контроллер качества/diagnostic backend;
4. после HIL/SITL можно добавить uORB bridge и параметры.

## 8. ArduPilot/AP_HAL

Практический вариант:

1. реализовать backend датчика ОП или использовать штатный;
2. передавать дальномер через штатный rangefinder backend;
3. использовать RTOS-ядро для независимой диагностики и safety decision;
4. не подменять штатный EKF без отдельной валидации.
