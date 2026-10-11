#!/usr/bin/env bash
# Build the shared ESKF workload for Cortex-M7 with the PX4 fmu-v6x code
# generation flags and count executed instructions in QEMU mps2-an500.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/build/qemu_m7}"
mkdir -p "$OUT"
CORE="$ROOT/firmware/rtos_core"
FLAGS=(-mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard -Os -std=gnu++14
       -fno-rtti -fno-exceptions -fno-threadsafe-statics -ffunction-sections -fdata-sections
       -fno-math-errno -freciprocal-math -fno-signed-zeros -fno-trapping-math
       -D__PX4_NUTTX -Wall -Wextra -Werror -Wdouble-promotion -I"$CORE/include")
arm-none-eabi-g++ "${FLAGS[@]}" \
  "$CORE/tests/target/qemu_m7/main.cpp" "$CORE/src/eskf15.cpp" "$CORE/src/eskf15_workload.cpp" \
  -T "$CORE/tests/target/qemu_m7/link.ld" -nostartfiles --specs=nano.specs --specs=nosys.specs \
  -u _printf_float -Wl,--gc-sections -o "$OUT/eskf15_m7.elf"
arm-none-eabi-size "$OUT/eskf15_m7.elf"
timeout 600 qemu-system-arm -M mps2-an500 -nographic -semihosting -icount shift=0 \
  -kernel "$OUT/eskf15_m7.elf" | tee "$OUT/qemu_m7_result.json"
