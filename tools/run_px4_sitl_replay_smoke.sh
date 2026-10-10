#!/usr/bin/env bash
# Runtime smoke test for the *built PX4 v1.16.2 POSIX SITL executable*.
# The replay is processed INSIDE PX4 C++ code, without publishing virtual
# sensor readings on uORB or changing EKF2/actuator state.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <PX4-Autopilot> <synthetic-replay.csv> <output-dir>" >&2
  exit 2
fi

PX4_DIR="$(realpath "$1")"
INPUT_CSV="$(realpath "$2")"
OUT="$(realpath -m "$3")"
BUILD="$PX4_DIR/build/px4_sitl_default"
PX4_BIN="$BUILD/bin/px4"
OFNAV_CLIENT="$BUILD/bin/px4-ofnav"

mkdir -p "$OUT" "$BUILD/tmp/rootfs"
test -s "$INPUT_CSV"
test -x "$PX4_BIN"
test -e "$OFNAV_CLIENT"

# PX4_SIM_MODEL=shell starts the minimal PX4 SITL POSIX runtime without
# Gazebo/vehicle physics, so there is NO claim of closed-loop simulation.
(
  cd "$BUILD/tmp/rootfs"
  PX4_SIM_MODEL=shell "$PX4_BIN" -d -s etc/init.d-posix/rcS "$BUILD/etc"     > "$OUT/px4_sitl_boot.log" 2>&1
) &
server_pid=$!

cleanup() {
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

succeeded=0
for attempt in $(seq 1 60); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "PX4 SITL server quit before module replay" >&2
    cat "$OUT/px4_sitl_boot.log" >&2
    exit 1
  fi
  if "$OFNAV_CLIENT" sitl-replay "$INPUT_CSV"        > "$OUT/ofnav_sitl_replay.log" 2>&1; then
    succeeded=1
    break
  fi
  sleep 0.2
done

if [[ "$succeeded" -ne 1 ]]; then
  echo "PX4 SITL module replay failed" >&2
  cat "$OUT/px4_sitl_boot.log" >&2
  cat "$OUT/ofnav_sitl_replay.log" >&2
  exit 1
fi

# The original C++ flight core is for 0.2-4m; the new ideal-camera
# synthetic scenario uses 100m. Rejection of all 100m flow is the
# REQUIRED safe behaviour. The smoke test must fail if any flow passes.
if ! grep -Eq 'OFNAV_SITL_REPLAY_OK rows=[1-9][0-9]* valid=[1-9][0-9]* accepted=0 height_rejected=[1-9][0-9]*'     "$OUT/ofnav_sitl_replay.log"; then
  echo "SITL did not explicitly reject the 100m flow measurements" >&2
  cat "$OUT/ofnav_sitl_replay.log" >&2
  exit 1
fi

echo "PX4_SITL_READ_ONLY_REPLAY_PASS"
cat "$OUT/ofnav_sitl_replay.log"
