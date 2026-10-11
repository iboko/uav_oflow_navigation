#!/usr/bin/env bash
# Runtime smoke test for the *built PX4 v1.16.2 POSIX SITL executable*.
# The replay is processed INSIDE PX4 C++ code, without publishing virtual
# sensor readings on uORB or changing EKF2/actuator state.
set -euo pipefail

if [[ $# -ne 3 && $# -ne 5 ]]; then
  echo "Usage: $0 <PX4-Autopilot> <synthetic-replay.csv> <output-dir> [<eskf-events.csv> <eskf-expected.json>]" >&2
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
  PX4_SIM_MODEL=shell exec "$PX4_BIN" -d -s etc/init.d-posix/rcS "$BUILD/etc"     > "$OUT/px4_sitl_boot.log" 2>&1
) &
server_pid=$!

cleanup() {
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

run_replay() {
  local profile="$1" log="$2"
  for attempt in $(seq 1 60); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "PX4 SITL server quit before module replay" >&2
      cat "$OUT/px4_sitl_boot.log" >&2
      return 1
    fi
    if "$OFNAV_CLIENT" sitl-replay "$INPUT_CSV" "$profile" > "$log" 2>&1; then
      return 0
    fi
    sleep 0.2
  done
  echo "PX4 SITL module replay failed for profile $profile" >&2
  cat "$OUT/px4_sitl_boot.log" >&2
  cat "$log" >&2
  return 1
}

# 1) Optical-flow SENSOR envelope is 0.2-4 m: rejection of every 100 m
# frame is the REQUIRED safe behaviour. Fail if any flow passes.
run_replay flow "$OUT/ofnav_sitl_replay.log"
if ! grep -Eq 'OFNAV_SITL_REPLAY_OK profile=OPTICAL_FLOW_SENSOR rows=[1-9][0-9]* valid=[1-9][0-9]* accepted=0 height_rejected=[1-9][0-9]*' \
    "$OUT/ofnav_sitl_replay.log"; then
  echo "SITL did not explicitly reject the 100m flow-sensor measurements" >&2
  cat "$OUT/ofnav_sitl_replay.log" >&2
  exit 1
fi

# 2) Downward camera VO envelope is 5-120 m with a height-scaled
# uncertainty model: the same 100 m frames must be accepted, none
# rejected by height.
run_replay camera "$OUT/ofnav_sitl_replay_camera.log"
if ! grep -Eq 'OFNAV_SITL_REPLAY_OK profile=DOWNWARD_CAMERA_VO rows=[1-9][0-9]* valid=[1-9][0-9]* accepted=[1-9][0-9]* height_rejected=0' \
    "$OUT/ofnav_sitl_replay_camera.log"; then
  echo "SITL camera-VO profile did not accept the 100m measurements" >&2
  cat "$OUT/ofnav_sitl_replay_camera.log" >&2
  exit 1
fi

# 3) The live module starts with the camera profile and the diagnostic
# 15-state ESKF (no sensors in the shell model: it must stay aligning).
"$OFNAV_CLIENT" start -p camera > "$OUT/ofnav_live.log" 2>&1
sleep 1
"$OFNAV_CLIENT" status >> "$OUT/ofnav_live.log" 2>&1
"$OFNAV_CLIENT" stop >> "$OUT/ofnav_live.log" 2>&1 || true
if ! grep -q 'profile=DOWNWARD_CAMERA_VO' "$OUT/ofnav_live.log" ||
   ! grep -q 'eskf mode=1' "$OUT/ofnav_live.log"; then
  echo "Live ofnav module with ESKF diagnostics did not start" >&2
  cat "$OUT/ofnav_live.log" >&2
  exit 1
fi

# 4) Optional: 15-state fixed-lag ESKF executed INSIDE PX4 on the synthetic
# IMU + delayed VO stream must reproduce the Python reference state.
if [[ $# -eq 5 ]]; then
  EVENTS_CSV="$(realpath "$4")"
  EXPECTED_JSON="$(realpath "$5")"
  "$OFNAV_CLIENT" sitl-eskf "$EVENTS_CSV" > "$OUT/ofnav_sitl_eskf.log" 2>&1
  python3 - "$OUT/ofnav_sitl_eskf.log" "$EXPECTED_JSON" <<'PY'
import json, re, sys
log = open(sys.argv[1]).read()
expected = json.load(open(sys.argv[2]))
m = re.search(r"OFNAV_SITL_ESKF_OK imu=(\d+) frames=(\d+) processed=(\d+) failed=(\d+) "
              r"accepted=(\d+) rejected=(\d+)", log)
s = re.search(r"OFNAV_SITL_ESKF_STATE pn=(\S+) pe=(\S+) vn=(\S+) ve=(\S+)", log)
if not m or not s:
    sys.exit("PX4 ESKF replay did not report OFNAV_SITL_ESKF_OK:\n" + log)
imu, frames, processed, failed, accepted, rejected = map(int, m.groups())
pn, pe, vn, ve = map(float, s.groups())
checks = [
    imu == expected["imu_events"], frames == expected["body_velocity_events"],
    failed == 0, accepted == expected["accepted"], rejected == expected["rejected"],
    abs(pn - expected["position_ned_m"][0]) < 1e-6,
    abs(pe - expected["position_ned_m"][1]) < 1e-6,
    abs(vn - expected["velocity_ned_m_s"][0]) < 1e-6,
    abs(ve - expected["velocity_ned_m_s"][1]) < 1e-6,
]
if not all(checks):
    sys.exit(f"PX4 C++ ESKF differs from Python reference: {checks}\n{log}\n{expected}")
print("PX4_SITL_ESKF_MATCHES_PYTHON")
PY
fi

echo "PX4_SITL_READ_ONLY_REPLAY_PASS"
cat "$OUT/ofnav_sitl_replay.log" "$OUT/ofnav_sitl_replay_camera.log" "$OUT/ofnav_live.log"
if [[ $# -eq 5 ]]; then cat "$OUT/ofnav_sitl_eskf.log"; fi
