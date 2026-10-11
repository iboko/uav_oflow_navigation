#!/usr/bin/env bash
# Static worst-case stack of the ofnav task for a BUILT PX4 NuttX target:
# recompiles the ofnav translation units with the exact compile commands of
# the build plus -fcallgraph-info=su and walks the call graph.
# Usage: tools/px4_v6x_stack_report.sh <PX4-Autopilot> [build-dir-name] [out-dir]
set -euo pipefail
PX4_DIR="$(realpath "$1")"
BUILD="$PX4_DIR/build/${2:-px4_fmu-v6x_default}"
OUT="$(realpath -m "${3:-build/px4_v6x_stack}")"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$OUT"
python3 - "$BUILD" "$OUT" <<'PY'
import json, os, shlex, subprocess, sys
build, out = sys.argv[1], sys.argv[2]
for c in json.load(open(os.path.join(build, "compile_commands.json"))):
    if "/modules/ofnav/" not in c["file"]:
        continue
    clean, skip = [], False
    for a in shlex.split(c["command"]):
        if skip:
            skip = False
            continue
        if a in ("-MT", "-MF", "-o"):
            skip = True
            continue
        if a == "-MD" or a.endswith("ccache"):
            continue
        clean.append(a)
    obj = os.path.join(out, os.path.basename(c["file"]) + ".o")
    subprocess.run(clean + ["-fcallgraph-info=su", "-o", obj], cwd=c["directory"], check=True)
PY
python3 "$ROOT/tools/stack_callgraph.py" "$OUT"/*.ci \
  --root _ZN5Ofnav3runEv \
  --root _ZN5ofnav10EskfRunner11addImuDeltaE \
  --root _ZN5ofnav10EskfRunner15addBodyVelocityE \
  --root _ZN7ofnav1511runWorkloadE \
  --json "$OUT/stack_report.json" | tee "$OUT/stack_report.txt"
