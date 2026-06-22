#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/install_px4_ofnav_module.sh <PX4-Autopilot-path> [board_config]

Examples:
  tools/install_px4_ofnav_module.sh ../PX4-Autopilot boards/px4/sitl/default.px4board
  tools/install_px4_ofnav_module.sh ../PX4-Autopilot boards/px4/fmu-v6x/default.px4board

The script installs the ofnav PX4 module into:
  <PX4-Autopilot>/src/modules/ofnav

It also copies the RTOS core into:
  <PX4-Autopilot>/src/modules/ofnav/ofnav_core

If src/modules/CMakeLists.txt exists, the script appends add_subdirectory(ofnav) when missing.
If src/modules/Kconfig exists, the script appends the ofnav Kconfig source when missing.
If board_config is provided, CONFIG_MODULES_OFNAV=y is appended when missing.
USAGE
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

PX4_DIR="$1"
BOARD_CONFIG="${2:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODULE_SRC="$REPO_ROOT/integrations/px4_ofnav_module"
CORE_SRC="$REPO_ROOT/firmware/rtos_core"
MODULE_DST="$PX4_DIR/src/modules/ofnav"

if [[ ! -d "$PX4_DIR" ]]; then
  echo "PX4 directory not found: $PX4_DIR" >&2
  exit 1
fi

if [[ ! -f "$PX4_DIR/CMakeLists.txt" || ! -d "$PX4_DIR/src/modules" ]]; then
  echo "Path does not look like PX4-Autopilot root: $PX4_DIR" >&2
  exit 1
fi

rm -rf "$MODULE_DST"
mkdir -p "$MODULE_DST/ofnav_core"

cp "$MODULE_SRC/CMakeLists.txt" "$MODULE_DST/CMakeLists.txt"
cp "$MODULE_SRC/Kconfig" "$MODULE_DST/Kconfig"
cp "$MODULE_SRC/ofnav_px4.cpp" "$MODULE_DST/ofnav_px4.cpp"
mkdir -p "$MODULE_DST/ofnav_core/include" "$MODULE_DST/ofnav_core/src"
cp -R "$CORE_SRC/include/ofnav" "$MODULE_DST/ofnav_core/include/"
cp "$CORE_SRC/src/ofnav.cpp" "$MODULE_DST/ofnav_core/src/ofnav.cpp"

MODULES_CMAKE="$PX4_DIR/src/modules/CMakeLists.txt"
if [[ -f "$MODULES_CMAKE" ]] && ! grep -q "add_subdirectory(ofnav)" "$MODULES_CMAKE"; then
  printf '\n# UAV optical-flow navigation module\nadd_subdirectory(ofnav)\n' >> "$MODULES_CMAKE"
  echo "Patched: $MODULES_CMAKE"
fi

MODULES_KCONFIG="$PX4_DIR/src/modules/Kconfig"
if [[ -f "$MODULES_KCONFIG" ]] && ! grep -q "src/modules/ofnav/Kconfig" "$MODULES_KCONFIG"; then
  printf '\nsource "src/modules/ofnav/Kconfig"\n' >> "$MODULES_KCONFIG"
  echo "Patched: $MODULES_KCONFIG"
fi

if [[ -n "$BOARD_CONFIG" ]]; then
  BOARD_CONFIG_PATH="$PX4_DIR/$BOARD_CONFIG"
  if [[ ! -f "$BOARD_CONFIG_PATH" ]]; then
    echo "Board config not found: $BOARD_CONFIG_PATH" >&2
    exit 1
  fi

  if ! grep -q "CONFIG_MODULES_OFNAV=y" "$BOARD_CONFIG_PATH"; then
    printf '\n# UAV optical-flow navigation monitor\nCONFIG_MODULES_OFNAV=y\n' >> "$BOARD_CONFIG_PATH"
    echo "Enabled module in: $BOARD_CONFIG_PATH"
  fi
fi

echo "Installed ofnav PX4 module into: $MODULE_DST"
echo "Next: cd $PX4_DIR && make px4_sitl gz_x500"
