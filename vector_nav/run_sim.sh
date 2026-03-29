#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  run_sim.sh  —  starts Gazebo + policy_server + ros_bridge
#
#  Usage:
#    ./run_sim.sh          →  3 agents, 3 goals (default)
#    ./run_sim.sh 3a1g     →  3 agents, 1 shared goal
#    ./run_sim.sh 3a2g     →  3 agents, 2 goals
#    ./run_sim.sh 3a3g     →  3 agents, 3 goals
#
#  Run from anywhere — no need to source ROS2 first.
# ─────────────────────────────────────────────────────────────────────────────
set -eo pipefail

VARIANT="${1:-3a3g}"
CONFIG="config/config_${VARIANT}.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_PYTHON="$HOME/miniconda3/envs/adico_nav/bin/python3"
SYSTEM_PYTHON="/usr/bin/python3"

if [ ! -f "$SCRIPT_DIR/$CONFIG" ]; then
  echo "ERROR: Config not found: $CONFIG"
  echo "Valid variants: 3a1g  3a2g  3a3g"
  exit 1
fi

if [ ! -f "$CONDA_PYTHON" ]; then
  echo "ERROR: conda env 'adico_nav' not found at $CONDA_PYTHON"
  exit 1
fi

echo "════════════════════════════════════════════════"
echo "  Vector Nav Sim  —  variant: $VARIANT"
echo "════════════════════════════════════════════════"

cd "$SCRIPT_DIR"

export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export LD_PRELOAD=/lib/x86_64-linux-gnu/libpthread.so.0

N_AGENTS="${VARIANT:0:1}"

cleanup() {
  echo ""; echo "Stopping..."
  kill "$GAZEBO_PID" 2>/dev/null || true
  kill "$POLICY_PID" 2>/dev/null || true
  kill "$BRIDGE_PID" 2>/dev/null || true
  wait 2>/dev/null
  echo "Done."
}
trap cleanup INT TERM

echo "[1/3] Starting Gazebo (n_agents=$N_AGENTS)..."
(
  source /opt/ros/jazzy/setup.bash
  source "$SCRIPT_DIR/ros2_ws/install/setup.bash"
  ros2 launch vector_sim sim_launch.py n_agents:="$N_AGENTS"
) &
GAZEBO_PID=$!

sleep 5

echo "[2/3] Starting policy_server..."
"$CONDA_PYTHON" "$SCRIPT_DIR/policy_server.py" --config "$SCRIPT_DIR/$CONFIG" &
POLICY_PID=$!

sleep 3

echo "[3/3] Starting ros_bridge..."
(
  source /opt/ros/jazzy/setup.bash
  source "$SCRIPT_DIR/ros2_ws/install/setup.bash"
  "$SYSTEM_PYTHON" "$SCRIPT_DIR/ros_bridge.py" --config "$SCRIPT_DIR/$CONFIG"
) &
BRIDGE_PID=$!

echo ""; echo "  All running. Press Ctrl+C to stop all."
echo "════════════════════════════════════════════════"

wait
