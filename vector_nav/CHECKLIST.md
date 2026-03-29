# Vector Nav – Verification Checklist

Work through each layer in order.  Only proceed when the current layer passes.

---

## Layer 0 – Ubuntu + GPU

```bash
lsb_release -a                        # Ubuntu 24.04
nvidia-smi                            # RTX 2060, driver 580
nvcc --version                        # CUDA 11.7 (after toolkit install)
```

---

## Layer 1 – Conda environment

```bash
conda activate adico_nav
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 1.13.1+cu117  True

python -c "import vmas; print('VMAS OK')"
python -c "import benchmarl; print('BenchMARL OK')"
python -c "from het_control.models.het_control_mlp_empirical import HetControlMlpEmpiricalConfig; print('ADiCo OK')"
```

---

## Layer 2 – Policy checkpoint

```bash
conda activate adico_nav
cd vector_nav/
python - <<'EOF'
from policy.loader import load_policy
fn = load_policy("checkpoints/<your_experiment_dir>/")
import torch
obs = torch.zeros(2, 8)   # 2 agents, obs_dim=8
actions = fn(obs)
print("Actions shape:", actions.shape)   # Expected: torch.Size([2, 2])
print("Actions:", actions)
EOF
```

---

## Layer 3 – ROS2 Jazzy

```bash
source /opt/ros/jazzy/setup.bash
ros2 --version                         # jazzy
ros2 topic list                        # should show /rosout at minimum
```

---

## Layer 4 – Gazebo Harmonic

```bash
gz sim --version                       # Gazebo Harmonic
# Launch the arena world:
gz sim vector_nav/ros2_ws/src/vector_sim/worlds/vector_arena.sdf -r
# You should see: 2 coloured boxes (robots) + 2 green/blue cylinders (goals) + arena walls
```

---

## Layer 5 – ROS2 workspace build

```bash
source /opt/ros/jazzy/setup.bash
cd vector_nav/ros2_ws/
colcon build --symlink-install
source install/setup.bash
ros2 pkg list | grep vector           # should show: vector_description, vector_driver, vector_sim
```

---

## Layer 6 – Simulation launch

```bash
# Terminal 1: Gazebo + bridge
source /opt/ros/jazzy/setup.bash
source vector_nav/ros2_ws/install/setup.bash
ros2 launch vector_sim sim_launch.py n_agents:=2

# Terminal 2: verify pose topics are live
source /opt/ros/jazzy/setup.bash
ros2 topic list | grep model          # /model/vector0/pose, /model/vector1/pose, etc.
ros2 topic echo /model/vector0/pose --once

# Terminal 3: send a test velocity command
ros2 topic pub /vector0/cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.1}, angular: {z: 0.0}}" --once
# Robot should move in Gazebo
```

---

## Layer 7 – Full policy loop (sim mode)

```bash
# Terminal 1: Gazebo running (from Layer 6)

# Terminal 2: policy loop
./vector_nav/run_nav.sh --config vector_nav/config/config.yaml
# Expected output:
#   Mode: sim | Agents: ['vector0', 'vector1'] | Goals: ['goal0', 'goal1'] | 10 Hz
#   World state complete. Starting episode.
#   Step 0/100
#   Step 10/100
#   ...
```

---

## Layer 8 – Real robot (defer until hardware available)

```bash
# Prerequisites:
#   - wire-pod running: sudo ./wire-pod/chipper/start.sh
#   - Robots connected to wire-pod: http://localhost:8080
#   - Overhead camera mounted and calibrated

# Calibrate homography:
python vector_nav/perception/calibrate_homography.py

# Update config.yaml: mode: real, camera.marker_ids, robots[].serial

# Run:
./vector_nav/run_nav.sh --config vector_nav/config/config.yaml
```

---

## Quick-fix reference

| Symptom | Fix |
|---------|-----|
| `torch.cuda.is_available()` = False | Check CUDA 11.7 toolkit installed; `nvcc --version` |
| `import benchmarl` fails | Activate `adico_nav` conda env; pip install benchmarl |
| `import het_control` fails | `pip install -e ./adico` inside conda env |
| Gazebo poses not on ROS2 | Check gz_bridge is running; verify bridge topic format |
| `WorldState` times out | Gazebo not publishing `/model/vectorN/pose`; restart bridge |
| Actions shape wrong | Check `n_agents` in config matches checkpoint |
| LD_LIBRARY_PATH conflict | Source ROS2 AFTER conda activate (see `run_nav.sh`) |
