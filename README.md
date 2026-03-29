# Vector Nav — MARL Sim-to-Real Navigation

Multi-agent reinforcement learning (MARL) navigation for 3 Anki Vector robots, using [ADiCo](https://github.com/Svar7769/AD2C-Diversity-Testing) policies trained in [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) and deployed in Gazebo Harmonic via ROS2 Jazzy.

**3 robots · 3 goals · decentralised execution**

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Gazebo Harmonic  (3 × Vector robot, DiffDrive physics)  │
└───────────────────────┬─────────────────────────────────┘
                        │  gz_bridge  (pose ↕ cmd_vel)
┌───────────────────────▼─────────────────────────────────┐
│  ros_bridge.py  (Python 3.12 / system)                   │
│  · subscribes /model/vectorN/pose                        │
│  · builds obs tensor (10-dim per agent)                  │
│  · smooth unicycle controller → /vectorN/cmd_vel         │
└───────────────────────┬─────────────────────────────────┘
                        │  TCP JSON  (port 5557)
┌───────────────────────▼─────────────────────────────────┐
│  policy_server.py  (Python 3.9 / conda env adico_nav)    │
│  · loads ADiCo checkpoint (HetControlMlpEmpirical)       │
│  · shared_mlp + per-agent MLPs (torchrl vmap format)     │
│  · returns [vx, vy] for each agent                       │
└─────────────────────────────────────────────────────────┘
```

### Observation space (10-dim per agent)
```
[x, y, vx, vy,  dx_goal, dy_goal,  dx_a1, dy_a1,  dx_a2, dy_a2]
```

### Action space (2-dim per agent)
```
[vx, vy]  — world-frame velocity, ±0.5 m/s
```
Mapped to DiffDrive via a smooth unicycle controller: `linear.x = speed · cos(heading_error)`, `angular.z = K · heading_error`.

---

## Project Structure

```
vector_nav/
├── checkpoints/                   # trained policy weights (.pt)
├── config/
│   └── config_3a3g.yaml           # all parameters (n_agents, obs_dim, goals, …)
├── policy/
│   └── loader.py                  # loads ADiCo checkpoint into PyTorch
├── ros2_ws/src/vector_sim/
│   ├── launch/sim_launch.py       # starts Gazebo + gz_bridge
│   ├── models/vector{0,1,2}/      # per-robot SDF models with DiffDrive
│   ├── models/vector/             # shared body mesh + geometry
│   └── worlds/vector_arena.sdf    # 2×2 m arena with walls and goal markers
├── environment.yaml               # conda env spec (Python 3.9, PyTorch 1.13.1)
├── policy_server.py               # policy inference TCP server
├── ros_bridge.py                  # ROS2 perception + control node
├── run_sim.sh                     # one-command launcher
└── setup.sh                       # one-time environment setup
```

---

## Requirements

| Component | Version |
|-----------|---------|
| OS | Ubuntu 24.04 |
| GPU | NVIDIA (CUDA 11.7 compatible) |
| ROS2 | Jazzy |
| Gazebo | Harmonic |
| Python (ROS2 node) | 3.12 (system) |
| Python (policy) | 3.9 (conda) |
| PyTorch | 1.13.1+cu117 |

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/Svar7769/ADiCo_anki.git
cd vector_nav
```

> The checkpoint file is not included in the repo (too large). Place your trained `.pt` file at:
> ```
> vector_nav/checkpoints/navigation_3a3g_seed0.pt
> ```

### 2. Run the setup script

```bash
cd vector_nav
chmod +x setup.sh
./setup.sh
```

This installs:
- System dependencies (cmake, curl, etc.)
- Miniconda + `adico_nav` conda env (Python 3.9, PyTorch 1.13.1+cu117)
- ROS2 Jazzy + Gazebo Harmonic
- Builds the ROS2 workspace (`ros2_ws/`)

To install components individually:
```bash
./setup.sh --ros     # ROS2 + Gazebo only
./setup.sh --conda   # conda env only
```

### 3. Build the ROS2 workspace (if not done by setup.sh)

```bash
cd vector_nav/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select vector_sim
```

### 4. Clone the ADiCo source (needed for policy_server)

```bash
# from the repo root (ADiCo_anki/)
git clone https://github.com/Svar7769/AD2C-Diversity-Testing.git adico
conda activate adico_nav
pip install -e adico/
```

---

## Running the simulation

```bash
cd vector_nav
./run_sim.sh
```

This starts all 3 processes in order:
1. **Gazebo** — opens the simulator with 3 Vector robots and 3 goal markers
2. **policy_server** — loads the checkpoint and listens on port 5557
3. **ros_bridge** — subscribes to poses, calls the policy, drives the robots

Press `Ctrl+C` to stop everything cleanly.

### Configuration

Edit `config/config_3a3g.yaml` to change:
- `goals` — x/y positions of the 3 goal markers
- `hz` — control frequency (default 10 Hz)
- `max_steps` — episode length
- `policy.u_range` — max velocity (default 0.5 m/s)

---

## How it works

1. **Gazebo** simulates 3 Vector robots in a 2×2 m walled arena. Each robot has a DiffDrive plugin receiving `cmd_vel` Twist messages.

2. **gz_bridge** (launched by `sim_launch.py`) bridges Gazebo topics to ROS2:
   - `/model/vectorN/pose` → ROS2 (robot positions)
   - `/vectorN/cmd_vel` → Gazebo (robot commands)

3. **ros_bridge.py** runs at 10 Hz:
   - Reads all robot and goal poses from ROS2
   - Estimates velocities via finite difference
   - Builds the 10-dim observation for each agent
   - Sends observations to **policy_server** over TCP
   - Receives `[vx, vy]` actions and converts to `Twist` via the unicycle controller

4. **policy_server.py** wraps the `HetControlMlpEmpirical` network:
   - **Shared MLP** — processes the shared observation
   - **Agent MLPs** — per-agent heads with batched weights `[n_agents, out, in]`
   - Runs under Python 3.9/conda to avoid PyTorch version conflicts with ROS2

---

## Training

Policies are trained with [BenchMARL](https://github.com/facebookresearch/BenchMARL) + the ADiCo algorithm on the `navigation_bounded` VMAS scenario.

Key training parameters:
```
n_agents=3, obs_dim=10, action_dim=2
observe_all_goals=False
agents_with_same_goal=1
```

To retrain:
```bash
conda activate adico_nav
cd adico
python het_control/train.py task=vmas/navigation_bounded \
    algorithm=adico n_agents=3 seed=0
```

---

## Troubleshooting

**`Package 'vector_sim' not found`**
```bash
source vector_nav/ros2_ws/install/setup.bash
# or just use ./run_sim.sh which handles this automatically
```

**Robots not moving**
- Check that `policy_server.py` is running and loaded the checkpoint
- Verify the gz_bridge is up: `ros2 topic list | grep cmd_vel`

**Mesh/URI errors in Gazebo**
```bash
cd vector_nav/ros2_ws
colcon build --symlink-install --packages-select vector_sim
```

**PyTorch CUDA errors in policy_server**
- Ensure `adico_nav` conda env has `pytorch-cuda=11.7`
- Run `conda activate adico_nav && python -c "import torch; print(torch.cuda.is_available())"`
