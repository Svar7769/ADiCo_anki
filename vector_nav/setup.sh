#!/usr/bin/env bash
# =============================================================================
#  vector_nav/setup.sh
#  Reproduces the complete environment on a fresh Ubuntu 24.04 machine.
#
#  USAGE:
#    chmod +x setup.sh
#    ./setup.sh           # full setup
#    ./setup.sh --ros     # ROS2 + Gazebo only
#    ./setup.sh --conda   # conda env only
#    ./setup.sh --wirepod # wire-pod only (needs Go)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONDA_ENV_NAME="adico_nav"
ROS_DISTRO="jazzy"

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Parse flags ───────────────────────────────────────────────────────────────
DO_ALL=true; DO_ROS=false; DO_CONDA=false; DO_WIREPOD=false
for arg in "$@"; do
  case $arg in
    --ros)     DO_ROS=true;     DO_ALL=false ;;
    --conda)   DO_CONDA=true;   DO_ALL=false ;;
    --wirepod) DO_WIREPOD=true; DO_ALL=false ;;
  esac
done
$DO_ALL && DO_ROS=true && DO_CONDA=true && DO_WIREPOD=true

# =============================================================================
#  STEP 1 – System dependencies
# =============================================================================
install_system_deps() {
  info "Installing system dependencies..."
  sudo apt-get update -qq
  sudo apt-get install -y \
    curl wget git build-essential cmake \
    python3-pip python3-dev \
    libopencv-dev python3-opencv \
    libssl-dev libffi-dev \
    gnupg lsb-release ca-certificates \
    software-properties-common \
    v4l-utils          # camera utilities (real mode)
}

# =============================================================================
#  STEP 2 – Miniconda
# =============================================================================
install_conda() {
  if command -v conda &>/dev/null; then
    info "conda already installed: $(conda --version)"
    return
  fi
  info "Installing Miniconda..."
  MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  wget -q "$MINICONDA_URL" -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  rm /tmp/miniconda.sh
  eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
  conda init bash
  info "Miniconda installed. Shell restart may be needed for full init."
}

# =============================================================================
#  STEP 3 – Conda environment (adico_nav)
# =============================================================================
create_conda_env() {
  # Make conda available in this script session
  if ! command -v conda &>/dev/null; then
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
  fi

  if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    warn "Conda env '${CONDA_ENV_NAME}' already exists. Skipping creation."
    warn "To rebuild: conda env remove -n ${CONDA_ENV_NAME}"
    return
  fi

  info "Creating conda env '${CONDA_ENV_NAME}' (Python 3.9, PyTorch 1.13.1+cu117)..."
  conda env create -f "${SCRIPT_DIR}/environment.yaml"

  info "Installing ADiCo stack into conda env..."
  conda run -n "$CONDA_ENV_NAME" pip install \
    --extra-index-url https://download.pytorch.org/whl/cu117 \
    torchrl==0.1.1 \
    tensordict==0.1.1

  # Install ADiCo (het_control) in editable mode
  ADICO_DIR="${PROJECT_ROOT}/adico"
  if [ -d "$ADICO_DIR" ]; then
    info "Installing ADiCo from ${ADICO_DIR}..."
    conda run -n "$CONDA_ENV_NAME" pip install -e "$ADICO_DIR"
  else
    warn "ADiCo source not found at ${ADICO_DIR}."
    warn "Clone it first:"
    warn "  git clone https://github.com/Svar7769/AD2C-Diversity-Testing.git adico"
    warn "Then rerun: ./setup.sh --conda"
  fi

  info "Conda env '${CONDA_ENV_NAME}' ready."
}

# =============================================================================
#  STEP 4 – ROS2 Jazzy (Ubuntu 24.04 → Jazzy is the supported distro)
# =============================================================================
install_ros2() {
  if command -v ros2 &>/dev/null; then
    info "ROS2 already installed: $(ros2 --version 2>&1 | head -1)"
    return
  fi

  info "Installing ROS2 ${ROS_DISTRO}..."

  # Add ROS2 apt repo
  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

  sudo apt-get update -qq
  sudo apt-get install -y \
    ros-${ROS_DISTRO}-desktop \
    ros-${ROS_DISTRO}-ros-gz \
    ros-${ROS_DISTRO}-gz-ros2-control \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool

  # rosdep init (skip if already done)
  sudo rosdep init 2>/dev/null || true
  rosdep update

  # Source in bashrc
  if ! grep -q "ros/${ROS_DISTRO}/setup.bash" ~/.bashrc; then
    echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
    info "Added ROS2 source to ~/.bashrc"
  fi

  info "ROS2 ${ROS_DISTRO} installed."
}

# =============================================================================
#  STEP 5 – Gazebo Harmonic (ships with ros-jazzy-ros-gz)
# =============================================================================
install_gazebo() {
  if command -v gz &>/dev/null; then
    info "Gazebo already installed: $(gz --version 2>&1 | head -1)"
    return
  fi

  info "Installing Gazebo Harmonic..."
  # Gazebo Harmonic is the default with ROS2 Jazzy via ros-jazzy-ros-gz
  # (already installed above). Install standalone tools as well:
  sudo apt-get install -y \
    gz-harmonic \
    ros-${ROS_DISTRO}-gz-bridge \
    ros-${ROS_DISTRO}-gz-image \
    ros-${ROS_DISTRO}-gz-sim

  info "Gazebo Harmonic installed."
}

# =============================================================================
#  STEP 6 – Build ROS2 workspace
# =============================================================================
build_ros2_ws() {
  WS="${SCRIPT_DIR}/ros2_ws"
  info "Building ROS2 workspace at ${WS}..."

  source /opt/ros/${ROS_DISTRO}/setup.bash

  cd "$WS"
  rosdep install --from-paths src --ignore-src -r -y 2>/dev/null || true
  colcon build --symlink-install
  info "ROS2 workspace built."

  # Source overlay in bashrc
  SETUP_LINE="source ${WS}/install/setup.bash"
  if ! grep -q "$SETUP_LINE" ~/.bashrc; then
    echo "$SETUP_LINE" >> ~/.bashrc
  fi
}

# =============================================================================
#  STEP 7 – wire-pod (optional, skip for sim-only setup)
# =============================================================================
install_wirepod() {
  info "─── wire-pod setup ─────────────────────────────────────────────"
  info "wire-pod requires Go ≥ 1.21. Checking..."

  if ! command -v go &>/dev/null; then
    info "Installing Go 1.22..."
    GO_VER="1.22.3"
    wget -q "https://go.dev/dl/go${GO_VER}.linux-amd64.tar.gz" -O /tmp/go.tar.gz
    sudo rm -rf /usr/local/go
    sudo tar -C /usr/local -xzf /tmp/go.tar.gz
    rm /tmp/go.tar.gz
    export PATH="$PATH:/usr/local/go/bin"
    if ! grep -q "/usr/local/go/bin" ~/.bashrc; then
      echo 'export PATH="$PATH:/usr/local/go/bin"' >> ~/.bashrc
    fi
  fi

  info "Go version: $(go version)"

  WIREPOD_DIR="${PROJECT_ROOT}/wire-pod"
  if [ ! -d "$WIREPOD_DIR" ]; then
    info "Cloning wire-pod..."
    git clone https://github.com/kercre123/wire-pod.git "$WIREPOD_DIR"
  fi

  info "Building wire-pod..."
  cd "$WIREPOD_DIR"
  sudo ./setup.sh   # wire-pod's own setup script

  info "wire-pod built. To start: sudo ./chipper/start.sh"
  info "Verify robots connect at: http://localhost:8080"
}

# =============================================================================
#  STEP 8 – ROS2/conda integration helper
# =============================================================================
write_ros_conda_wrapper() {
  WRAPPER="${SCRIPT_DIR}/run_nav.sh"
  cat > "$WRAPPER" <<'EOF'
#!/usr/bin/env bash
# Activates adico_nav conda env AND sources ROS2, then runs main.py
# Usage: ./run_nav.sh [--config config/config.yaml]

eval "$(conda shell.bash hook)"
conda activate adico_nav

# Source ROS2 AFTER conda activate (order matters for LD_LIBRARY_PATH)
source /opt/ros/jazzy/setup.bash
source "$(dirname "$0")/ros2_ws/install/setup.bash" 2>/dev/null || true

# Prevent conda's libstdc++ from shadowing ROS2's
export LD_LIBRARY_PATH="/opt/ros/jazzy/lib:$LD_LIBRARY_PATH"

python "$(dirname "$0")/main.py" "$@"
EOF
  chmod +x "$WRAPPER"
  info "Created run wrapper: ${WRAPPER}"
}

# =============================================================================
#  Main
# =============================================================================
info "Ubuntu $(lsb_release -rs) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
install_system_deps

$DO_CONDA   && { install_conda; create_conda_env; }
$DO_ROS     && { install_ros2; install_gazebo; build_ros2_ws; }
$DO_WIREPOD && install_wirepod
write_ros_conda_wrapper

info "══════════════════════════════════════════════════════"
info "Setup complete!  Next steps:"
info "  1. source ~/.bashrc"
info "  2. conda activate adico_nav"
info "  3. Place BenchMARL checkpoint in: vector_nav/checkpoints/"
info "  4. gz sim ros2_ws/src/vector_sim/worlds/vector_arena.sdf"
info "  5. ./run_nav.sh --config config/config.yaml"
info "══════════════════════════════════════════════════════"
