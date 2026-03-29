"""
ros_bridge.py  —  run with Python 3.12 (system, ROS2 sourced)
──────────────────────────────────────────────────────────────
ROS2 node that:
  1. Subscribes to /model/vectorN/pose  (Gazebo → WorldState)
  2. Connects to policy_server over TCP
  3. Sends observations, receives actions
  4. Publishes geometry_msgs/Twist to /vectorN/cmd_vel

Must be run AFTER policy_server.py is ready.

Usage:
  # (system Python, ROS2 sourced, NO conda)
  source /opt/ros/jazzy/setup.bash
  source vector_nav/ros2_ws/install/setup.bash
  cd vector_nav/
  python3 ros_bridge.py --config config/config_3a3g.yaml
"""
import argparse
import json
import logging
import math
import socket
import sys
import threading
import time
import yaml
from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] ros_bridge: %(message)s",
)
log = logging.getLogger(__name__)

POLICY_HOST = "127.0.0.1"
POLICY_PORT = 5557


# ── WorldState (no-dep copy; avoids importing from conda path) ────────────────

class AgentState:
    def __init__(self):
        self.x = self.y = self.vx = self.vy = self.heading = 0.0
        self.prev_x = self.prev_y = 0.0
        self.prev_time = time.time()
        self.updated = False

class GoalState:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y
        self.updated = True   # static goals are pre-populated from config


# ── ROS2 Node ────────────────────────────────────────────────────────────────

class VectorNavBridge(Node):

    def __init__(self, cfg: dict, policy_socket: socket.socket):
        super().__init__("vector_nav_bridge")
        self._cfg = cfg
        self._sock = policy_socket
        self._sock_buf = ""
        self._lock = threading.Lock()

        n_agents = cfg["n_agents"]
        agent_ids = [r["id"] for r in cfg["robots"]]
        goal_ids  = [g["name"] for g in cfg["goals"]]
        self._agent_ids = agent_ids
        self._goal_ids  = goal_ids
        self._hz        = cfg.get("hz", 10)
        self._max_steps = cfg.get("max_steps", 100)
        self._u_range   = cfg["policy"].get("u_range", 0.5)
        self._step      = 0

        prefix = cfg["ros2"]["pose_topic_prefix"]

        # Agent states (live-updated by pose callbacks)
        self._agents = {aid: AgentState() for aid in agent_ids}

        # Goal states (pre-filled from config; also subscribed for completeness)
        self._goals = {
            g["name"]: GoalState(g["x"], g["y"])
            for g in cfg["goals"]
        }

        # Subscribe to agent poses
        for aid in agent_ids:
            self.create_subscription(
                Pose, f"{prefix}/{aid}/pose",
                lambda msg, a=aid: self._agent_pose_cb(msg, a), 10
            )

        # Subscribe to goal poses (Gazebo may move them; keeps us in sync)
        for gid in goal_ids:
            self.create_subscription(
                Pose, f"{prefix}/{gid}/pose",
                lambda msg, g=gid: self._goal_pose_cb(msg, g), 10
            )

        # Publishers: /vectorN/cmd_vel
        ros_ns = cfg["ros2"]["namespace_prefix"]
        self._pubs = {
            aid: self.create_publisher(
                Twist, f"/{aid}/cmd_vel", 10
            )
            for aid in agent_ids
        }

        # Control loop timer
        self._timer = self.create_timer(1.0 / self._hz, self._control_loop)
        log.info(f"Bridge ready — {n_agents} agents, {len(goal_ids)} goals, {self._hz} Hz")

    # ── Pose callbacks ────────────────────────────────────────────────────────

    def _agent_pose_cb(self, msg: Pose, agent_id: str):
        now = time.time()
        with self._lock:
            a = self._agents[agent_id]
            dt = now - a.prev_time
            if dt > 0 and a.updated:
                a.vx = (msg.position.x - a.prev_x) / dt
                a.vy = (msg.position.y - a.prev_y) / dt
            a.prev_x, a.prev_y = a.x, a.y
            a.x = msg.position.x
            a.y = msg.position.y
            a.heading = self._yaw_from_quat(msg.orientation)
            a.prev_time = now
            a.updated = True

    def _goal_pose_cb(self, msg: Pose, goal_id: str):
        with self._lock:
            self._goals[goal_id].x = msg.position.x
            self._goals[goal_id].y = msg.position.y

    @staticmethod
    def _yaw_from_quat(q) -> float:
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    # ── Control loop ──────────────────────────────────────────────────────────

    def _control_loop(self):
        if self._step >= self._max_steps:
            log.info("Episode complete.")
            self._zero_all()
            self._timer.cancel()
            return

        with self._lock:
            # Check all agents have received at least one pose update
            if not all(a.updated for a in self._agents.values()):
                log.info("Waiting for all agent poses...")
                return

            obs = self._build_obs()

        # Log full obs on first step so we can verify format
        if self._step == 0:
            with self._lock:
                for i, aid in enumerate(self._agent_ids):
                    log.info(f"INIT obs[{aid}] = {[f'{v:+.4f}' for v in obs[i]]}")
                for gid in self._goal_ids:
                    g = self._goals[gid]
                    log.info(f"INIT goal {gid}: ({g.x:+.3f}, {g.y:+.3f})")

        # Send obs to policy_server, get actions
        try:
            actions = self._query_policy(obs)
        except Exception as e:
            log.error(f"Policy query failed: {e}")
            return

        # Publish cmd_vel for each agent + diagnostics every 5 steps
        awsg = self._cfg["policy"].get("agents_with_same_goal", 1)
        agent_radius = self._cfg["task"].get("agent_radius", 0.1)
        with self._lock:
            for i, aid in enumerate(self._agent_ids):
                goal_idx = i // awsg
                gid = self._goal_ids[goal_idx]
                g = self._goals[gid]
                a = self._agents[aid]
                dist = math.sqrt((a.x - g.x)**2 + (a.y - g.y)**2)
                if dist < agent_radius:
                    # Agent reached its goal — stop and hold
                    self._publish_twist(aid, 0.0, 0.0)
                else:
                    vx = float(actions[i][0])
                    vy = float(actions[i][1])
                    self._publish_twist(aid, vx, vy)

        if self._step % 5 == 0:
            self._log_diagnostics(obs, actions)
        self._step += 1

    def _log_diagnostics(self, obs: list, actions: list):
        log.info(f"── Step {self._step}/{self._max_steps} ──────────────────")
        with self._lock:
            for i, aid in enumerate(self._agent_ids):
                a = self._agents[aid]
                # Distance to each goal
                dists = []
                for gid in self._goal_ids:
                    g = self._goals[gid]
                    d = math.sqrt((a.x - g.x)**2 + (a.y - g.y)**2)
                    dists.append(f"{gid}:{d:.3f}m")
                vx, vy = float(actions[i][0]), float(actions[i][1])
                log.info(
                    f"  {aid}: pos=({a.x:+.3f},{a.y:+.3f}) "
                    f"vel=({a.vx:+.3f},{a.vy:+.3f}) "
                    f"heading={math.degrees(a.heading):+.1f}° "
                    f"action=({vx:+.3f},{vy:+.3f}) "
                    f"dist_to_goals=[{', '.join(dists)}]"
                )

    # ── Observation builder ───────────────────────────────────────────────────

    def _build_obs(self) -> list:
        """
        Build obs list-of-lists matching navigation_bounded training format.

        obs_dim = 4 + 2 + 2*(n_agents-1)  =  10 for 3 agents

        Layout per agent i:
          [x, y, vx, vy,                          # absolute pose + velocity  (4)
           dx_to_own_goal, dy_to_own_goal,         # relative to assigned goal (2)
           dx_to_other0, dy_to_other0, ...]        # relative to every other agent (2*(n-1))

        Goal assignment: agent i → goal index floor(i / agents_with_same_goal).
        Other-agent order: all agents in self._agent_ids order, skipping self
          (matches navigation_bounded.py observation() loop order).

        observe_all_goals=True (legacy 3a1g / 3a2g):
          Uses the old format: all goal deltas, no relative agent positions.
        """
        observe_all = self._cfg["task"].get("observe_all_goals", False)
        awsg = self._cfg["policy"].get("agents_with_same_goal", 1)
        obs_dim = self._cfg["policy"].get("obs_dim", None)

        obs_all = []
        for idx, aid in enumerate(self._agent_ids):
            a = self._agents[aid]
            obs = [a.x, a.y, a.vx, a.vy]

            if observe_all:
                # Legacy format: relative to every goal
                for gid in self._goal_ids:
                    g = self._goals[gid]
                    obs += [a.x - g.x, a.y - g.y]
            else:
                # Own goal only
                goal_idx = idx // awsg
                gid = self._goal_ids[goal_idx]
                g = self._goals[gid]
                obs += [a.x - g.x, a.y - g.y]

                # Relative positions of other agents (navigation_bounded format)
                for other_aid in self._agent_ids:
                    if other_aid == aid:
                        continue
                    o = self._agents[other_aid]
                    obs += [a.x - o.x, a.y - o.y]

            obs_all.append(obs)

        # Sanity check on first call (step 0 log will catch mismatches)
        if obs_dim is not None and len(obs_all[0]) != obs_dim:
            log.warning(
                f"obs_dim mismatch: built {len(obs_all[0])} but config says {obs_dim}. "
                "Check observe_all_goals and n_agents settings."
            )

        return obs_all

    # ── Policy socket ─────────────────────────────────────────────────────────

    def _query_policy(self, obs: list) -> list:
        """Send obs, receive actions via TCP socket."""
        req = json.dumps({"obs": obs, "step": self._step}) + "\n"
        self._sock.sendall(req.encode("utf-8"))

        # Read response (wait for newline)
        while "\n" not in self._sock_buf:
            chunk = self._sock.recv(4096).decode("utf-8")
            if not chunk:
                raise ConnectionError("policy_server closed connection")
            self._sock_buf += chunk

        line, self._sock_buf = self._sock_buf.split("\n", 1)
        resp = json.loads(line)

        if "error" in resp:
            raise RuntimeError(f"policy_server error: {resp['error']}")

        return resp["actions"]   # [[vx0,vy0], [vx1,vy1], ...]

    # ── Command publishing ────────────────────────────────────────────────────

    # P-gain for heading controller (rad/s per rad of error).
    _K_HEADING = 2.5

    def _publish_twist(self, agent_id: str, vx: float, vy: float):
        """
        Convert world-frame policy (vx, vy) to DiffDrive Twist (linear.x, angular.z).

        Strategy (smooth unicycle controller):
          - linear.x  = speed * max(0, cos(err))  — naturally fades to 0 when
            misaligned (>90°) and ramps up smoothly as heading corrects.
          - angular.z = K_HEADING * err            — proportional heading correction.
          - No hard threshold → no abrupt stop/start jerk.
        """
        speed = math.sqrt(vx**2 + vy**2)
        msg = Twist()
        if speed < 1e-3:
            msg.linear.x = 0.0
            msg.angular.z = 0.0
        else:
            heading = self._agents[agent_id].heading
            desired = math.atan2(vy, vx)
            err = math.atan2(math.sin(desired - heading), math.cos(desired - heading))
            msg.linear.x = float(speed * max(0.0, math.cos(err)))
            msg.angular.z = float(self._K_HEADING * err)
        self._pubs[agent_id].publish(msg)

    def _zero_all(self):
        for aid in self._agent_ids:
            self._publish_twist(aid, 0.0, 0.0)


# ── Main ──────────────────────────────────────────────────────────────────────

def connect_to_policy_server(host: str, port: int, retries: int = 20) -> socket.socket:
    """Retry connecting to policy_server until it's ready."""
    for i in range(retries):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
            log.info(f"Connected to policy_server at {host}:{port}")
            return s
        except ConnectionRefusedError:
            log.info(f"Waiting for policy_server... ({i+1}/{retries})")
            time.sleep(1.0)
    raise RuntimeError(f"Could not connect to policy_server at {host}:{port}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config_3a3g.yaml")
    parser.add_argument("--policy-host", default=POLICY_HOST)
    parser.add_argument("--policy-port", type=int, default=POLICY_PORT)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Connect to policy_server (waits up to 20 s)
    sock = connect_to_policy_server(args.policy_host, args.policy_port)

    rclpy.init()
    node = VectorNavBridge(cfg, sock)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        node._zero_all()
        node.destroy_node()
        rclpy.shutdown()
        sock.close()


if __name__ == "__main__":
    main()
