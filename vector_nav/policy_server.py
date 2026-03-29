"""
policy_server.py  —  run with Python 3.9 (conda adico_nav)
────────────────────────────────────────────────────────────
TCP server that loads the ADiCo policy and serves inference requests.

Protocol (newline-delimited JSON):
  Request:  {"config": "3a3g", "obs": [[...], [...], ...], "step": N}
  Response: {"actions": [[vx0,vy0], [vx1,vy1], ...]}
  Error:    {"error": "message"}

Usage:
  conda activate adico_nav
  cd vector_nav/
  python policy_server.py --config config/config_3a3g.yaml
"""
import argparse
import json
import logging
import socket
import sys
import threading
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] policy_server: %(message)s",
)
log = logging.getLogger(__name__)

HOST = "127.0.0.1"
PORT = 5557


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def handle_client(conn: socket.socket, policy_fn, obs_dim: int, n_agents: int):
    """Handle one persistent client connection (ros_bridge)."""
    log.info("ros_bridge connected.")
    buf = ""
    try:
        while True:
            data = conn.recv(65536).decode("utf-8")
            if not data:
                break
            buf += data
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                    obs_list = req["obs"]               # [[...], [...], ...]
                    obs = torch.tensor(obs_list, dtype=torch.float32)

                    # Validate shape
                    if obs.shape != (n_agents, obs_dim):
                        raise ValueError(
                            f"Expected obs shape ({n_agents}, {obs_dim}), got {list(obs.shape)}"
                        )

                    actions = policy_fn(obs)             # (N, 2)
                    resp = {"actions": actions.tolist()}
                except Exception as e:
                    resp = {"error": str(e)}

                conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        conn.close()
        log.info("ros_bridge disconnected.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config_3a3g.yaml")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pcfg = cfg["policy"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(f"Loading policy: {pcfg['checkpoint_path']}")
    log.info(f"obs_dim={pcfg['obs_dim']}  n_agents={pcfg['n_agents']}  device={device}")

    from policy.loader import load_policy
    policy_fn = load_policy(
        checkpoint_path=pcfg["checkpoint_path"],
        obs_dim=pcfg["obs_dim"],
        n_agents=pcfg["n_agents"],
        num_cells=pcfg.get("num_cells", [256, 256]),
        action_dim=pcfg.get("action_dim", 2),
        u_range=pcfg.get("u_range", 0.5),
        device=device,
    )
    log.info("Policy ready.")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    log.info(f"Listening on {args.host}:{args.port}  (waiting for ros_bridge...)")

    try:
        while True:
            conn, addr = srv.accept()
            t = threading.Thread(
                target=handle_client,
                args=(conn, policy_fn, pcfg["obs_dim"], pcfg["n_agents"]),
                daemon=True,
            )
            t.start()
    except KeyboardInterrupt:
        log.info("Shutting down.")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
