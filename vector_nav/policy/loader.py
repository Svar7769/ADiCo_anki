"""
policy/loader.py
────────────────
Loads a trained ADiCo / BenchMARL policy from a checkpoint .pt file.

Checkpoint format (confirmed by inspection of navigation_3a3g_seed0.pt):
  - Saved by BenchMARL/TorchRL as nested OrderedDict
  - Actor weights live under:
      collector.policy_state_dict.module.0.module.0.module.0.{shared_mlp,agent_mlps}...
  - Architecture: HetControlMlpEmpirical (ADiCo)
      shared_mlp  (share_params=True)  : obs_dim → [256→Tanh]×2 → action_dim*2  (loc+scale)
      agent_mlps  (share_params=False) : obs_dim → [256→Tanh]×2 → action_dim    (loc offset)
  - Forward (scaling_ratio=1.0 when desired_snd=-1.0):
      agent_loc_i = shared_loc_i + agent_mlps[i](obs_i)
      action_i    = tanh(agent_loc_i) * u_range

Usage
-----
  from policy.loader import load_policy
  policy_fn = load_policy("checkpoints/navigation_3a3g_seed0.pt",
                           obs_dim=6, n_agents=3)
  obs = torch.zeros(3, 6)
  actions = policy_fn(obs)   # Tensor[3, 2]
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Prefix in the BenchMARL checkpoint that leads to the actor module weights
_CKPT_ACTOR_PREFIX = (
    "collector.policy_state_dict."
    "module.0.module.0.module.0."
)


# ── Public entry point ────────────────────────────────────────────────────────

def load_policy(
    checkpoint_path: str,
    obs_dim: int,
    n_agents: int,
    num_cells: List[int] = None,
    action_dim: int = 2,
    u_range: float = 0.5,
    device: str = "cpu",
) -> Callable:
    """
    Load a trained ADiCo navigation policy for inference.

    Parameters
    ----------
    checkpoint_path : path to .pt checkpoint file
    obs_dim         : observation dimension per agent (6 / 8 depending on n_goals)
    n_agents        : number of agents (3 for all current checkpoints)
    num_cells       : MLP hidden layer sizes (default [256, 256])
    action_dim      : output action dimension (2 for navigation)
    u_range         : action bound — actions scaled by tanh(·) * u_range
    device          : "cpu" or "cuda"

    Returns
    -------
    Callable: f(obs: Tensor[N, obs_dim]) → actions: Tensor[N, action_dim]
    """
    num_cells = num_cells or [256, 256]
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint: {ckpt_path.name}  (obs_dim={obs_dim}, n_agents={n_agents})")

    raw = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # Flatten nested OrderedDict → {dot.separated.key: Tensor}
    flat = _flatten(raw)

    # Extract actor weights under the known BenchMARL prefix
    actor_weights = {
        k[len(_CKPT_ACTOR_PREFIX):]: v
        for k, v in flat.items()
        if k.startswith(_CKPT_ACTOR_PREFIX) and isinstance(v, torch.Tensor)
    }

    if not actor_weights:
        raise RuntimeError(
            f"No weights found under prefix '{_CKPT_ACTOR_PREFIX}'.\n"
            f"Top-level flat keys (first 10): {list(flat.keys())[:10]}"
        )

    logger.info(f"Extracted {len(actor_weights)} actor tensors.")

    # Build DiCoPolicy with matching key structure and load weights.
    # Filter to only keys the model expects — the checkpoint also stores
    # non-parameter scalars (e.g. desired_snd, estimated_snd) under the same prefix.
    policy = DiCoPolicy(obs_dim, n_agents, num_cells, action_dim)
    expected = set(policy.state_dict().keys())
    weights = {k: v for k, v in actor_weights.items() if k in expected}
    missing = expected - set(weights.keys())
    if missing:
        raise RuntimeError(f"Missing expected policy weights: {missing}")
    policy.load_state_dict(weights, strict=True)
    logger.info(f"Policy loaded. Parameters: {sum(p.numel() for p in policy.parameters()):,}")

    policy = policy.to(device).eval()
    _u_range = u_range

    def policy_fn(obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return policy(obs.to(device), _u_range)

    return policy_fn


# ── DiCoPolicy: mirrors HetControlMlpEmpirical key structure ─────────────────
#
# Checkpoint key structure (torchrl MultiAgentMLP vmap format):
#   shared_mlp.params.{0,2,4}.{weight,bias}   — single network, shapes [out,in]
#   agent_mlps.params.{0,2,4}.{weight,bias}   — batched, shapes [n_agents,out,in]
# Tanh layers at indices 1 and 3 have no parameters (not in checkpoint).


class _SharedMLP(nn.Module):
    """
    Shared network stored as self.params (ModuleList).
    Key structure: shared_mlp.params.{0,2,4}.{weight,bias}
    Indices 1,3 are Tanh — present in ModuleList but absent from checkpoint.
    """

    def __init__(self, obs_dim: int, num_cells: List[int], out_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        in_d = obs_dim
        for hidden in num_cells:
            layers += [nn.Linear(in_d, hidden), nn.Tanh()]
            in_d = hidden
        layers.append(nn.Linear(in_d, out_dim))
        self.params = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.params:
            x = layer(x)
        return x


class _BatchedLinear(nn.Module):
    """
    Linear layer with batched weights [n_agents, out_features, in_features].
    Matches the torchrl vmap-style parameter format stored in the checkpoint.
    """

    def __init__(self, n_agents: int, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_agents, out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(n_agents, out_features))

    def forward_agent(self, x: torch.Tensor, i: int) -> torch.Tensor:
        return x @ self.weight[i].T + self.bias[i]


class _AgentMLPs(nn.Module):
    """
    Per-agent networks stored as self.params (ModuleList of _BatchedLinear + Tanh).
    Key structure: agent_mlps.params.{0,2,4}.{weight,bias}
    Weights have shape [n_agents, out, in] — extracted per-agent at inference time.
    """

    def __init__(self, obs_dim: int, num_cells: List[int], out_dim: int, n_agents: int):
        super().__init__()
        layers: List[nn.Module] = []
        in_d = obs_dim
        for hidden in num_cells:
            layers += [_BatchedLinear(n_agents, in_d, hidden), nn.Tanh()]
            in_d = hidden
        layers.append(_BatchedLinear(n_agents, in_d, out_dim))
        self.params = nn.ModuleList(layers)

    def forward_agent(self, x: torch.Tensor, i: int) -> torch.Tensor:
        for layer in self.params:
            if isinstance(layer, _BatchedLinear):
                x = layer.forward_agent(x, i)
            else:          # Tanh
                x = torch.tanh(x)
        return x


class DiCoPolicy(nn.Module):
    """
    Inference-only reconstruction of HetControlMlpEmpirical.

    Attribute names and parameter shapes match the BenchMARL/torchrl checkpoint
    exactly so that load_state_dict(strict=True) works without key remapping.

    Checkpoint keys (under collector.policy_state_dict.module.0.module.0.module.0.):
      shared_mlp.params.0.weight  [256, obs_dim]
      shared_mlp.params.2.weight  [256, 256]
      shared_mlp.params.4.weight  [action_dim*2, 256]
      agent_mlps.params.0.weight  [n_agents, 256, obs_dim]
      agent_mlps.params.2.weight  [n_agents, 256, 256]
      agent_mlps.params.4.weight  [n_agents, action_dim, 256]
    """

    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        num_cells: List[int],
        action_dim: int,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.shared_mlp = _SharedMLP(obs_dim, num_cells, action_dim * 2)
        self.agent_mlps = _AgentMLPs(obs_dim, num_cells, action_dim, n_agents)

    def forward(self, obs: torch.Tensor, u_range: float = 0.5) -> torch.Tensor:
        """
        obs    : Tensor[n_agents, obs_dim]
        returns: Tensor[n_agents, action_dim]
        """
        actions = []
        for i, obs_i in enumerate(obs):
            shared_out = self.shared_mlp(obs_i)             # [action_dim * 2]
            shared_loc = shared_out[: self.action_dim]      # [action_dim]
            agent_out  = self.agent_mlps.forward_agent(obs_i, i)  # [action_dim]
            agent_loc  = shared_loc + agent_out
            actions.append(torch.tanh(agent_loc) * u_range)
        return torch.stack(actions)                         # [n_agents, action_dim]


# ── Checkpoint key helpers ────────────────────────────────────────────────────

def _flatten(obj, prefix: str = "") -> dict:
    """Recursively flatten a nested dict/OrderedDict into {dot.key: value}."""
    result = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            full = f"{prefix}.{k}" if prefix else str(k)
            result.update(_flatten(v, full))
    elif hasattr(obj, "items"):
        # TensorDict or similar mapping
        for k, v in obj.items():
            full = f"{prefix}.{k}" if prefix else str(k)
            result.update(_flatten(v, full))
    else:
        result[prefix] = obj
    return result
