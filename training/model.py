"""Actor-Critic MLP for dogfight PPO training."""

import torch
import torch.nn as nn
import numpy as np

OBS_SIZE = 46
ACTION_SIZE = 3


def orthogonal_init(module, gain=np.sqrt(2)):
    """Apply orthogonal initialization to a module's weight and zero its bias."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ActorCritic(nn.Module):
    """Separate-backbone actor-critic network.

    Actor outputs:
        - cont_head: 2 raw floats (yaw_mean, throttle_mean) — NO squashing
        - shoot_head: 1 logit (Bernoulli)

    Critic outputs:
        - value: scalar

    Raw means are stored in the buffer so log_prob is consistent between
    collection and PPO re-evaluation. Clamping happens only before env.step().
    """

    def __init__(self, obs_dim: int = OBS_SIZE, hidden: int = 256):
        super().__init__()
        self.backbone_actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.backbone_critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Continuous head: 2 outputs (yaw_mean, throttle_mean) — raw, unbounded
        self.cont_head = nn.Linear(hidden, 2)
        # Discrete head (shoot)
        self.shoot_head = nn.Linear(hidden, 1)
        # Critic
        self.value_head = nn.Linear(hidden, 1)
        # Learned log_std for continuous actions (yaw, throttle)
        self.log_std = nn.Parameter(torch.full((2,), -1.0))

        # Orthogonal init
        for module in self.backbone_actor.modules():
            orthogonal_init(module)
        for module in self.backbone_critic.modules():
            orthogonal_init(module)
        orthogonal_init(self.cont_head, gain=0.01)
        orthogonal_init(self.shoot_head, gain=0.01)
        orthogonal_init(self.value_head, gain=1.0)

    def forward(self, obs: torch.Tensor):
        """Returns (cont_mean, shoot_logit, value, log_std).

        cont_mean is [batch, 2] with raw (unbounded) yaw and throttle means.
        """
        h_actor = self.backbone_actor(obs)
        h_critic = self.backbone_critic(obs)
        cont_mean = self.cont_head(h_actor)          # [batch, 2]
        shoot_logit = self.shoot_head(h_actor).squeeze(-1)
        value = self.value_head(h_critic).squeeze(-1)
        return cont_mean, shoot_logit, value, self.log_std

    def get_action_and_value(self, obs: torch.Tensor, action=None):
        """Sample or evaluate actions. Returns (action, log_prob, entropy, value).

        Actions are RAW (unbounded). Caller must clamp before sending to env.
        """
        cont_mean, shoot_logit, value, log_std = self(obs)
        log_std = log_std.clamp(-2.0, 0.0)
        std = log_std.exp()

        # Continuous distributions over raw (unbounded) values
        yaw_dist = torch.distributions.Normal(cont_mean[:, 0], std[0])
        throttle_dist = torch.distributions.Normal(cont_mean[:, 1], std[1])
        # Discrete distribution
        shoot_dist = torch.distributions.Bernoulli(logits=shoot_logit)

        if action is None:
            yaw = yaw_dist.sample()
            throttle = throttle_dist.sample()
            shoot = shoot_dist.sample()
        else:
            yaw = action[:, 0]
            throttle = action[:, 1]
            shoot = action[:, 2]

        # Log probs (computed on raw values — consistent between collect and update)
        log_prob = (
            yaw_dist.log_prob(yaw)
            + throttle_dist.log_prob(throttle)
            + shoot_dist.log_prob(shoot)
        )

        # Entropy
        entropy = (
            yaw_dist.entropy() + throttle_dist.entropy() + shoot_dist.entropy()
        )

        # Return RAW actions — no clamping here
        action_out = torch.stack([yaw, throttle, shoot], dim=-1)

        return action_out, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.backbone_critic(obs)
        return self.value_head(h).squeeze(-1)

    def get_deterministic_action(self, obs: torch.Tensor) -> np.ndarray:
        """For evaluation: use means clamped to valid range (matching training), threshold shoot at 0.5."""
        with torch.no_grad():
            cont_mean, shoot_logit, _, _ = self(obs)
            yaw = cont_mean[:, 0].clamp(-1.0, 1.0)
            throttle = cont_mean[:, 1].clamp(0.0, 1.0)
            shoot = (torch.sigmoid(shoot_logit) > 0.5).float()
            action = torch.stack([yaw, throttle, shoot], dim=-1)
        return action.cpu().numpy()


class ActorOnly(nn.Module):
    """Actor-only network for ONNX export (no critic head).

    Uses clamp to match training behavior (NOT tanh/sigmoid).
    """

    def __init__(self, obs_dim: int = OBS_SIZE, hidden: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.cont_head = nn.Linear(hidden, 2)
        self.shoot_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.backbone(obs)
        cont = self.cont_head(h)                    # [batch, 2]
        yaw = cont[:, 0:1].clamp(-1.0, 1.0)        # [-1, 1]
        throttle = cont[:, 1:2].clamp(0.0, 1.0)    # [0, 1]
        shoot = self.shoot_head(h)                   # raw logit; sim uses > 0 threshold
        return torch.cat([yaw, throttle, shoot], dim=-1)

    @staticmethod
    def from_actor_critic(ac: ActorCritic) -> "ActorOnly":
        """Extract actor weights from a trained ActorCritic."""
        actor = ActorOnly()
        actor.backbone.load_state_dict(ac.backbone_actor.state_dict())
        actor.cont_head.load_state_dict(ac.cont_head.state_dict())
        actor.shoot_head.load_state_dict(ac.shoot_head.state_dict())
        return actor
