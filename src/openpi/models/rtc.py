"""Real-Time Chunking (RTC) for flow-matching action policies (pi0 / pi05).

While the robot is still executing the previous action chunk, the next chunk is generated as an inpainting problem: its
first steps are guided towards the not-yet-executed actions of the previous chunk (the "prefix"), so that consecutive
chunks join smoothly.

References:
- Black et al., "Real-Time Execution of Action Chunking Flow Policies", arXiv:2506.07339.
- LeRobot's `lerobot.policies.rtc.modeling_rtc.RTCProcessor`, which this module is numerically aligned with.

Conventions follow openpi's `sample_actions`: time goes from 1 (noise) to 0 (actions), and each Euler step is
`x_t + dt * v_t` with `dt < 0`. Prefixes and weights are expressed in model action space (normalized, padded to the
model action dim).

The guidance correction uses the identity Jacobian approximation `d x1 / d x_t ~= I`, which is also what LeRobot's
implementation effectively computes. It avoids backpropagating through the action expert.
"""

import dataclasses
import enum
import math

import jax.numpy as jnp
import numpy as np
import torch


class PrefixSchedule(enum.Enum):
    """How the guidance weight decays over the steps of the new chunk."""

    # Only the first `inference_delay` steps are guided.
    ZEROS = "zeros"
    # The first `execution_horizon` steps are fully guided.
    ONES = "ones"
    # Fully guided for `inference_delay` steps, then linear decay to zero at `execution_horizon`.
    LINEAR = "linear"
    # Like LINEAR, but with an exponential decay.
    EXP = "exp"


@dataclasses.dataclass(frozen=True)
class RTCConfig:
    # Number of steps of the new chunk that are guided towards the prefix. If None, all prefix steps are guided, which
    # corresponds to the soft-masking horizon of the RTC paper. Always clipped to the prefix length.
    execution_horizon: int | None = None
    # Upper bound of the guidance weight (beta in the RTC paper).
    max_guidance_weight: float = 10.0
    # How the guidance decays from the first `inference_delay` steps to `execution_horizon`.
    prefix_schedule: PrefixSchedule = PrefixSchedule.LINEAR
    # If true, only the action dims sent by the client are guided, not the zero padding up to the model action dim.
    mask_padded_action_dims: bool = True

    def __post_init__(self):
        if self.max_guidance_weight <= 0:
            raise ValueError(f"max_guidance_weight must be positive, got {self.max_guidance_weight}")
        if self.execution_horizon is not None and self.execution_horizon < 0:
            raise ValueError(f"execution_horizon must be non-negative, got {self.execution_horizon}")


def get_prefix_weights(
    inference_delay: int, execution_horizon: int, action_horizon: int, schedule: PrefixSchedule
) -> np.ndarray:
    """Per-step guidance weights of shape [action_horizon]."""
    start = min(inference_delay, execution_horizon)
    end = execution_horizon
    total = action_horizon

    if schedule == PrefixSchedule.ZEROS:
        weights = np.zeros(total)
        weights[:start] = 1.0
    elif schedule == PrefixSchedule.ONES:
        weights = np.ones(total)
        weights[end:] = 0.0
    elif schedule in (PrefixSchedule.LINEAR, PrefixSchedule.EXP):
        decay = _linear_decay(start, end, total)
        if schedule == PrefixSchedule.EXP:
            decay = decay * np.expm1(decay) / (math.e - 1)
        weights = np.concatenate([np.ones(min(start, total)), decay, np.zeros(max(total - end, 0))])
    else:
        raise ValueError(f"Unknown prefix schedule: {schedule}")

    return weights.astype(np.float32)


def _linear_decay(start: int, end: int, total: int) -> np.ndarray:
    num_steps = total - max(total - end, 0) - start
    if end <= start or num_steps <= 0:
        return np.zeros(0)
    return np.linspace(1, 0, num_steps + 2)[1:-1]


def make_guidance_inputs(
    prefix: np.ndarray | None,
    *,
    inference_delay: int,
    action_horizon: int,
    action_dim: int,
    config: RTCConfig,
    num_guided_action_dims: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds the fixed-shape `rtc_prefix` and `rtc_weights` arrays consumed by `sample_actions`.

    Args:
        prefix: Not-yet-executed actions of the previous chunk in model action space, shape [k, action_dim]. Steps
            beyond `action_horizon` are dropped. None or k == 0 means no guidance.
        inference_delay: Number of control steps that elapse while the new chunk is generated.
        action_horizon: Model action horizon.
        action_dim: Model action dim.
        config: RTC config.
        num_guided_action_dims: If set, only the first dims are guided.

    Returns:
        prefix and weights, both float32 of shape [action_horizon, action_dim]. With all-zero weights, guided sampling
        is identical to unguided sampling, so the same compiled function serves both cases.
    """
    if inference_delay < 0:
        raise ValueError(f"inference_delay must be non-negative, got {inference_delay}")

    out_prefix = np.zeros((action_horizon, action_dim), dtype=np.float32)
    out_weights = np.zeros((action_horizon, action_dim), dtype=np.float32)
    if prefix is None or len(prefix) == 0:
        return out_prefix, out_weights

    prefix = np.asarray(prefix)
    if prefix.ndim != 2 or prefix.shape[-1] != action_dim:
        raise ValueError(f"Expected prefix of shape [k, {action_dim}], got {prefix.shape}")

    prefix_len = min(len(prefix), action_horizon)
    out_prefix[:prefix_len] = prefix[:prefix_len]

    execution_horizon = config.execution_horizon if config.execution_horizon is not None else prefix_len
    execution_horizon = min(execution_horizon, prefix_len)
    step_weights = get_prefix_weights(inference_delay, execution_horizon, action_horizon, config.prefix_schedule)

    out_weights[:] = step_weights[:, None]
    if num_guided_action_dims is not None:
        out_weights[:, num_guided_action_dims:] = 0.0
    return out_prefix, out_weights


def guidance_weight_jax(time, max_guidance_weight):
    tau = 1.0 - jnp.asarray(time, dtype=jnp.float32)
    one_minus_tau_sq = (1.0 - tau) ** 2
    inv_r2 = (one_minus_tau_sq + tau**2) / one_minus_tau_sq
    c = jnp.nan_to_num((1.0 - tau) / tau, posinf=max_guidance_weight)
    weight = jnp.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
    return jnp.minimum(weight, max_guidance_weight)


def guide_velocity_jax(x_t, v_t, time, prefix, weights, max_guidance_weight):
    """Returns the RTC-guided velocity. All arrays have shape [b, action_horizon, action_dim]."""
    x1 = x_t - time * v_t
    err = (prefix - x1) * weights
    return v_t - guidance_weight_jax(time, max_guidance_weight) * err


def guidance_weight_torch(time, max_guidance_weight: float):
    tau = 1.0 - torch.as_tensor(time, dtype=torch.float32)
    one_minus_tau_sq = (1.0 - tau) ** 2
    inv_r2 = (one_minus_tau_sq + tau**2) / one_minus_tau_sq
    c = torch.nan_to_num((1.0 - tau) / tau, posinf=max_guidance_weight)
    weight = torch.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
    return torch.clamp(weight, max=max_guidance_weight)


def guide_velocity_torch(x_t, v_t, time, prefix, weights, max_guidance_weight: float):
    """Returns the RTC-guided velocity. All tensors have shape [b, action_horizon, action_dim]."""
    x1 = x_t - time * v_t
    err = (prefix - x1) * weights
    return v_t - guidance_weight_torch(time, max_guidance_weight) * err
