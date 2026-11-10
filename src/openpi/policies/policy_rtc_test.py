import types

import flax.nnx as nnx
import numpy as np
import pytest
import torch

from openpi import transforms as _transforms
from openpi.models import rtc as _rtc
from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.shared import download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


class _TorchEchoModel(torch.nn.Module):
    """Returns the RTC prefix as the action chunk and records the sampling kwargs."""

    def __init__(self, action_horizon: int, action_dim: int):
        super().__init__()
        self.config = types.SimpleNamespace(action_horizon=action_horizon, action_dim=action_dim)
        self.calls: list[dict] = []

    def sample_actions(self, device, observation, **kwargs):
        self.calls.append(kwargs)
        return kwargs["rtc_prefix"]


class _JaxEchoModel(nnx.Module):
    """Returns the RTC prefix as the action chunk."""

    def __init__(self, action_horizon: int, action_dim: int):
        self.action_horizon = action_horizon
        self.action_dim = action_dim

    def sample_actions(self, rng, observation, *, rtc_prefix, rtc_weights, rtc_max_guidance_weight):
        del rng, observation, rtc_weights, rtc_max_guidance_weight
        return rtc_prefix


def _make_policy(backend: str, action_horizon: int, action_dim: int, **kwargs) -> _policy.Policy:
    if backend == "torch":
        return _policy.Policy(_TorchEchoModel(action_horizon, action_dim), is_pytorch=True, **kwargs)
    return _policy.Policy(_JaxEchoModel(action_horizon, action_dim), **kwargs)


def _model_space_obs(action_dim: int) -> dict:
    return {
        "image": {"base_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8)},
        "image_mask": {"base_0_rgb": np.True_},
        "state": np.zeros(action_dim, dtype=np.float32),
    }


def test_rtc_request_requires_rtc_config():
    policy = _make_policy("torch", 10, 8)
    obs = {**_model_space_obs(8), "rtc": {"prev_actions": np.zeros((3, 8)), "inference_delay": 1}}
    with pytest.raises(ValueError, match="without an RTC config"):
        policy.infer(obs)


def test_request_without_rtc_entry_is_unguided():
    policy = _make_policy("torch", 10, 8, rtc_config=_rtc.RTCConfig())
    policy.infer(_model_space_obs(8))
    (call,) = policy._model.calls  # noqa: SLF001
    assert not call["rtc_prefix"].any()
    assert not call["rtc_weights"].any()


def test_guidance_inputs_are_built_from_request():
    action_horizon, action_dim, robot_dim, delay = 10, 8, 3, 2
    policy = _make_policy(
        "torch",
        action_horizon,
        action_dim,
        transforms=[_transforms.PadStatesAndActions(action_dim)],
        rtc_config=_rtc.RTCConfig(prefix_schedule=_rtc.PrefixSchedule.LINEAR),
    )
    prev_actions = np.arange(6 * robot_dim, dtype=np.float32).reshape(6, robot_dim)
    obs = {
        **_model_space_obs(robot_dim),
        "rtc": {"prev_actions": prev_actions, "inference_delay": delay},
    }
    policy.infer(obs)
    assert "rtc" in obs, "infer must not modify its input"

    (call,) = policy._model.calls  # noqa: SLF001
    prefix, weights = call["rtc_prefix"][0].numpy(), call["rtc_weights"][0].numpy()
    assert prefix.shape == weights.shape == (action_horizon, action_dim)
    np.testing.assert_array_equal(prefix[:6, :robot_dim], prev_actions)
    np.testing.assert_array_equal(prefix[6:], 0.0)
    np.testing.assert_array_equal(weights[:delay, :robot_dim], 1.0)
    np.testing.assert_array_equal(weights[:, robot_dim:], 0.0)
    np.testing.assert_array_equal(weights[6:], 0.0)
    assert call["rtc_max_guidance_weight"] == _rtc.RTCConfig().max_guidance_weight


@pytest.fixture(scope="module")
def aloha_towel_transforms():
    """Input/output transforms of the pi0_aloha_towel policy: Aloha<->pi joint conversion, delta actions, normalization.

    Needs the norm stats of gs://openpi-assets/checkpoints/pi0_aloha_towel/assets (a few KB) and the PaliGemma tokenizer.
    """
    train_config = _config.get_config("pi0_aloha_towel")
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    assets_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_aloha_towel/assets")
    norm_stats = _checkpoints.load_norm_stats(assets_dir, data_config.asset_id)
    inputs = [
        _transforms.InjectDefaultPrompt(None),
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    outputs = [
        *data_config.model_transforms.outputs,
        _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
    ]
    assert any(isinstance(t, _transforms.DeltaActions) for t in inputs), "expected a delta-action config"
    return train_config.model, inputs, outputs


def _aloha_example(rng: np.random.Generator) -> dict:
    example = aloha_policy.make_aloha_example()
    state = rng.uniform(-1.0, 1.0, 14)
    state[[6, 13]] = rng.uniform(0.2, 0.8, 2)
    example["state"] = state
    return example


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_aloha_prev_actions_round_trip(aloha_towel_transforms, backend: str):
    model_config, inputs, outputs = aloha_towel_transforms
    policy = _make_policy(
        backend,
        model_config.action_horizon,
        model_config.action_dim,
        transforms=inputs,
        output_transforms=outputs,
        rtc_config=_rtc.RTCConfig(),
    )
    rng = np.random.default_rng(0)
    prev_actions = rng.uniform(-1.0, 1.0, (20, 14)).astype(np.float32)
    prev_actions[:, [6, 13]] = rng.uniform(0.2, 0.8, (20, 2))

    # The echo model returns the model-space prefix, so the output transforms must recover the absolute prev_actions,
    # for any current state.
    for _ in range(2):
        example = _aloha_example(rng)
        example["rtc"] = {"prev_actions": prev_actions, "inference_delay": 5}
        result = policy.infer(example)
        assert result["actions"].shape == (model_config.action_horizon, 14)
        np.testing.assert_allclose(result["actions"][:20], prev_actions, atol=1e-4)


def test_aloha_prefix_is_relative_to_current_state(aloha_towel_transforms):
    model_config, inputs, outputs = aloha_towel_transforms
    policy = _make_policy(
        "torch",
        model_config.action_horizon,
        model_config.action_dim,
        transforms=inputs,
        output_transforms=outputs,
        rtc_config=_rtc.RTCConfig(),
    )
    rng = np.random.default_rng(1)
    prev_actions = rng.uniform(-1.0, 1.0, (8, 14)).astype(np.float32)
    for _ in range(2):
        example = _aloha_example(rng)
        example["rtc"] = {"prev_actions": prev_actions, "inference_delay": 2}
        policy.infer(example)

    first, second = (call["rtc_prefix"][0, :8].numpy() for call in policy._model.calls)  # noqa: SLF001
    joint_dims = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    gripper_dims = [6, 13]
    # Joints are delta actions, so their model-space prefix depends on the state. Grippers are absolute.
    assert not np.allclose(first[:, joint_dims], second[:, joint_dims])
    np.testing.assert_allclose(first[:, gripper_dims], second[:, gripper_dims], atol=1e-6)
    np.testing.assert_array_equal(first[:, 14:], 0.0)
