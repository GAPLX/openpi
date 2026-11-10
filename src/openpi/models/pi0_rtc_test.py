import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from openpi.models import gemma as _gemma
from openpi.models import pi0_config
from openpi.models import rtc as _rtc
from openpi.models_pytorch import pi0_pytorch

ACTION_HORIZON = 16
PREFIX_LEN = 10
INFERENCE_DELAY = 4


def _make_config(*, pi05: bool) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=ACTION_HORIZON,
        pi05=pi05,
        dtype="float32",
        pytorch_compile_mode=None,
    )


def _make_prefix(
    config: pi0_config.Pi0Config, inference_delay: int = INFERENCE_DELAY
) -> tuple[np.ndarray, np.ndarray]:
    target = (
        np.random.default_rng(1)
        .standard_normal((PREFIX_LEN, config.action_dim))
        .astype(np.float32)
    )
    prefix, weights = _rtc.make_guidance_inputs(
        target,
        inference_delay=inference_delay,
        action_horizon=config.action_horizon,
        action_dim=config.action_dim,
        config=_rtc.RTCConfig(),
    )
    return prefix[None], weights[None]


def _prefix_error(actions: np.ndarray, prefix: np.ndarray) -> float:
    return float(
        np.abs(actions[0, :INFERENCE_DELAY] - prefix[0, :INFERENCE_DELAY]).mean()
    )


@pytest.fixture(scope="module", params=[False, True], ids=["pi0", "pi05"])
def jax_setup(request):
    config = _make_config(pi05=request.param)
    model = config.create(jax.random.key(0))
    graphdef, state = nnx.split(model)

    @jax.jit
    def sample(state, observation, noise, rtc_prefix=None, rtc_weights=None):
        model = nnx.merge(graphdef, state)
        return model.sample_actions(
            jax.random.key(0),
            observation,
            noise=noise,
            rtc_prefix=rtc_prefix,
            rtc_weights=rtc_weights,
        )

    observation = config.fake_obs()
    noise = jax.random.normal(
        jax.random.key(2), (1, config.action_horizon, config.action_dim)
    )
    return (
        config,
        lambda *args, **kwargs: sample(state, observation, noise, *args, **kwargs),
        sample,
    )


def test_jax_zero_weights_match_unguided(jax_setup):
    config, sample, _ = jax_setup
    unguided = np.asarray(sample())
    prefix, weights = _make_prefix(config)
    guided = np.asarray(sample(prefix, np.zeros_like(weights)))
    np.testing.assert_allclose(guided, unguided, atol=1e-5)


def test_jax_guidance_follows_prefix(jax_setup):
    config, sample, _ = jax_setup
    prefix, weights = _make_prefix(config)
    unguided = np.asarray(sample())
    guided = np.asarray(sample(prefix, weights))
    assert _prefix_error(guided, prefix) < 0.2 * _prefix_error(unguided, prefix)
    # Steps beyond the guided horizon are not pulled towards the (zero) padding of the prefix.
    assert not np.allclose(guided[0, PREFIX_LEN:], 0.0, atol=1e-2)


def test_jax_no_recompile_across_delays(jax_setup):
    config, sample, jitted = jax_setup
    sample(*_make_prefix(config))
    cache_size = jitted._cache_size()  # noqa: SLF001
    for delay in (0, 2, 7):
        sample(*_make_prefix(config, inference_delay=delay))
    assert jitted._cache_size() == cache_size  # noqa: SLF001


@pytest.fixture(scope="module", params=[False, True], ids=["pi0", "pi05"])
def torch_setup(request):
    config = _make_config(pi05=request.param)
    torch.manual_seed(0)
    model = pi0_pytorch.PI0Pytorch(config).eval()
    # PaliGemmaWithExpertModel hardcodes the image projection to the gemma_2b width, so resize it for the dummy LLM.
    projector = model.paligemma_with_expert.paligemma.model.multi_modal_projector
    projector.linear = torch.nn.Linear(
        projector.linear.in_features, _gemma.get_config("dummy").width
    )

    observation = jax.tree.map(
        lambda x: torch.from_numpy(np.array(x)), config.fake_obs()
    )
    # The PyTorch model takes channels-first images.
    observation = dataclasses.replace(
        observation,
        images={k: v.permute(0, 3, 1, 2) for k, v in observation.images.items()},
    )
    noise = torch.from_numpy(
        np.array(
            jax.random.normal(
                jax.random.key(2), (1, config.action_horizon, config.action_dim)
            )
        )
    )

    def sample(rtc_prefix=None, rtc_weights=None):
        return model.sample_actions(
            torch.device("cpu"),
            observation,
            noise=noise.clone(),
            rtc_prefix=None if rtc_prefix is None else torch.from_numpy(rtc_prefix),
            rtc_weights=None if rtc_weights is None else torch.from_numpy(rtc_weights),
        ).numpy()

    return config, sample


def test_torch_zero_weights_match_unguided(torch_setup):
    config, sample = torch_setup
    unguided = sample()
    prefix, weights = _make_prefix(config)
    np.testing.assert_allclose(
        sample(prefix, np.zeros_like(weights)), unguided, atol=1e-5
    )


def test_torch_guidance_follows_prefix(torch_setup):
    config, sample = torch_setup
    prefix, weights = _make_prefix(config)
    unguided = sample()
    guided = sample(prefix, weights)
    assert _prefix_error(guided, prefix) < 0.2 * _prefix_error(unguided, prefix)


def test_rtc_kwargs_must_be_paired():
    config = _make_config(pi05=False)
    model = nnx.eval_shape(config.create, jax.random.key(0))
    with pytest.raises(ValueError, match="provided together"):
        model.sample_actions(
            jax.random.key(0),
            config.fake_obs(),
            rtc_prefix=jnp.zeros((1, ACTION_HORIZON, 32)),
        )
