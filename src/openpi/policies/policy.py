from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import rtc as _rtc
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        rtc_config: _rtc.RTCConfig | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            rtc_config: If set, enables Real-Time Chunking. Requests may then contain an `rtc` entry, see `infer`.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._rtc_config = rtc_config
        shape_source = model.config if is_pytorch else model
        self._action_horizon = shape_source.action_horizon
        self._action_dim = shape_source.action_dim

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        """Infers an action chunk.

        If the policy was created with an RTC config, `obs` may contain an `rtc` entry with:
        - `prev_actions`: [k, action_dim] not-yet-executed actions of the previous chunk, in the same action space as
          the `actions` returned by this method. They are mapped to model space with the input transforms, i.e.
          relative to the current state for delta-action policies.
        - `inference_delay`: number of control steps that will be executed while this request is processed.
        """
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        rtc_request = inputs.pop("rtc", None)
        if rtc_request is not None and self._rtc_config is None:
            raise ValueError(
                "Received an RTC request, but the policy was created without an RTC config."
            )

        prev_actions = None
        if rtc_request is not None:
            prev_actions = np.array(rtc_request["prev_actions"], dtype=np.float32)
            if prev_actions.ndim != 2:
                raise ValueError(
                    f"Expected rtc.prev_actions of shape [k, action_dim], got {prev_actions.shape}"
                )
            if len(prev_actions) > 0:
                inputs["actions"] = prev_actions.copy()

        inputs = self._input_transform(inputs)

        rtc_kwargs = {}
        if self._rtc_config is not None:
            model_prev_actions = (
                inputs.pop("actions", None) if rtc_request is not None else None
            )
            mask_dims = (
                self._rtc_config.mask_padded_action_dims and prev_actions is not None
            )
            prefix, weights = _rtc.make_guidance_inputs(
                model_prev_actions,
                inference_delay=int(rtc_request["inference_delay"])
                if rtc_request is not None
                else 0,
                action_horizon=self._action_horizon,
                action_dim=self._action_dim,
                config=self._rtc_config,
                num_guided_action_dims=prev_actions.shape[-1] if mask_dims else None,
            )
            rtc_kwargs = {
                "rtc_prefix": prefix,
                "rtc_weights": weights,
                "rtc_max_guidance_weight": self._rtc_config.max_guidance_weight,
            }

        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            for key in ("rtc_prefix", "rtc_weights"):
                if key in rtc_kwargs:
                    rtc_kwargs[key] = jnp.asarray(rtc_kwargs[key])[np.newaxis, ...]
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(
                lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[
                    None, ...
                ],
                inputs,
            )
            for key in ("rtc_prefix", "rtc_weights"):
                if key in rtc_kwargs:
                    rtc_kwargs[key] = torch.from_numpy(rtc_kwargs[key]).to(
                        self._pytorch_device
                    )[None, ...]
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        sample_kwargs.update(rtc_kwargs)
        if noise is not None:
            noise = (
                torch.from_numpy(noise).to(self._pytorch_device)
                if self._is_pytorch_model
                else jnp.asarray(noise)
            )

            if (
                noise.ndim == 2
            ):  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(
                sample_rng_or_pytorch_device, observation, **sample_kwargs
            ),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(
                lambda x: np.asarray(x[0, ...].detach().cpu()), outputs
            )
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
