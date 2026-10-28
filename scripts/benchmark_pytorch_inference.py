"""Steady-state latency benchmark for a converted PyTorch checkpoint.

Weight loading and the first torch.compile / autotune calls are warmup and are
not included in the reported statistics. Each measured sample is one
CUDA-synchronized ``policy.infer`` on a freshly drawn xlerobot observation.
Observation generation itself is outside the timer.
"""

import dataclasses
import json
import logging
import os
import pathlib
import time

# The policy stack imports JAX. Without this, JAX preallocates most of the GPU
# before PyTorch loads the checkpoint.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import numpy as np
import torch
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.policies import xlerobot_policy
from openpi.training import config as _config

logger = logging.getLogger(__name__)

# ``XlerobotOutputs`` keeps 14 action dims, then ``_encode_action`` drops two
# padded joints, so the policy returns (action_horizon, 12).
_XLEROBOT_ACTION_DIM = 12
# Default of ``PI0Pytorch.sample_actions`` when ``sample_kwargs`` is omitted.
_DEFAULT_DIFFUSION_STEPS = 10
_COMPILE_MODES = (
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)


@dataclasses.dataclass
class Args:
    """Benchmark steady-state PyTorch policy inference latency.

    Startup, weight loading, and torch.compile are excluded. Timing wraps
    ``policy.infer`` with CUDA synchronization and starts only after warmup
    latency has settled.
    """

    # Directory of a converted PyTorch checkpoint. Must contain model.safetensors.
    checkpoint_dir: pathlib.Path
    # Training config name. Must match the checkpoint (for example, pi0_xlerobot_lora).
    config: str = "pi0_xlerobot_lora"
    # CUDA device. CPU is rejected: bfloat16 weights would be cast to float32.
    device: str = "cuda"
    # Number of timed inferences after warmup. Startup is not included.
    num_steps: int = 50
    # Minimum discarded warmup inferences before the stability check can pass.
    min_warmup: int = 5
    # Stop warming up and measure anyway after this many discarded inferences.
    max_warmup: int = 30
    # Trailing warmup samples used to decide that latency has settled.
    stability_window: int = 5
    # Warmup ends when the trailing window's sample std / mean is at or below this.
    stability_cv: float = 0.05
    # Seed for ``make_xlerobot_example``. Pixel noise does not change the compute graph.
    seed: int = 0
    # Flow-matching steps passed to ``sample_actions``. Omit to use the model default (10).
    num_diffusion_steps: int | None = None
    # Override ``Pi0Config.pytorch_compile_mode``. Omit to keep the config value (max-autotune).
    compile_mode: str | None = None
    # Optional JSON path for the environment info, summary, and raw sample times.
    output: pathlib.Path | None = None


def main(args: Args) -> None:
    _validate_args(args)
    device = _require_cuda(args.device)
    # Input shapes are fixed, so the cuDNN autotuner should pick once during warmup.
    torch.backends.cudnn.benchmark = True
    np.random.seed(args.seed)

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    weight_path = checkpoint_dir / "model.safetensors"
    if not weight_path.is_file():
        raise SystemExit(
            f"PyTorch checkpoint required: {weight_path} not found. JAX checkpoints are not supported."
        )

    train_config = _load_config(args)
    diffusion_steps = args.num_diffusion_steps if args.num_diffusion_steps is not None else _DEFAULT_DIFFUSION_STEPS
    sample_kwargs = None if args.num_diffusion_steps is None else {"num_steps": args.num_diffusion_steps}
    expected_shape = (train_config.model.action_horizon, _XLEROBOT_ACTION_DIM)
    norm_stats = _norm_stats(train_config, checkpoint_dir)

    logger.info("Loading PyTorch policy from %s", checkpoint_dir)
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        sample_kwargs=sample_kwargs,
        norm_stats=norm_stats,
        pytorch_device=device,
    )

    warmup_ms, stable = _warmup(policy, device, args, expected_shape)
    cold_start_ms = warmup_ms[0]
    window_cv = _window_cv(warmup_ms, args.stability_window)
    if stable:
        logger.info(
            "Warmup settled after %d inferences (window cv=%.4f, threshold %.4f).",
            len(warmup_ms),
            window_cv,
            args.stability_cv,
        )
    else:
        logger.warning(
            "Warmup reached max_warmup=%d without settling (window cv=%s, threshold %.4f). Measuring anyway.",
            args.max_warmup,
            "n/a" if window_cv is None else f"{window_cv:.4f}",
            args.stability_cv,
        )

    samples_ms, unsynced_model_ms = _measure(policy, device, args.num_steps, expected_shape)
    metrics = _metrics(samples_ms, train_config.model.action_horizon)
    report = _report(
        args=args,
        device=device,
        checkpoint_dir=checkpoint_dir,
        train_config=train_config,
        diffusion_steps=diffusion_steps,
        warmup_ms=warmup_ms,
        stable=stable,
        window_cv=window_cv,
        cold_start_ms=cold_start_ms,
        samples_ms=samples_ms,
        unsynced_model_ms=unsynced_model_ms,
        metrics=metrics,
    )
    _print_report(report, metrics)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote %s", args.output)


def _validate_args(args: Args) -> None:
    if args.num_steps < 2:
        raise SystemExit("num_steps must be >= 2 so the sample standard deviation is defined.")
    if args.min_warmup < 1:
        raise SystemExit("min_warmup must be >= 1. The first warmup call absorbs torch.compile.")
    if args.stability_window < 2:
        raise SystemExit("stability_window must be >= 2 so the sample standard deviation is defined.")
    if args.max_warmup < args.min_warmup:
        raise SystemExit("max_warmup must be >= min_warmup.")
    if args.max_warmup < args.stability_window:
        raise SystemExit("max_warmup must be >= stability_window, otherwise the run can never be marked stable.")
    if not args.stability_cv > 0:
        raise SystemExit("stability_cv must be positive.")
    if args.num_diffusion_steps is not None and args.num_diffusion_steps < 1:
        raise SystemExit("num_diffusion_steps must be >= 1.")
    if args.compile_mode is not None and args.compile_mode not in _COMPILE_MODES:
        raise SystemExit(f"compile_mode must be one of {', '.join(_COMPILE_MODES)}.")


def _require_cuda(device: str) -> str:
    """Return a ``cuda:N`` device and make it current.

    ``torch.cuda.set_device`` rejects the bare string ``"cuda"`` because it has no index.
    """
    if not device.startswith("cuda"):
        raise SystemExit(
            f"This benchmark only measures CUDA inference (got device={device!r}). "
            "CPU timings are not comparable: bfloat16 weights are cast to float32."
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Refusing to fall back to CPU.")
    if device == "cuda":
        index = torch.cuda.current_device()
    else:
        index_str = device.removeprefix("cuda:")
        if not index_str.isdigit():
            raise SystemExit(f"Unrecognized CUDA device {device!r}. Use 'cuda' or 'cuda:N'.")
        index = int(index_str)
        if index >= torch.cuda.device_count():
            raise SystemExit(
                f"CUDA device {device!r} is out of range (visible devices: {torch.cuda.device_count()})."
            )
    torch.cuda.set_device(index)
    return f"cuda:{index}"


def _norm_stats(train_config: _config.TrainConfig, checkpoint_dir: pathlib.Path):
    """Norm stats used by ``create_trained_policy``.

    The policy loader reads ``<checkpoint>/assets/<asset_id>/norm_stats.json``.
    Converted checkpoints sometimes omit that copy; the same file then lives
    under the training config assets directory and is passed in explicitly.
    """
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise SystemExit("Asset id is required to load norm stats.")
    checkpoint_norm = checkpoint_dir / "assets" / data_config.asset_id / "norm_stats.json"
    if checkpoint_norm.is_file():
        # ``None`` tells create_trained_policy to load the checkpoint copy itself.
        return None
    if data_config.norm_stats is None:
        raise SystemExit(
            f"Norm stats not found at {checkpoint_norm} or under {train_config.assets_dirs / data_config.asset_id}."
        )
    logger.info(
        "Checkpoint has no %s; using norm stats from %s.",
        checkpoint_norm,
        train_config.assets_dirs / data_config.asset_id,
    )
    return data_config.norm_stats


def _load_config(args: Args) -> _config.TrainConfig:
    train_config = _config.get_config(args.config)
    if args.compile_mode is None:
        return train_config
    if not hasattr(train_config.model, "pytorch_compile_mode"):
        raise SystemExit(f"Config {args.config!r} has no pytorch_compile_mode to override.")
    return dataclasses.replace(
        train_config,
        model=dataclasses.replace(train_config.model, pytorch_compile_mode=args.compile_mode),
    )


def _sync(device: str) -> None:
    torch.cuda.synchronize(device)


def _time_infer(policy, device: str) -> tuple[float, float, np.ndarray]:
    """Time one inference.

    The random observation is built first so ``np.random`` is not on the clock.
    Synchronization before the start keeps leftover GPU work out of the sample;
    synchronization after the end waits for ``sample_actions``, which launches
    asynchronously. ``policy_timing['infer_ms']`` does not do this, so it is
    returned only as a diagnostic.
    """
    example = xlerobot_policy.make_xlerobot_example()
    _sync(device)
    start = time.perf_counter()
    result = policy.infer(example)
    _sync(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    unsynced_ms = float(result["policy_timing"]["infer_ms"])
    return elapsed_ms, unsynced_ms, np.asarray(result["actions"])


def _check_actions(actions: np.ndarray, expected_shape: tuple[int, int]) -> None:
    if actions.shape != expected_shape or actions.dtype.kind not in "fc" or not np.isfinite(actions).all():
        raise SystemExit(
            f"Inference returned invalid actions: shape={actions.shape} dtype={actions.dtype}, "
            f"expected shape={expected_shape} with finite values."
        )


def _coefficient_of_variation(samples: list[float]) -> float:
    values = np.asarray(samples, dtype=np.float64)
    mean = float(values.mean())
    if mean <= 0:
        return float("inf")
    return float(values.std(ddof=1) / mean)


def _window_cv(samples: list[float], window: int) -> float | None:
    if len(samples) < window:
        return None
    return _coefficient_of_variation(samples[-window:])


def _warmup(policy, device: str, args: Args, expected_shape: tuple[int, int]) -> tuple[list[float], bool]:
    samples: list[float] = []
    while len(samples) < args.max_warmup:
        step = len(samples) + 1
        if step == 1:
            logger.info(
                "Warmup 1/%d: first inference includes torch.compile and is excluded from steady-state stats.",
                args.max_warmup,
            )
        elapsed_ms, _, actions = _time_infer(policy, device)
        _check_actions(actions, expected_shape)
        samples.append(elapsed_ms)
        logger.info("Warmup %d/%d: %.1f ms (discarded)", step, args.max_warmup, elapsed_ms)
        if len(samples) >= args.min_warmup and len(samples) >= args.stability_window:
            cv = _coefficient_of_variation(samples[-args.stability_window :])
            logger.info("Warmup window cv=%.4f (threshold %.4f)", cv, args.stability_cv)
            if cv <= args.stability_cv:
                return samples, True
    return samples, False


def _measure(
    policy, device: str, num_steps: int, expected_shape: tuple[int, int]
) -> tuple[list[float], list[float]]:
    samples: list[float] = []
    unsynced: list[float] = []
    for step in range(1, num_steps + 1):
        elapsed_ms, unsynced_ms, actions = _time_infer(policy, device)
        _check_actions(actions, expected_shape)
        samples.append(elapsed_ms)
        unsynced.append(unsynced_ms)
        logger.info("Measure %d/%d: %.1f ms", step, num_steps, elapsed_ms)
    return samples, unsynced


def _metrics(samples: list[float], action_horizon: int) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    minimum = float(values.min())
    maximum = float(values.max())
    mean = float(values.mean())
    if mean <= 0:
        raise SystemExit("Measured mean latency is not positive, so throughput is undefined.")
    std = float(values.std(ddof=1))
    p25, median, p75, p90, p95, p99 = (float(x) for x in np.quantile(values, [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))
    throughput_hz = 1000.0 / mean
    return {
        "min_ms": minimum,
        "max_ms": maximum,
        "mean_ms": mean,
        "std_ms": std,
        "cv": std / mean,
        "median_ms": median,
        "p90_ms": p90,
        "p95_ms": p95,
        "p99_ms": p99,
        "iqr_ms": p75 - p25,
        "range_ms": maximum - minimum,
        "throughput_hz": throughput_hz,
        "actions_per_second": throughput_hz * action_horizon,
    }


def _report(
    *,
    args: Args,
    device: str,
    checkpoint_dir: pathlib.Path,
    train_config: _config.TrainConfig,
    diffusion_steps: int,
    warmup_ms: list[float],
    stable: bool,
    window_cv: float | None,
    cold_start_ms: float,
    samples_ms: list[float],
    unsynced_model_ms: list[float],
    metrics: dict[str, float],
) -> dict:
    device_index = torch.cuda.current_device()
    return {
        "config": args.config,
        "checkpoint_dir": str(checkpoint_dir),
        "device": device,
        "gpu": torch.cuda.get_device_name(device_index),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "compile_mode": train_config.model.pytorch_compile_mode,
        "num_diffusion_steps": diffusion_steps,
        "action_horizon": train_config.model.action_horizon,
        "seed": args.seed,
        "warmup_iters": len(warmup_ms),
        "min_warmup": args.min_warmup,
        "max_warmup": args.max_warmup,
        "stability_window": args.stability_window,
        "stability_cv": args.stability_cv,
        "warmup_window_cv": window_cv,
        "stable": stable,
        "cold_start_ms": cold_start_ms,
        "num_steps": args.num_steps,
        "metrics": metrics,
        # Diagnostic only. This is policy.py's timer around sample_actions, which
        # returns before the GPU kernels finish.
        "unsynced_model_ms": unsynced_model_ms,
        "samples_ms": samples_ms,
        "warmup_ms": warmup_ms,
    }


def _format_ms(milliseconds: float) -> str:
    if milliseconds >= 1000.0:
        return f"{milliseconds:.1f} ms ({milliseconds / 1000.0:.2f} s)"
    return f"{milliseconds:.3f} ms"


def _print_report(report: dict, metrics: dict[str, float]) -> None:
    window_cv = report["warmup_window_cv"]
    header = [
        ("config", report["config"]),
        ("checkpoint", report["checkpoint_dir"]),
        ("device", f"{report['gpu']} ({report['device']})"),
        ("torch / cuda", f"{report['torch_version']} / {report['cuda_version']}"),
        ("compile mode", str(report["compile_mode"])),
        ("diffusion steps", str(report["num_diffusion_steps"])),
        ("seed", str(report["seed"])),
        ("warmup iters", f"{report['warmup_iters']} / {report['max_warmup']}"),
        ("stable", "yes" if report["stable"] else "no"),
        ("warmup window cv", "n/a" if window_cv is None else f"{window_cv:.4f}"),
        ("cold start", _format_ms(report["cold_start_ms"])),
        ("measured steps", str(report["num_steps"])),
    ]
    rows = [
        ("min", _format_ms(metrics["min_ms"])),
        ("max", _format_ms(metrics["max_ms"])),
        ("mean", _format_ms(metrics["mean_ms"])),
        ("std", _format_ms(metrics["std_ms"])),
        ("cv", f"{metrics['cv']:.4f}"),
        ("median", _format_ms(metrics["median_ms"])),
        ("p90", _format_ms(metrics["p90_ms"])),
        ("p95", _format_ms(metrics["p95_ms"])),
        ("p99", _format_ms(metrics["p99_ms"])),
        ("iqr", _format_ms(metrics["iqr_ms"])),
        ("range", _format_ms(metrics["range_ms"])),
        ("throughput", f"{metrics['throughput_hz']:.3f} infer/s"),
        ("actions/s", f"{metrics['actions_per_second']:.3f}"),
    ]
    label_width = max(len(label) for label, _ in (*header, *rows))
    print()
    print("PyTorch inference benchmark")
    for label, value in header:
        print(f"  {label:<{label_width}}  {value}")
    print()
    print(f"Steady-state latency (n={report['num_steps']}, CUDA-synchronized policy.infer)")
    for label, value in rows:
        print(f"  {label:<{label_width}}  {value}")
    unsynced = np.asarray(report["unsynced_model_ms"], dtype=np.float64)
    print()
    print(
        "note: policy_timing infer_ms mean "
        f"{float(unsynced.mean()):.3f} ms is not CUDA-synchronized and is excluded from the stats above."
    )


if __name__ == "__main__":
    # Imports (JAX, absl) may have already configured the root logger, which makes
    # a later basicConfig a no-op and hides warmup progress during torch.compile.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    main(tyro.cli(Args))
