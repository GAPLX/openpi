"""Generates `rtc_lerobot_reference.npz`, the reference outputs of LeRobot's RTC implementation used by `rtc_test.py`.

Must be run from a LeRobot checkout that contains `lerobot.policies.rtc`, e.g.:

    cd /path/to/lerobot
    uv run python /path/to/openpi/src/openpi/models/testdata/generate_rtc_lerobot_reference.py
"""

import pathlib

from lerobot.configs import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
import numpy as np
import torch

OUTPUT_PATH = pathlib.Path(__file__).parent / "rtc_lerobot_reference.npz"

SCHEDULES = ["ZEROS", "ONES", "LINEAR", "EXP"]

# (inference_delay, execution_horizon, action_horizon)
WEIGHT_CASES = [
    (0, 10, 50),
    (3, 8, 10),
    (3, 10, 10),
    (5, 30, 50),
    (8, 5, 20),
    (0, 0, 10),
    (4, 60, 50),
]

# (batch, action_horizon, action_dim, prefix_len, prefix_dim, inference_delay, execution_horizon, time, max_weight)
DENOISE_CASES = [
    (1, 50, 32, 40, 32, 5, 40, 1.0, 10.0),
    (2, 50, 32, 30, 14, 4, 20, 0.7, 10.0),
    (1, 10, 8, 6, 8, 2, 10, 0.3, 5.0),
    (2, 16, 32, 16, 7, 3, 8, 0.1, 10.0),
    (1, 50, 32, 12, 32, 6, 10, 0.5, 2.0),
]


def main() -> None:
    data = {}
    for schedule in SCHEDULES:
        processor = RTCProcessor(RTCConfig(prefix_attention_schedule=RTCAttentionSchedule[schedule]))
        for i, (delay, horizon, total) in enumerate(WEIGHT_CASES):
            weights = processor.get_prefix_weights(delay, horizon, total)
            data[f"weights/{schedule}/{i}"] = weights.numpy()
    data["weights/cases"] = np.array(WEIGHT_CASES)

    generator = torch.Generator().manual_seed(0)
    for schedule in SCHEDULES:
        for i, case in enumerate(DENOISE_CASES):
            batch, horizon, dim, prefix_len, prefix_dim, delay, exec_horizon, time, max_weight = case
            processor = RTCProcessor(
                RTCConfig(
                    prefix_attention_schedule=RTCAttentionSchedule[schedule],
                    max_guidance_weight=max_weight,
                    execution_horizon=exec_horizon,
                )
            )
            x_t = torch.randn(batch, horizon, dim, generator=generator)
            v_t = torch.randn(batch, horizon, dim, generator=generator)
            prefix = torch.randn(batch, prefix_len, prefix_dim, generator=generator)
            guided = processor.denoise_step(
                x_t=x_t,
                prev_chunk_left_over=prefix,
                inference_delay=delay,
                time=time,
                original_denoise_step_partial=lambda _, v_t=v_t: v_t,
                execution_horizon=exec_horizon,
            )
            key = f"denoise/{schedule}/{i}"
            data[f"{key}/x_t"] = x_t.numpy()
            data[f"{key}/v_t"] = v_t.numpy()
            data[f"{key}/prefix"] = prefix.numpy()
            data[f"{key}/guided"] = guided.detach().numpy()
    data["denoise/cases"] = np.array(DENOISE_CASES)

    np.savez_compressed(OUTPUT_PATH, **data)
    print(f"Wrote {len(data)} arrays to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
