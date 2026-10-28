---
name: PyTorch 推理测速
overview: 新增一个只接受 PyTorch checkpoint 的推理测速脚本：用 xlerobot 随机观测走完整 policy.infer，丢掉加载和 torch.compile 的启动开销，只统计模型稳定后、经 CUDA 同步的端到端延迟。
todos:
  - id: add-benchmark-script
    content: 新增 scripts/benchmark_pytorch_inference.py：仅 PyTorch checkpoint、xlerobot 随机输入、CUDA 同步计时、稳定期预热与指标汇总
    status: completed
  - id: run-on-checkpoint
    content: 用 checkpoints/pi0_xlerobot_lora/debug_001/29999_pytorch 实跑，确认稳定期统计与 action 形状
    status: completed
isProject: false
---

# PyTorch 推理速度 Benchmark

新增 [`scripts/benchmark_pytorch_inference.py`](scripts/benchmark_pytorch_inference.py)。只走 README PyTorch 推理路径：`create_trained_policy` 发现 `model.safetensors` 后加载，再调用 `policy.infer`。没有 `model.safetensors` 时直接退出，不测 JAX。

默认对着已准备的权重跑：

```bash
uv run scripts/benchmark_pytorch_inference.py \
  --config pi0_xlerobot_lora \
  --checkpoint-dir checkpoints/pi0_xlerobot_lora/debug_001/29999_pytorch
```

[`Pi0Config.pytorch_compile_mode`](src/openpi/models/pi0_config.py) 默认是 `max-autotune`。第一次 `sample_actions` 会编译，可能要几分钟，这段不计入统计。

## 测什么

每次迭代先调用 [`make_xlerobot_example`](src/openpi/policies/xlerobot_policy.py) 生成一条随机观测（12 维 state、三张 `224x224x3` uint8 图、prompt `"do something"`），**生成放在计时之外**。

计时包住整次 `policy.infer`，前后各一次 `torch.cuda.synchronize()`，用 `time.perf_counter()`：

```python
example = make_xlerobot_example()          # 不计时
torch.cuda.synchronize()
t0 = time.perf_counter()
result = policy.infer(example)
torch.cuda.synchronize()
elapsed_ms = (time.perf_counter() - t0) * 1000
```

这样计入的是机器人实际等待的时间：输入变换（解析图像、state 解码、归一化、tokenize）、H2D、`sample_actions`（prefix KV cache + 默认 10 步去噪）、`.cpu()` 回传、输出变换（反归一化、绝对动作、编码成 12 维）。不计进程启动、读权重、`torch.compile` / autotune、随机数生成。

不把 [`policy.py`](src/openpi/policies/policy.py) 里的 `policy_timing["infer_ms"]` 当作主指标。那个计时只包住 `sample_actions`，且没有 CUDA 同步，GPU 内核还在队列里计时就结束了，会系统性偏小。`infer()` 末尾的 `.cpu()` 才会把 GPU 工作等完，所以从外部对整次 `infer` 做同步计时才是准的。脚本仍会记下这个内部数值，标成 `unsynced_model_ms`，只作对照，不进汇总。

第一次有效推理会检查 `actions` 形状为 `(action_horizon, 12)`（配置里 horizon 是 15）且全部有限，避免把一次失败的前向算进速度。

## 稳定期

```mermaid
flowchart LR
  load[加载权重] --> cold[第1次推理_编译]
  cold --> warmup[继续预热直到窗口稳定]
  warmup --> measure[正式计时N次]
  measure --> report[汇总并写JSON]
```

- 预热样本全部丢弃。至少 `min_warmup`（默认 5）次，覆盖编译、CUDA caching allocator 和 GPU 时钟爬升。
- 之后看最近 `stability_window`（默认 5）次的变异系数 `std/mean`（样本标准差，`ddof=1`）。低于 `stability_cv`（默认 0.05）就结束预热。
- 预热上限 `max_warmup`（默认 30）。到顶仍不稳定就警告，并在报告里写明未达稳定，然后仍做正式测量。
- 正式测量 `num_steps`（默认 50），与预热样本不混用。
- 单独报告第一次预热耗时 `cold_start_ms`，不进入 min/max/mean。

固定输入尺寸下设置 `torch.backends.cudnn.benchmark = True`。`numpy` 用 `--seed`（默认 0）播种，方便复现同一批观测；像素内容对这条固定 shape 的计算图几乎无影响。

## 指标

只对正式测量的毫秒样本计算：

- 最小、最大、平均
- 样本标准差、变异系数 `std/mean`
- 中位数，以及 p90 / p95 / p99（控制回路更关心尾延迟）
- 四分位距 IQR = p75 − p25
- 极差 max − min
- 吞吐量：`1000 / mean_ms` 次推理每秒
- 动作吞吐量：推理吞吐 × `action_horizon`（本配置每次推理产出 15 步动作）

报告头同时打印设备名、PyTorch / CUDA 版本、compile mode、checkpoint、实际预热次数、是否达到稳定、`cold_start_ms`。

## CLI

沿用 [`scripts/serve_policy.py`](scripts/serve_policy.py) 的 `tyro` 风格，不新增依赖（主 `pyproject.toml` 没有 `rich` / `polars`），终端用纯文本表。

- `--config` 默认 `pi0_xlerobot_lora`
- `--checkpoint-dir` 必填
- `--device` 默认 `cuda`；不可用时退出，避免静默落到 CPU（CPU 上 bfloat16 会被换成 float32，数字没有可比性）
- `--num-steps` `--min-warmup` `--max-warmup` `--stability-window` `--stability-cv` `--seed`
- `--num-diffusion-steps` 默认不传，即模型默认 10；传入则作为 `sample_kwargs["num_steps"]`
- `--compile-mode` 可选，用 `dataclasses.replace` 覆盖冻结的 `Pi0Config.pytorch_compile_mode`（`None` 表示沿用配置，即 `max-autotune`）
- `--output` 可选，把上述指标、环境信息和原始样本毫秒数写成 JSON

实现后用给定的 `checkpoints/pi0_xlerobot_lora/debug_001/29999_pytorch` 实跑一次。第一次会卡在 `max-autotune` 编译上，属预期；确认脚本能打出稳定期统计和一次形状正确的 action。