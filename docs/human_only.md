# Agent do not read this file

机器人实际等待的时间：输入变换（解析图像、state 解码、归一化、tokenize）、H2D、`sample_actions`（prefix KV cache + 默认 10 步去噪）、`.cpu()` 回传、输出变换（反归一化、绝对动作、编码成 12 维）

windows infer time

| 指标 | 数值 |
| --- | --- |
| 最小 | 94.702 ms |
| 最大 | 96.613 ms |
| 平均 | 95.520 ms |
| 标准差 | 0.469 ms |
| 变异系数 | 0.0049 |
| 中位数 | 95.485 ms |
| p90 / p95 / p99 | 96.293 / 96.416 / 96.534 ms |
| 吞吐 | 10.469 次推理/秒，157.0 个动作/秒 |


