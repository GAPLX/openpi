# Action chunk 与 delta

LIBERO 训练数据从磁盘到模型、再回到环境，action 的形状和含义会变几次。下面只记这条链路上实际发生的变换。

## Chunk 在进 `LiberoInputs` 之前已经拼好

磁盘上每帧 action 是 `(7,)`。`create_torch_dataset` 用 `action_horizon`（π0 默认 50）给 `action_sequence_keys`（默认 `"actions"`）设置 `delta_timestamps`。LeRobot 在 `__getitem__` 里从当前帧向后取 50 帧并 `stack`，所以 `LiberoInputs` 看到的是 `(50, 7)`。

`state` 和图像不在这个 key 列表里，仍是当前帧。

超出当前 episode 结尾的偏移会被夹到该 episode 的最后一帧，并标上 `actions_is_pad`。openpi 训练不读这个 mask，重复帧仍参与 loss。

## `PadStatesAndActions` 在动作维补零

它沿最后一维用 `0` 补到 `action_dim`（π0 默认 32），时间维不变：`(50, 7)` → `(50, 32)`，`state` 的 `(8,)` → `(32,)`。这和上面的末帧重复不是一回事。

## 两种 delta

**数据集里的逐步增量。** LIBERO 采集时每步 action 已是 OSC 控制器指令，不是绝对末端位姿，也不是 `state[t] - state[t-1]`：

- 前 3 维：末端位置增量，约 ±5 cm/步
- 中 3 维：末端姿态增量，约 ±0.5 rad/步
- 最后 1 维：夹爪，绝对指令

chunk 里的 50 行是先后执行的小指令，每条相对该步开始时的末端位姿，不是都相对 chunk 第一帧。

**`extra_delta_transform`。** 另一套可选变换。`DeltaActions` 用 chunk 起点的观测 `state` 去减 action（mask 为真的维）；`AbsoluteActions` 在推理时加回来。LIBERO 配置里的 mask 是 `make_bool_mask(6, -1)`：前 6 维参与加减，夹爪不动。

默认 `extra_delta_transform=False`，不跑这套变换。`pi0_libero` 等设为 `True`，是为了对齐旧 checkpoint：在已经是逐步增量的 action 上再减一次当前 state。`pi05_libero` 为 `False`。

被减的维必须和 state 是同一物理量、同一单位、同一顺序。

## 推理返回值

`policy.infer` 的结果直接送进 `env.step`。

- 开了 `extra_delta_transform`：网络预测的是「训练目标空间里的增量」，输出链里的 `AbsoluteActions` 加回当前 state。返回值回到训练数据在做 `DeltaActions` 之前的空间。
- 没开：网络直接预测该空间里的 action，只做反归一化并裁到前 7 维。

因此，原始 action 若已是逐步增量，返回的仍是逐步增量。原始 action 若是绝对量并打开了该开关，返回的是绝对 action（mask 为假的维全程保持原样）。mask 要按自己的动作布局修改，训练和推理共用同一份。
