# 当前代码精简与训练契约诊断报告

检查范围仅限当前代码。诊断报告是可重复生成的观测产物，不参与训练、续训、播放或导出的准入判断。

## 1. 结论

- S0 teacher latent 消融已恢复为真实网络宽度 `8/16/24/32`，没有固定 16 维投影或兼容层。
- PPO rollout 消融已按 transition 总预算对齐；不同 rollout 长度不再获得不同样本量。
- `fat2_weight`、`latent_dim` 和 `rollout_steps` 由 S0 checkpoint 记录，S1/S2 自动继承，续训拒绝跨配置混用。
- S1 rollout 固定为 `4096 x 64 = 262,144` 条 transition，不受 PPO rollout 消融影响。
- reset 后策略接收环境 forward 后的当前真实观测；student history 由该真实观测复制填满 61 帧。
- reset pose 的下肢、腰部、手臂力矩限制统一使用 `0.86`，无旧阈值运行时入口。
- 域随机化保留：启动前采样一次，之后每 `200 x 48` 个 policy step 全局重采样；普通 episode reset 不重采样。

## 2. Latent 消融契约

| latent_dim | Teacher encoder 输出 | Student encoder 输出 | Actor 输入宽度 |
|---:|---:|---:|---:|
| 8 | 8 | 8 | 104 |
| 16 | 16 | 16 | 112 |
| 24 | 24 | 24 | 120 |
| 32 | 32 | 32 | 128 |

Actor 输入为 `96 + latent_dim`。Critic 不接收 latent，仍使用 `96` 维当前观察和 `64` 维原始特权信息，因此 critic 容量不会随 latent 消融变化。

Checkpoint 加载会同时核对记录的 latent 维度、encoder 输出权重和 actor 首层输入宽度。部署 manifest 也使用 checkpoint 中的实际维度。

FAT2 权重受控支持 `0.0/0.1/0.2`，默认 `0.1`。每条 lineage 的环境 reward、rollout、S1、S2 和诊断均使用 checkpoint 记录值。

## 3. Rollout 样本预算

基准为 `48` steps/update。S0 和 S2 的默认 update 数量及 checkpoint/domain cadence 按 rollout 长度反向缩放：

| rollout steps | 静载结束 update | S0 updates | S0 steps/env | S2 updates | S2 steps/env | save/domain updates |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 4000 | 12000 | 288000 | 4000 | 96000 | 400 |
| 48 | 2000 | 6000 | 288000 | 2000 | 96000 | 200 |
| 64 | 1500 | 4500 | 288000 | 1500 | 96000 | 150 |

课程和域随机化进度使用累计 policy steps 换算为 48-step 基准 iteration。新训练和恢复训练都使用同一换算，不按 PPO update 数直接推进课程。

## 4. 主训练线

- S0：teacher temporal encoder 融合 61 帧 observation history、61 帧 dynamic privilege 和 episode-static privilege，输出所选 latent。
- Critic：直接读取当前 observation 与原始 64 维 privilege，不复用 teacher latent。
- S1：只训练 student context encoder；actor 从 teacher 初始化并冻结。每 200 iteration 计算验证 action KL，最终保存历史最低 KL 对应的完整 student 状态。
- S2：从 S1 actor/context 和 S0 critic 构造 bootstrap checkpoint，继承同一 latent/rollout 配置后继续 PPO。
- S0 的前 2000 个基准 iteration 不创建真实 rickshaw，双手施加 reset 静力学得到的恒定 wrench；剩余 4000 个基准 iteration 恢复真实 rickshaw 和 D6 约束。

## 5. 已清理逻辑

- 删除训练主线中的文件哈希、代码提交绑定、接受门槛和候选策略评估入口。
- 删除 extrinsics/context 随机化脚手架；保留明确的物理域随机化。
- 删除 D6 incoming wrench 的重复状态，诊断与特权观察共用完整左右 D6 wrench。
- 删除无调用的 passed-report evidence 校验链和其专用 helper，共减少 `validation.py` 607 行。
- 删除记录旧 `0.85` reset 阈值的 search 报告，并用当前代码重新生成 alignment 报告。

## 6. 验证状态

- 全仓 CPU 回归：`337 passed, 2 skipped, 4 subtests passed`。
- Isaac Lab/Hydra 配置往返：`4 passed`。
- 四种 latent 的真实 RSL teacher/student forward、critic forward 和 JIT 构造均通过。
- `latent_dim=8, rollout_steps=24` 的真实 S0 与 S2 headless 单 update 冒烟训练通过，checkpoint 张量宽度和训练参数正确。
- `fat2_weight=0.0` 的真实 S0 headless 单 update 冒烟训练通过，环境奖励权重、checkpoint 训练参数和 TensorBoard event 均正确。
- S0 真实断点续训从 `iter=0` 恢复到 `iter=1`，只执行目标上限内的剩余 update。
- 8 GPU 消融流水线的 8 个唯一配置、GPU `0..7` 映射、断点身份、rollout 分片及诊断复用契约通过测试；当前双 4090 节点仅执行了无副作用的 8 卡调度预检。
- reset solver 参数与路径测试：`61 passed`；`--validate-existing config/reset_poses.yaml` 可直接复用旧 pose 库。
- reset alignment：固定物理下 19 个坡度各 1000 步全部通过，`status=passed`、`failures=[]`，三组静态 preload 阈值均为 `0.86`。

复验沿用旧 pose 库而未重新搜索。静态 preload 最大硬件力矩比分别为下肢 `0.18938`、腰部 `0.60952`、手臂 `0.76477`；rollout 手臂峰值为 `0.85891`，仍低于统一 `0.86` 限制。
