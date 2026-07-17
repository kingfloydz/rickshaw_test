# 项目运行指令

命令以当前 `scripts/*.py` 的 CLI 为准。`scripts/_*.py` 是内部模块，不直接运行。

## 1. 环境

```bash
cd <PROJECT_ROOT>
export ISAACLAB_PATH="$(cd ../IsaacLab && pwd)"
export PYTHON=/root/miniconda3/envs/env_isaaclab/bin/python
"$PYTHON" -m pip install -e source/g1_rickshaw_lab
```

服务器运行 Isaac Lab 时通常追加 `--headless`。完整参数使用：

```bash
"$PYTHON" scripts/<SCRIPT_NAME>.py --help
```

## 2. 主训练线

### 2.1 S0 Teacher PPO

```bash
"$PYTHON" scripts/train_teacher.py \
  --task Isaac-G1-Rickshaw-Directional-Slope-v0 \
  --num-envs 4096 \
  --run_name mainline-s0 \
  --headless
```

默认训练 6000 iterations。课程阶段为：

- `0..999`：无真实 rickshaw，双手施加 reset 静力学恒定 wrench；
- `>=1000`：启用真实 rickshaw 和 D6；
- 19 个坡度始终均衡分配；
- 域参数在训练开始前采样并固定，每 200 个基准 PPO iteration 全局重采样；
- 普通 episode reset 不重采样。

续训：

```bash
"$PYTHON" scripts/train_teacher.py \
  --resume-checkpoint <S0_CHECKPOINT> \
  --headless
```

### 2.2 S1 Context 蒸馏

缺少 rollout 时，`train_context.py` 会自动调用采集器：

```bash
"$PYTHON" scripts/train_context.py \
  --teacher <S0_CHECKPOINT> \
  --output logs/rsl_rl/g1_rickshaw_context/s1_context.pt
```

S1 固定为 16 维 student latent，只优化 context encoder。每 200 iterations 在固定
验证集计算 action KL，最终 checkpoint 恢复并保存历史最低 action KL 的完整 student
状态。

已有 rollout 时：

```bash
"$PYTHON" scripts/train_context.py \
  --teacher <S0_CHECKPOINT> \
  --rollout-dir <TEACHER_ROLLOUT_DIR> \
  --output <S1_CHECKPOINT>
```

单独采集 rollout：

```bash
"$PYTHON" scripts/collect_teacher_rollouts.py \
  --teacher <S0_CHECKPOINT> \
  --output-dir <TEACHER_ROLLOUT_DIR> \
  --num-envs 4096 \
  --num-steps 64 \
  --headless
```

### 2.3 S2 Student PPO

```bash
"$PYTHON" scripts/finetune_student.py \
  --teacher <S0_CHECKPOINT> \
  --context <S1_CHECKPOINT> \
  --num-envs 4096 \
  --run_name mainline-s2 \
  --headless
```

续训：

```bash
"$PYTHON" scripts/finetune_student.py \
  --teacher <S0_CHECKPOINT> \
  --context <S1_CHECKPOINT> \
  --resume-checkpoint <S2_CHECKPOINT> \
  --headless
```

## 3. 策略诊断

诊断报告不阻止训练、续训、播放或导出。

```bash
# S0
"$PYTHON" scripts/evaluate_policy.py \
  --checkpoint <S0_CHECKPOINT> \
  --output outputs/diagnostics/s0.json \
  --headless

# S1；teacher 仅用于记录 teacher-student KL
"$PYTHON" scripts/evaluate_policy.py \
  --checkpoint <S1_CHECKPOINT> \
  --teacher-checkpoint <S0_CHECKPOINT> \
  --output outputs/diagnostics/s1.json \
  --headless

# S2；S1 报告只用于同条件 return 对比
"$PYTHON" scripts/evaluate_policy.py \
  --checkpoint <S2_CHECKPOINT> \
  --teacher-checkpoint <S0_CHECKPOINT> \
  --s1-baseline-report outputs/diagnostics/s1.json \
  --output outputs/diagnostics/s2.json \
  --headless
```

Reward 分布定标也是独立诊断：

```bash
"$PYTHON" scripts/calibrate_rewards.py \
  --checkpoint <S0_CHECKPOINT> \
  --policy-kind teacher \
  --output-dir outputs/reward_calibration \
  --headless
```

## 4. 播放与导出

```bash
# 播放 nominal、真实 rickshaw 配置
"$PYTHON" scripts/play_student.py --checkpoint <S2_CHECKPOINT>

# 仅导出 JIT、ONNX、deployment controller 与 manifest
"$PYTHON" scripts/play_student.py \
  --checkpoint <S2_CHECKPOINT> \
  --export-only \
  --headless
```

播放入口只接受展示和运行规模参数，如 `--headless`、`--video`、`--device`、
`--num_envs`；策略或环境 Hydra override 会被拒绝。

## 5. Reset 与资产

Reset pose 的下肢、腰部、手臂力矩比上限统一固定为 `0.86`，无旧阈值兼容入口。

```bash
# 转换 USD
"$PYTHON" scripts/convert_assets.py --asset all --headless

# 可选资产诊断
"$PYTHON" scripts/inspect_assets.py \
  --output outputs/validation/asset_inspection.json \
  --headless

# 生成 19 坡度 reset pose 并做 1000 步复验
"$PYTHON" scripts/solve_reset_poses.py \
  --output config/reset_poses.yaml \
  --alignment-output outputs/validation/reset_alignment_1000.json \
  --steps 1000 \
  --device cuda:0 \
  --headless

# 只复验当前 pose 库
"$PYTHON" scripts/solve_reset_poses.py \
  --validate-existing config/reset_poses.yaml \
  --alignment-output outputs/validation/reset_alignment_1000.json \
  --steps 1000 \
  --headless

# 可选物理诊断
"$PYTHON" scripts/validate_feasibility.py --quick --headless
"$PYTHON" scripts/validate_dynamics.py --headless

# Reset 可视化
"$PYTHON" scripts/render_reset_multiview.py \
  --output-dir outputs/reset_render \
  --headless
```

资产、reset alignment、feasibility、dynamics 和 reward 报告都属于诊断产物。
训练入口直接读取当前配置和资产，不读取这些报告作为接受门槛。

## 6. 测试

```bash
PYTHONPATH=source/g1_rickshaw_lab "$PYTHON" -m pytest -q
```

常用路径覆盖：

```bash
PYTHONPATH=source/g1_rickshaw_lab "$PYTHON" -m pytest -q \
  tests/test_observation_and_tcn.py \
  tests/test_reset_observation_lifecycle.py \
  tests/test_runner_domain_refresh.py \
  tests/test_two_stage_curriculum.py \
  tests/test_training_contract.py
```

可切换的外部路径只有：

```bash
export ISAACLAB_PATH=/absolute/path/to/IsaacLab
export G1_RICKSHAW_FEASIBILITY_ENVELOPE=/absolute/path/to/feasibility_envelope.yaml
export G1_RICKSHAW_RESET_POSES=/absolute/path/to/reset_poses.yaml
```

其余 `G1_RICKSHAW_*` 变量由入口脚本按 checkpoint stage/lineage 设置，不手工覆盖。
