# 项目运行指令汇总

本文汇总当前项目中所有面向用户的运行入口。命令以 `scripts/*.py` 的实际 CLI
为准，并按正式流水线的依赖顺序排列。`scripts/_*.py` 是内部辅助模块，不应直接
运行。

## 1. 环境准备

在项目根目录执行：

```bash
cd /inspire/hdd/project/leverage-robot/ky26212/humanoid_rickshaw_1

export ISAACLAB_PATH="$(cd ../IsaacLab && pwd)"
export PYTHON=/root/miniconda3/envs/env_isaaclab/bin/python

"$PYTHON" -m pip install -e source/g1_rickshaw_lab
```

也可以先激活环境，后续将 `"$PYTHON"` 替换为 `python`：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export ISAACLAB_PATH="$(cd ../IsaacLab && pwd)"
```

项目产物全部写入当前文件系统：RSL-RL 日志和 checkpoint、验证与评估 JSON、
汇总 CSV，以及策略回放视频。多计算节点运行时，所有节点必须使用同一个共享项目
目录和绝对 `--output-dir`。S0/S2 的 `--run_name` 只用于命名对应配置的 RSL-RL
运行，不代表外部实验或上传目标。

本文中的尖括号表示必须替换的路径或值，例如 `<S0_CHECKPOINT>`。Isaac Lab
图形程序在服务器上通常追加 `--headless`；需要窗口时去掉该参数。所有脚本的
完整参数可通过以下形式查询：

```bash
"$PYTHON" scripts/<SCRIPT_NAME>.py --help
```

## 2. 正式流水线

以下命令给出从资产检查到最终播放/导出的完整顺序。训练产生的 checkpoint
文件名由 RSL-RL 日志目录决定，需要将占位符替换成实际路径。

当前训练入口强制校验资产检查；19 坡度、1000 步 reset alignment 由 reset
求解器自身完成，不再作为训练启动 gate。
feasibility/dynamics 扫描保留为可选物理诊断，不再阻止训练启动。

### 2.1 资产转换与训练前验证

```bash
# 1. 将 G1+Dex1 与 rickshaw URDF 转换为 USD
"$PYTHON" scripts/convert_assets.py --asset all --headless

# 2. 检查源资产与组合后的 USD physics ABI
"$PYTHON" scripts/inspect_assets.py \
  --num_envs 1 \
  --output outputs/validation/asset_inspection.json \
  --headless

# 3. 生成并复验 19 个坡度的 reset pose
"$PYTHON" scripts/solve_reset_poses.py \
  --output config/reset_poses.yaml \
  --alignment-output outputs/validation/reset_alignment_1000.json \
  --steps 1000 \
  --device cuda:0 \
  --headless

# 4. 可选：独立诊断 coast-down 与车辆力平衡
"$PYTHON" scripts/validate_dynamics.py \
  --task Isaac-G1-Rickshaw-Directional-Slope-Play-v0 \
  --headless
```

默认关键产物：

- `config/reset_poses.yaml`
- `config/feasibility_envelope.yaml`
- `outputs/validation/asset_inspection.json`
- `outputs/validation/reset_alignment_1000.json`
- `outputs/validation/feasibility_report.json`
- `outputs/validation/dynamics_report.json`

### 2.2 S0 教师策略

```bash
"$PYTHON" scripts/train_teacher.py \
  --task Isaac-G1-Rickshaw-Directional-Slope-v0 \
  --num-envs 4096 \
  --run_name mainline-s0 \
  --headless
```

正式默认值为 seed `42`、`6000` iterations、FAT2 weight `0.1`、物理滚阻开启、
rollout steps `48`。续训必须使用脚本自己的参数：

```bash
"$PYTHON" scripts/train_teacher.py \
  --resume-checkpoint <S0_CHECKPOINT> \
  --headless
```

### 2.3 Reward 定标

从固定 S0 checkpoint 的 TRAINING 分布采集样本并生成内容寻址报告：

```bash
"$PYTHON" scripts/calibrate_rewards.py \
  --checkpoint <S0_CHECKPOINT> \
  --policy-kind teacher \
  --output-dir outputs/reward_calibration \
  --headless
```

使用已经导出的原始样本重新计算，不启动 PhysX：

```bash
"$PYTHON" scripts/calibrate_rewards.py \
  --samples <RAW_REWARD_SAMPLES.pt> \
  --output-dir outputs/reward_calibration
```

`--checkpoint` 与 `--samples` 必须且只能提供一个。

### 2.4 S1 Context 蒸馏

推荐命令会自动采集缺失的教师 rollout，并执行候选策略的固定种子 task-return
选择：

```bash
"$PYTHON" scripts/train_context.py \
  --teacher <S0_CHECKPOINT> \
  --reward-calibration-report <REWARD_CALIBRATION_REPORT.json> \
  --latent-dim 16 \
  --output logs/rsl_rl/g1_rickshaw_context/s1_context.pt
```

latent dim 消融允许值为 `8`、`16`、`24`。已有 rollout 时可显式指定：

```bash
"$PYTHON" scripts/train_context.py \
  --teacher <S0_CHECKPOINT> \
  --reward-calibration-report <REWARD_CALIBRATION_REPORT.json> \
  --rollout-dir <TEACHER_ROLLOUT_DIR> \
  --output <S1_CHECKPOINT>
```

单独采集教师 rollout：

```bash
"$PYTHON" scripts/collect_teacher_rollouts.py \
  --teacher <S0_CHECKPOINT> \
  --output-dir <TEACHER_ROLLOUT_DIR> \
  --num-envs 4096 \
  --num-steps 64 \
  --headless
```

覆盖已有 rollout 时追加 `--overwrite`。

单独比较 S1 候选 checkpoint：

```bash
"$PYTHON" scripts/evaluate_context_candidates.py \
  --candidates <S1_CANDIDATE_1.pt> <S1_CANDIDATE_2.pt> \
  --output <S1_SELECTION_REPORT.json> \
  --num-envs 380 \
  --episodes-per-slope 100 \
  --seeds 42 43 44 45 46 \
  --headless
```

### 2.5 S2 Student PPO 微调

```bash
"$PYTHON" scripts/finetune_student.py \
  --teacher <S0_CHECKPOINT> \
  --context <S1_CHECKPOINT> \
  --num-envs 4096 \
  --run_name mainline-s2 \
  --headless
```

S2 续训：

```bash
"$PYTHON" scripts/finetune_student.py \
  --teacher <S0_CHECKPOINT> \
  --context <S1_CHECKPOINT> \
  --resume-checkpoint <S2_CHECKPOINT> \
  --headless
```

### 2.6 固定种子验收

评估 S0 的单一 TRAINING 分布：

```bash
"$PYTHON" scripts/evaluate_policy.py \
  --checkpoint <S0_CHECKPOINT> \
  --output <S0_ACCEPTANCE_REPORT.json> \
  --curriculum-stages training \
  --headless
```

评估 S1，生成供 S2 return-floor 比较使用的基线报告：

```bash
"$PYTHON" scripts/evaluate_policy.py \
  --checkpoint <S1_CHECKPOINT> \
  --teacher-checkpoint <S0_CHECKPOINT> \
  --output <S1_ACCEPTANCE_REPORT.json> \
  --headless
```

正式评估 S2：

```bash
"$PYTHON" scripts/evaluate_policy.py \
  --checkpoint <S2_CHECKPOINT> \
  --teacher-checkpoint <S0_CHECKPOINT> \
  --s1-baseline-report <S1_ACCEPTANCE_REPORT.json> \
  --output <S2_ACCEPTANCE_REPORT.json> \
  --thresholds config/final_thresholds.yaml \
  --num-envs 380 \
  --episodes-per-slope 100 \
  --seeds 42 43 44 45 46 \
  --curriculum-stages training \
  --headless
```

`FINAL_THRESHOLDS.yaml` 必须包含完整的 `stages.training` 最终验收项。仓库默认使用
`config/final_thresholds.yaml`，也可通过环境变量覆盖。S1 命令不传阈值时生成状态为
`recorded` 的固定种子基线报告，供 S2 return-floor 验证使用。也可以重复传入
`--threshold 'path.to.metric<=value'`，但正式运行建议使用受版本控制的 YAML。

### 2.7 三组策略消融

消融矩阵必须包含恰好 8 个矩阵项：FAT2 weight `0/0.1`、rollout steps
`24/48/64`、latent dim `8/16/24`。物理滚阻在全部训练、评估和播放中始终开启，
保留真实轮心切向阻力实现，不再作为 `off/on` 消融因素。每个矩阵项需绑定对应的
S2 checkpoint、S0 teacher 和 S1 baseline report。

三个默认矩阵项共用一套 baseline，因此实际训练以下 6 个唯一配置：

| 唯一配置 ID | FAT2 weight | rollout steps | latent dim | 对应矩阵项 |
|---|---:|---:|---:|---|
| `baseline` | 0.1 | 48 | 16 | `fat2_weight_0.1`、`rollout_steps_48`、`latent_dim_16` |
| `fat2_weight_0.0` | 0.0 | 48 | 16 | `fat2_weight_0.0` |
| `rollout_steps_24` | 0.1 | 24 | 16 | `rollout_steps_24` |
| `rollout_steps_64` | 0.1 | 64 | 16 | `rollout_steps_64` |
| `latent_dim_8` | 0.1 | 48 | 8 | `latent_dim_8` |
| `latent_dim_24` | 0.1 | 48 | 24 | `latent_dim_24` |

每个唯一配置独立执行 S0、reward calibration、S1、S1 baseline evaluation 和 S2。

#### 共享存储、多计算节点运行

所有节点必须看到同一个项目根目录，并使用完全相同的绝对共享输出路径。每个节点
只能负责不同的唯一配置；`--worker-only` 会在完成所选配置后退出，不扫描完整矩阵，
也不会启动正式评估。共享文件系统必须支持 POSIX advisory lock；共享锁会拒绝
两个节点同时运行同一配置。训练期间不得修改共享源码、配置或 reset pose。

在每个节点先执行：

```bash
cd /inspire/hdd/project/leverage-robot/ky26212/humanoid_rickshaw_1
export ISAACLAB_PATH="$(cd ../IsaacLab && pwd)"
export PYTHON=/root/miniconda3/envs/env_isaaclab/bin/python
export SHARED_OUTPUT="$PWD/outputs/ablation_pipeline"
```

然后在 6 个计算节点分别设置对应的 `RUN_ID`，每个节点只执行一次相同命令：

```bash
# 节点 1
export RUN_ID=baseline

# 节点 2
export RUN_ID=fat2_weight_0.0

# 节点 3
export RUN_ID=rollout_steps_24

# 节点 4
export RUN_ID=rollout_steps_64

# 节点 5
export RUN_ID=latent_dim_8

# 节点 6
export RUN_ID=latent_dim_24

# 每个节点执行；GPU 0 指当前节点内可见的 GPU 0
"$PYTHON" scripts/run_ablation_pipeline.py \
  --final-thresholds config/final_thresholds.yaml \
  --output-dir "$SHARED_OUTPUT" \
  --runs "$RUN_ID" \
  --worker-only \
  --gpus 0 \
  --resume
```

`--resume` 会逐阶段复验 checkpoint、报告哈希和 lineage，只复用通过验证的产物。
某个节点中断后，使用相同 `RUN_ID` 重复执行该节点命令即可。

六个计算节点全部成功退出后，只在一个汇总节点运行：

```bash
"$PYTHON" scripts/run_ablation_pipeline.py \
  --final-thresholds config/final_thresholds.yaml \
  --output-dir "$SHARED_OUTPUT" \
  --finalize-only \
  --selected-run-id fat2_weight_0.1 \
  --gpus 0 \
  --resume \
  --skip-video
```

该命令不重新训练；它要求 6 个唯一配置均通过验证，然后生成 8 项正式矩阵、运行
全部正式评估并汇总 JSON/CSV。需要策略视频时去掉 `--skip-video`。只生成
`ablation_matrix.yaml` 而不评估时追加 `--skip-postprocess`。

也可使用包装器执行相同模式：

```bash
# 每个计算节点设置不同 RUNS
MODE=worker RUNS=baseline GPUS=0 \
  OUTPUT_DIR="$SHARED_OUTPUT" bash scripts/run_mainline_pipeline.sh

# 所有计算节点结束后，只在一个汇总节点执行
unset RUNS
MODE=finalize GPUS=0 SKIP_VIDEO=1 \
  OUTPUT_DIR="$SHARED_OUTPUT" bash scripts/run_mainline_pipeline.sh
```

`MODE` 只允许 `all`、`worker`、`finalize`。`MODE=worker` 必须设置 `RUNS`；
`MODE=finalize` 必须先 `unset RUNS`。包装器默认启用续跑复验；设置 `RESUME=0`
只适合全新的单进程输出目录，不适合共享 worker 模式。

#### 单节点或单服务器运行完整矩阵

一个进程可管理同一服务器上的多张 GPU，每张 GPU 同时只运行一个唯一配置：

```bash
"$PYTHON" scripts/run_ablation_pipeline.py \
  --final-thresholds config/final_thresholds.yaml \
  --output-dir outputs/ablation_pipeline \
  --gpus 0 1 2 3 \
  --selected-run-id fat2_weight_0.1 \
  --resume \
  --skip-video
```

省略 `--runs` 表示运行全部 6 个唯一配置。也可以显式限制配置；非
`--worker-only` 模式只有在全部 6 个配置完成后才会生成正式矩阵：

```bash
"$PYTHON" scripts/run_ablation_pipeline.py \
  --final-thresholds config/final_thresholds.yaml \
  --output-dir outputs/ablation_pipeline \
  --runs latent_dim_8 latent_dim_24 \
  --gpus 0 1 \
  --resume \
  --skip-postprocess
```

默认选择项为矩阵 ID `fat2_weight_0.1`，它映射到唯一训练配置 `baseline`。完整
后处理会生成以下产物：

- `outputs/ablation_pipeline/results/evaluation/*.json`：8 个矩阵项的正式评估报告；
- `outputs/ablation_pipeline/results/evaluation/manifest.json`：评估及选择项绑定；
- `outputs/ablation_pipeline/results/metrics.csv`：全部评估报告的数值指标表；
- `outputs/ablation_pipeline/results/results.json`：输入、哈希、评估、导出和视频总索引；
- `outputs/ablation_pipeline/results/policy.mp4`：选择项的无界面策略回放。

覆盖选择项、视频长度或录像环境数时，在汇总命令中追加参数：

```bash
"$PYTHON" scripts/run_ablation_pipeline.py \
  --final-thresholds config/final_thresholds.yaml \
  --output-dir outputs/ablation_pipeline \
  --gpus 0 \
  --finalize-only \
  --resume \
  --selected-run-id latent_dim_24 \
  --video-length 1500 \
  --video-num-envs 1
```

每个配置只写入共享目录下自己的 `runs/<唯一配置 ID>/`；`.locks/` 用于防止重复
调度。计算节点不得手工共享同一个 `RUN_ID`，也不得在 worker 尚未退出时启动
finalizer。

#### 仅评估已有消融产物

先校验矩阵并生成包含计划命令的 manifest，不执行评估：

```bash
"$PYTHON" scripts/run_policy_ablations.py \
  --matrix <ABLATION_MATRIX.yaml> \
  --output-dir outputs/policy_ablations \
  --dry-run
```

执行全部消融并指定最终选择项：

```bash
"$PYTHON" scripts/run_policy_ablations.py \
  --matrix <ABLATION_MATRIX.yaml> \
  --output-dir outputs/policy_ablations \
  --selected-run-id <SELECTED_RUN_ID>
```

已有完整评估 manifest 时，可独立汇总指标并为选择项录制视频：

```bash
"$PYTHON" scripts/generate_training_artifacts.py \
  --evaluation-manifest outputs/policy_ablations/manifest.json \
  --output-dir outputs/policy_ablations/results \
  --selected-run-id <SELECTED_RUN_ID> \
  --validation-dir <SELECTED_RUN_VALIDATION_DIR>
```

该命令默认生成 `metrics.csv`、`results.json` 和 `policy.mp4`；可用
`--video-length`、`--video-num-envs` 调整录像。使用 `--skip-video` 只生成数据结果时，
不需要传 `--validation-dir`。

也可以直接调用 `evaluate_policy.py` 运行单个消融，例如 latent dim 24：

```bash
"$PYTHON" scripts/evaluate_policy.py \
  --checkpoint <LATENT_24_S2_CHECKPOINT> \
  --teacher-checkpoint <LATENT_24_S0_CHECKPOINT> \
  --s1-baseline-report <LATENT_24_S1_REPORT.json> \
  --output outputs/policy_ablations/latent_dim_24.json \
  --thresholds config/final_thresholds.yaml \
  --ablation-id latent_dim_24 \
  --ablation-group latent_dim \
  --latent-dim 24 \
  --ablation-matrix-sha256 <MATRIX_SHA256> \
  --headless
```

### 2.8 播放与导出

播放最终 S2 策略：

```bash
"$PYTHON" scripts/play_student.py \
  --checkpoint <S2_CHECKPOINT> \
  --acceptance-report <S2_ACCEPTANCE_REPORT.json> \
  --ablation-manifest <POLICY_ABLATION_MANIFEST.json>
```

仅验证并导出 JIT/ONNX，不进入播放循环：

```bash
"$PYTHON" scripts/play_student.py \
  --checkpoint <S2_CHECKPOINT> \
  --acceptance-report <S2_ACCEPTANCE_REPORT.json> \
  --ablation-manifest <POLICY_ABLATION_MANIFEST.json> \
  --export-only \
  --headless
```

播放入口只接受展示/运行规模参数，例如 `--headless`、`--video`、`--device`、
`--num_envs`；策略和环境 Hydra override 会被拒绝。

## 3. Reset Pose 单一流水线

`solve_reset_poses.py` 是唯一 reset 入口。它执行每坡度 50 次静态多起点
求解、Isaac Lab 候选 rollout、跨坡度 winner 组装，以及最终整库复验。只有
候选级和整库硬门全部通过时，才会原子发布输出文件。
每个候选 rollout batch 都在一次性的独立 Isaac Sim 进程中运行，进程完成报告
后退出；流水线不会在同一 Kit/PhysX 上下文中销毁并重建下一批环境。

### 3.1 生成并认证正式姿态库

```bash
"$PYTHON" scripts/solve_reset_poses.py \
  --output config/reset_poses.yaml \
  --alignment-output outputs/validation/reset_alignment_1000.json \
  --steps 1000 \
  --device cuda:0 \
  --headless
```

Stage A 每完成一个坡度就原子更新 `outputs/reset_pose_candidates.json`。如果后续
坡度失败，修正参数或代码后可复用契约仍匹配的已完成坡度，只求解缺失部分：

```bash
"$PYTHON" scripts/solve_reset_poses.py \
  --output config/reset_poses.yaml \
  --candidate-output outputs/reset_pose_candidates.json \
  --alignment-output outputs/validation/reset_alignment_1000.json \
  --reuse-candidates \
  --steps 1000 \
  --device cuda:0 \
  --headless
```

完整缓存会直接跳过 Stage A；部分缓存会按 continuation parent 重建 seed 后续算。
执行顺序为 `0.00 -> +0.01 -> ... -> +0.10`，然后从 `0.00` 重新开始
`-0.01 -> ... -> -0.08`；`+0.09` 以 `+0.08` 为 parent，`+0.10` 以 `+0.09`
为 parent。缓存契约比较坡度网格、continuation 计划、seed 和全部 Stage A 求解
参数；这些内容变化时缓存会被拒绝，避免混用。

### 3.2 只复验现有姿态库

```bash
"$PYTHON" scripts/solve_reset_poses.py \
  --validate-existing config/reset_poses.yaml \
  --steps 1000 \
  --alignment-output outputs/validation/reset_alignment_1000.json \
  --headless
```

记录复验时序可追加 `--timeseries-stride <N>`。车辆 coast-down、滚阻和
解析/实测力一致性仍由通用 `validate_dynamics.py` 验证，不再维护第二套 reset
入口。

## 4. 资产与物理验证的其他模式

只检查 URDF/YAML 等源文件，不启动 USD stage。该模式不能产生正式通过结论：

```bash
"$PYTHON" scripts/inspect_assets.py \
  --num_envs 1 \
  --static-only \
  --output outputs/validation/asset_inspection_static.json
```

只转换一种资产：

```bash
"$PYTHON" scripts/convert_assets.py --asset g1_dex --headless
"$PYTHON" scripts/convert_assets.py --asset rickshaw --headless
```

强制重新转换：

```bash
"$PYTHON" scripts/convert_assets.py --asset all --force --headless
```

只校验 feasibility YAML schema，不运行 PhysX，也不产生 gate report：

```bash
"$PYTHON" scripts/validate_feasibility.py --schema-only
```

运行不具备放行效力的快速 feasibility 诊断：

```bash
"$PYTHON" scripts/validate_feasibility.py --quick --headless
```

显式指定物理验证输入/输出：

```bash
"$PYTHON" scripts/validate_dynamics.py \
  --feasibility config/feasibility_envelope.yaml \
  --reset-poses config/reset_poses.yaml \
  --output outputs/validation/dynamics_report.json \
  --headless
```

### 4.1 Reset 诊断与历史迁移工具

渲染当前配置中全部坡度的 reset pose，每个坡度输出 side、front-oblique 和 top
三张图片，并写出 manifest：

```bash
"$PYTHON" scripts/render_reset_multiview.py \
  --output-dir outputs/reset_render \
  --width 960 \
  --height 720 \
  --device cuda:0 \
  --headless
```

运行 MuJoCo 抓握宽度、肘部间隙和局部 IK 诊断：

```bash
"$PYTHON" scripts/analyze_grasp_width_clearance.py \
  --reset-poses config/reset_poses.yaml \
  --output outputs/diagnostics/grasp_width_clearance.json
```

`upgrade_reset_pose_schema.py` 仅用于将历史 schema-v3 文件补齐 schema-v4 静力学
字段，不能补齐缺失坡度，也不能替代 `solve_reset_poses.py`。不要对当前正式姿态库
执行原地升级；历史文件应显式写到新路径：

```bash
"$PYTHON" scripts/upgrade_reset_pose_schema.py \
  <LEGACY_SCHEMA_V3_RESET_POSES.json> \
  --output outputs/migrations/reset_poses_schema_v4.yaml
```

## 5. 测试

运行完整测试套件：

```bash
"$PYTHON" -m pytest -q
```

运行训练、评估与模型契约测试：

```bash
"$PYTHON" -m pytest -q \
  tests/test_slope_contract.py \
  tests/test_training_contract.py \
  tests/test_ablation_pipeline.py \
  tests/test_policy_evaluation.py \
  tests/test_observation_and_tcn.py \
  tests/test_play_contract.py
```

## 6. 可覆盖的运行路径

通常优先使用 CLI 参数。以下环境变量只在需要切换生成物或 Isaac Lab checkout
时设置：

```bash
export ISAACLAB_PATH=/absolute/path/to/IsaacLab
export G1_RICKSHAW_FEASIBILITY_ENVELOPE=/absolute/path/to/feasibility_envelope.yaml
export G1_RICKSHAW_RESET_POSES=/absolute/path/to/reset_poses.yaml
export G1_RICKSHAW_VALIDATION_DIR=/absolute/path/to/validation
```

训练、播放和导出所用的其余 `G1_RICKSHAW_*` 变量由入口脚本根据 checkpoint
lineage 自动设置，不应手工伪造。

## 7. 注意事项

- `train_teacher.py` 会在启动 PPO 前验证资产检查报告；reset alignment 和
  feasibility/dynamics 报告不再是训练 gate。
- `train_context.py` 的正式运行需要通过 reward calibration，并要求覆盖单一
  `TRAINING` 分布的 on-policy rollout。
- `finetune_student.py`、`evaluate_policy.py` 和 `play_student.py` 都会核验 checkpoint
  provenance 与 lineage，不能混用不同流水线的产物。
- `play_student.py` 当前要求同时提供通过的 S2 acceptance report 和完整三组消融
  manifest；实施指南中的旧简写命令缺少这两个必填参数。
- 实施指南中早期出现的 `scripts/tools/convert_urdf.py` 不存在于当前项目；当前资产
  转换入口是 `scripts/convert_assets.py`。
