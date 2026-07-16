# G1 拉黄包车速度跟踪：Isaac Lab 实施规范

## 1. 范围与固定配置

目标：使用 Isaac Lab manager-based task 和 RSL-RL PPO，使 G1 在平地、上坡和下坡沿单一方向跟踪 `0–1 m/s` 速度；所有训练阶段均连接同一 rickshaw。

```text
G1 body                         29 DoF
Dex1-1                          4 gripper joints, excluded from RL action
RL action                       29-D joint-position target
Simulation / policy frequency   200 / 50 Hz
Episode                         20 s
Parallel environments           4096
Sampled speed command           v_sample=0...1 m/s
Tracking target                 v_ref, acceleration/jerk limited
Slope gradient                  0, +/-{0.01, ..., 0.06}
Terrain                         6 rows, 21 columns, 26 m x 6 m per patch
Action filter                   first-order Butterworth, 4 Hz
History                         61 x 96, causal TCN
Context latent                  16
Actor and critic MLP            [512, 256, 128], ELU
PPO rollout                     48 steps/env
```

必须固定并写入 checkpoint：Isaac Sim 版本、Isaac Lab commit、RSL-RL commit、PyTorch/CUDA 版本、关节顺序和配置文件 hash。

## 2. 上游代码位置


| 用途                    | 代码位置                                                                                                                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| External task 模板      | [`tools/template/templates/external`](https://github.com/isaac-sim/IsaacLab/tree/main/tools/template/templates/external)                                                                          |
| Manager-based task 模板 | [`manager-based_single-agent`](https://github.com/isaac-sim/IsaacLab/tree/main/tools/template/templates/tasks/manager-based_single-agent)                                                         |
| G1 资产                 | [`isaaclab_assets/robots/unitree.py`](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_assets/isaaclab_assets/robots/unitree.py)                                                   |
| G1 velocity 配置        | [`config/g1/rough_env_cfg.py`](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py)                 |
| G1 PPO 配置             | [`config/g1/agents/rsl_rl_ppo_cfg.py`](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py) |
| Locomotion MDP          | [`locomotion/velocity/mdp`](https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp)                                           |
| Joint-position action   | [`joint_actions.py`](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py)                                                                  |
| Stateful action term    | [`joint_actions_to_limits.py`](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions_to_limits.py)                                              |
| Terrain generator       | [`terrain_generator_cfg.py`](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/terrains/terrain_generator_cfg.py)                                                          |
| Trimesh terrains        | [`terrains/trimesh`](https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab/isaaclab/terrains/trimesh)                                                                                   |
| URDF 转 USD             | [`scripts/tools/convert_urdf.py`](https://github.com/isaac-sim/IsaacLab/blob/main/scripts/tools/convert_urdf.py)                                                                                  |
| RSL-RL wrapper          | [`vecenv_wrapper.py`](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py)                                                                    |
| RSL-RL models           | [`rsl_rl/models`](https://github.com/leggedrobotics/rsl_rl/tree/main/rsl_rl/models)                                                                                                               |
| G1 + Dex1-1 URDF        | [`g1_29dof_mode_15_with_dex1_1.urdf`](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf)                                         |
| PhysX D6 示例           | [`D6JointDemo.py`](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/omni/extensions/ux/source/omni.physx.demos/python/scenes/D6JointDemo.py)                                                   |

## 3. 工程结构

```text
g1_rickshaw_lab/
  source/g1_rickshaw_lab/g1_rickshaw_lab/
    assets/
      g1_dex1.py
      rickshaw.py
    tasks/manager_based/rickshaw_velocity/
      __init__.py
      env_cfg.py
      terrain_cfg.py
      agents/rsl_rl_cfg.py
      mdp/
        actions.py
        curricula.py
        dynamics.py
        events.py
        observations.py
        rewards.py
        terminations.py
    rl/
      actor_critic.py
      teacher_model.py
      context_encoder.py
      distillation.py
  assets/
    g1_dex1/g1_29dof_mode_15_with_dex1_1.usd
    rickshaw/{rickshaw.urdf,body.stl,left_wheel.stl,right_wheel.stl,rickshaw.usd}
  scripts/
    inspect_assets.py
    solve_reset_poses.py
    validate_dynamics.py
    validate_feasibility.py
    train_teacher.py
    train_context.py
    finetune_student.py
    play_student.py
```

```bash
python -m pip install -e source/g1_rickshaw_lab
```

## 4. 资产与约束

### 4.1 G1 与 Dex1-1

将官方 G1+Dex1-1 URDF 转换为一个 articulation。G1 本体仍为 29 DoF；组合资产含 29 个 G1 关节和 4 个 Dex 关节。

```bash
python scripts/tools/convert_urdf.py \
  g1_29dof_mode_15_with_dex1_1.urdf \
  g1_29dof_mode_15_with_dex1_1.usd
```

不要合并 fixed joints。沿用 `G1_29DOF_CFG` 的 29 关节 actuator 配置；Dex actuator 参数必须来自驱动规格或实机辨识。

```python
G1_RICKSHAW_CFG = G1_29DOF_CFG.copy()
G1_RICKSHAW_CFG.prim_path = "{ENV_REGEX_NS}/Robot"
G1_RICKSHAW_CFG.spawn.usd_path = G1_DEX1_USD
G1_RICKSHAW_CFG.spawn.activate_contact_sensors = True
```

```python
lower_ids, lower_names = robot.find_joints(".*_(hip|knee|ankle)_.*")
waist_ids, waist_names = robot.find_joints("waist_.*_joint")
arm_ids, arm_names = robot.find_joints(".*_(shoulder|elbow|wrist)_.*")
dex_ids, dex_names = robot.find_joints("(left|right)_dex1_finger_joint_[12]")

assert [len(lower_ids), len(waist_ids), len(arm_ids), len(dex_ids)] == [12, 3, 14, 4]
assert len(set(lower_ids + waist_ids + arm_ids)) == 29
assert len(set(lower_ids + waist_ids + arm_ids + dex_ids)) == 33
```

RL action 只包含 `lower_names + waist_names + arm_names`。将其实际顺序写入 checkpoint，训练、导出和部署不得重新依赖正则排序。

Dex 没有力传感器。夹持状态机只使用标定后的 `q_open`、`q_grasp`、速度和超时；仿真中由整车动量平衡恢复的手部合力、joint effort 和 D6 incoming-joint constraint proxy 只能用于 critic、reward、安全门或日志，不能进入 actor 或实机状态机。

### 4.2 Rickshaw

```bash
python scripts/tools/convert_urdf.py \
  assets/rickshaw/rickshaw.urdf assets/rickshaw/rickshaw.usd \
  --joint-target-type none
```

两个 wheel joints 保持 passive，USD 必须保留 `0.02 N*m*s/rad` joint damping。两个 hitch links 及其内部 fixed joints 不得合并；外部 grasp-hitch 连接使用 D6。

```python
RICKSHAW_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Rickshaw",
    spawn=sim_utils.UsdFileCfg(
        usd_path=RICKSHAW_USD,
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={".*_wheel_joint": 0.0},
        joint_vel={".*_wheel_joint": 0.0},
    ),
    actuators={},
)
```


| URDF 参数                 |                                                数值 |
| ------------------------- | --------------------------------------------------: |
| 坐标轴                    |                           `+X` 前、`+Y` 左、`+Z` 上 |
| `base_link` 质量/惯量对角 | `36.0 kg` / `(7.393572, 22.277208, 17.829456) kg*m^2` |
| 单轮质量/惯量对角         |     `2.0 kg` / `(0.071184, 0.140624, 0.071184) kg*m^2` |
| 单 hitch link 质量        |                                           `0.02 kg` |
| 整车总质量/质心           |            `40.04 kg` / `(0.651664, 0, 0.669432) m` |
| 轮半径/宽度/轮距          |                  `0.374999 / 0.072548 / 0.756462 m` |
| 左/右 hitch               |         `(1.85049373, +/-0.225, 0.18164719) m` |
| 车身 visual 材质          |                 深红色 RGBA `(0.18, 0.004, 0.008, 1)` |

启动时从 USD 读取并断言质量、惯量、关节轴和 frame。高面数 triangle mesh 只用于 visual；body collision 使用简化凸分解，wheel collision 使用轴向 `Y` 的 cylinder。

### 4.3 双 D6

```text
left Dex grasp  <-> left_tow_hitch_link
right Dex grasp <-> right_tow_hitch_link
```

Hitch link frame 已位于连接点，因此 hitch 侧 D6 local pose 为 identity；grasp 侧 frame 使用 Dex 实际夹持中心标定。D6 在 scene setup 时创建一次，episode reset 不重复创建 prim。

```python
@configclass
class HandleConstraintCfg:
    linear_stiffness: float = MISSING
    linear_damping: float = MISSING
    angular_stiffness: float = MISSING
    angular_damping: float = MISSING
    max_force: float = MISSING
    max_torque: float = MISSING
    linear_limit: float = MISSING
    angular_limit: float = MISSING
```

平移轴使用小范围 limit 和力驱动；旋转轴按真实手柄自由度分别配置。真实自由旋转轴不设置 drive。上述值与 USD drive 保持单一数据源，禁止用两个 external fixed joints 代替 D6。

## 5. 坡面、目标姿态与 Reset

### 5.1 单向平面坡

每个地块只沿局部 `+X` 变化；上坡和下坡通过有符号 gradient 表示，不使用 pyramid terrain。

```python
def directional_plane_slope(difficulty, cfg):
    length, width = cfg.size
    level = min(int(difficulty * 6), 5)
    slope = cfg.direction * (0.01 + 0.01 * level)
    z0 = -slope * cfg.spawn_x
    z1 = slope * (length - cfg.spawn_x)
    zb = min(z0, z1) - 1.0

    vertices = np.array([
        [0, 0, z0], [length, 0, z1], [length, width, z1], [0, width, z0],
        [0, 0, zb], [length, 0, zb], [length, width, zb], [0, width, zb],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ], dtype=np.int64)
    return [trimesh.Trimesh(vertices=vertices, faces=faces, process=False)], \
        np.array([cfg.spawn_x, width / 2, 0.0])


@configclass
class DirectionalPlaneSlopeCfg(SubTerrainBaseCfg):
    function = directional_plane_slope
    direction: int = 0
    spawn_x: float = 4.0


DIRECTIONAL_SLOPES_CFG = TerrainGeneratorCfg(
    seed=42,
    curriculum=True,
    size=(26.0, 6.0),
    num_rows=6,
    num_cols=21,
    border_width=0.0,
    use_cache=False,
    sub_terrains={
        "flat": DirectionalPlaneSlopeCfg(proportion=1 / 3, direction=0),
        "uphill": DirectionalPlaneSlopeCfg(proportion=1 / 3, direction=1),
        "downhill": DirectionalPlaneSlopeCfg(proportion=1 / 3, direction=-1),
    },
)
```

`26 m = 20 s * 1 m/s + 4 m robot-cart length + 2 m margin`。Robot spawn 位于地块 `x=4 m`；实测组合长度超过 `4 m` 时同步增大 `spawn_x` 和地块长度。

### 5.2 坡面坐标

```python
level = env.scene.terrain.terrain_levels
column = env.scene.terrain.terrain_types
magnitude = 0.01 + 0.01 * level
sign = torch.where(
    column < 7,
    torch.zeros_like(magnitude),
    torch.where(column < 14, torch.ones_like(magnitude), -torch.ones_like(magnitude)),
)
slope = sign * magnitude
gamma = torch.atan(slope)
e_s = torch.stack((torch.cos(gamma), torch.zeros_like(gamma), torch.sin(gamma)), -1)
e_y = torch.zeros_like(e_s)
e_y[:, 1] = 1.0
e_n = torch.stack((-torch.sin(gamma), torch.zeros_like(gamma), torch.cos(gamma)), -1)
slope_quat = quat_from_matrix(torch.stack((e_s, e_y, e_n), dim=-1))

env.path_tangent_w = e_s
env.path_lateral_w = e_y
env.path_normal_w = e_n
```

上坡 `gamma>0`、下坡 `gamma<0`；两种环境均面向运动方向，yaw 为 0。每次 terrain level 变化后、写入 root state 前重新计算这些量。

### 5.3 Hitch 高度与 Rickshaw 前抬角

Wheel joint position 只控制滚动相位。Hitch 高度由整个 rickshaw 绕轮轴的前抬角 `alpha` 控制。

```python
@configclass
class RickshawPoseTargetCfg:
    wheel_radius: float = 0.374999
    hitch_x: float = 1.85049373
    hitch_z: float = 0.18164719
    hitch_half_width: float = 0.225
    hitch_height_target: float = MISSING
    hitch_height_tolerance: float = MISSING
    hitch_vertical_speed_tolerance: float = MISSING


def target_pitch_from_hitch_height(cfg):
    radius = math.hypot(cfg.hitch_x, cfg.hitch_z - cfg.wheel_radius)
    phase = math.atan2(cfg.hitch_z - cfg.wheel_radius, cfg.hitch_x)
    ratio = (cfg.hitch_height_target - cfg.wheel_radius) / radius
    if not -1.0 <= ratio <= 1.0:
        raise ValueError("infeasible hitch_height_target")
    return math.asin(ratio) - phase
```

几何关系：

\[
H=r+L\sin\alpha+(h-r)\cos\alpha,
\qquad h_B=r(1-\cos\alpha)
\]

`hitch_height_target=0.85 m` 对应 `alpha_target=0.362266 rad (20.76 deg)`，仅作为 IK 搜索候选。按当前 `40.04 kg` 质量和质心估算，最终高度必须通过 G1 静力矩、真实硬件力矩和 CoP 验证；若载荷超限，应调整车辆质量、质心或轮轴，不能靠增大奖励权重补偿。

```python
alpha_target = target_pitch_from_hitch_height(cfg.rickshaw_pose)
zeros = torch.zeros_like(gamma)
pitch_rel = quat_from_euler_xyz(zeros, -alpha_target * torch.ones_like(gamma), zeros)
cart_quat_w = quat_mul(slope_quat, pitch_rel)
cart_root_height = cfg.rickshaw_pose.wheel_radius * (1.0 - math.cos(alpha_target))
```

### 5.4 G1 闭链初始化

为 19 个坡度离线求解并保存 reset pose。IK 硬约束：

```text
feet:  soles coplanar with terrain, +X forward, +Z=e_n
hands: left/right Dex grasp 6-D pose equals corresponding hitch pose
robot: joint limits, no self/robot-cart collision
cart:  both wheel centers have normal height 0.374999 m
```

软目标：torso/pelvis 接近自然姿态、静力 CoP 居中、左右对称、归一化关节力矩最小。肩 roll/yaw 强约束在默认值附近，shoulder pitch 只补偿位置，elbow 负责主要伸缩，wrist 对齐 Dex 朝向。Dex 局部 `+X` 以坡面路径 `+X` 为目标，初始及柔顺 settling 全过程均允许绕横杆 `+Y` 在 `+/-70 deg` 内转动；初始化横向投影上限为 `0.01`，柔顺 settling 瞬态上限为 `0.03`（约 `1.72 deg` 偏航），不得回到局部 `+X` 朝坡面 `-Z` 的下夹分支。离线 reset 姿态在解析 FAT2 目标之外加入单独记录的 `0.12 rad` 坡上静态倾斜裕量，解析 FAT2 本身不修改，最终仍以 1000 步 PhysX 站立、位移、D6 和真实硬件力矩指标验收。上坡和下坡使用同一有符号模型；膝、肘弯曲不简单取反。

实机启动前使用 IMU、双足姿态和静态接触估计 `gamma_init`，选择或插值同一组 IK pose 和 `q_ref`；禁止用仿真 terrain truth 形成实机不存在的初始化接口。初版只部署在已覆盖的坡度范围内。

由 IK 的 grasp midpoint 反解 cart root：

```python
hitch_mid_local = torch.tensor((1.85049373, 0.0, 0.18164719), device=env.device)
cart_pos_w = grasp_mid_w - quat_apply(
    cart_quat_w, hitch_mid_local.expand(env.num_envs, -1)
)
actual_height = ((cart_pos_w - terrain_origin_w) * e_n).sum(-1)
assert torch.max(torch.abs(actual_height - cart_root_height)) \
    <= cfg.rickshaw_pose.hitch_height_tolerance
```

Wheel phase 与纵向位置一致；它不参与高度计算：

```python
wheel_phase = torch.remainder(-cart_path_position / 0.374999 + math.pi, 2 * math.pi) - math.pi
wheel_pos = torch.stack((wheel_phase, wheel_phase), dim=-1)
wheel_vel = torch.zeros_like(wheel_pos)
```

Reset 顺序：关闭 D6 drive；写 cart root/wheels；写 G1 root、29 关节 IK pose 和 Dex `q_grasp`；更新运动学；检查双足、双轮、碰撞和 grasp residual；清零速度、动作滤波器、FAT 与 history 状态；将 D6 gains 从 0 平滑增加到标定值。Settling 期间不累计 command、reward、history 或 termination counter。

### 5.5 Scene 与命令

```python
@configclass
class RickshawDirectionalSlopeSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=DIRECTIONAL_SLOPES_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    robot = G1_RICKSHAW_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    rickshaw = RICKSHAW_CFG.replace(prim_path="{ENV_REGEX_NS}/Rickshaw")
    robot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    wheel_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Rickshaw/.*_wheel_link", history_length=3
    )
```

```text
sim.dt             0.005 s
decimation         4
episode_length_s   20.0
scene.num_envs     4096
v_sample resample  10.0 s
standing envs      2%
v_sample           [0, 1] m/s
```

`v_sample` 不能直接进入 observation 或 reward。每个环境保存 `v_ref` 和 `a_ref`，每个 policy step 先做 jerk 限制，再做加速度限制和积分：

```python
@configclass
class SpeedReferenceCfg:
    acceleration_limit: float = MISSING  # m/s^2，由可行性扫描确定
    jerk_limit: float = MISSING          # m/s^3，由可行性扫描确定
    response_time: float = 0.5           # s
    velocity_tolerance: float = 1.0e-3   # m/s


def update_speed_reference(state, v_sample, dt, cfg):
    # 若从当前 a_ref 以最大反向 jerk 制动，v_stop 是 a_ref 回到 0 时的速度。
    v_stop = state.v_ref + state.a_ref * torch.abs(state.a_ref) / (
        2.0 * cfg.jerk_limit
    )
    a_des = torch.clamp(
        (v_sample - v_stop) / cfg.response_time,
        -cfg.acceleration_limit,
        cfg.acceleration_limit,
    )
    da = torch.clamp(
        a_des - state.a_ref,
        -cfg.jerk_limit * dt,
        cfg.jerk_limit * dt,
    )
    a_next = torch.clamp(
        state.a_ref + da,
        -cfg.acceleration_limit,
        cfg.acceleration_limit,
    )
    v_next = state.v_ref + a_next * dt
    settled = (
        (torch.abs(v_sample - v_next) <= cfg.velocity_tolerance)
        & (torch.abs(a_next) <= cfg.jerk_limit * dt)
    )
    state.v_ref[:] = torch.where(settled, v_sample, v_next)
    state.a_ref[:] = torch.where(settled, torch.zeros_like(a_next), a_next)
```

Reset 时统一令 `v_sample=v_ref=a_ref=0`；settling 完成后再采样 `v_sample`，从第一步开始经过 limiter。只有 `|a_next|<=jerk_limit*dt` 时才允许吸附到目标，因此吸附步骤仍满足 jerk bound。`acceleration_limit` 和 `jerk_limit` 必须位于第 11.4 节可行性扫描通过的范围内，并写入 checkpoint。

路径命令固定为：`v_ref`、机器人与车辆中点的横向路径误差 `e_y`、机身航向相对坡面切向的误差 `e_psi`。零 yaw-rate 不再作为任务目标，因为它不能约束累计航向偏差。C1 关闭 push、base mass/CoM randomization 和 height scanner。

## 6. Action

三个 ActionTerm 合计 29 维，Dex 不进入 RL action：

```text
lower 12      scale=0.40
waist 3       scale=0.20
shoulder 6    scale=0.25
elbow 2       scale=0.30
wrist 6       scale=0.15
```

以上 scale 单位为 `rad / normalized_action`。关节分组在启动时按 checkpoint 中的固定 joint names 建立并断言维度合计为 29；不得在训练和部署时分别用正则重新排序。

每个环境保存坡度对应的闭链 IK pose `q_ref`。Action 0 必须映射到 `q_ref`，不能使用资产的固定 default offset；否则 reset 后第一步会破坏闭链姿态。在缩放并加 `q_ref` 后应用 50 Hz、一阶 4 Hz Butterworth：

```python
B0 = B1 = 0.20430082
A1 = -0.59139835

def process_actions(self, actions):
    self._raw_actions[:] = actions
    x = actions * self._scale + self.q_ref
    y = B0 * x + B1 * self.x_prev - A1 * self.y_prev
    self._processed_actions[:] = y
    self.x_prev[:] = x
    self.y_prev[:] = y
```

Reset 时更新对应环境的 `q_ref`，并令 `x_prev=y_prev=q_ref`；`q_ref` 在 episode 内固定。History 使用各 ActionTerm 的 `processed_actions`；`mdp.last_action()` 是 raw action，不能替代它。

## 7. Rickshaw 真实动力学与 FAT2

### 7.1 滚动阻力进入 PhysX

`c_rr` 必须改变仿真轨迹，不能只出现在 reward、teacher latent 或解析 `T_s` 中。本项目采用“在两个轮心施加坡面切向阻力”这一种实现；不要同时再施加 `c_rr N r` 轮轴力矩，否则会重复计入滚阻。URDF 中的 `0.02 N*m*s/rad` wheel joint damping 保留，它描述轴承粘性阻尼，与库仑型滚阻不是同一项。

每个 physics step 在积分前执行：

```python
@configclass
class RollingResistanceCfg:
    c_rr: tuple[float, float] = MISSING
    velocity_epsilon: float = 0.05       # m/s，仅平滑换向
    normal_force_filter_hz: float = 20.0


def apply_rolling_resistance(env, cfg):
    cart = env.scene["rickshaw"]
    wheel_vel = cart.data.body_lin_vel_w[:, env.wheel_body_ids]
    v_s = torch.sum(wheel_vel * env.path_tangent_w[:, None, :], dim=-1)

    contact_force = env.scene["wheel_contacts"].data.net_forces_w[:, env.wheel_sensor_ids]
    n_raw = torch.clamp(
        torch.sum(contact_force * env.path_normal_w[:, None, :], dim=-1),
        min=0.0,
    )
    env.rickshaw_state.wheel_normal_force[:] = low_pass(
        env.rickshaw_state.wheel_normal_force,
        n_raw,
        cutoff_hz=cfg.normal_force_filter_hz,
        dt=env.physics_dt,
    )
    direction = torch.tanh(v_s / cfg.velocity_epsilon)
    f_rr = -env.c_rr[:, None] * env.rickshaw_state.wheel_normal_force * direction
    force_w = f_rr[..., None] * env.path_tangent_w[:, None, :]

    zeros = torch.zeros_like(force_w)
    cart.permanent_wrench_composer.set_forces_and_torques(
        force_w,
        zeros,
        body_ids=env.wheel_body_ids,
        is_global=True,
    )
```

`env.c_rr` 在 reset 时从统一训练分布采样；同一个 tensor 同时供物理力、privileged observation、解析校验和日志使用。切向方向来自轮心实际速度，禁止使用 `v_ref` 或 `v_sample` 决定阻力方向。若 Isaac Lab 当前版本要求显式 `write_data_to_sim()`，则在所有 external wrench 写完后、`sim.step()` 前调用一次。

必须通过单环境动力学校验：断开 G1/D6，将车辆置于平地并给定初速度；`c_rr=0` 与 `c_rr>0` 的减速度必须不同，且测得的总阻力满足

\[
F_{rr}=c_{rr}(N_L+N_R)
\]

在速度平滑区 `|v_s|<3 velocity_epsilon` 内不做该幅值验收。

### 7.2 车辆切向力和轮轴力矩平衡

定义坡面坐标系 `(e_s,e_y,e_n)`：`e_s` 沿前进方向，`e_n` 离开坡面。`T_s,T_n` 是 G1 双手通过 D6 施加给车辆的合力分量；正 `T_s` 向前拉车，正 `T_n` 向上托举手柄。`O` 是左右轮轴中点。所有位置均从仿真实际 pose 计算，不使用 URDF 名义坐标代替：

\[
r_h=p_h-O=x_h e_s+z_h e_n,\qquad
r_c=p_{CoM,c}-O=x_c e_s+z_c e_n
\]

切向动力学为：

\[
T_s=m_{eff}\dot v_s+m_cg\sin\gamma
+c_{rr}N_w\tanh(v_s/\epsilon)+b_{eff}v_s
\]

\[
m_{eff}=m_c+\sum_i I_{w,i}/r_i^2,\qquad
b_{eff}=\sum_i b_{w,i}/r_i^2,\qquad N_w=N_L+N_R
\]

车辆绕轮轴、以手柄抬升方向为正的力矩平衡为：

\[
x_hT_n-z_hT_s-m_cg(x_c\cos\gamma-z_c\sin\gamma)
=I_{O,y}\ddot\alpha
\]

因此 FAT2 所需的法向载荷估计为：

\[
T_n=
\frac{I_{O,y}\ddot\alpha+z_hT_s
+m_cg(x_c\cos\gamma-z_c\sin\gamma)}{x_h}
\]

`I_O,y` 是车辆 body 与 payload 关于轮轴的 pitch inertia；wheel spin inertia 已进入 `m_eff`，不得再次加到 `I_O,y`。`alpha`、`v_s` 用 20 Hz 二阶低通后的差分计算，reset 时令两级历史等于当前值，使 `dot(v_s)=ddot(alpha)=0`。当 `x_h <= 0.5 m`、双轮法向力低于安全阈值或 wheel lift 时，不计算 FAT reward，直接由安全项处理。

Payload reset 后必须用 parallel-axis theorem 重新计算 `m_cart`、`p_CoM,c` 和 `I_O,y`，再更新 `m_eff`；禁止只增加质量而保留 nominal CoM/inertia。符号单元测试固定为：平地静态且 `T_s=alpha_ddot=0` 时，`T_n=m_cart*g*x_c/x_h>0`；在 `z_h>0` 时增大正 `T_s` 必须增大 `T_n`。Isaac Sim 5.1 对 `excludeFromArticulation` 外部 D6 不提供可靠的 tensor reaction wrench；retained hitch link 的 incoming joint wrench 只作为 residual/impulse 的保守约束代理，不作为手部合力真值。

解析实现：

```python
v_s = dot(cart.data.root_lin_vel_w, env.path_tangent_w)
alpha = rickshaw_pitch(env)
a_s = filtered_first_derivative(v_s, state.v_hist, env.step_dt, cutoff_hz=20.0)
alpha_ddot = filtered_second_derivative(
    alpha, state.alpha_hist, env.step_dt, cutoff_hz=20.0
)

n_w = state.wheel_normal_force.sum(-1)
rr_mag = env.c_rr * n_w * torch.tanh(v_s / cfg.velocity_epsilon)
t_s = state.m_eff * a_s + state.m_cart * 9.81 * torch.sin(env.gamma) \
    + rr_mag + state.b_eff * v_s
t_n = (
    state.pitch_inertia_about_axle * alpha_ddot
    + state.handle_z_from_axle * t_s
    + state.m_cart * 9.81 * (
        state.com_x_from_axle * torch.cos(env.gamma)
        - state.com_z_from_axle * torch.sin(env.gamma)
    )
) / state.handle_x_from_axle
```

物理交互合力由整个黄包车的线动量平衡独立测量，而不是从 retained hitch link 的 incoming wrench 推断。每个 physics substep 累积两个车轮的地面接触合力和实际施加在轮心的滚阻；不得直接减去 all-body cart contact sensor 的合力，因为其中还包含预期的 Dex/横杆接触，而该接触正是待测机器人-车辆总交互力的一部分。policy 边界用整车质心速度差得到 `a_C`：

\[
F_{D,cart}=m_{cart}a_C-m_{cart}g-F_{contact}-F_{rr},\qquad F_h=-F_{D,cart}.
\]

pre-physics 的首个 contact sample 属于上一 policy interval，因此在边界用当前最终 sample 替换它，保证平均窗口严格覆盖本 interval 的 4 个 physics substep。该测量使用实际 PhysX 质量、接触、速度和外加滚阻，不使用解析 `T_s/T_n`。随后将 `F_{D,cart}` 投影到 `e_s/e_n` 做符号和幅值校验。在线 FAT2 门使用 25 个 policy step（0.5 s）滑窗。对每个分量令 `bar(T)=mean(T)`、`bar(D)=mean(F_D)`、`S_T=mean(abs(T))`，检查

\[
\epsilon_T=\frac{|\bar D-\bar T|}{\max(S_T,335\mathrm{N})}\le 0.35
\]

只有当两侧净力都超过同一 `0.35*max(S_T,335 N)` 不确定度带时才要求符号相同，避免加速/制动换向时的大幅正负力相消造成伪失效。`335 N` 是 G1 标称质量 `34.1299349 kg` 的一个向上取整自重尺度，仅用于低力区 FAT2 弱姿态先验的归一化；它不改变任何 D6 safety limit。强制 `validate_dynamics.py` 实测/解析闭环验证使用独立且更严格的 `12 N` 绝对下限，不能通过放宽 FAT2 floor 掩盖力平衡错误。窗口必须填满且其中每步解析力和动量测量均有效；FAT2 reference 使用该窗口的 `bar(T_s)/bar(T_n)`，不追逐步态冲击。D6 incoming proxy 的峰值、冲量和 residual 仍由独立 safety/termination 保守处理；窗口不一致只令弱 FAT2 reward 无效，不触发终止。解析量只进入 critic、reward 和日志，不进入 actor。

`T_s/T_n` 是诊断和 FAT2 reference，不作为 external force 再施加给 G1 或车辆；真实交互力只由双 D6 闭链产生。否则会把同一手柄载荷施加两次。

### 7.3 FAT2 参考与稳定性

[Thor/FAT2](https://arxiv.org/abs/2510.26280v3) 从含外力的 ZMP/力矩平衡构造随交互力变化的 torso tilt。论文的准静态标量式对小力臂项作了近似；本任务使用完整二维 wrench。这里不能只使用 `T_s`：车辆的轮轴平衡通常产生同量级的 `T_n`。

车辆施加给 G1 的手部合力为：

\[
F_h=-T_s e_s-T_n e_n
\]

令 `p` 为当前双足支撑多边形中心，`r_{ph}=p_h-p=h_s e_s+h_n e_n`，则手部外力绕 `p` 的前倾力矩为：

\[
M_h=h_sF_{h,n}-h_nF_{h,s}
\]

在准静态近似下，期望 torso 相对世界竖直的前倾角为：

\[
\theta_{FAT}=\arcsin\left(
\operatorname{clip}\left(\frac{M_h}{m_rgR_c},
-\sin\theta_{max},\sin\theta_{max}\right)\right)
\]

`m_r` 从 G1 资产计算；`R_c` 是当前支撑中心到机器人全身 CoM 在
`e_s/e_n` 平面内的距离。在线实现将有效支撑下的 `R_c` 裁剪到标定的
`[0.5, 0.85] m` 物理范围，并使用与 wrench consistency 相同的 25 个 policy
step 滑窗均值；无有效支撑或非有限样本时保持上一个滤波值，reset 时回到
标定半径 `0.715092420262594 m`。这避免单/双支撑切换把瞬时几何跳变直接
传入弱姿态 reward。`theta_FAT>0` 表示沿 `+e_s` 前倾。该项是弱姿态先验，
不是主稳定性判据：

```python
def fat2_prior_exp(env, sigma):
    theta = torso_pitch_from_world_vertical(env)
    valid = env.rickshaw_state.fat_valid
    reward = torch.exp(-torch.square((theta - env.rickshaw_state.theta_fat) / sigma))
    return reward * valid
```

主稳定性项使用含手部外力的坡面 ZMP 裕量。设机器人 CoM 为 `(c_s,c_n)`、手部作用点为 `(h_s,h_n)`，均在坡面坐标中表达；机器人受到的手部 wrench 为 `(F_hs,F_hn,tau_h)`。由平动力学先计算地面合力：

\[
R_s=m_r(a_s+g\sin\gamma)-F_{h,s},\qquad
R_n=m_r(a_n+g\cos\gamma)-F_{h,n}
\]

第一版令质心角动量变化率 `dot(L_y)=0`，坡面切向 ZMP 为：

\[
p_{zmp,s}=c_s+
\frac{-c_nR_s-
[(h_s-c_s)F_{h,n}-(h_n-c_n)F_{h,s}]-\tau_h}{R_n}
\]

`R_n<=min_ground_reaction` 时 ZMP 无效并触发安全计数。双支撑 polygon 取两脚四角点的 convex hull，单支撑取接触脚 polygon；在坡面二维坐标中计算 ZMP 到所有边的最小有符号距离 `d_zmp`。D6 wrench 用于训练期 ZMP，解析 `T_s/T_n` 用于独立校验；二者均不进入 actor。FAT2、ZMP margin 和 torso orientation 不得同时使用大权重；默认仅 ZMP margin 为主项，FAT2 为小权重消融项。

以下值在资产读取或标定前保持 `MISSING`：payload mass/CoM、`c_rr` 范围、`I_O,y`、`theta_max` 和 D6 gains/limits。它们必须由 `validate_feasibility.py` 和第 13 节验收确定，不能通过 PPO 自动“补偿”不可行参数。FAT2 sigma/weight 和 ZMP margin 使用第 11.1 节的明确初值，并按第 11.2 节定标。

## 8. Observation 与 History

Actor current observation 固定为 96 维：


| 分量                           | 维度 |           scale |
| ------------------------------ | ---: | --------------: |
| base angular velocity          |    3 |            0.25 |
| projected gravity              |    3 |             1.0 |
| task signal`(v_ref,e_y,e_psi)` |    3 | `(2.0,2.0,1.0)` |
| `q-q_ref`                      |   29 |             1.0 |
| joint velocity                 |   29 |            0.05 |
| previous processed action      |   29 |             1.0 |

`e_y` 是机器人 base 与 rickshaw base 中点到路径中心线的有符号坡面横向距离；`e_psi` 用 `atan2(sin(delta_yaw),cos(delta_yaw))` 包裹到 `[-pi,pi]`。Actor 不读取真实 base linear velocity、坡度、车辆参数、D6 wrench、接触力、`T_s/T_n` 或 ZMP。

History 只有一个：

```text
history [N, 61, 96] = 动作时刻 t 之前的 o[t-61]...o[t-1]
current [N, 96]      = 动作时刻 t 的 o[t]
```

History 明确排除 `current`，避免 TCN 复制 current branch。每帧保留 previous processed action，因此 history 同时包含状态响应和实际下发目标。顺序固定为：处理并滤波 action；执行 4 个 physics steps；读取 `o[t+1]`；先把旧 current 追加到 history，再将新 observation 设为 current。Reset 后用第一个有效 observation 填满 61 帧，并清零 TCN/滤波差分状态；settling 数据不进入 history。

61 帧在 50 Hz 下为 `1.22 s`，与第 9 节 TCN 的精确感受野一致。禁止再创建 5 帧 short-history MLP 或第二个 recurrent/history encoder。

Observation corruption：

```text
base angular velocity   uniform [-0.2, 0.2] rad/s
projected gravity       uniform [-0.05, 0.05]
joint position          uniform [-0.01, 0.01] rad
joint velocity          uniform [-1.5, 1.5] rad/s
```

Teacher extrinsics 只包含独立随机变量：payload mass/CoM、`c_rr`、terrain friction、左右 wheel damping、D6 stiffness/damping/limits、motor strength、control delay 和 observation delay。Critic privileged group在此基础上增加 slope、base/cart velocity、rickshaw pitch、wheel normal force、D6 wrench/residual、filtered acceleration、`T_s/T_n` 和 ZMP margin。由上述变量计算出的量不得再次进入 teacher encoder。

所有连续 extrinsics 按训练上下界映射到 `[-1,1]`；固定参数不进入 teacher encoder。Actor 和 critic 不使用运行时 empirical normalizer，严格使用本节固定 scale；scale、clip limit 和 observation 顺序写入 checkpoint。

## 9. 单一 Context 网络、Actor 与 Critic

信息通路固定为：

```text
history [61,96] -> causal TCN -> feature 128 -> z_hat 16 --+
current 96 ------------------------------------------------+-> actor [512,256,128] -> Gaussian 29

independent extrinsics -> teacher [256,128] -> z_star 16   (仅训练)
current + z + privileged -> critic [512,256,128] -> value  (仅训练)
```

`current` 提供即时反馈；`z_hat` 只负责从过去响应估计载荷、滚阻、延迟和车辆相位等隐状态。Actor 输入只能是 `concat(current,z)`，维度 `112`。

TCN 使用 kernel `5`、dilation `(1,2,4,8)`、stride `1`。每个 block 只有一个 dilated convolution，因此最后一个输出的感受野为

\[
1+(5-1)(1+2+4+8)=61\text{ frames}.
\]

可直接实现为：

```python
class CausalBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.left_pad = 4 * dilation
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=5, dilation=dilation
        )
        self.mix = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        y = self.conv(F.pad(x, (self.left_pad, 0)))
        y = self.mix(F.elu(y))
        return F.elu(x + y)


class ContextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Conv1d(96, 128, kernel_size=1)
        self.blocks = nn.Sequential(*[
            CausalBlock(128, d) for d in (1, 2, 4, 8)
        ])
        self.context = nn.Sequential(
            nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 16)
        )
        self.phase = nn.Linear(128, 2)          # sin/cos
        self.frequency = nn.Linear(128, 1)      # normalized step frequency
        self.contact = nn.Linear(128, 2)        # left/right contact logits
        self.cart_lag = nn.Linear(128, 1)       # normalized phase lag

    def forward(self, history):
        x = self.blocks(self.input(history.transpose(1, 2)))
        feature = x[:, :, -1]
        return self.context(feature), {
            "phase": self.phase(feature),
            "frequency": self.frequency(feature),
            "contact": self.contact(feature),
            "cart_lag": self.cart_lag(feature),
        }
```

Auxiliary heads 仅在 student 训练时使用，部署时删除。Phase label 由左右足触地事件形成的连续 gait phase 生成；frequency 是相邻同侧触地间隔的倒数；contact 是当前左右足二值接触；cart lag 是 `v_ref` 与 cart speed 的归一化互相关峰值延迟。没有有效 gait 周期或速度低于 `0.1 m/s` 时，用 mask 排除 phase/frequency/cart-lag loss。

Teacher encoder 为 `extrinsics_dim -> 256 -> 128 -> 16`，ELU。Actor MLP 为 `112 -> 512 -> 256 -> 128 -> 29`，ELU；Gaussian log-std 为独立参数，lower-body 初始 std `0.4`、waist/arm 初始 std `0.25`。Critic 不与 actor 共享 privileged trunk，防止 privileged information 通过共享层泄漏到部署策略。

## 10. 训练流程与 PPO

### 10.1 S0 Privileged Teacher PPO

完整创建 rickshaw、双 D6、真实滚阻、车辆奖励和安全终止。Teacher 使用：

```text
z_star       = teacher_encoder(independent_extrinsics)
actor input  = current + z_star
critic input = current + z_star + privileged
```

全部环境始终使用同一个 `TRAINING` 分布，总计最多 `6000` iterations。这里没有 off-policy replay buffer。每 200 iterations 在固定 validation seeds 上评估并保存 checkpoint；连续 10 次评估无提升则停止。19 个坡度中的每个坡度至少评估 100 episodes。

### 10.2 S1 On-policy Student Distillation

冻结 teacher。由 teacher 在同一个 `TRAINING` 分布中生成 on-policy rollout；student 用同一 `current` 和过去 61 帧 history。主要目标是动作分布蒸馏，不以不可完全辨识的 extrinsics latent MSE 作为主目标：

```python
with torch.no_grad():
    z_star = teacher_encoder(extrinsics)
    teacher_dist = teacher_actor(current, z_star)

z_hat, aux = context_encoder(history)
student_dist = student_actor(current, z_hat)

loss = (
    gaussian_kl(teacher_dist, student_dist).mean()
    + 0.05 * F.mse_loss(z_hat, z_star)
    + 0.10 * masked_phase_loss(aux["phase"], phase_target, gait_mask)
    + 0.05 * masked_mse(aux["frequency"], frequency_target, gait_mask)
    + 0.05 * F.binary_cross_entropy_with_logits(aux["contact"], contact_target)
    + 0.05 * masked_mse(aux["cart_lag"], cart_lag_target, lag_mask)
)
```

Student actor 从 teacher actor 权重初始化。Adam：context learning rate `3e-4`、actor learning rate `1e-4`、batch `65536`、mini-batch `8192`、gradient clip `1.0`，最多 `2000` iterations。模型选择依据固定验证集的 teacher action KL 与 task return；latent MSE 只用于诊断。

### 10.3 S2 Student PPO Fine-tune

用 `z_hat` 替换 `z_star`，critic 继续读取 privileged group。Context encoder 不冻结：context learning rate `1e-4`，actor/critic learning rate `3e-4`，继续 PPO `2000` iterations。S2 不再保留 distillation loss，防止 student 被 teacher 在不可辨识状态下的 latent 表示限制。

### 10.4 PPO 配置

```python
num_steps_per_env = 48
save_interval = 200

algorithm = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.005,
    num_learning_epochs=5,
    num_mini_batches=8,
    learning_rate=3.0e-4,
    schedule="adaptive",
    gamma=0.99,
    lam=0.97,
    desired_kl=0.01,
    max_grad_norm=1.0,
)
```

每次 rollout 为 `0.96 s`、`196,608` transitions；8 个 mini-batch 时每个为 `24,576`。必须用相同 seeds 消融 `24/48/64`，默认保留 48，除非 64 在独立 `TRAINING` validation 的 speed RMSE、fall rate 和 ZMP margin 三项均改善且 wall-clock sample throughput 降低不超过 15%。

## 11. Reward、Termination 与课程

### 11.1 Reward 定义

所有 dense term 先归一化为无量纲量，再乘权重。以下权重是 Isaac Lab `RewardTermCfg.weight`，由 RewardManager 再乘 `step_dt`。第一版只启用表中项目：


| 类别     | term                         |    weight | 定义/尺度                                        |
| -------- | ---------------------------- | --------: | ------------------------------------------------ |
| 任务     | `track_speed_exp`            |    `+2.0` | `exp(-((v_ref-v_robot,s)/0.25)^2)`               |
| 任务     | `lateral_error_l2`           |    `-0.5` | `(e_y/0.30 m)^2`                                 |
| 任务     | `heading_error_l2`           |    `-0.5` | `(e_psi/0.30 rad)^2`                             |
| 稳定     | `zmp_margin_barrier`         |    `-2.0` | `(relu(0.02-d_zmp)/0.02)^2`                      |
| 车辆     | `hitch_height_exp`           |    `+0.5` | `exp(-((H-H_target)/0.02)^2)`，仅双轮接触有效    |
| 先验     | `fat2_prior_exp`             |    `+0.1` | 第 7.3 节，`sigma=0.12 rad`，仅 `fat_valid` 有效 |
| 步态     | `feet_air_time`              |   `+0.10` | threshold`0.4 s`                                 |
| 步态     | `feet_slide`                 |   `-0.10` | 仅接触足                                         |
| 运动质量 | `terrain_normal_velocity_l2` |    `-0.5` | `(v_n/0.25 m/s)^2`                               |
| 运动质量 | `joint_power_l1`             | `-1.0e-4` | `sum(abs(tau*qd))`                               |
| 运动质量 | `processed_action_rate_l2`   |   `-0.01` | `mean(((u_t-u_t-1)/0.05 rad)^2)`                 |
| 运动质量 | `processed_action_jerk_l2`   |  `-0.005` | `mean(((u_t-2u_t-1+u_t-2)/0.03 rad)^2)`          |
| 约束     | `joint_position_limits`      |    `-1.0` | Isaac Lab soft-limit term                        |
| 失败     | `termination`                |  `-200.0` | 仅非 timeout 终止                                |

实现：

```python
def track_speed_exp(env):
    v_s = torch.sum(
        env.scene["robot"].data.root_lin_vel_w * env.path_tangent_w, dim=-1
    )
    return torch.exp(-torch.square((env.command_state.v_ref - v_s) / 0.25))


def lateral_error_l2(env):
    return torch.square(env.path_state.lateral_error / 0.30)


def heading_error_l2(env):
    e = env.path_state.heading_error
    e = torch.atan2(torch.sin(e), torch.cos(e))
    return torch.square(e / 0.30)


def zmp_margin_barrier(env):
    return torch.square(torch.relu(0.02 - env.stability_state.zmp_margin) / 0.02)


def hitch_height_exp(env):
    error = env.rickshaw_state.hitch_height - env.rickshaw_pose_cfg.hitch_height_target
    valid = env.rickshaw_state.two_wheel_contact
    return torch.exp(-torch.square(error / 0.02)) * valid


def processed_action_rate_l2(env):
    delta = env.action_state.target - env.action_state.prev_target
    return torch.mean(torch.square(delta / 0.05), dim=-1)


def processed_action_jerk_l2(env):
    jerk = (
        env.action_state.target
        - 2.0 * env.action_state.prev_target
        + env.action_state.prev_prev_target
    )
    return torch.mean(torch.square(jerk / 0.03), dim=-1)
```

禁用上游 `track_lin_vel_xy_exp`、固定零 yaw-rate tracking、`flat_orientation_l2`、world-frame `lin_vel_z_l2`、raw action-rate、joint acceleration penalty 和单独 joint torque penalty。后两项已由 power、processed rate 和 processed jerk 覆盖。

不为 wheel contact、wheel height、D6 residual、D6 asymmetry、arm saturation、overspeed 或 cart pitch-rate设置额外 dense reward；这些量进入 barrier、termination 和日志。这样不会对同一物理失败重复计分。FAT2 初值 `+0.1` 必须与 `0.0` 做消融；不得高于 ZMP barrier 的绝对权重。

### 11.2 Reward 定标

先在固定 `TRAINING` policy rollout 上保留每个样本的坡度标签，并记录每个未加权 term 的 `p50/p90/p99`。要求总体样本以及 19 个坡度中的每一个，任一 term 加权后的 `p90` 绝对值均不超过该层 speed term 加权 `p90` 的 50%，termination 除外；任一单坡失败都使定标报告失败，不能用总体分位数掩盖。只允许按最严格单坡比例调整表中 weight，不调整 term 的物理尺度；`0.25 m/s`、`0.30 m`、`0.30 rad`、`0.02 m` 和 `0.02 m ZMP margin` 若需修改，必须先修改验收阈值和可行性扫描，再重训全部阶段。

### 11.3 Termination

立即终止：NaN/Inf、非法 body contact、robot-cart collision、车体触地、任一 wheel normal force 低于标定 lift threshold、D6 residual/impulse 超安全值、joint hard limit。以下条件连续 `10` 个 policy steps 后终止：G1 坡面法向 root height `<0.31 m`、torso tilt 超 `theta_max`、rickshaw hitch-height/pitch 超安全包线、`|e_y|` 超 corridor、`|e_psi|` 超航向包线、实际速度超过 `v_ref+overspeed_margin`、arm torque 超连续安全值、ZMP 位于支撑多边形外。

除 `0.31 m` 和持续步数外，硬件相关阈值从 `validate_feasibility.py` 输出并写入 cfg；cfg 中不得存在无来源的默认值。Timeout 不触发 `-200`。

### 11.4 训练前可行性扫描

`validate_feasibility.py` 在 RL 前扫描 19 个坡度、payload、`c_rr`、`acceleration_limit`、`jerk_limit`、terrain friction 和 D6 参数。每个组合检查：

```text
两轮法向力均大于 lift margin
双足摩擦锥可行
ZMP margin >= 0.02 m
arm/leg torque <= 0.7 * actuator limit
D6 force/torque <= 0.7 * configured limit
q_ref 到 joint limit 的余量达标
```

任一条件失败的参数组合不进入训练采样范围。`acceleration_limit` 取全部保留组合可行上界的 `0.8` 倍；`jerk_limit` 在该加速度下扫描，并取不触发 D6 impulse threshold 的最大值的 `0.8` 倍。输出 `feasibility_envelope.yaml`，训练 cfg 必须加载并断言采样范围包含于该文件。

### 11.5 环境课程


| 训练分布   | 配置 |
| ---------- | ---- |
| `TRAINING` | 使用同一车辆和 D6 拓扑，覆盖 `-0.08` 到 `+0.10` 的 19 个坡度；物理、执行器、延迟和观测参数由统一 runtime randomization 配置决定。 |

训练全过程不设阶段切换或验证 gate；保存、固定种子评估和 lineage 记录均以 200 iterations 为周期。

26 m 地块不使用上游 `terrain_levels_vel`。每个 episode 计算：

\[
s=\operatorname{mean}\exp[-((v_{ref}-v_s)/0.25)^2]
\]

Timeout 且 `s>=0.8`、无安全计数器触发时升一级；提前终止或 `s<0.5` 时降一级；其余保持。更新 level 后重新计算 slope frame，再执行闭链 reset。

## 12. Task 注册与命令

注册训练和 Play 两个配置；二者使用同一资产、动力学和 observation schema：

```python
gym.register(
    id="Isaac-G1-Rickshaw-Directional-Slope-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity:G1RickshawDirectionalSlopeEnvCfg",
        "rsl_rl_cfg_entry_point": "g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.agents:G1RickshawTeacherPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Rickshaw-Directional-Slope-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity:G1RickshawDirectionalSlopePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.agents:G1RickshawStudentPPORunnerCfg",
    },
)
```

```bash
PYTHON=/root/miniconda3/envs/env_isaaclab/bin/python
"$PYTHON" scripts/inspect_assets.py --num_envs 1 --headless \
  --output outputs/validation/asset_inspection.json
"$PYTHON" scripts/solve_reset_poses.py --headless --steps 1000 \
  --output config/reset_poses.yaml \
  --alignment-output outputs/validation/reset_alignment_1000.json
"$PYTHON" scripts/train_teacher.py --task Isaac-G1-Rickshaw-Directional-Slope-v0
"$PYTHON" scripts/calibrate_rewards.py --checkpoint <fixed-C1-policy-checkpoint>
"$PYTHON" scripts/train_context.py --teacher <checkpoint> --reward-calibration-report <reward_calibration.<sha>.json>
"$PYTHON" scripts/finetune_student.py --teacher <checkpoint> --context <checkpoint>
"$PYTHON" scripts/play_student.py --task Isaac-G1-Rickshaw-Directional-Slope-Play-v0 --checkpoint <checkpoint>
```

当前工作区没有通过上述固定点认证：正式 alignment 与 feasibility report
缺失，现有 authority 和 dynamics report 已过期。完整重签顺序及关键产物见
`RUN_COMMANDS.md`；运行时 gate 通过前不得开始训练。

Play 配置只关闭 observation corruption、push 和 curriculum；保留 command limiter、物理 `c_rr`、action filter、61 帧 history、rickshaw pose target 和全部安全检查。导出包只包含 observation scale、TCN、student actor、action filter、关节顺序和安全参数；teacher、critic、privileged group 和 auxiliary heads不得导出。

## 13. 验收

1. USD 的 mass/inertia/joint axis、Dex/G1 关节数量和固定顺序与源资产一致；所有 checkpoint hash 可复现。
2. 19 个训练 terrain gradient、法向和 origin 精确；闭链 IK 满足足底、grasp、joint margin、静态 ZMP margin 和无碰撞约束。
3. Command 单元测试覆盖上升、下降、换向和抵达目标；每一步满足 `|a_ref|<=acceleration_limit`、`|delta(a_ref)/dt|<=jerk_limit`，reward/observation 只使用 `v_ref`。
4. Action filter DC gain 为 1、4 Hz gain 为 `1/sqrt(2)`；reset 后第一步 processed target 等于 `q_ref`。
5. 断开 D6 的车辆滑行测试使用完整 `40.04 kg` Rickshaw articulation，通过 world-X prismatic joint 隔离 X 向 PhysX 力响应；水平车辆法向载荷取 `m_cart*g`，`c_rr*N` 分配到两个真实 wheel body center。改变 `c_rr` 必须改变实测 X 减速度；平滑区外的阻力幅值误差在配置容差内，且无双重滚阻项。该夹具不依赖车身/地面摩擦产生减速度。
6. `H -> alpha -> cart root height -> H` 几何往返误差小于容差；reset 后双轮接触、无 penetration，D6 ramp 无冲量尖峰。
7. 平地静态、平地匀速、上坡加速和下坡制动四种工况中，解析 `T_s/T_n` 与由整车质心动量平衡独立重构的交互力同号；跳过滤波写入后的 5 步，再以 25 policy-step 窗口报告平均相对误差，并使用独立的 `12 N` 归一化下限，超限则不得训练。人为初速度/外力工况隔离 RL overspeed termination，但每一步解析力和动量测量有效性仍必须通过；incoming-joint wrench 只用于保守的 D6 residual/impulse 安全代理。
8. TCN 输入严格为 `[N,61,96]`，感受野单元测试为 61；扰动 history 之外或 future frame 不改变 `z_hat`。
9. S0/S1/S2 在固定 19 坡度、固定 randomization seeds 上报告相同指标；除总体和逐坡结果外，必须完整报告 standing/accelerating/cruising/decelerating 和 19 坡度×命令阶段。S1 以 action KL 和 return 选模，S2 的 `TRAINING` return 不低于 S1，latent MSE 不作为通过条件。
10. `z_hat` 置零或跨环境 shuffle 后，`TRAINING` return 应显著下降；若无下降，说明 context 被 actor 忽略，应缩减或删除该分支，而不是扩大 latent。
11. 完成 `FAT2 weight=0/0.1`、rollout `24/48/64` 和 latent dim `8/16/24` 三组消融；矩阵共 8 项、对应 6 个唯一训练配置。物理滚阻在全部配置中始终开启并保留真实轮心切向阻力实现，不再作为消融因素；只保留改善独立验证集而非训练 return 的配置。

必须记录：speed RMSE、fall rate、termination cause histogram、overspeed rate、`e_y/e_psi` RMS/max、rickshaw pitch/hitch-height error、双轮接触率和 normal-force percentile、foot slip、processed action rate/jerk、power、D6 residual/force/torque/asymmetry、解析 `T_s/T_n` 瞬时误差、FAT2 窗口一致率及窗口误差、ZMP margin、arm/leg torque margin、teacher-student action KL、`z_hat` zero/shuffle return drop 和 curriculum level 分布；上述样本指标同时按命令阶段、cross-case 及其与坡度的组合分层。

## 14. 设计依据

- [Thor/FAT2](https://arxiv.org/abs/2510.26280v3)：FAT2 基于含手部外力的 ZMP/力矩平衡生成 torso tilt reward。本任务先由车辆绕轮轴平衡求 `T_n`，再与 `T_s` 一起形成手部 wrench；不采用只看水平拉力的简化式。
- [RMA](https://arxiv.org/abs/2107.04034)：privileged teacher 与 history adaptation 的分工依据。本文档限制 teacher 只编码独立 extrinsics，并将 action distillation 设为主目标。
- [TCN](https://arxiv.org/abs/1803.01271)：使用确定感受野的 causal dilated convolution。61 帧与 kernel/dilation 精确匹配，不保留未被网络读取的历史帧。
- [Asymmetric Actor-Critic](https://arxiv.org/abs/1710.06542)：privileged state 仅供训练 critic，部署 actor 不读取仿真真值。

最终部署信息路径只有：`sensor/estimator -> current + 61-frame history -> TCN z_hat -> actor -> 4 Hz action filter`。任何新增输入必须说明其传感器来源；任何新增 history、latent 或 reward 必须先证明不与现有通路表达同一信息。
