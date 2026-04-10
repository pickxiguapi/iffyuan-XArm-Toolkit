# XArm Toolkit

ER Group XArm 6 机械臂工具包，包含真机控制、相机采集、遥操作、数据采集、VLA 部署。

## 硬件一览

| 组件 | 型号 | 关键参数 |
|------|------|---------|
| 机械臂 | XArm 6 (6-DOF) | IP: `192.168.31.232` |
| 夹爪 | 内置 | 行程 0（闭合）~ 840（全开） |
| 力传感器 | 内置 6 轴 FT | 支持阻抗控制 |
| 臂上相机 | Intel RealSense D435i | serial: `327122075644` |
| 固定相机 | Intel RealSense L515 | serial: `f1271506` |
| 遥操作 | 3DConnexion SpaceMouse | 6D 输入 + 左右按钮控制夹爪 |

## 快速开始

### 1. 安装

```bash
git clone https://github.com/pickxiguapi/iffyuan-XArm-Toolkit && cd iffyuan-XArm-Toolkit
pip install -e .
```

### 2. 最小示例：连接机械臂并复位

```python
from xarm_toolkit.env import XArmEnv

env = XArmEnv(
    addr="192.168.31.232",
    use_force=False,
    action_mode="delta_eef",
    initial_gripper_position=840,
)

obs = env.reset(close_gripper=True)
print(obs["cart_pos"])     # [470, 0, 530, π, 0, -π/2]
print(obs["servo_angle"])  # 6 个关节角 (rad)
```

### 3. 最小示例：打开相机看画面

```python
import cv2, numpy as np
from xarm_toolkit.env import RealsenseEnv

cam = RealsenseEnv(serial="327122075644", mode="rgb")
obs = cam.step()
rgb = np.asarray(obs["rgb"])  # (480, 640, 3) uint8
cv2.imshow("arm", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
```

### 4. SpaceMouse 遥操作（不保存数据）

```bash
python scripts/teleop_demo.py                    # 默认参数
python scripts/teleop_demo.py --force             # 启用力传感器
python scripts/teleop_demo.py --trans-scale 3     # 调低灵敏度
```

| 操作 | 说明 |
|------|------|
| SpaceMouse 6D | 控制机械臂末端位姿 |
| 左键 | 切换夹爪开/关 |
| 键盘 q | 退出 |
| 键盘 r | 复位 |
| 键盘 o/c | 手动开/关夹爪 |

### 5. 数据采集

```bash
# 默认任务，采集 3 个 episode
python scripts/collect_data.py --dataset datasets/demo.zarr

# 指定任务 + episode 数量
python scripts/collect_data.py --task plug --episodes 10 --dataset datasets/plug.zarr

# 无力传感器
python scripts/collect_data.py --no-force --dataset datasets/test.zarr
```

| 操作 | 说明 |
|------|------|
| Space | 开始录制当前 episode |
| Enter | 结束当前 episode |
| Ctrl+C | 中止采集 |
| 左键/右键 | 等待阶段调整夹爪；录制阶段开/关夹爪 |

任务配置文件在 `configs/tasks/` 目录下（YAML），新任务只需新建一个文件：
```yaml
# configs/tasks/plug.yaml
start_bias: [0, 0, -200]
random_bias:
  x: [-50, 70]
  y: [-20, 10]
gripper_always_closed: false
```

### 6. Web 键盘控制 Demo

```bash
python scripts/web_control_demo.py          # 真机模式
python scripts/web_control_demo.py --mock   # mock 模式调试 UI
```

打开 `http://localhost:8080`：

| 按键 | 功能 | 按键 | 功能 |
|------|------|------|------|
| W/S | Y 前/后 | I/K | Roll +/- |
| A/D | X 左/右 | J/L | Pitch +/- |
| Q/E | Z 上/下 | U/O | Yaw +/- |
| Space | 切换夹爪 | R | 复位 Home |
| Esc | 急停 | | |

## 核心概念速查

### 三种控制模式

| 模式 | `action_mode` | action 含义 | 典型场景 |
|------|---------------|------------|---------|
| 增量末端 | `"delta_eef"` | `[dx,dy,dz,dr,dp,dy]` 相对位移 | SpaceMouse 遥操作 |
| 绝对末端 | `"absolute_eef"` | `[x,y,z,r,p,y]` 目标位姿 | VLA 笛卡尔部署 |
| 绝对关节 | `"absolute_joint"` | `[j1..j6]` 目标关节角 (rad) | Pi0 关节部署 |

### Observation 字典

`env.reset()` 和 `env.step()` 返回的 obs：

| 字段 | 类型 | 说明 |
|------|------|------|
| `cart_pos` | `np.ndarray (6,)` | 末端位姿 `[x,y,z,roll,pitch,yaw]`，xyz 单位 mm，角度 rad |
| `servo_angle` | `np.ndarray (6,)` | 6 个关节角 (rad) |
| `ext_force` | `np.ndarray (6,)` 或 `None` | 外部力/力矩（需 `use_force=True`） |
| `raw_force` | `np.ndarray (6,)` 或 `None` | 原始力传感器读数 |
| `goal_pos` | `np.ndarray (6,)` | 当前目标位姿 |
| `gripper_position` | `np.ndarray` | 夹爪位置（0=闭合，840=全开） |

### SpaceMouse 灵敏度参数

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `translation_scale` | 5.0 | 平移灵敏度（mm/tick）|
| `rotation_scale` | 0.004 | 旋转灵敏度（rad/tick）|
| `deadzone` | 0.0 | 死区，过滤手抖（建议 0.05~0.15）|

### Zarr 数据集格式

```
dataset.zarr/
├── data/
│   ├── rgb_arm        # (N, 3, 240, 320) uint8  Blosc 压缩
│   ├── rgb_fix        # (N, 3, 240, 320) uint8  Blosc 压缩
│   ├── pos            # (N, 6) float32 — 末端位姿
│   ├── force          # (N, 6) float32 — 力传感器
│   ├── action         # (N, 6) float32 — SpaceMouse 动作
│   ├── gripper_state  # (N, 1) float32 — 夹爪状态 (0/1)
│   ├── gripper_action # (N, 1) float32 — 夹爪目标动作 (0/1)
│   └── episode        # (N,) uint16 — episode 编号
└── meta/
    └── episode_ends   # (M,) uint32 — 各 episode 的结束索引
```

## 项目结构

```
iffyuan-XArm-Toolkit/
├── xarm_toolkit/           # 主 Python 包
│   ├── env/
│   │   ├── xarm_env.py         # 统一环境接口（力控/位控可选）
│   │   └── realsense_env.py    # RealSense D435i + L515 封装
│   ├── teleop/
│   │   ├── spacemouse_expert.py # SERL SpaceMouseExpert（原文件，勿改）
│   │   └── spacemouse.py       # SpacemouseAgent — 缩放/重映射/夹爪切换
│   ├── collect/
│   │   └── collector.py        # Collector — env + cameras + teleop → Zarr
│   ├── deploy/                 # VLA 部署（开发中）
│   └── utils/
│       └── logger.py           # 统一日志
├── configs/
│   ├── hardware.yaml           # 硬件配置（IP、serial、采集参数）
│   └── tasks/                  # 任务配置
│       ├── default.yaml
│       ├── plug.yaml
│       └── stamp.yaml
├── scripts/
│   ├── teleop_demo.py          # SpaceMouse 遥操作（不保存）
│   ├── collect_data.py         # 数据采集入口
│   └── web_control_demo.py     # Web 键盘控制 Demo
├── reference/              # 旧代码参考（只读，勿改）
├── xarm/                   # xarm-python-sdk（勿改）
├── tests/                  # 测试（mock 硬件）
├── pyproject.toml
└── requirements.txt
```
