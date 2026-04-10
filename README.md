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

## 安装

```bash
git clone https://github.com/pickxiguapi/iffyuan-XArm-Toolkit && cd iffyuan-XArm-Toolkit
pip install -e .
```

## ⚠️ 运动速度 (`speed`) 参数说明

`speed` 是机械臂笛卡尔空间运动速度（单位 mm/s），**直接决定机械臂移动快慢，务必注意安全**。

| 速度 (mm/s) | 级别 | 适用场景 |
|:-----------:|:----:|---------|
| **50~100** | 🐢 慢速 | 初次调试、精细操作、接近物体 |
| **200~400** | 🚶 中速 | **日常遥操作、数据采集（推荐默认 400）** |
| **500~800** | 🏃 快速 | 空间大范围移动、复位 |
| **1000+** | ⚡ 极速 | 仅限空旷环境，**新手慎用** |

各脚本 `--speed` 默认值均为 **400 mm/s**，可通过命令行覆盖。

## 真机验证流程

按以下顺序逐步验证，每一步通过了再往下走。

---

### Step 1: 验证相机

确认双 RealSense 相机能正常出图。

```bash
# 双相机预览（按 q 退出）
python scripts/test_cameras.py

# 只测臂上相机
python scripts/test_cameras.py --arm-only

# 只测固定相机
python scripts/test_cameras.py --fix-only
```

**预期:**
- 看到两个相机窗口，画面流畅
- 日志显示 shape、dtype、min/max 等信息
- FPS ≥ 25

---

### Step 2: 验证机械臂环境

逐步测试: 连接 → 读状态 → 复位 → 小幅移动 → 夹爪。每步暂停等你确认，确保安全。

```bash
# 完整测试（交互式，每步需确认）
python scripts/test_env.py

# 带力传感器
python scripts/test_env.py --force

```

**预期:**
- 笛卡尔位姿、关节角、夹爪位置正常打印
- 复位后到达 Home 位姿
- 小幅移动 X±10mm 后回到原位
- 夹爪开合正常

---

### Step 3: Web 键盘遥操作

网页界面 + 双相机画面 + 键盘控制机械臂。

```bash
python scripts/web_control_demo.py          # 真机模式 (默认 speed=400)
python scripts/web_control_demo.py --mock   # mock 模式调试 UI

# 调整速度
python scripts/web_control_demo.py --speed 100                  # 慢速精细操作
python scripts/web_control_demo.py --step-pos 2 --speed 600     # 大步快速移动
```

打开 `http://<机器IP>:8080`

| 按键 | 功能 | 按键 | 功能 |
|------|------|------|------|
| W/S | Y 前/后 | I/K | Roll +/- |
| A/D | X 左/右 | J/L | Pitch +/- |
| Q/E | Z 上/下 | U/O | Yaw +/- |
| Space | 切换夹爪 | R | 复位 Home |
| Esc | 急停 | | |

---

### Step 4: SpaceMouse 遥操作

3D SpaceMouse 控制机械臂末端。

```bash
# 基础遥操作
python scripts/teleop_demo.py

# 启用力传感器
python scripts/teleop_demo.py --force

# 调整灵敏度
python scripts/teleop_demo.py --trans-scale 3 --rot-scale 0.003

# 自定义频率和速度
python scripts/teleop_demo.py --hz 50 --speed 400
```

**终端键盘:**
- `q` / `Ctrl+C` — 退出
- `r` — 复位
- `o` — 开夹爪
- `c` — 关夹爪
- SpaceMouse 左键 — 切换夹爪开/关

---

### Step 5: 数据采集

SpaceMouse 遥操作 + 双相机 + 力传感器 → Zarr 数据集。

```bash
# 默认采集 rgbd（RGB + depth 分开存储，默认不采集力）
python scripts/collect_data.py --dataset datasets/demo.zarr

# 只采集 RGB（不需要 depth）
python scripts/collect_data.py --cam-mode rgb --dataset datasets/rgb_only.zarr

# 指定任务 + episode 数量
python scripts/collect_data.py --task plug --episodes 10 --dataset datasets/plug.zarr

# 需要力传感器数据时手动开启
python scripts/collect_data.py --force --dataset datasets/force_demo.zarr

# 保存视频方便回看（每个 episode 生成两个 MP4: arm + fix）
python scripts/collect_data.py --save-video --dataset datasets/demo.zarr
```

| 操作 | 说明 |
|------|------|
| Space | 开始录制当前 episode |
| Enter | 结束当前 episode |
| Ctrl+C | 中止采集 |

---

### Step 6: VLA 部署

Server/Client 架构，GPU 机器跑推理，机器人端执行动作。

**GPU 机器（启动 Server）:**
```bash
pip install websockets msgpack
# 还需要 openpi 环境（torch + transformers 等）

# 自动下载 checkpoint
python scripts/start_server.py --config pi05_droid --port 10093

# 指定本地 checkpoint
python scripts/start_server.py --config pi05_droid --ckpt-dir /data/ckpts/pi05_droid
```

**机器人端（启动 Client）:**
```bash
python scripts/deploy_vla.py \
    --server-host <GPU_IP> \
    --server-port 10093 \
    --instruction "pick up the cup" \
    --max-steps 5
```

> ⚠️ **安全提醒:** 先用 `--max-steps 5` 小步测试，手放在急停按钮上。

---

## 核心概念速查

### 三种控制模式

| 模式 | `action_mode` | action 含义 | 典型场景 |
|------|---------------|------------|---------|
| 增量末端 | `"delta_eef"` | `[dx,dy,dz,dr,dp,dy]` 相对位移 | SpaceMouse 遥操作/VLA使用 |
| 绝对末端 | `"absolute_eef"` | `[x,y,z,r,p,y]` 目标位姿 | VLA 使用 |
| 绝对关节 | `"absolute_joint"` | `[j1..j6]` 目标关节角 (rad) | VLA 关节部署 |

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
│   ├── rgb_arm        # (N, 3, 240, 320) uint8   — 臂上相机 RGB
│   ├── rgb_fix        # (N, 3, 240, 320) uint8   — 固定相机 RGB
│   ├── depth_arm      # (N, 1, 240, 320) uint16  — 臂上相机 depth (rgbd/pcd 模式)
│   ├── depth_fix      # (N, 1, 240, 320) uint16  — 固定相机 depth (rgbd/pcd 模式)
│   ├── pos            # (N, 6) float32 — 末端位姿
│   ├── force          # (N, 6) float32 — 力传感器
│   ├── action         # (N, 6) float32 — SpaceMouse 动作
│   ├── gripper_state  # (N, 1) float32 — 夹爪状态 (0/1)
│   ├── gripper_action # (N, 1) float32 — 夹爪目标动作 (0/1)
│   └── episode        # (N,) uint16 — episode 编号
└── meta/
    └── episode_ends   # (M,) uint32 — 各 episode 的结束索引
```

> `--cam-mode rgb` 时不创建 `depth_arm` / `depth_fix`。

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
│   ├── deploy/
│   │   ├── msgpack_numpy.py    # ndarray ↔ msgpack 序列化
│   │   ├── server.py           # VLAServer — WebSocket 推理服务
│   │   └── client.py           # VLAClient — 同步客户端
│   └── utils/
│       └── logger.py           # 统一日志
├── configs/
│   ├── hardware.yaml           # 硬件配置（IP、serial、采集参数）
│   └── tasks/                  # 任务配置
│       ├── default.yaml
│       ├── plug.yaml
│       └── stamp.yaml
├── scripts/
│   ├── test_cameras.py         # 相机验证
│   ├── test_env.py             # 机械臂验证
│   ├── web_control_demo.py     # Web 键盘遥操作
│   ├── teleop_demo.py          # SpaceMouse 遥操作
│   ├── collect_data.py         # 数据采集入口
│   ├── deploy_vla.py           # VLA 机器人端
│   └── start_server.py         # VLA GPU 端
├── xarm/                   # xarm-python-sdk（勿改）
├── tests/                  # 测试（mock 硬件）
├── pyproject.toml
└── requirements.txt
```

