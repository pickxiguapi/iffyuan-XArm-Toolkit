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
python scripts/web_control_demo.py          # 真机模式
python scripts/web_control_demo.py --mock   # mock 模式调试 UI
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
python scripts/teleop_demo.py --hz 50 --speed 500
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
├── reference/              # 旧代码参考（只读，勿改）
├── xarm/                   # xarm-python-sdk（勿改）
├── tests/                  # 测试（mock 硬件）
├── pyproject.toml
└── requirements.txt
```

## 运行测试

```bash
python -m pytest tests/ -v
```
