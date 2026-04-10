# XArm Toolkit

所有分析和交流都用中文。任何操作都要称呼我为iff
xarm/ 不能改任何东西，这是SDK
我这是6自由度的xarm6，别给我瞎说xarm7，记住了，是xarm6

## 项目目标
重构 reference/ 中的代码为一套生产级 XArm 机械臂工具包：
不要改动reference/的代码，有需要可以复制出来
- **Env**: 真机环境（含双 RealSense 相机 observation），兼容力控/位控切换
- **Teleop**: 3D SpaceMouse 遥操作，用于数据采集
- **Collect**: 数据采集流程，env + teleop → Zarr 存储
- **Deploy**: Pi0.5 等 VLA 模型推理 + 真机部署与评测

## 开发约束
- **机器 A**（本机）仅有代码，无硬件 → 修改代码后需 mock 数据做本地验证
- **机器 B** 有 XArm + 相机 + 力传感器 → 真机测试结果由我反馈
- 不要自行运行依赖真实硬件的脚本，会直接报错

## 硬件参数
| 组件 | 型号 | 备注 |
|------|------|------|
| 机械臂 | XArm 6 (6-DOF) | IP: 192.168.31.232 |
| 夹爪 | 内置 | 行程 0-840 |
| 力传感器 | 内置 6 轴 FT | 支持阻抗控制 |
| 臂上相机 | Intel D435i | serial: 327122075644 |
| 固定相机 | Intel L515 | serial: f1271506 |
| 遥操作 | 3DConnexion SpaceMouse | 6D 输入 + 2 按钮控制夹爪 |

## 代码结构
```
iffyuan-XArm-Toolkit/
├── reference/              # 旧代码，只读参考（勿改）
├── xarm/                   # xarm-python-sdk（勿改）
├── xarm_toolkit/           # 主 Python 包
│   ├── __init__.py
│   ├── env/                # 机械臂环境 + 相机 observation
│   │   ├── xarm_env.py         # 统一环境接口（力控/位控可选）
│   │   └── realsense_env.py    # RealSense D435i + L515 封装
│   ├── teleop/             # 遥操作
│   │   ├── spacemouse_expert.py # SERL SpaceMouseExpert（原文件，勿改）
│   │   └── spacemouse.py       # SpacemouseAgent — 缩放/重映射/夹爪切换
│   ├── collect/            # 数据采集
│   │   └── collector.py        # Collector — env + cameras + teleop → Zarr
│   ├── deploy/             # VLA 模型部署与评测（待开发）
│   └── utils/
│       └── logger.py           # 统一日志
├── configs/                # 配置文件（YAML）
│   ├── hardware.yaml           # 机械臂IP、相机serial、采集参数
│   └── tasks/                  # 任务配置（各任务的初始位姿偏移等）
│       ├── default.yaml
│       ├── plug.yaml
│       └── stamp.yaml
├── scripts/                # 入口脚本
│   ├── teleop_demo.py          # SpaceMouse 遥操作（不保存数据）
│   ├── collect_data.py         # 数据采集入口
│   └── web_control_demo.py     # Web 键盘控制 Demo
├── tests/                  # 测试（mock 硬件）
├── pyproject.toml
└── requirements.txt
```

## 关键接口
```python
# Env
env = XArmEnv(addr, use_force, action_mode, initial_gripper_position)
obs = env.reset()           # → dict: cart_pos, servo_angle, ext_force, goal_pos, gripper_position
obs = env.step(action, gripper_action, speed)

# Camera
cam = RealsenseEnv(serial, mode="rgbd")
obs = cam.step()            # → dict: rgbd, intrinsic_matrix, depth_scale

# Teleop
agent = SpacemouseAgent(config=SpacemouseConfig(...))
action, gripper = agent.act(obs)  # → (6D delta, int gripper_pos)

# Collect
collector = Collector(env, cam_arm, cam_fix, agent, dataset_path, task_config, num_episodes)
collector.run()             # 阻塞式采集循环
```

## 数据采集流程
```
SpaceMouse → SpacemouseAgent(缩放/重映射) → XArmEnv(delta_eef) → 机械臂执行
                                                  ↓
                               并行采集: FT力 + 关节角 + 末端位姿 + 双相机 RGBD(320×240)
                                                  ↓
                                    Collector → Zarr 数据集 (Blosc 压缩)
```

Zarr 存储格式 (默认 rgbd 模式，rgb 和 depth 分开存):
```
dataset.zarr/
├── data/
│   ├── rgb_arm        (N, 3, 240, 320) uint8
│   ├── rgb_fix        (N, 3, 240, 320) uint8
│   ├── depth_arm      (N, 1, 240, 320) uint16  — rgbd/pcd 模式
│   ├── depth_fix      (N, 1, 240, 320) uint16  — rgbd/pcd 模式
│   ├── pos            (N, 6) float32 — goal_pos
│   ├── force          (N, 6) float32 — ext_force
│   ├── action         (N, 6) float32 — SpaceMouse delta
│   ├── gripper_state  (N, 1) float32 — 0/1
│   ├── gripper_action (N, 1) float32 — 0/1
│   └── episode        (N,) uint16
└── meta/
    └── episode_ends   (M,) uint32
```

## 安装
```bash
pip install -e .          # 或 pip install -r requirements.txt
pip install -e ".[dev]"   # 额外安装 pytest / ruff（开发用）
```

## 依赖
| 包 | 用途 |
|---|---|
| numpy | 数值计算 |
| opencv-python | 图像处理 |
| open3d | 点云 & RGBD |
| pytransform3d | 坐标变换 |
| zarr | 数据采集存储（Blosc 压缩） |
| numcodecs | Zarr 压缩编解码 |
| h5py | HDF5 读写 |
| pyrealsense2 | RealSense 相机驱动 |
| pyyaml | 配置文件解析 |
| pyspacemouse | SpaceMouse 输入 |
| hidapi | USB HID（SpaceMouse 底层） |

> Deploy 相关依赖（torch / transformers 等）不在默认依赖中，按需另行安装。
