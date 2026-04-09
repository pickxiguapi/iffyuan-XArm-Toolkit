import os
import sys
import threading
import termios
import tty
import select
import cv2

os.environ["DISPLAY"] = ":0"

import time

import cv2
import h5py
import numpy as np
import zarr
from termcolor import cprint

from CleanDiffuser.image_codecs import jpeg
from env.realsense_env import RealsenseEnv
from env.spacemouse_slow import SpacemouseAgent
from env.xarm_env_force import XarmEnv

NUM_EPISODES = 3
DATASET_PATH = "datasets/demo_fjh.zarr"

WARMUP_TIME = 1

class KeyboardListener:
    def __init__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        self.key_pressed = None
        self.listening = True
        
    def start_listening(self):
        tty.setraw(sys.stdin.fileno())
        
    def stop_listening(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        self.listening = False
        
    def get_key(self):
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            key = sys.stdin.read(1)
            return key.lower()
        return None

if __name__ == "__main__":
    env = XarmEnv()
    rs_arm_env = RealsenseEnv(serial="327122075644")
    rs_fix_env = RealsenseEnv(serial="f1271506")
    agent = SpacemouseAgent()

    obs = env.reset()
    obs_rs_arm = rs_arm_env.reset()
    obs_rs_fix = rs_fix_env.reset()
    time.sleep(1)


    cprint("Initializing ...", "cyan")
    try:
        goal_gripper_action = 0

        if "plug" in DATASET_PATH or "insert" in DATASET_PATH:
            x_bias = np.random.uniform(-50, 70)
            y_bias = np.random.uniform(-20, 10)
            z_bias = -200
        elif "whiteboard" in DATASET_PATH:
            x_bias = -230
            y_bias = -40
            z_bias = -280
        elif "stamp" in DATASET_PATH:
            x_bias = 0
            y_bias = 0
            z_bias = -270
        elif "button" in DATASET_PATH:
            x_bias = 0
            y_bias = 0
            z_bias = 0
        elif "vase" in DATASET_PATH:
            x_bias = -117
            y_bias = 53
            z_bias = 0
        elif "push" in DATASET_PATH:
            x_bias = -120
            y_bias = 0
            z_bias = -170

        else:
            x_bias = 0
            y_bias = 0
            z_bias = 0
            print(f"Warning: Unknown dataset: {DATASET_PATH}, using default initialization")
        
        obs = env.step([x_bias, y_bias, z_bias, 0, 0, 0], goal_gripper_action, speed=100)
        obs_rs_arm = rs_arm_env.step()
        obs_rs_fix = rs_fix_env.step()
        cprint(f"Initialization done! Initial position: {obs['goal_pos'][:3]}", "green")
    except Exception as e:
        cprint(f"{e}", "red")
    cprint(f"Warming up... ({WARMUP_TIME}s)", "green")
    time.sleep(WARMUP_TIME)



    
    keyboard_listener = KeyboardListener()


    dataset_path = DATASET_PATH
    dataset_exists = os.path.exists(dataset_path)
    
    if dataset_exists:
        cprint(f"Found existing dataset: {dataset_path}", "green")

        dataset = zarr.open(dataset_path, "a")
        dataset_data = dataset["data"]
        dataset_meta = dataset["meta"]
        
        if "episode" in dataset_data:
            existing_episodes = np.unique(dataset_data["episode"][:])
            start_episode = len(existing_episodes)
            cprint(f"Found {start_episode} episodes, starting from episode {start_episode}", "cyan")
        else:
            start_episode = 0
            cprint("Dataset exists but no episode data, starting from episode 0", "cyan")
    else:
        cprint(f"Creating new dataset: {dataset_path}", "green")

        dataset = zarr.open(dataset_path, "w")
        dataset_data = dataset.create_group("data")
        dataset_meta = dataset.create_group("meta")
        start_episode = 0
 

    if not dataset_exists:
        dataset_data.require_dataset(
            "rgb_arm",
            shape=(0, 3, 240, 320),
            dtype=np.uint8,
            chunks=(1, 3, 240, 320),
            compressor=jpeg((1, 3, 240, 320), quality=90),
        )
        dataset_data.require_dataset(
            "rgb_fix",
            shape=(0, 3, 240, 320),
            dtype=np.uint8,
            chunks=(1, 3, 240, 320),
            compressor=jpeg((1, 3, 240, 320), quality=90),
        )
        # dataset_data.require_dataset(
        #     "depth",
        #     shape=(0, 1, 240, 320),
        #     dtype=np.uint16,
        #     chunks=(1, 1, 240, 320),
        #     compressor=jpeg((1, 1, 240, 320), quality=90),
        # )
        dataset_data.require_dataset("pos", shape=(0, 6), dtype=np.float32)  # eef position
        dataset_data.require_dataset("force", shape=(0, 6), dtype=np.float32)  # force sensor data (using as ext_force)
        dataset_data.require_dataset("action", shape=(0, 6), dtype=np.float32)  # 3dmouse action
        dataset_data.require_dataset("gripper_state", shape=(0, 1), dtype=np.float32)  # gripper state (binary)
        dataset_data.require_dataset("gripper_action", shape=(0, 1), dtype=np.float32)  # gripper action (binary)
        dataset_data.require_dataset("episode", shape=(0,), dtype=np.uint16)

    for episode in range(NUM_EPISODES):
        current_episode = start_episode + episode
        cprint(
            f"\rPreparing to record Episode {current_episode} ({episode + 1}/{NUM_EPISODES} new episodes)",
            "yellow",
        )
        cprint("\rPress space to start recording, press enter to end recording", "cyan")
        

        if episode == 0:
            keyboard_listener.start_listening()
        
        # wait for space key to start recording, you can also adjust the gripper here
        goal_gripper_action = 0
        while True:
            
            action, buttons = agent.act()
            if buttons[0]:
                goal_gripper_action += 40
                if goal_gripper_action > 840:
                    goal_gripper_action = 840
            elif buttons[1]:
                goal_gripper_action -= 40
                if goal_gripper_action < 0:
                    goal_gripper_action = 0
            action = [0, 0, 0, 0, 0, 0]

            obs = env.step(action, goal_gripper_action, speed=100)
            obs_rs_arm = rs_arm_env.step()
            obs_rs_fix = rs_fix_env.step()
            
            key = keyboard_listener.get_key()
            if key == ' ':  # space key
                break
            elif key == '\x03':  # Ctrl+C
                cprint("Recording cancelled", "red")
                keyboard_listener.stop_listening()
                sys.exit(0)
            time.sleep(0.1)
        
        # 在开始采集前调零力传感器
        cprint("Zeroing force sensor...", "yellow")
        if env.reset_force_sensor_zero():
            cprint("Force sensor zeroed successfully!", "green")
        else:
            cprint("Warning: Force sensor zero failed, but continuing...", "yellow")
        time.sleep(0.2)  # 等待调零稳定
        
        cprint(
            f"\rEpisode {current_episode} recording started...",
            "green",
        )
        cprint("\rRecording... press enter to end recording", "cyan")
        
        buffer = {
            "rgb_arm": [],
            "rgb_fix": [],
            "pos": [],
            "force": [],
            "action": [],
            "gripper_state": [],
            "gripper_action": [],
            "episode": [],

        }

        goal_gripper_action = 0

        episode_steps = 0
        start_time = time.time()
        last_time = time.time()

        try:
            while True:
                # check keyboard input
                key = keyboard_listener.get_key()
                if key == '\r':  # enter key
                    cprint("\rEpisode recording ended", "red")
                    time.sleep(1)
                    break
                elif key == '\x03':  # Ctrl+C
                    cprint("\rRecording cancelled", "red")
                    keyboard_listener.stop_listening()
                    sys.exit(0)

                action, buttons = agent.act()
                if buttons[0]:
                    goal_gripper_action = 840
                elif buttons[1]:
                    goal_gripper_action = 0

                obs_np = {
                    "rgb_arm": np.asarray(obs_rs_arm["im_rgbd"].color),  # (480, 640, 3) uint8
                    "rgb_fix": np.asarray(obs_rs_fix["im_rgbd"].color),  # (480, 640, 3) uint8
                    # "depth": np.asarray(obs["im_rgbd"].depth),  # (480, 640, 1) uint16
                    # "intrinsic_matrix": obs["intrinsic_matrix"],  # (3, 3) float64
                    # "depth_scale": np.asarray(obs["depth_scale"]),  # (,) float64
                    # "servo_angle": obs["servo_angle"],  # (6,) float64
                    # "cart_pos": obs["cart_pos"],  # (6,) float64
                    "goal_pos": obs["goal_pos"],  # (6,) float64
                    "force": obs["ext_force"],  # (6,) float64
                    "gripper_position": obs["gripper_position"],  # (1,) float64
                    "action": action,  # (6,) float64
                }

                rgb_arm = cv2.resize(obs_np["rgb_arm"], (320, 240))
                rgb_fix = cv2.resize(obs_np["rgb_fix"], (320, 240))
                # depth = cv2.resize(obs_np["image_depth"], (320, 240))
                buffer["rgb_arm"].append(rgb_arm.transpose(2, 0, 1)[None])
                buffer["rgb_fix"].append(rgb_fix.transpose(2, 0, 1)[None])
                # buffer["depth"].append(depth[None, None])
                buffer["pos"].append(obs_np["goal_pos"].astype(np.float32)[None])

                # force sensor data
                buffer["force"].append(obs_np["force"].astype(np.float32)[None])

                buffer["episode"].append(np.array(current_episode, dtype=np.uint16)[None])
                buffer["action"].append(action.astype(np.float32)[None])

                # Gripper state and action processing
                if "stamp" in DATASET_PATH or "whiteboard" in DATASET_PATH or "vase" in DATASET_PATH:
                    # stamp任务：夹爪始终保持闭合状态
                    current_gripper_state = 0.0  # Always closed
                    gripper_action = 0.0  # Always close action
                else:
                    # 其他任务：根据实际夹爪位置二元化状态
                    current_gripper_state = 0.0 if obs_np["gripper_position"] <= 420 else 1.0
                    
                    # 根据目标夹爪动作二元化 (0=闭合, 840=张开)
                    gripper_action = 1.0 if goal_gripper_action == 840 else 0.0
                
                # 保存到buffer
                buffer["gripper_state"].append(np.array([current_gripper_state], dtype=np.float32)[None])
                buffer["gripper_action"].append(np.array([gripper_action], dtype=np.float32)[None])

                obs = env.step(action, goal_gripper_action, speed=100)
                obs_rs_arm = rs_arm_env.step()
                obs_rs_fix = rs_fix_env.step()
                print(f"\rforce_z: {obs['ext_force'][2]}")

                # calculate fps and time
                current_time = time.time()
                delta_time = current_time - last_time
                episode_steps += 1

                # print debug info every 50 steps
                if episode_steps % 50 == 0:
                    elapsed_time = current_time - start_time
                    fps = episode_steps / elapsed_time if elapsed_time > 0 else 0
                    avg_step_time = elapsed_time / episode_steps if episode_steps > 0 else 0
                    cprint(f"Episode {current_episode}: Step {episode_steps}, "
                          f"Time: {elapsed_time:.1f}s, FPS: {fps:.1f}, "
                          f"Avg Step: {avg_step_time*1000:.1f}ms", "cyan")

                last_time = current_time
        except KeyboardInterrupt:
            cprint("\rRecording cancelled", "red")
            keyboard_listener.stop_listening()
            sys.exit(0)



        cprint(f"\rEpisode {current_episode} done. Collected {episode_steps} steps.", "green")
        
        # save episode data
        for key, val in buffer.items():
            dataset_data[key].append(np.concatenate(val, axis=0))
        
        # reset environment after each episode
        cprint("\rResetting environment for next episode...", "cyan")
        obs = env.reset()
        obs_rs_arm = rs_arm_env.reset()
        obs_rs_fix = rs_fix_env.reset()
        time.sleep(1)

        if "plug" in DATASET_PATH or "insert" in DATASET_PATH:
            x_bias = np.random.uniform(-50, 70)
            y_bias = np.random.uniform(-20, 10)
            z_bias = -200
        elif "clean_whiteboard" in DATASET_PATH:
            x_bias = -230
            y_bias = -40
            z_bias = -280
        elif "stamp" in DATASET_PATH:
            x_bias = 0
            y_bias = 0
            z_bias = -270
        elif "button" in DATASET_PATH:
            x_bias = 0
            y_bias = 0
            z_bias = 0
        elif "vase" in DATASET_PATH:
            x_bias = -117
            y_bias = 53
            z_bias = 0
        elif "push" in DATASET_PATH:
            x_bias = -120
            y_bias = 0
            z_bias = -170
        else:
            x_bias = 0
            y_bias = 0
            z_bias = 0
            print(f"Warning: Unknown dataset: {DATASET_PATH}, using default initialization")

        obs = env.step([x_bias, y_bias, z_bias, 0, 0, 0], gripper_action=0, speed=100)
        obs_rs_arm = rs_arm_env.step()
        obs_rs_fix = rs_fix_env.step()
        # cprint(f"\rEnvironment reset complete! Initial position: {obs['goal_pos'][:3]}, target range: x = [420, 540], y = [-20, 10], z = 330", "green")
        
        # if not last episode, add short delay
        if episode < NUM_EPISODES - 1:
            cprint("\rPreparing for next episode...", "yellow")
            time.sleep(1)

    keyboard_listener.stop_listening()
    print("\rSaving...")
    print(f"Total new episodes: {NUM_EPISODES}")
    
    # create episode_ends data
    print("\rCreating episode_ends data...")
    episode_ends = []
    current_end = 0
    
    # get all episode data
    all_episodes = dataset_data["episode"][:]
    unique_episodes = np.unique(all_episodes)
    
    for episode in unique_episodes:
        episode_mask = all_episodes == episode
        episode_length = np.sum(episode_mask)
        current_end += episode_length
        episode_ends.append(current_end)
    
    # if episode_ends already exists, cover it
    if "episode_ends" in dataset_meta:
        del dataset_meta["episode_ends"]
    
    dataset_meta.require_dataset("episode_ends", shape=(len(episode_ends),), dtype=np.uint32)
    dataset_meta["episode_ends"][:] = np.array(episode_ends, dtype=np.uint32)
    
    print(f"\rCreated episode_ends: {episode_ends}")
    print(f"Total episodes: {len(unique_episodes)}")
    print("\rDataset saved successfully!")
