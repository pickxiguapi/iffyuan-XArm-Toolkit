from xarm.wrapper import XArmAPI
import numpy as np
import time
import atexit
import math
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr

class XarmEnv:
    def __init__(self, addr="192.168.31.232", action_mode="relative"):
        print("Connecting")
        self.arm = XArmAPI(
            addr, 
            # report_type="real" # for fast force feedback
        )
        print("Connected")
        
        self.action_mode = action_mode
        
        self._clear_error_states()
        
        # self._set_impedance_control()

        self._set_gripper()

        atexit.register(self.cleanup)
    
    def _clear_error_states(self, mode=7):
        assert(self.arm)
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        self.arm.set_mode(mode) # https://github.com/xArm-Developer/xArm-Python-SDK/issues/122
        self.arm.set_state(state=0)
        # time.sleep(0.1)
        
    # def _set_impedance_control(self):
    #     # set tool impedance parameters:
    #     K_pos = 200         #  x/y/z linear stiffness coefficient, range: 0 ~ 2000 (N/m)
    #     K_ori = K_pos * 0.01           #  Rx/Ry/Rz rotational stiffness coefficient, range: 0 ~ 20 (Nm/rad)
    #     B_pos = 0
    #     B_ori = B_pos * 0.01

    #     # Attention: for M and J, smaller value means less effort to drive the arm, but may also be less stable, please be careful. 
    #     M = float(0.05)  #  x/y/z equivalent mass; range: 0.02 ~ 1 kg
    #     J = M * 0.01     #  Rx/Ry/Rz equivalent moment of inertia, range: 1e-4 ~ 0.01 (Kg*m^2)

    #     c_axis = [1,1,1,1,1,1] # set z axis as compliant axis
    #     ref_frame = 0         # 0 : base , 1 : tool

    #     self.arm.set_impedance_mbk([M, M, M, J, J, J], [K_pos, K_pos, K_pos, K_ori, K_ori, K_ori], [B_pos, B_pos, B_pos, B_ori, B_ori, B_ori]) # B(damping) is reserved, give zeros
    #     self.arm.set_impedance_config(ref_frame, c_axis)

    #     # enable ft sensor communication
    #     self.arm.ft_sensor_enable(1)
    #     # will overwrite previous sensor zero and payload configuration
    #     self.arm.ft_sensor_set_zero() # remove this if zero_offset and payload already identified & compensated!
    #     time.sleep(0.2) # wait for writing zero operation to take effect, do not remove

    #     # move robot in impedance control application
    #     self.arm.ft_sensor_app_set(1)
    #     # will start after set_state(0)
    #     self.arm.set_state(0)
        
    # def _unset_impedance_control(self):
    #     self.arm.ft_sensor_app_set(0)
    #     self.arm.ft_sensor_enable(0)
    
    def _set_gripper(self):
        self.arm.set_gripper_mode(0)
        self.arm.set_gripper_enable(True)
        self.arm.set_gripper_speed(5000)
        
    def _get_observation(self):
        code, cart_pos = self.arm.get_position(is_radian=True)
        while code != 0:
            print(f"Error code {code} in get_position().")
            self._clear_error_states()
            code, cart_pos = self.arm.get_position(is_radian=True)
        
        code, servo_angle = self.arm.get_servo_angle(is_radian=True)
        while code != 0:
            print(f"Error code {code} in get_servo_angle().")
            self._clear_error_states()
            code, servo_angle = self.arm.get_servo_angle(is_radian=True)
        servo_angle = servo_angle[:6]

        code, gripper_position = self.arm.get_gripper_position()
        while code != 0:
            print(f"Error code {code} in get_gripper_position().")
            self._clear_error_states()
            code, gripper_position = self.arm.get_gripper_position()
        
        return {
            "cart_pos": np.array(cart_pos),
            "servo_angle": np.array(servo_angle),
            # "force": np.array(self.arm.ft_ext_force),
            # "raw_force": np.array(self.arm.ft_raw_force),
            "goal_pos": np.array(self.goal_pos),
            "gripper_position": np.array(gripper_position),
        }
        
    def reset(self, close_gripper=False):
        print("resetting")

        if close_gripper:
            self.arm.set_gripper_position(0, wait=False, speed=8000)
            time.sleep(1)  # 添加延迟再移动
        else:
            self.arm.set_gripper_position(840, wait=False, speed=8000)

        # self._unset_impedance_control()
        self._clear_error_states(6) # mode 7 is not good, 0/6 is ok
        time.sleep(0.1)
        
        self.goal_pos = np.array([470, 0, 530, 180 / 180 * math.pi, 0, 0])
        
        while True:
            # code = self.arm.set_position(
            #     x=self.goal_pos[0],
            #     y=self.goal_pos[1], 
            #     z=self.goal_pos[2], 
            #     roll=self.goal_pos[3], 
            #     pitch=self.goal_pos[4], 
            #     yaw=self.goal_pos[5], 
            #     speed=1000, wait=False, is_radian=True)

            # use position control to fast reset
            code = self.arm.set_servo_angle(angle=[0, 0, -math.pi/2, 0, math.pi/2, 0, 0], speed=1, is_radian=True)

            code, curr_pos = self.arm.get_position(is_radian=True)
            time.sleep(0.02)
            
            pos_dist = math.dist(self.goal_pos[:3], curr_pos[:3])
            rot_dist = pr.quaternion_dist(
                pr.quaternion_from_extrinsic_euler_xyz(self.goal_pos[3:6]),
                pr.quaternion_from_extrinsic_euler_xyz(curr_pos[3:6])
            )
            
            print(f"resetting, pos_dist {pos_dist} rot_dist {rot_dist}, curr_pos {curr_pos}")
            
            if (pos_dist < 20 and rot_dist < 0.02):
                break

        self._clear_error_states()
        # self._set_impedance_control()

        print("resetting done")

        return self._get_observation()


    
    def step(self, action, gripper_action=None, speed=1000) -> dict[str, np.ndarray]:
        if self.action_mode == "relative":
            self.goal_pos += action
        else:
            self.goal_pos = action
        
        code = self.arm.set_position(
            x=self.goal_pos[0],
            y=self.goal_pos[1], 
            z=self.goal_pos[2], 
            roll=self.goal_pos[3], 
            pitch=self.goal_pos[4], 
            yaw=self.goal_pos[5], 
            speed=speed, wait=False, is_radian=True)
        while code != 0:
            print(f"Error code {code} in set_position().")
            self._clear_error_states()
            code = self.arm.set_position(
                x=self.goal_pos[0],
                y=self.goal_pos[1], 
                z=self.goal_pos[2], 
                roll=self.goal_pos[3], 
                pitch=self.goal_pos[4], 
                yaw=self.goal_pos[5], 
                speed=speed, wait=False, is_radian=True)
        
        if gripper_action is not None:            
            code = self.arm.set_gripper_position(gripper_action, wait=False)

            while code != 0:
                print(f"Error code {code} in set_gripper_position().")
                self._clear_error_states()
                code = self.arm.set_gripper_position(gripper_action, wait=False)

        return self._get_observation()

    def cleanup(self):
        # remember to reset ft_sensor_app when finished
        # self._unset_impedance_control()
        self.arm.disconnect()
        
if __name__ == "__main__":
    env = XarmEnv()
    env.reset()
    
    while True:
        env.step([0, 0, 0, 0, 0, 0])
        time.sleep(0.1)
