from env.spacemouse_expert import SpaceMouseExpert

class SpacemouseAgent():
    def __init__(self):
        self.mouse = SpaceMouseExpert()
        
    def act(self):
        action, buttons = self.mouse.get_action()
        # action[:2] *= 1.5
        action[:2] *= 5
        # action[2] = 0

        # 调整扭矩映射以适应夹爪旋转90度后的操作习惯
        # 原始: [roll, pitch, yaw] 
        # 旋转90度后需要重新映射坐标系
        original_torque = action[3:] * 0.004

        


        
        # 重新映射扭矩：考虑夹爪已旋转90度的情况
        # roll和pitch需要交换并调整方向
        action[3] = -original_torque[1]  # roll = -pitch

        action[4] = original_torque[0]   # pitch = roll

        # action[5] = original_torque[2]
        # action[5] = original_torque[2]   # yaw保持不变

        action[5] = original_torque[2]

        # action[3] = 0
        # action[4] = 0
        # action[5] = 0


        
        return action, buttons
