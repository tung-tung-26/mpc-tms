import pdb
from collections import deque

'''
# 从FMU info中提取必要数据
cabin_temp = info['cabin_temp']
battery_temp = info['battery_temp']
motor_temp = info['motor_temp']
# 直接从FMU获取各回路的功率（单位：W 或 kW）
power_cabin = info['power_cabin']  # 座舱回路功率
power_refrigerant = info['power_refrigerant']  # 冷媒回路功率
power_coolant = info['power_coolant']  # 冷却液回路功率
# 计算奖励（选择使用历史平均版本或单步版本）
# 使用历史平均（更稳定，符合长期目标），应对热反馈慢？
rewards = [
    reward_calculator.calculate_cabin_reward(cabin_temp, power_cabin),
    reward_calculator.calculate_refrigerant_reward(cabin_temp, battery_temp, power_refrigerant),
    reward_calculator.calculate_coolant_reward(motor_temp, battery_temp, power_coolant)
]
'''
class RewardCalculator:
    def __init__(self, T_cabin_set, T_bat_set, T_motor_set, window_size=20):
        # === 三个智能体的超参数配置 ===
        # 座舱智能体 (Cabin Agent)
        self.cabin_weights = {
            'temp_control': -0.7,  # 温度控制权重
            'power': -0.3  # 功率权重
        }
        # 冷媒回路智能体 (Refrigerant Circuit Agent)
        self.refrigerant_weights = {
            'power': -1  # 功率权重
        }
        # 冷却液回路智能体 (Coolant Circuit Agent)
        self.coolant_weights = {
            'temp_control': -0.7,  # 温度控制权重
            'power': -0.3  # 功率权重
        }
        # === 系统目标温度 ===
        self.TARGET_CABIN_TEMP = T_cabin_set  # °C
        self.TARGET_BATTERY_TEMP = T_bat_set  # °C
        self.TARGET_MOTOR_TEMP = T_motor_set  # °C
        # === 内部状态跟踪 ===
        self.window_size = window_size  # TODO 滑动窗口大小
        self.cabin_temp_history = deque(maxlen=window_size)
        self.battery_temp_history = deque(maxlen=window_size)
        self.motor_temp_history = deque(maxlen=window_size)


    def calculate_cabin_reward(self, cabin_temp, power_consumption_cabin):
        self.cabin_temp_history.append(cabin_temp)
        # 1. 温度控制误差
        avg_cabin_temp = sum(self.cabin_temp_history) / len(self.cabin_temp_history)
        temp_error = abs(avg_cabin_temp - self.TARGET_CABIN_TEMP)
        # 2. 功率惩罚（直接从FMU获取）
        power_penalty = power_consumption_cabin  # 单位：W 或 kW
        # 3. 总奖励
        cabin_reward = (self.cabin_weights['temp_control'] * temp_error +
                        self.cabin_weights['power'] * power_penalty)
        # print(f"\ncomfort_penalty, power_penalty, {comfort_penalty, power_penalty}")
        return cabin_reward

    def calculate_refrigerant_reward(self, power_consumption_refrigerant):
        # 1. 温度控制误差
        # 2. 功率惩罚（直接从FMU获取）
        power_penalty = power_consumption_refrigerant
        # 3. 总奖励
        refrigerant_reward = (self.refrigerant_weights['power'] * power_penalty)
        # print(f"power_penalty, {power_penalty}")
        return refrigerant_reward

    def calculate_coolant_reward(self, battery_temp, motor_temp, battery_power, motor_power):
        self.motor_temp_history.append(motor_temp)
        self.battery_temp_history.append(battery_temp)
        # 1. 温度控制误差
        avg_motor_temp = sum(self.motor_temp_history) / len(self.motor_temp_history)
        avg_battery_temp = sum(self.battery_temp_history) / len(self.battery_temp_history)
        temp_error = (0.5 * abs(avg_motor_temp - self.TARGET_MOTOR_TEMP) +
                      0.5 * abs(avg_battery_temp - self.TARGET_BATTERY_TEMP))
        # 2. 功率惩罚（直接从FMU获取）
        power_penalty = battery_power + motor_power
        # 3. 总奖励
        coolant_reward = (self.coolant_weights['temp_control'] * temp_error +
                          self.coolant_weights['power'] * power_penalty)
        # print(f"temp_error, power_penalty, {temp_error, power_penalty}")
        return coolant_reward

    def reset(self):
        self.cabin_temp_history = []
        self.battery_temp_history = []
        self.motor_temp_history = []


