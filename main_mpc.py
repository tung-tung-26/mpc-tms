"""
热管理系统的短视野MPC控制 - 优化版本（参数管理集中化）
========================================================

优化内容:
1. 创建集中式参数管理类 (ConfigManager)
2. 所有矩阵、权重、约束统一管理
3. 改进电机温度控制问题 - 增强系统响应性
4. 添加Cabin初始温度设置
5. 支持可配置的step_size参数，plot记录所有步骤

演示如何使用简单的短视野MPC（无终止成本）来控制
黑盒热管理FMU系统。

主要组件:
- ConfigManager: 集中式参数管理
- ThermalMPCCost: 热管理的成本函数
- MPCThermalSystem: 简化的系统模型（用于MPC预测）
- ThermalMPCControllerDelta: 增量型短视野MPC（无终止成本）
- EnhancedFMUITMS: FMU环境包装
- 完整的仿真和绘图工具

重要说明:
- FMU初期不稳定，添加了预热阶段（50步无控制）
- 使用增量型MPC控制（更平稳）
- 支持中文绘图
- 功耗从FMU直接获取
- 温度误差为带符号的值（用于MPC优化方向判断）
- 支持step_size > 1的情况，同时记录所有FMU观察值
- 所有配置通过 ConfigManager 统一管理

作者: 优化版本（参数管理集中化）
日期: 2026-02-28
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import casadi as ca
from typing import Tuple, Optional, Dict, Any, List, Union
import warnings
import os
from datetime import datetime
from collections import defaultdict
import pathlib
from tqdm import tqdm
from dataclasses import dataclass, field
import json

# ============================================================================
# 配置中文字体
# ============================================================================
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

# ============================================================================
# 常数定义
# ============================================================================
K_TO_C = 273.15  # 开尔文到摄氏度的转换常数

# ============================================================================
# 集中式参数管理类
# ============================================================================
@dataclass
class ThermalConfig:
    """
    热管理系统的集中式配置管理。
    所有参数的单一真实来源（Single Source of Truth）。
    """

    # ========================================================================
    # 1. FMU和仿真配置
    # ========================================================================
    fmu_path: Optional[str] = r"..\env\MyITMS.fmu"
    warmup_steps: int = 50  # FMU预热步数（不施加控制）
    max_steps: int = 1800    # 每个episode的最大仿真步数

    # ========================================================================
    # 2. 预热阶段配置
    # ========================================================================
    # 预热阶段的固定转速值（list形式，为4个部件分别设定）
    # [鼓风机, 压缩机, 电池泵, 电机泵]
    warmup_rpm: List[float] = field(default_factory=lambda: [5.0, 2000.0, 1000.0, 1000.0])

    # ========================================================================
    # 3. 转速约束配置
    # ========================================================================
    # 绝对转速约束上下界（每个部件分别设定）
    # 格式: list of (min, max)
    rpm_bounds: List[Tuple[float, float]] = field(default_factory=lambda: [
        (1.0, 10.0),       # 鼓风机（最大300 RPM）
        (100.0, 4000.0),    # 压缩机（500-2000 RPM）
        (100.0, 2000.0),    # 电池泵（500-2000 RPM）
        (100.0, 2000.0)     # 电机泵（500-2000 RPM）
    ])

    # 增量型MPC的增量约束（每个部件分别设定）
    delta_rpm_bounds: List[Tuple[float, float]] = field(default_factory=lambda: [
        (-5.0, 5.0),        # 鼓风机（更小的变化范围）
        (-100.0, 100.0),    # 压缩机
        (-100.0, 100.0),    # 电池泵
        (-100.0, 100.0)     # 电机泵
    ])

    # ========================================================================
    # 4. 温度目标值配置
    # ========================================================================
    temp_targets: Dict[str, float] = field(default_factory=lambda: {
        'cabin': 25.0,
        'battery': 35.0,
        'motor': 40.0
    })

    # ========================================================================
    # 5. 初始温度设置（开尔文）
    # ========================================================================
    init_cabin_temp_K: float = 308.15    # 车舱初始温度35°C (开尔文)
    init_battery_temp_K: float = 313.15  # 电池初始温度40°C (开尔文)
    init_motor_temp_K: float = 333.15    # 电机初始温度60°C (开尔文)

    # ========================================================================
    # 6. MPC成本函数权重配置
    # ========================================================================
    # COST_Q: 状态误差权重矩阵 (3x3 对角矩阵)
    # 状态误差权重。值越大，温度控制越精准
    # [车舱, 电池, 电机] - 对角线元素
    cost_q_diag: List[float] = field(default_factory=lambda: [100.0, 5.0, 5.0])

    # COST_R: 控制增量权重矩阵 (4x4 对角矩阵)
    # 控制增量权重。值越大，动作越平滑。
    # [鼓风机, 压缩机, 电池泵, 电机泵] - 对角线元素
    cost_r_diag: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1, 0.1])

    # COST_ALPHA_POWER: 功耗权重系数
    # 功耗惩罚系数。直接惩罚总功率消耗。值越大越省电，但可能牺牲温度跟踪性能。
    cost_alpha_power: float = 1e-8

    # ========================================================================
    # 7. 系统动态模型配置 (A矩阵和B矩阵)
    # ========================================================================
    # A矩阵：衰减系数 (x_{k+1} = A*x_k + B*u_k)
    # 状态转移矩阵。代表温度的自然衰减/惯性系数（对角线 0.95 表示每一步保留 95% 的温度状态）。
    # 这里使用标量形式（对角矩阵）
    dynamics_a_decay: float = 0.95

    # B矩阵：控制输入影响系数 (3x4 矩阵)
    #         B = np.array([
    #         [0.002,  0.0,    0.0,     0.0],       # ← 行0: 车舱温度 (T_cabin)
    #         [0.0,    0.003, -0.002,   0.0],       # ← 行1: 电池温度 (T_battery)
    #         [0.0,    0.001,  0.0,    -0.004]      # ← 行2: 电机温度 (T_motor)
    #     ])
    #          ↓      ↓       ↓        ↓
    #        RPM_  RPM_    RPM_     RPM_
    #      blower  comp    batt    motor
    # 【优化】改进了系数以增强电机控制响应性
    # 行: [车舱温度, 电池温度, 电机温度]
    # 列: [鼓风机, 压缩机, 电池泵, 电机泵]
    dynamics_b_matrix: List[List[float]] = field(default_factory=lambda: [
        [-0.002, -0.002, 0.0, 0.0],      # ← 车舱温度 (T_cabin)
        [0.0, -0.002, -0.002, 0.0],    # ← 电池温度 (T_battery)
        [0.0, -0.002, 0.0, -0.002]     # ← 电机温度 (T_motor)
    ])
    #    ↓       ↓       ↓       ↓
    #   RPM_    RPM_    RPM_    RPM_
    #   blower  comp    batt    motor

    # ========================================================================
    # 8. MPC求解器配置
    # ========================================================================
    mpc_horizon: int = 5  # 预测视野
    mpc_max_iter: int = 1000
    mpc_tol: float = 1e-4
    mpc_acceptable_tol: float = 1e-3

    # ========================================================================
    # 9. Step size配置
    # ========================================================================
    step_size_multiplier: int = 1  # 默认=1时等价于原功能

    # ========================================================================
    # 10. 观测和控制变量配置
    # ========================================================================
    observation_names: List[str] = field(default_factory=lambda: [
        "cabinVolume.summary.T",
        "battery.Batt_top[1].T",
        "machine.heatCapacitor.T"
    ])

    action_names: List[str] = field(default_factory=lambda: [
        "RPM_blower",
        "RPM_comp",
        "RPM_batt",
        "RPM_motor"
    ])

    # FMU功耗变量名称
    power_names: List[str] = field(default_factory=lambda: [
        "TableDC3.Pe",       # 鼓风机功率
        "TableDC.Pe",        # 压缩机功率
        "TableDC1.Pe",       # 电池泵功率
        "TableDC2.Pe"        # 电机泵功率
    ])

    # 额外监测变量名称
    extra_var_names: List[str] = field(default_factory=lambda: [
        "driverPerformance.controlBus.driverBus._acc_pedal_travel",    # 油门踏板
        "driverPerformance.controlBus.driverBus._brake_pedal_travel",  # 制动踏板
        "driverPerformance.controlBus.vehicleStatus.vehicle_velocity",  # 车速
        "battery.controlBus.batteryBus.battery_SOC[1]",                # 电池SOC
        "fan2Table.qflow"                                               # 鼓风机流量 (m^3/h)
    ])

    # ========================================================================
    # 11. 温度边界
    # ========================================================================
    temp_bounds: Tuple[float, float] = (10.0, 60.0)  # 单位: °C

    # ========================================================================
    # 12. 仿真状态初始值（仅仿真模式下使用）
    # ========================================================================
    sim_state_init_cabin: float = 35.0   # °C
    sim_state_init_battery: float = 35.0 # °C
    sim_state_init_motor: float = 40.0   # °C
    sim_state_noise_std: float = 0.5     # 标准差

    def __post_init__(self):
        """初始化后的验证和处理。"""
        self._validate_config()
        self._build_derived_matrices()

    def _validate_config(self):
        """验证配置的有效性。"""
        # 检查维度
        assert len(self.warmup_rpm) == 4, "warmup_rpm 必须有4个元素"
        assert len(self.rpm_bounds) == 4, "rpm_bounds 必须有4个元组"
        assert len(self.delta_rpm_bounds) == 4, "delta_rpm_bounds 必须有4个元组"
        assert len(self.cost_q_diag) == 3, "cost_q_diag 必须有3个元素"
        assert len(self.cost_r_diag) == 4, "cost_r_diag 必须有4个元素"
        assert len(self.dynamics_b_matrix) == 3, "B矩阵必须有3行"
        assert all(len(row) == 4 for row in self.dynamics_b_matrix), "B矩阵每行必须有4列"

        # 检查温度目标
        assert 'cabin' in self.temp_targets
        assert 'battery' in self.temp_targets
        assert 'motor' in self.temp_targets

        # 检查变量名数量
        assert len(self.observation_names) == 3, "观测变量必须有3个"
        assert len(self.action_names) == 4, "控制变量必须有4个"
        assert len(self.power_names) == 4, "功耗变量必须有4个"
        assert len(self.extra_var_names) == 5, "额外变量必须有5个"

        print("✅ 配置验证通过")

    def _build_derived_matrices(self):
        """构建派生的矩阵（从配置中计算）。"""
        # Q矩阵：对角矩阵
        self.cost_Q = np.diag(self.cost_q_diag)

        # R矩阵：对角矩阵
        self.cost_R = np.diag(self.cost_r_diag)

        # A矩阵：对角矩阵（衰减）
        self.dynamics_A = np.eye(3) * self.dynamics_a_decay

        # B矩阵：转换为numpy数组
        self.dynamics_B = np.array(self.dynamics_b_matrix, dtype=float)

    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典（便于保存和查看）。"""
        return {
            'fmu_path': self.fmu_path,
            'warmup_steps': self.warmup_steps,
            'warmup_rpm': self.warmup_rpm,
            'rpm_bounds': self.rpm_bounds,
            'delta_rpm_bounds': self.delta_rpm_bounds,
            'temp_targets': self.temp_targets,
            'cost_q_diag': self.cost_q_diag,
            'cost_r_diag': self.cost_r_diag,
            'cost_alpha_power': self.cost_alpha_power,
            'dynamics_a_decay': self.dynamics_a_decay,
            'mpc_horizon': self.mpc_horizon,
            'step_size_multiplier': self.step_size_multiplier,
        }

    def save_to_json(self, filepath: str):
        """将配置保存为JSON文件。"""
        config_dict = self.to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"✅ 配置已保存到: {filepath}")

    @classmethod
    def load_from_json(cls, filepath: str) -> 'ThermalConfig':
        """从JSON文件加载配置。"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        # 只加载已知的字段
        config_dict = {k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__}
        print(f"✅ 配置已从 {filepath} 加载")
        return cls(**config_dict)


# ============================================================================
# 成本函数
# ============================================================================
class ThermalMPCCost:
    """
    热管理控制的成本函数。
    惩罚温度偏离设定值和控制功耗。

    成本函数形式：
    J = (x - x_ref)^T @ Q @ (x - x_ref) + u_delta^T @ R @ u_delta + alpha_power * P_total

    其中：
    - x: 当前状态 [T_cabin, T_battery, T_motor]
    - x_ref: 目标状态 [T_cabin_set, T_bat_set, T_motor_set]
    - u_delta: 控制增量 [ΔU_blower, ΔU_comp, ΔU_batt, ΔU_motor]
    - Q: 状态误差权重矩阵 (3x3)
    - R: 控制增量权重矩阵 (4x4)
    - alpha_power: 功耗权重系数
    """

    def __init__(self, config: ThermalConfig):
        """
        参数说明
        ----------
        config : ThermalConfig
            集中式配置对象
        """
        self.config = config

        self.x_ref = np.array([
            config.temp_targets['cabin'],
            config.temp_targets['battery'],
            config.temp_targets['motor']
        ], dtype=float)

        self.Q = config.cost_Q
        self.R = config.cost_R
        self.alpha_power = config.cost_alpha_power

    def stage_cost(self, x: np.ndarray, delta_u: np.ndarray,
                   power_dict: Optional[Dict[str, float]] = None) -> float:
        """
        计算单步成本。

        参数说明
        ----------
        x : np.ndarray
            状态向量 [T_cabin, T_battery, T_motor]（单位: °C）
        delta_u : np.ndarray
            控制增量 [ΔU_blower, ΔU_comp, ΔU_batt, ΔU_motor]（单位: RPM/step）
        power_dict : dict, 可选
            功耗字典，包含各个部件的功率（单位: kW）

        返回值
        -------
        float
            总的单步成本
        """
        x = np.asarray(x).flatten()
        delta_u = np.asarray(delta_u).flatten()

        # 状态跟踪成本（使用带符号的误差）
        error = x - self.x_ref
        state_cost = float(error.T @ self.Q @ error)

        # 控制增量成本（平滑性）
        control_cost = float(delta_u.T @ self.R @ delta_u)

        # 功耗成本
        if power_dict is not None:
            total_power = (
                power_dict.get('power_1', 0.0) +
                power_dict.get('power_2', 0.0) +
                power_dict.get('power_3', 0.0) +
                power_dict.get('power_4', 0.0)
            )
            power_cost = self.alpha_power * total_power
        else:
            power_cost = 0.0
        print(state_cost, control_cost, power_cost)
        return float(state_cost + control_cost + power_cost)


# ============================================================================
# 简化的系统模型（用于MPC）
# ============================================================================
class MPCThermalSystem:
    """
    用于MPC的简化系统模型，包装FMU观测。
    只用于轨迹预测，实际动态在FMU中。

    【优化说明】
    改进了B矩阵的系数以更好地反映实际系统响应，
    特别是增强了电机控制的响应性。
    """

    def __init__(self, config: ThermalConfig):
        """
        参数说明
        ----------
        config : ThermalConfig
            集中式配置对象
        """
        self.config = config
        self.n = 3  # 状态维数（温度：车舱、电池、电机）
        self.m = 4  # 控制维数（4个泵/压缩机转速）
        self.dt = 1.0  # 采样时间（秒）

        # 从配置加载矩阵
        self.A = config.dynamics_A
        self.B = config.dynamics_B
        self.Q = config.cost_Q
        self.R = config.cost_R

        # 边界
        self.temp_bounds = config.temp_bounds
        self.rpm_bounds = config.rpm_bounds
        self.delta_rpm_bounds = config.delta_rpm_bounds

        # 为了兼容archive的MPC类
        self.v_bounds = (0.0, 5000.0)
        self.omega_bounds = (0.0, 0.0)  # 在热管理中未使用

    def stage_cost(self, x: np.ndarray, delta_u: np.ndarray) -> float:
        """二次型成本。"""
        x = np.asarray(x).flatten()
        delta_u = np.asarray(delta_u).flatten()

        state_cost = float(x.T @ self.Q @ x)
        control_cost = float(delta_u.T @ self.R @ delta_u)
        print(state_cost, control_cost)
        return state_cost + control_cost

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        简化的线性动态：x_{k+1} = A*x_k + B*u_k
        这是占位符；真实动态在FMU中实现。

        注意：这里的u是绝对转速，不是增量

        【优化点1】改进B矩阵系数以增强电机控制响应性
        """
        x = np.asarray(x).flatten()
        u = np.asarray(u).flatten()

        x_next = self.A @ x + self.B @ u
        x_next = np.clip(x_next, self.temp_bounds[0], self.temp_bounds[1])

        return x_next


# ============================================================================
# 辅助函数：转换配置参数
# ============================================================================
def convert_warmup_rpm(warmup_rpm: Union[float, int, List]) -> np.ndarray:
    """
    转换WARMUP_RPM为numpy数组。

    参数说明
    ----------
    warmup_rpm : float, int, or list
        可以是标量（所有部件相同）或list/tuple（每个部件不同）

    返回值
    -------
    np.ndarray
        形状为(4,)的数组，表示4个部件的预热RPM
    """
    if isinstance(warmup_rpm, (list, tuple)):
        warmup_array = np.array(warmup_rpm, dtype=float)
        if len(warmup_array) != 4:
            raise ValueError(f"WARMUP_RPM列表长度必须为4，但得到{len(warmup_array)}")
        return warmup_array
    else:
        # 标量情况：所有部件相同
        return np.ones(4, dtype=float) * float(warmup_rpm)


def convert_bounds(bounds: Union[Tuple, List]) -> List[Tuple]:
    """
    转换边界约束为标准格式。

    参数说明
    ----------
    bounds : tuple or list of tuples
        可以是单tuple (min, max)（所有部件相同）
        或 list of tuples（每个部件不同）

    返回值
    -------
    list of tuples
        长度为4的list，每个元素是(min, max)元组
    """
    if isinstance(bounds[0], (tuple, list)):
        # 已是list of tuples格式
        bounds_list = list(bounds)
        if len(bounds_list) != 4:
            raise ValueError(f"边界列表长度必须为4，但得到{len(bounds_list)}")
        return bounds_list
    else:
        # 标量tuple情况：所有部件相同
        min_val, max_val = bounds
        return [(min_val, max_val) for _ in range(4)]


# ============================================================================
# 增量型短视野MPC控制器（无终止成本）
# ============================================================================
class ThermalMPCControllerDelta:
    """
    热管理用的增量型短视野MPC。
    - 不使用终止成本，只在预测视野内累积单步成本
    - 使用增量控制（更平稳）
    - 支持每个部件不同的增量约束和绝对转速约束
    """

    def __init__(self,
                 system: MPCThermalSystem,
                 cost_fn: ThermalMPCCost):
        """
        参数说明
        ----------
        system : MPCThermalSystem
            用于预测的系统模型
        cost_fn : ThermalMPCCost
            成本函数
        """
        self.system = system
        self.horizon = system.config.mpc_horizon
        self.cost_fn = cost_fn
        self.config = system.config
        self.last_solution = None

        self._build_nlp()

    def _build_nlp(self):
        """构建CasADi NLP问题（基于增量）。"""
        n = self.system.n
        m = self.system.m
        N = self.horizon

        # 决策变量：控制增量
        dU = ca.SX.sym('dU', m, N)
        x0 = ca.SX.sym('x0', n)
        u_prev = ca.SX.sym('u_prev', m)  # 前一个时刻的绝对控制值

        # 成本累加
        J = 0.0
        x = x0
        u = u_prev  # 初始化为前一个时刻的值

        for i in range(N):
            dU_i = dU[:, i]

            # 更新绝对控制值（累积增量）
            u = u + dU_i

            # 约束：绝对转速边界（每个部件可能不同）
            # 使用列表推导式构建约束后的控制值
            u_clipped_list = []
            for j in range(m):
                u_j_clipped = ca.fmax(ca.fmin(u[j], self.system.rpm_bounds[j][1]),
                                     self.system.rpm_bounds[j][0])
                u_clipped_list.append(u_j_clipped)

            # 使用 ca.vertcat 构建列向量
            u_clipped = ca.vertcat(*u_clipped_list)

            # 单步成本（使用带符号的误差）
            x_ref = ca.DM(self.cost_fn.x_ref)
            error = x - x_ref
            state_cost = ca.mtimes([error.T, ca.DM(self.system.Q), error])
            control_cost = ca.mtimes([dU_i.T, ca.DM(self.system.R), dU_i])
            J += state_cost + control_cost

            # 线性动态预测（使用约束后的控制值）
            A = ca.DM(self.system.A)
            B = ca.DM(self.system.B)

            x = A @ x + B @ u_clipped

        # 展平决策变量
        dU_flat = ca.reshape(dU, m * N, 1)

        # 增量约束（每个部件可能不同）
        lbx = []
        ubx = []
        for i in range(N):
            for j in range(m):
                lbx.append(self.system.delta_rpm_bounds[j][0])
                ubx.append(self.system.delta_rpm_bounds[j][1])

        # NLP表述
        nlp = {
            'x': dU_flat,
            'f': J,
            'p': ca.vertcat(x0, u_prev)
        }

        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
            'ipopt.max_iter': self.config.mpc_max_iter,
            'ipopt.tol': self.config.mpc_tol,
            'ipopt.acceptable_tol': self.config.mpc_acceptable_tol,
            'ipopt.warm_start_init_point': 'yes',
        }

        self.solver = ca.nlpsol('mpc_solver', 'ipopt', nlp, opts)

        self.lbx = lbx
        self.ubx = ubx
        self.n = n
        self.m = m
        self.N = N

    def solve(self, x0: np.ndarray, u_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        求解MPC优化问题。

        参数说明
        ----------
        x0 : np.ndarray
            当前状态
        u_prev : np.ndarray
            前一个时刻的绝对控制值

        返回值
        -------
        delta_u_seq : np.ndarray (m x N)
            最优增量控制序列
        cost_seq : np.ndarray (N,)
            视野上的单步成本
        x_traj : np.ndarray (n x N+1)
            预测的状态轨迹
        """
        x0 = np.asarray(x0).flatten()
        u_prev = np.asarray(u_prev).flatten()
        N = self.N
        m = self.m

        # 初始猜测：增量都为0（保持不变）
        dU0 = np.zeros(m * N)

        # 使用热启动
        if self.last_solution is not None:
            last_sol_len = len(self.last_solution)
            expected_len = m * N

            if last_sol_len == expected_len:
                # 将前一步的解整体向前移动，最后一步设为0
                dU0[:-m] = self.last_solution[:-m]
                dU0[-m:] = 0.0
            else:
                dU0 = np.zeros(m * N)

        try:
            # 参数：当前状态和前一个控制值
            p = np.concatenate([x0, u_prev])

            sol = self.solver(
                x0=dU0,
                lbx=self.lbx,
                ubx=self.ubx,
                p=p
            )

            dU_opt = np.array(sol['x']).flatten()
            stats = self.solver.stats()

            if not stats['success']:
                # 这表示求解器未达到最优收敛条件，但返回了可接受的解
                pass

        except Exception as e:
            warnings.warn(f"MPC求解器失败: {str(e)}。使用初始猜测。")
            dU_opt = dU0

        # 验证输出大小
        if dU_opt.shape[0] != m * N:
            warnings.warn(f"MPC输出大小不匹配: 得到 {dU_opt.shape[0]}, 期望 {m * N}。使用零输入。")
            dU_opt = np.zeros(m * N)

        self.last_solution = dU_opt.copy()

        try:
            delta_u_seq = dU_opt.reshape((N, m)).T
        except ValueError as e:
            warnings.warn(f"reshape失败: {str(e)}。使用零输入。")
            delta_u_seq = np.zeros((m, N))

        # 计算轨迹和成本
        x_traj = np.zeros((self.n, N + 1))
        x_traj[:, 0] = x0
        cost_seq = np.zeros(N)

        u_traj = np.zeros((m, N + 1))
        u_traj[:, 0] = u_prev

        for i in range(N):
            # 更新控制值（累积增量）
            u_curr = u_traj[:, i] + delta_u_seq[:, i]
            # 应用每个部件的边界约束
            for j in range(m):
                u_curr[j] = np.clip(u_curr[j], self.system.rpm_bounds[j][0],
                                   self.system.rpm_bounds[j][1])
            u_traj[:, i + 1] = u_curr

            # 计算成本
            cost_seq[i] = self.system.stage_cost(x_traj[:, i], delta_u_seq[:, i])

            # 更新状态
            x_traj[:, i + 1] = self.system.dynamics(x_traj[:, i], u_curr)

        return delta_u_seq, cost_seq, x_traj


# ============================================================================
# 增强的FMU环境包装
# ============================================================================
class EnhancedFMUITMS:
    """
    FMU热管理系统的增强包装。
    改编自 archive/fmu_env_itms.py，更好地与MPC集成。

    重要说明：
    - 当 fmu_path=None 时，将进入"仅仿真模式"
    - 在仅仿真模式下，返回的温度始终为摄氏度
    - 当有实际FMU时，返回的温度为开尔文，需要转换
    - 功耗和其他变量从FMU中直接获取
    """

    def __init__(self, config: ThermalConfig):
        """
        参数说明
        ----------
        config : ThermalConfig
            集中式配置对象
        """
        self.config = config
        self.fmu_path = pathlib.Path(config.fmu_path) if config.fmu_path else None
        self.step_size = 1.0

        # 观测和控制变量名称
        self.observation_names = config.observation_names
        self.action_names = config.action_names
        self.power_names = config.power_names
        self.extra_var_names = config.extra_var_names

        self.max_steps = config.max_steps

        self.n_obs = len(self.observation_names)
        self.n_act = len(self.action_names)
        self.n_power = len(self.power_names)
        self.n_extra = len(self.extra_var_names)

        # 尝试加载FMU（如果存在）
        self.fmu_available = False
        try:
            # 如果fmu_path为None，直接跳过FMU加载
            if self.fmu_path is None:
                print("⚠️  FMU路径为None，进入仅仿真模式（Simulation-Only Mode）")
                print("   在此模式下，所有返回值为模拟值")
                self.fmu_available = False
            elif self.fmu_path.exists():
                from fmpy import extract, read_model_description
                from fmpy.fmi2 import FMU2Slave

                print(f"✅ 找到FMU文件: {self.fmu_path}")
                self.unzip_dir = extract(str(self.fmu_path))
                self.md = read_model_description(str(self.fmu_path), validate=False)
                self.cs = self.md.coSimulation

                if self.cs is not None:
                    self.fmu_available = True
                    print("✅ FMU已成功加载（Co-Simulation模式）")
                    print(f"   将获取功耗和其他监测变量")
                    self._build_vr_map()
                    self._create_fmu_instance()
            else:
                print(f"⚠️  FMU文件不存在: {self.fmu_path}")
                print("   进入仅仿真模式（Simulation-Only Mode）")
                self.fmu_available = False
        except Exception as e:
            print(f"⚠️  FMU加载失败: {e}")
            print("   进入仅仿真模式（Simulation-Only Mode）")
            self.fmu_available = False

        self.current_step = 0
        self.current_time = 0.0
        self._last_obs = None
        # 仿真模式下的状态（初始温度，单位: °C）
        self._sim_state = np.array([
            config.sim_state_init_cabin,
            config.sim_state_init_battery,
            config.sim_state_init_motor
        ])

    def _build_vr_map(self):
        """构建变量名到valueReference的映射。"""
        self.vrs = {}
        for var in self.md.modelVariables:
            self.vrs[var.name] = (var.valueReference, var.type)

    def _create_fmu_instance(self):
        """创建并初始化FMU实例。"""
        try:
            from fmpy.fmi2 import FMU2Slave

            self._fmu = FMU2Slave(
                guid=self.md.guid,
                unzipDirectory=str(self.unzip_dir),
                modelIdentifier=self.cs.modelIdentifier,
                instanceName="thermal_mpc_instance"
            )
            self._fmu.instantiate()
            self._fmu.setupExperiment(startTime=0.0)
            self._fmu.enterInitializationMode()

            # 设置初始温度
            init_dict = {
                'MY_socinit': 0.5,
                'MY_battT0': self.config.init_battery_temp_K,
                'MY_motorT0': self.config.init_motor_temp_K,
                'MY_cabinT0': self.config.init_cabin_temp_K
            }

            for k, v in init_dict.items():
                if k in self.vrs:
                    self._fmu.setReal([self.vrs[k][0]], [float(v)])

            self._fmu.exitInitializationMode()
        except Exception as e:
            warnings.warn(f"未能创建FMU实例: {e}")
            self.fmu_available = False

    def reset(self, seed: Optional[int] = None):
        """重置环境并返回初始观测。"""
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.current_time = 0.0

        if self.fmu_available:
            try:
                self._fmu.terminate()
            except:
                pass
            self._create_fmu_instance()
        else:
            # 仅仿真模式：初始化状态
            self._sim_state = np.array([
                self.config.sim_state_init_cabin,
                self.config.sim_state_init_battery,
                self.config.sim_state_init_motor
            ]) + np.random.randn(3) * self.config.sim_state_noise_std

        # 如果FMU不可用，返回虚拟初始观测
        obs = self._read_observations()
        self._last_obs = obs
        return obs

    def _read_observations(self) -> Dict[str, float]:
        """
        从FMU读取当前观测或返回仿真值。

        重要：FMU返回的温度为开尔文，需要转换为摄氏度
        """
        obs = {}

        if self.fmu_available:
            try:
                for name in self.observation_names:
                    if name in self.vrs:
                        vr, fmi_type = self.vrs[name]
                        if fmi_type in ("Real", "Radians"):
                            # FMU返回的是开尔文，转换为摄氏度
                            temp_K = float(self._fmu.getReal([vr])[0])
                            obs[name] = temp_K - K_TO_C
                        elif fmi_type == "Integer":
                            obs[name] = float(self._fmu.getInteger([vr])[0])
            except Exception as e:
                warnings.warn(f"读取FMU观测出错: {e}")

        # 用仿真值填充缺失的观测
        if not obs:
            # 仿真的稳态温度（直接返回摄氏度）
            obs = {
                self.observation_names[0]: self._sim_state[0],   # 车舱
                self.observation_names[1]: self._sim_state[1],   # 电池
                self.observation_names[2]: self._sim_state[2],   # 电机
            }

        return obs

    def _read_power(self) -> Dict[str, float]:
        """
        从FMU读取功耗真实值。

        返回值
        -------
        power_dict : dict
            功耗字典 {'power_1': ..., 'power_2': ..., 'power_3': ..., 'power_4': ...}
        """
        power_dict = {}

        if self.fmu_available:
            try:
                for i, name in enumerate(self.power_names):
                    if name in self.vrs:
                        vr, fmi_type = self.vrs[name]
                        if fmi_type in ("Real", "Radians"):
                            # 从FMU获取功率（单位为瓦特，需要转换为千瓦）
                            power_val = float(self._fmu.getReal([vr])[0])
                            # 单位转换: W -> kW
                            power_dict[f'power_{i+1}'] = power_val / 1000.0
                    else:
                        power_dict[f'power_{i+1}'] = 0.0
            except Exception as e:
                warnings.warn(f"读取FMU功耗出错: {e}")
                # 出错时返回零
                for i in range(len(self.power_names)):
                    power_dict[f'power_{i+1}'] = 0.0
        else:
            # 仿真模式下返回零
            for i in range(len(self.power_names)):
                power_dict[f'power_{i+1}'] = 0.0

        return power_dict

    def _read_extra_vars(self) -> Dict[str, float]:
        """
        从FMU读取额外监测变量（踏板、车速、SOC、鼓风机流量等）。

        返回值
        -------
        extra_dict : dict
            额外变量字典
        """
        extra_dict = {}

        if self.fmu_available:
            try:
                for i, name in enumerate(self.extra_var_names):
                    if name in self.vrs:
                        vr, fmi_type = self.vrs[name]
                        if fmi_type in ("Real", "Radians"):
                            val = float(self._fmu.getReal([vr])[0])
                            # 简化显示名称
                            short_name = name.split('.')[-1]

                            # 特殊处理：鼓风机流量单位转换 (m^3/h -> m^3/s)
                            if 'qflow' in short_name.lower():
                                val = val / 3600.0  # m^3/h -> m^3/s

                            extra_dict[short_name] = val
                        elif fmi_type == "Integer":
                            val = float(self._fmu.getInteger([vr])[0])
                            short_name = name.split('.')[-1]
                            extra_dict[short_name] = val
            except Exception as e:
                warnings.warn(f"读取FMU额外变量出错: {e}")
        else:
            # 仿真模式下返回模拟值
            extra_dict = {
                '_acc_pedal_travel': 0.0,
                '_brake_pedal_travel': 0.0,
                'vehicle_velocity': 0.0,
                'battery_SOC[1]': 0.5,
                'qflow': 0.0  # m^3/s
            }

        return extra_dict

    def step(self, action: Dict[str, float]) -> Tuple[Dict, Dict, Dict, bool, bool]:
        """
        前进环境一步。

        参数说明
        ----------
        action : dict
            控制输入字典，包含各个泵和压缩机的RPM设定值（绝对值）

        返回值
        -------
        obs : dict
            观测（温度已转换为摄氏度）
        power : dict
            功耗字典（从FMU获取真实值）
        extra_vars : dict
            额外监测变量（踏板、车速、SOC、鼓风机流量等）
        terminated : bool
            是否因错误而终止
        truncated : bool
            是否达到最大步数而截断
        """
        terminated = False
        truncated = False

        # 写入动作到FMU
        if self.fmu_available and action:
            try:
                for name, value in action.items():
                    if name in self.vrs:
                        vr, fmi_type = self.vrs[name]
                        self._fmu.setReal([vr], [float(value)])
            except Exception as e:
                warnings.warn(f"向FMU写入动作出错: {e}")

        # 步进FMU
        if self.fmu_available:
            try:
                self._fmu.doStep(self.current_time, self.step_size)
            except Exception as e:
                terminated = True
                warnings.warn(f"FMU步进失败: {e}")
        else:
            # 仅仿真模式：更新模拟状态
            u = np.array([
                action.get(name, 0.0) for name in self.action_names
            ])

            # 简单的线性动态（基于MPC系统模型）
            self._sim_state = self.config.dynamics_A @ self._sim_state + self.config.dynamics_B @ u
            self._sim_state = np.clip(self._sim_state, self.config.temp_bounds[0], self.config.temp_bounds[1])

        # 读取观测（温度已是摄氏度）
        obs = self._read_observations()

        # 读取功耗（从FMU获取真实值）
        power = self._read_power()

        # 读取额外变量
        extra_vars = self._read_extra_vars()

        # 检查终止条件
        truncated = (self.current_step + 1 >= self.max_steps)

        self.current_step += 1
        self.current_time += self.step_size
        self._last_obs = obs

        return obs, power, extra_vars, terminated, truncated

    def close(self):
        """清理FMU资源。"""
        if self.fmu_available:
            try:
                self._fmu.terminate()
            except:
                pass

    def obs_to_array(self, obs: Dict[str, float]) -> np.ndarray:
        """
        将观测字典转换为状态数组。

        返回的数组已是摄氏度。
        """
        return np.array([obs.get(name, 0.0) for name in self.observation_names], dtype=float)

    def action_array_to_dict(self, u: np.ndarray) -> Dict[str, float]:
        """将控制数组转换为动作字典。"""
        return {name: float(u[i]) for i, name in enumerate(self.action_names)}


# ============================================================================
# 主要仿真和绘图
# ============================================================================
def run_thermal_mpc_simulation(
    config: ThermalConfig,
    sim_steps: int = 280,
    seed: int = 42,
) -> Dict:
    """
    运行MPC控制的热管理FMU仿真。

    参数说明
    ----------
    config : ThermalConfig
        集中式配置对象
    sim_steps : int
        仿真步数（不包括预热步数）
    seed : int
        随机种子

    返回值
    -------
    results : dict
        包含所有仿真结果
    """
    np.random.seed(seed)

    # 【优化3】处理step_size_multiplier
    step_size_multiplier = config.step_size_multiplier
    warmup_steps = config.warmup_steps
    warmup_steps_adjusted = max(1, warmup_steps // step_size_multiplier)

    print(f"\n【信息】step_size_multiplier = {step_size_multiplier}")
    if step_size_multiplier > 1:
        print(f"  原始 warmup_steps = {warmup_steps}")
        print(f"  调整后 warmup_steps = {warmup_steps_adjusted}")
        print(f"  每 {step_size_multiplier} 步执行一次MPC控制")

    # 初始化系统和控制器
    sys = MPCThermalSystem(config)

    cost_fn = ThermalMPCCost(config)
    controller = ThermalMPCControllerDelta(sys, cost_fn)

    # 初始化FMU环境
    env = EnhancedFMUITMS(config)

    # 运行仿真
    controller_name = f"增量型MPC (N={config.mpc_horizon}, step_size={step_size_multiplier})"
    print(f"\n开始 {controller_name} 仿真")
    print(f"预热阶段: {warmup_steps_adjusted} 步（MPC控制周期）")
    print(f"  - RPM设定: blower={config.warmup_rpm[0]:.0f}, comp={config.warmup_rpm[1]:.0f}, "
          f"batt={config.warmup_rpm[2]:.0f}, motor={config.warmup_rpm[3]:.0f}")
    print(f"MPC控制: {sim_steps // step_size_multiplier} 步（MPC控制周期）")
    print(f"  - 预测视野: {config.mpc_horizon}")
    print(f"  - 增量约束与绝对转速约束已配置")
    print("=" * 150)

    obs = env.reset(seed=seed)
    x_k = env.obs_to_array(obs)

    # 【优化3】使用FMU实际步数初始化历史
    total_fmu_steps = warmup_steps + sim_steps
    x_history = np.zeros((3, total_fmu_steps + 1))
    u_history = np.zeros((4, total_fmu_steps))
    delta_u_history = np.zeros((4, total_fmu_steps))
    cost_history = np.zeros(total_fmu_steps)
    power_history = np.zeros((total_fmu_steps, 4))
    extra_vars_history = np.zeros((total_fmu_steps, 5))
    timestamps = []

    x_history[:, 0] = x_k

    # 初始化控制值
    u_current = np.array(config.warmup_rpm, dtype=float)
    u_history[:, 0] = u_current

    # 使用tqdm显示进度条
    pbar = tqdm(range(total_fmu_steps), desc=f"{controller_name} 运行中", unit="步")

    for k in pbar:
        timestamps.append(k)

        # 【优化3】判断是否执行MPC（每step_size_multiplier步）
        is_mpc_step = (k >= warmup_steps_adjusted * step_size_multiplier) and \
                      ((k - warmup_steps_adjusted * step_size_multiplier) % step_size_multiplier == 0)
        is_warmup = k < warmup_steps_adjusted * step_size_multiplier

        if is_warmup:
            u_k = np.array(config.warmup_rpm, dtype=float)
            delta_u_k = np.zeros(4)
            phase_label = "预热"
        elif is_mpc_step:
            # 每step_size_multiplier步执行一次MPC
            try:
                delta_u_seq, _, _ = controller.solve(x_k, u_current)
                delta_u_k = delta_u_seq[:, 0]
            except Exception as e:
                warnings.warn(f"MPC求解在步骤 {k} 失败: {e}")
                delta_u_k = np.zeros(4)

            phase_label = "控制"
        else:
            # 非MPC步，保持上一步的控制
            delta_u_k = np.zeros(4)
            phase_label = "保持"

        # 更新绝对控制值
        u_k = u_current + delta_u_k
        for j in range(4):
            u_k[j] = np.clip(u_k[j], config.rpm_bounds[j][0], config.rpm_bounds[j][1])

        # 步进环境
        action = env.action_array_to_dict(u_k)
        obs, power, extra_vars, terminated, truncated = env.step(action)
        x_next = env.obs_to_array(obs)

        # 记录数据
        cost_history[k] = cost_fn.stage_cost(x_k, delta_u_k, power)
        u_history[:, k] = u_k
        delta_u_history[:, k] = delta_u_k
        x_history[:, k + 1] = x_next

        power_history[k, 0] = power.get('power_1', 0.0)
        power_history[k, 1] = power.get('power_2', 0.0)
        power_history[k, 2] = power.get('power_3', 0.0)
        power_history[k, 3] = power.get('power_4', 0.0)

        extra_vars_history[k, 0] = extra_vars.get('_acc_pedal_travel', 0.0)
        extra_vars_history[k, 1] = extra_vars.get('_brake_pedal_travel', 0.0)
        extra_vars_history[k, 2] = extra_vars.get('vehicle_velocity', 0.0)
        extra_vars_history[k, 3] = extra_vars.get('battery_SOC[1]', 0.5)
        extra_vars_history[k, 4] = extra_vars.get('qflow', 0.0)

        temp_errors = x_next - np.array([
            config.temp_targets['cabin'],
            config.temp_targets['battery'],
            config.temp_targets['motor']
        ])

        pbar.set_postfix({
            '阶段': phase_label,
            'T_cabin': f'{x_next[0]:6.2f}°C',
            'T_battery': f'{x_next[1]:6.2f}°C',
            'T_motor': f'{x_next[2]:6.2f}°C',
            'err_cabin': f'{temp_errors[0]:+6.2f}°C',
            'err_battery': f'{temp_errors[1]:+6.2f}°C',
            'err_motor': f'{temp_errors[2]:+6.2f}°C',
            'RPM_blower': f'{u_k[0]:7.0f}',
            'RPM_comp': f'{u_k[1]:7.0f}',
            'RPM_batt': f'{u_k[2]:7.0f}',
            'RPM_motor': f'{u_k[3]:7.0f}',
        })

        x_k = x_next
        u_current = u_k.copy()

        if terminated or truncated:
            print(f"\n仿真在步骤 {k + 1} 结束")
            x_history = x_history[:, :k + 2]
            u_history = u_history[:, :k + 1]
            delta_u_history = delta_u_history[:, :k + 1]
            cost_history = cost_history[:k + 1]
            power_history = power_history[:k + 1, :]
            extra_vars_history = extra_vars_history[:k + 1, :]
            break

    pbar.close()
    env.close()

    print("=" * 150)
    print(f"{controller_name} 仿真完成！")
    print(f"预热阶段摘要: {min(warmup_steps_adjusted * step_size_multiplier, len(timestamps))} 步")
    print(f"MPC控制阶段摘要: {max(0, len(timestamps) - warmup_steps_adjusted * step_size_multiplier)} 步")

    results = {
        'x_history': x_history,
        'u_history': u_history,
        'delta_u_history': delta_u_history,
        'cost_history': cost_history,
        'power_history': power_history,
        'extra_vars_history': extra_vars_history,
        'timestamps': np.array(timestamps),
        'cost_fn': cost_fn,
        'controller_name': controller_name,
        'fmu_available': env.fmu_available,
        'warmup_steps': warmup_steps_adjusted * step_size_multiplier,
        'mpc_steps': len(timestamps) - warmup_steps_adjusted * step_size_multiplier,
        'config': config,
    }

    return results


def plot_thermal_mpc_results(results: Dict):
    """
    生成MPC控制结果的综合绘图。
    包括鼓风机流量（QFLOW_blower）的显示。
    """
    x_hist = results['x_history']
    u_hist = results['u_history']
    delta_u_hist = results['delta_u_history']
    cost_hist = results['cost_history']
    power_hist = results['power_history']
    extra_vars_hist = results['extra_vars_history']
    timestamps = results['timestamps']
    controller_name = results.get('controller_name', 'MPC')
    fmu_available = results.get('fmu_available', False)
    warmup_steps = results.get('warmup_steps', 0)
    config = results['config']

    action_names = config.action_names
    temp_targets = config.temp_targets

    n_steps = x_hist.shape[1] - 1
    time_x = np.arange(x_hist.shape[1])
    time_u = np.arange(n_steps)

    # 标题前缀
    mode_str = "FMU" if fmu_available else "仿真模式"

    # ============================================================
    # 创建一个大Figure，包含所有subplot
    # ============================================================
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f'热管理系统MPC控制结果 - {controller_name} [{mode_str}]',
                 fontsize=16, fontweight='bold')

    # ---- 第1个subplot: 温度控制 ----
    ax1 = plt.subplot(4, 2, 1)
    ax1.plot(time_x, x_hist[0, :], 'b-', linewidth=2, label='车舱温度')
    ax1.plot(time_x, x_hist[1, :], 'r-', linewidth=2, label='电池温度')
    ax1.plot(time_x, x_hist[2, :], 'g-', linewidth=2, label='电机温度')
    ax1.axhline(temp_targets['cabin'], color='b', linestyle='--', alpha=0.3)
    ax1.axhline(temp_targets['battery'], color='r', linestyle='--', alpha=0.3)
    ax1.axhline(temp_targets['motor'], color='g', linestyle='--', alpha=0.3)
    if warmup_steps > 0:
        ax1.axvline(warmup_steps, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax1.set_ylabel('温度 (°C)', fontsize=10)
    ax1.set_title('1) 驾舱、电池、电机温度', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)

    # ---- 第2个subplot: 部件能耗 ----
    ax2 = plt.subplot(4, 2, 2)
    ax2.plot(time_u, power_hist[:, 0], linewidth=2, label='鼓风机 (P1)')
    ax2.plot(time_u, power_hist[:, 1], linewidth=2, label='压缩机 (P2)')
    ax2.plot(time_u, power_hist[:, 2], linewidth=2, label='电池泵 (P3)')
    ax2.plot(time_u, power_hist[:, 3], linewidth=2, label='电机泵 (P4)')
    total_power = power_hist.sum(axis=1)
    ax2.plot(time_u, total_power, 'k--', linewidth=2.5, label='总功率')
    if warmup_steps > 0:
        ax2.axvline(warmup_steps, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax2.set_ylabel('功率 (kW)', fontsize=10)
    ax2.set_title('2) 部件能耗', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)

    # ---- 第3个subplot: 踏板行程、车速、电池SOC ----
    ax3 = plt.subplot(4, 2, 3)
    ax3_twin1 = ax3.twinx()
    ax3_twin2 = ax3.twinx()
    ax3_twin2.spines['right'].set_position(('outward', 60))

    p1, = ax3.plot(time_u, extra_vars_hist[:, 0], 'b-', linewidth=2, label='油门踏板')
    p2, = ax3.plot(time_u, extra_vars_hist[:, 1], 'r-', linewidth=2, label='制动踏板')
    p3, = ax3_twin1.plot(time_u, extra_vars_hist[:, 2], 'g-', linewidth=2, label='车速')
    p4, = ax3_twin2.plot(time_u, extra_vars_hist[:, 3] * 100, 'purple', linewidth=2, label='电池SOC')

    ax3.set_ylabel('踏板行程 (%)', fontsize=10, color='k')
    ax3_twin1.set_ylabel('车速 (km/h)', fontsize=10, color='g')
    ax3_twin2.set_ylabel('电池SOC (%)', fontsize=10, color='purple')

    if warmup_steps > 0:
        ax3.axvline(warmup_steps, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax3.set_title('3) 踏板行程、车速、电池SOC', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    lines = [p1, p2, p3, p4]
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left', fontsize=9)

    # ---- 第4个subplot: 控制输入（各部件转速）+ 鼓风机流量 ----
    ax4 = plt.subplot(4, 2, 4)
    colors_rpm = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (name, color) in enumerate(zip(action_names, colors_rpm)):
        ax4.plot(time_u, u_hist[i, :], linewidth=2, label=name, color=color)

    # 添加鼓风机流量（在第二y轴上）
    ax4_twin = ax4.twinx()
    qflow_data = extra_vars_hist[:, 4] * 1000  # 转换为 L/s 以便显示
    ax4_twin.plot(time_u, qflow_data, 'purple', linewidth=2, linestyle='--', label='QFLOW_blower (L/s)')
    ax4_twin.set_ylabel('鼓风机流量 (L/s)', fontsize=10, color='purple')

    if warmup_steps > 0:
        ax4.axvline(warmup_steps, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax4.set_ylabel('转速 (RPM)', fontsize=10)
    ax4.set_title('4) 各部件转速 + 鼓风机流量', fontsize=11, fontweight='bold')
    ax4.set_ylim([0, 5500])
    ax4.grid(True, alpha=0.3)

    # 合并图例
    lines1 = [plt.Line2D([0], [0], color=c, linewidth=2) for c in colors_rpm]
    lines1.append(plt.Line2D([0], [0], color='purple', linewidth=2, linestyle='--'))
    labels1 = action_names + ['QFLOW_blower']
    ax4.legend(lines1, labels1, loc='best', fontsize=9)

    # ---- 第5个subplot: 控制增量 ----
    ax5 = plt.subplot(4, 2, 5)
    for i, (name, color) in enumerate(zip(action_names, colors_rpm)):
        ax5.plot(time_u, delta_u_hist[i, :], linewidth=2, label=name, color=color)
    ax5.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    if warmup_steps > 0:
        ax5.axvline(warmup_steps, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax5.set_ylabel('转速增量 (RPM/step)', fontsize=10)
    ax5.set_title('5) 控制增量', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='best', fontsize=9)

    # ---- 第6个subplot: 累积成本 ----
    ax6 = plt.subplot(4, 2, 6)
    cumulative_cost = np.cumsum(cost_hist)
    ax6.plot(time_u, cost_hist, label='瞬时成本', linewidth=2, alpha=0.7)
    ax6.plot(time_u, cumulative_cost, label='累积成本', linewidth=2.5)
    if warmup_steps > 0:
        ax6.axvline(warmup_steps, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax6.set_ylabel('成本', fontsize=10)
    ax6.set_title('6) 控制成本演化', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='best', fontsize=9)

    # ---- 第7个subplot: 温度跟踪误差（带符号） ----
    ax7 = plt.subplot(4, 2, 7)
    targets = np.array([temp_targets['cabin'], temp_targets['battery'], temp_targets['motor']])
    signed_errors = x_hist - targets.reshape(-1, 1)

    ax7.plot(time_x, signed_errors[0, :], linewidth=2, label='车舱误差')
    ax7.plot(time_x, signed_errors[1, :], linewidth=2, label='电池误差')
    ax7.plot(time_x, signed_errors[2, :], linewidth=2, label='电机误差')
    ax7.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    if warmup_steps > 0:
        ax7.axvline(warmup_steps, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax7.set_xlabel('时间步', fontsize=10)
    ax7.set_ylabel('温度误差 (°C)', fontsize=10)
    ax7.set_title('7) 温度跟踪误差（带符号）', fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.legend(loc='best', fontsize=9)

    # ---- 第8个subplot: 增量约束监测 ----
    ax8 = plt.subplot(4, 2, 8)
    delta_bounds = config.delta_rpm_bounds
    colors_names = ['blower', 'comp', 'batt', 'motor']

    for i, (delta, name, color) in enumerate(zip(delta_u_hist, colors_names, colors_rpm)):
        ax8.scatter(time_u, delta, alpha=0.6, s=20, label=name, color=color)

    # 绘制每个部件的约束线
    for i, (bounds, name) in enumerate(zip(delta_bounds, colors_names)):
        ax8.axhline(bounds[0], color='r', linestyle='--', linewidth=1, alpha=0.3)
        ax8.axhline(bounds[1], color='g', linestyle='--', linewidth=1, alpha=0.3)

    ax8.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    if warmup_steps > 0:
        ax8.axvline(warmup_steps, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax8.set_xlabel('时间步', fontsize=10)
    ax8.set_ylabel('增量 (RPM/step)', fontsize=10)
    ax8.set_title('8) 增量约束监测', fontsize=11, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.legend(loc='best', fontsize=8)

    plt.tight_layout()
    plt.show()

    # 保存图表
    if not os.path.exists('results'):
        os.makedirs('results')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.savefig(f'results/mpc_results_{timestamp}.png', dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: results/mpc_results_{timestamp}.png")


def print_summary_statistics(results: Dict):
    """打印仿真结果的摘要统计。"""
    x_hist = results['x_history']
    u_hist = results['u_history']
    delta_u_hist = results['delta_u_history']
    cost_hist = results['cost_history']
    power_hist = results['power_history']
    extra_vars_hist = results['extra_vars_history']
    controller_name = results.get('controller_name', 'MPC')
    fmu_available = results.get('fmu_available', False)
    warmup_steps = results.get('warmup_steps', 0)
    mpc_steps = results.get('mpc_steps', 0)
    config = results['config']

    mode_str = "FMU" if fmu_available else "仿真模式"
    temp_targets = config.temp_targets
    targets = np.array([temp_targets['cabin'], temp_targets['battery'], temp_targets['motor']])
    action_names = config.action_names
    power_names_short = ['鼓风机', '压缩机', '电池泵', '电机泵']

    print("\n" + "=" * 150)
    print(f"仿真摘要统计 - {controller_name} [{mode_str}]")
    print("=" * 150)

    print(f"\n仿真配置:")
    print(f"  step_size_multiplier: {config.step_size_multiplier}")
    print(f"  预热阶段: {warmup_steps} 步（MPC控制周期）")
    print(f"    - 实际FMU步数: {warmup_steps * config.step_size_multiplier}")
    print(f"    - RPM设定: blower={config.warmup_rpm[0]:.0f}, comp={config.warmup_rpm[1]:.0f}, "
          f"batt={config.warmup_rpm[2]:.0f}, motor={config.warmup_rpm[3]:.0f}")
    print(f"  MPC控制阶段: {mpc_steps} 步（MPC控制周期）")
    print(f"    - 实际FMU步数: {mpc_steps * config.step_size_multiplier}")
    print(f"    - 增量约束:")
    for i, (name, bounds) in enumerate(zip(['blower', 'comp', 'batt', 'motor'], config.delta_rpm_bounds)):
        print(f"      {name:>6}: [{bounds[0]:+6.0f}, {bounds[1]:+6.0f}] RPM/step")
    print(f"    - 绝对转速约束:")
    for i, (name, bounds) in enumerate(zip(['blower', 'comp', 'batt', 'motor'], config.rpm_bounds)):
        print(f"      {name:>6}: [{bounds[0]:6.0f}, {bounds[1]:6.0f}] RPM")
    print(f"  总步数: {len(cost_hist)}")

    # 温度统计
    for i, (name, target) in enumerate(zip(
        ['车舱', '电池', '电机'],
        targets
    )):
        print(f"\n{name}温度:")

        if warmup_steps > 0:
            temps_warmup = x_hist[i, :warmup_steps+1]
            error_warmup = temps_warmup - target
            print(f"  预热阶段:")
            print(f"    平均温度: {np.mean(temps_warmup):.2f}°C")
            print(f"    标准差: {np.std(temps_warmup):.2f}°C")
            print(f"    平均误差: {np.mean(error_warmup):+.2f}°C")

        temps_ctrl = x_hist[i, warmup_steps:]
        error_ctrl = temps_ctrl - target
        print(f"  控制阶段:")
        print(f"    目标温度: {target:.1f}°C")
        print(f"    平均温度: {np.mean(temps_ctrl):.2f}°C")
        print(f"    标准差: {np.std(temps_ctrl):.2f}°C")
        print(f"    最小值: {np.min(temps_ctrl):.2f}°C，最大值: {np.max(temps_ctrl):.2f}°C")
        print(f"    平均误差: {np.mean(error_ctrl):+.2f}°C")
        print(f"    误差范围: {np.min(error_ctrl):+.2f}°C ~ {np.max(error_ctrl):+.2f}°C")

    # 控制输入统计
    print(f"\n控制输入（控制阶段）:")
    for i, name in enumerate(action_names):
        u_ctrl = u_hist[i, warmup_steps:]
        delta_u_ctrl = delta_u_hist[i, warmup_steps:]
        mean_rpm = np.mean(u_ctrl)
        max_rpm = np.max(u_ctrl)
        min_rpm = np.min(u_ctrl)
        mean_delta = np.mean(delta_u_ctrl)
        max_delta = np.max(np.abs(delta_u_ctrl))

        print(f"  {name:>12}:")
        print(f"    转速范围: {min_rpm:6.0f} ~ {max_rpm:6.0f} RPM (平均={mean_rpm:6.0f})")
        print(f"    增量统计: 平均={mean_delta:+6.0f}, 最大幅度={max_delta:6.0f} RPM/step")

    # 功耗统计
    print(f"\n功耗分析（从FMU获取，控制阶段）:")
    power_ctrl = power_hist[warmup_steps:]
    for i, name in enumerate(power_names_short):
        mean_p = np.mean(power_ctrl[:, i])
        max_p = np.max(power_ctrl[:, i])
        print(f"  {name:>6}: 平均={mean_p:7.3f}kW, 最大={max_p:7.3f}kW")

    total_power = power_ctrl.sum(axis=1)
    print(f"  总功率: 平均={np.mean(total_power):.3f}kW, 最大={np.max(total_power):.3f}kW, 总计={np.sum(total_power):.2f}kWh")

    # 额外变量统计
    print(f"\n驾驶工况与鼓风机流量（控制阶段）:")
    extra_ctrl = extra_vars_hist[warmup_steps:]
    print(f"  油门踏板: 平均={np.mean(extra_ctrl[:, 0]):.1f}%, 最大={np.max(extra_ctrl[:, 0]):.1f}%")
    print(f"  制动踏板: 平均={np.mean(extra_ctrl[:, 1]):.1f}%, 最大={np.max(extra_ctrl[:, 1]):.1f}%")
    print(f"  车速: 平均={np.mean(extra_ctrl[:, 2]):.1f}km/h, 最大={np.max(extra_ctrl[:, 2]):.1f}km/h")
    print(f"  电池SOC: 平均={np.mean(extra_ctrl[:, 3])*100:.1f}%, 范围=[{np.min(extra_ctrl[:, 3])*100:.1f}%, {np.max(extra_ctrl[:, 3])*100:.1f}%]")
    print(f"  鼓风机流量: 平均={np.mean(extra_ctrl[:, 4]):.4f}m³/s, 最大={np.max(extra_ctrl[:, 4]):.4f}m³/s")

    # 控制性能
    print(f"\n控制性能:")
    cost_ctrl = cost_hist[warmup_steps:]
    print(f"  总成本: {np.sum(cost_ctrl):.2f}")
    print(f"  平均单步成本: {np.mean(cost_ctrl):.4f}")
    print(f"  控制步数: {len(cost_ctrl)}")

    # MPC 超���数
    print(f"\nMPC 超参数:")
    print(f"  预测视野: {config.mpc_horizon}")
    print(f"  成本权重矩阵 Q (状态权重):")
    for i, name in enumerate(['车舱', '电池', '电机']):
        print(f"    {name}: {config.cost_q_diag[i]:.2f}")
    print(f"  成本权重矩阵 R (控制增量权重):")
    for i, name in enumerate(['blower', 'comp', 'batt', 'motor']):
        print(f"    {name}: {config.cost_r_diag[i]:.3f}")
    print(f"  功耗权重系数 (alpha_power): {config.cost_alpha_power:.2e}")

    print("=" * 150 + "\n")


# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 150)
    print("短视野增量型MPC热管理系统控制（FMU）- 优化版本（参数管理集中化）")
    print("=" * 150 + "\n")

    # ========================================================================
    # 方案1：使用默认配置
    # ========================================================================
    print("\n【方案1】使用默认配置运行")
    config_default = ThermalConfig()

    results = run_thermal_mpc_simulation(
        config=config_default,
        sim_steps=50,
        seed=42,
    )

    print_summary_statistics(results)
    plot_thermal_mpc_results(results)

    # 【可选】保存配置到JSON
    config_default.save_to_json('results/default_config.json')

    # ========================================================================
    # 方案2：自定义配置
    # ========================================================================
    # 【示例】修改某个参数，其他保持默认
    # print("\n【方案2】自定义配置 - 修改MPC视野和成本权重")
    # config_custom = ThermalConfig(
    #     mpc_horizon=8,  # 增加预测视野
    #     cost_q_diag=[15.0, 8.0, 8.0],  # 增加温度权重
    #     cost_alpha_power=5e-4,  # 增加功耗权重
    #     step_size_multiplier=2  # 每2步执行一次MPC
    # )
    #
    # results2 = run_thermal_mpc_simulation(
    #     config=config_custom,
    #     sim_steps=100,
    #     seed=42,
    # )
    #
    # print_summary_statistics(results2)
    # plot_thermal_mpc_results(results2)
    # config_custom.save_to_json('results/custom_config.json')

    # ========================================================================
    # 方案3：从JSON加载配置
    # ========================================================================
    # 【示例】从之前保存的配置文件加载
    # print("\n【方案3】从JSON配置文件加载")
    # config_loaded = ThermalConfig.load_from_json('results/custom_config.json')
    #
    # results3 = run_thermal_mpc_simulation(
    #     config=config_loaded,
    #     sim_steps=100,
    #     seed=42,
    # )
    #
    # print_summary_statistics(results3)
    # plot_thermal_mpc_results(results3)

    print("\n所有结果已生成成功！")
