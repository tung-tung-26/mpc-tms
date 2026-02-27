"""
热管理系统的短视野MPC控制
========================================================

演示如何使用简单的短视野MPC（无终止成本）来控制
黑盒热管理FMU系统。

主要组件:
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

作者: 自动生成的集成版本
日期: 2026-02-27
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

# ============================================================================
# 配置中文字体
# ============================================================================
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

# ============================================================================
# 常数定义
# ============================================================================
K_TO_C = 273.15  # 开尔文到摄氏度的转换常数
WARMUP_STEPS = 50  # FMU预热步数（不施加控制）

# 预热阶段的固定转速值（list形式，为4个部件分别设定）
# [鼓风机, 压缩机, 电池泵, 电机泵]
WARMUP_RPM = [50.0, 2000.0, 1000.0, 1000.0]

# FMU路径配置
FMU_PATH = r"..\env\MyITMS.fmu"  # FMU文件路径（相对路径）

# ============================================================================
# MPC 超参数配置
# ============================================================================
# 增量型MPC的增量约束（可为每个部件分别设定）
DELTA_RPM_BOUNDS = [
    (-5.0, 5.0),        # 鼓风机（更小的变化范围）
    (-100.0, 100.0),      # 压缩机
    (-100.0, 100.0),      # 电池泵
    (-100.0, 100.0)       # 电机泵
]

# MPC 的绝对转速约束上下界（可为每个部件分别设定）
# 格式: list of (min, max) 或 单个 (min, max) 适用于所有部件
RPM_BOUNDS = [
    (0.0, 100.0),       # 鼓风机（最大300 RPM）
    (100.0, 4000.0),    # 压缩机（500-2000 RPM）
    (100.0, 2000.0),    # 电池泵（500-2000 RPM）
    (100.0, 2000.0)     # 电机泵（500-2000 RPM）
]

# MPC 成本函数的权重超参数
# 温度目标值（°C）
TEMP_TARGETS = {
    'cabin': 25.0,
    'battery': 35.0,
    'motor': 40.0
}

# 成本函数权重矩阵（已归一化）
# Q: 状态误差权重 (温度跟踪误差)
COST_Q = np.diag([10.0, 5.0, 5.0])  # [车舱, 电池, 电机]

# R: 控制增量权重 (平滑性)
COST_R = np.diag([0.1, 0.1, 0.1, 0.1])  # [鼓风机, 压缩机, 电池泵, 电机泵]

# alpha_power: 功耗权重系数
COST_ALPHA_POWER = 1e-4

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

    def __init__(
        self,
        T_cabin_set: float = 25.0,
        T_bat_set: float = 35.0,
        T_motor_set: float = 40.0,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        alpha_power: float = 1e-4
    ):
        """
        参数说明
        ----------
        T_cabin_set, T_bat_set, T_motor_set : float
            车舱、电池、电机的目标温度（单位: °C��
        Q : np.ndarray, 可选
            状态误差权重矩阵（3x3）
            - 对角线元素越大，该状态对成本的贡献越大
            - 建议值范围: 0.1 ~ 100.0
        R : np.ndarray, 可选
            控制增量权重矩阵（4x4）
            - 对角线元素越大，越倾向于平稳控制（增量小）
            - 建议值范围: 0.001 ~ 1.0
        alpha_power : float
            功耗惩罚的权重系数
            - 建议值范围: 1e-6 ~ 1e-2
        """
        self.x_ref = np.array([
            T_cabin_set,
            T_bat_set,
            T_motor_set
        ], dtype=float)

        if Q is None:
            # 默认权重：车舱权重更大（舒适性优先）
            self.Q = np.diag([10.0, 5.0, 5.0])
        else:
            self.Q = Q

        if R is None:
            # 增量控制的权重：对转速变化的惩罚
            self.R = np.diag([0.1, 0.1, 0.1, 0.1])
        else:
            self.R = R

        self.alpha_power = alpha_power

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

        return float(state_cost + control_cost + power_cost)


# ============================================================================
# 简化的系统模型（用于MPC）
# ============================================================================
class MPCThermalSystem:
    """
    用于MPC的简化系统模型，包装FMU观测。
    只用于轨迹预测，实际动态在FMU中。
    """

    def __init__(
        self,
        n: int = 3,
        m: int = 4,
        dt: float = 1.0,
        rpm_bounds: Optional[List[Tuple]] = None,
        delta_rpm_bounds: Optional[List[Tuple]] = None
    ):
        """
        参数说明
        ----------
        n : int
            状态维数（温度：车舱、电池、电机）
        m : int
            控制维数（4个泵/压缩机转速）
        dt : float
            采样时间（秒）
        rpm_bounds : list of tuples
            绝对转速边界 [(min1, max1), (min2, max2), ...]
        delta_rpm_bounds : list of tuples
            增量转速边界 [(min1, max1), (min2, max2), ...]
        """
        self.n = n
        self.m = m
        self.dt = dt

        # 成本函数权重
        self.Q = np.diag([10.0, 5.0, 5.0])
        self.R = np.diag([0.1, 0.1, 0.1, 0.1])  # 增量控制的权重

        # 边界
        self.temp_bounds = (10.0, 60.0)  # 单位: °C

        # 转速边界
        if rpm_bounds is None:
            self.rpm_bounds = [(0.0, 5000.0) for _ in range(m)]
        else:
            self.rpm_bounds = rpm_bounds if isinstance(rpm_bounds[0], (tuple, list)) else [(rpm_bounds[0], rpm_bounds[1]) for _ in range(m)]

        if delta_rpm_bounds is None:
            self.delta_rpm_bounds = [(-500.0, 500.0) for _ in range(m)]
        else:
            self.delta_rpm_bounds = delta_rpm_bounds if isinstance(delta_rpm_bounds[0], (tuple, list)) else [(delta_rpm_bounds[0], delta_rpm_bounds[1]) for _ in range(m)]

        # 为了兼容archive的MPC类
        self.v_bounds = (0.0, 5000.0)
        self.omega_bounds = (0.0, 0.0)  # 在热管理中未使用

    def stage_cost(self, x: np.ndarray, delta_u: np.ndarray) -> float:
        """二次型成本。"""
        x = np.asarray(x).flatten()
        delta_u = np.asarray(delta_u).flatten()

        state_cost = float(x.T @ self.Q @ x)
        control_cost = float(delta_u.T @ self.R @ delta_u)

        return state_cost + control_cost

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        简化的线性动态：x_{k+1} = A*x_k + B*u_k
        这是占位符；真实动态在FMU中实现。

        注意：这里的u是绝对转速，不是增量
        """
        x = np.asarray(x).flatten()
        u = np.asarray(u).flatten()

        # 非常简单的衰减动态（不进行控制时温度缓慢回到环境温度）
        A = np.eye(3) * 0.95
        B = np.array([
            [0.001, 0.0, 0.0, 0.0],      # 车舱：受空调影响
            [0.0, 0.002, 0.001, 0.0],    # 电池：受冷却液影响
            [0.0, 0.0, 0.001, 0.002]     # 电机：受冷却液和电机泵影响
        ])

        x_next = A @ x + B @ u
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
                 horizon: int = 5,
                 cost_fn: Optional[ThermalMPCCost] = None):
        """
        参数说明
        ----------
        system : MPCThermalSystem
            用于预测的系统模型
        horizon : int
            预测视野（N）
        cost_fn : ThermalMPCCost, 可选
            成本函数（若为None则使用默认值）
        """
        self.system = system
        self.horizon = horizon
        self.cost_fn = cost_fn or ThermalMPCCost()
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
                u_j_clipped = ca.fmax(ca.fmin(u[j], self.system.rpm_bounds[j][1]), self.system.rpm_bounds[j][0])
                u_clipped_list.append(u_j_clipped)

            # 使用 ca.vertcat 构建列向量
            u_clipped = ca.vertcat(*u_clipped_list)

            # 单步成本（使用带符号的误差）
            x_ref = ca.DM(self.cost_fn.x_ref)
            error = x - x_ref
            state_cost = ca.mtimes([error.T, self.system.Q, error])
            control_cost = ca.mtimes([dU_i.T, self.system.R, dU_i])
            J += state_cost + control_cost

            # 简单的线性动态预测（使用约束后的控制值）
            A = ca.DM(np.eye(3) * 0.95)
            B = ca.DM(np.array([
                [0.001, 0.0, 0.0, 0.0],
                [0.0, 0.002, 0.001, 0.0],
                [0.0, 0.0, 0.001, 0.002]
            ]))

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
            'ipopt.max_iter': 1000,
            'ipopt.tol': 1e-4,  # 降低精度要求以加快求解
            'ipopt.acceptable_tol': 1e-3,
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
                u_curr[j] = np.clip(u_curr[j], self.system.rpm_bounds[j][0], self.system.rpm_bounds[j][1])
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

    def __init__(
        self,
        fmu_path: Optional[str] = None,
        step_size: float = 1.0,
        observation_names: Optional[List[str]] = None,
        action_names: Optional[List[str]] = None,
        power_names: Optional[List[str]] = None,
        extra_var_names: Optional[List[str]] = None,
        init_dict: Optional[Dict[str, Any]] = None,
        max_steps: int = 600
    ):
        """
        参数说明
        ----------
        fmu_path : str, 可选
            .fmu文件路径。若为None，进入仅仿真模式
        step_size : float
            FMU步长（秒）
        observation_names : list of str
            观测的温度变量名称
        action_names : list of str
            控制的变量名称（4个泵/压缩机）
        power_names : list of str
            功耗的变量名称
        extra_var_names : list of str
            额外的监测变量（踏板行程、车速、SOC、鼓风机流量等）
        init_dict : dict
            FMU变量的初始值
        max_steps : int
            每个episode的最大仿真步数
        """
        self.fmu_path = pathlib.Path(fmu_path) if fmu_path else None
        self.step_size = step_size

        # 观测和控制变量名称
        self.observation_names = observation_names or [
            "cabinVolume.summary.T",
            "battery.Batt_top[1].T",
            "machine.heatCapacitor.T"
        ]
        self.action_names = action_names or [
            "RPM_blower",
            "RPM_comp",
            "RPM_batt",
            "RPM_motor"
        ]
        # 功耗变量名称（从FMU获取）
        # 对应关系：TableDC3->鼓风机, TableDC->压缩机, TableDC1->电池泵, TableDC2->电机泵
        self.power_names = power_names or [
            "TableDC3.Pe",       # 鼓风机功率
            "TableDC.Pe",        # 压缩机功率
            "TableDC1.Pe",       # 电池泵功率
            "TableDC2.Pe"        # 电机泵功率
        ]
        # 额外监测变量（包括鼓风机流量）
        self.extra_var_names = extra_var_names or [
            "driverPerformance.controlBus.driverBus._acc_pedal_travel",    # 油门踏板
            "driverPerformance.controlBus.driverBus._brake_pedal_travel",  # 制动踏板
            "driverPerformance.controlBus.vehicleStatus.vehicle_velocity",  # 车速
            "battery.controlBus.batteryBus.battery_SOC[1]",                # 电池SOC
            "fan2Table.qflow"                                               # 鼓风机流量 (m^3/h)
        ]

        self.init_dict = init_dict
        self.max_steps = max_steps

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
        self._sim_state = np.array([25.0, 35.0, 40.0])

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

            if self.init_dict:
                for k, v in self.init_dict.items():
                    if k in self.vrs:
                        self._fmu.setReal([self.vrs[k][0]], [float(v)])

            self._fmu.exitInitializationMode()
        except Exception as e:
            warnings.warn(f"未能创建FMU实例: {e}")
            self.fmu_available = False

    def reset(self, init_dict: Optional[Dict] = None, seed: Optional[int] = None):
        """重置环境并返回初始观测。"""
        if seed is not None:
            np.random.seed(seed)

        if init_dict is not None:
            self.init_dict = init_dict

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
            self._sim_state = np.array([25.0, 35.0, 40.0]) + np.random.randn(3) * 0.5

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
            额外变量字���
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
            A = np.eye(3) * 0.95
            B = np.array([
                [0.001, 0.0, 0.0, 0.0],
                [0.0, 0.002, 0.001, 0.0],
                [0.0, 0.0, 0.001, 0.002]
            ])

            self._sim_state = A @ self._sim_state + B @ u
            self._sim_state = np.clip(self._sim_state, 10.0, 60.0)

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
        """将控制数组转换为动作字典��"""
        return {name: float(u[i]) for i, name in enumerate(self.action_names)}


# ============================================================================
# 主要仿真和绘图
# ============================================================================
def run_thermal_mpc_simulation(
    fmu_path: Optional[str] = None,
    sim_steps: int = 280,
    horizon: int = 5,
    seed: int = 42,
    controller_name: str = "增量型MPC (N=5)",
    warmup_steps: int = WARMUP_STEPS,
    warmup_rpm: Union[float, int, List] = WARMUP_RPM,
    delta_rpm_bounds: Union[Tuple, List] = DELTA_RPM_BOUNDS,
    rpm_bounds: Union[Tuple, List] = RPM_BOUNDS,
    cost_q: Optional[np.ndarray] = None,
    cost_r: Optional[np.ndarray] = None,
    cost_alpha_power: float = COST_ALPHA_POWER
) -> Dict:
    """
    运行MPC控制的热管理FMU仿真。

    参数说明
    ----------
    fmu_path : str, 可选
        FMU文件路径。若为None，进入仅仿真模式
    sim_steps : int
        仿真步数（不包括预热步数）
    horizon : int
        MPC预测视野
    seed : int
        随机种子
    controller_name : str
        控制器名称（用于绘图标签）
    warmup_steps : int
        FMU预热步数（不施加MPC控制）
    warmup_rpm : float, int, or list
        预热阶段的固定RPM值
    delta_rpm_bounds : tuple or list of tuples
        增量约束
    rpm_bounds : tuple or list of tuples
        绝对转速约束
    cost_q : np.ndarray
        成本函数的Q矩阵
    cost_r : np.ndarray
        成本函数的R矩阵
    cost_alpha_power : float
        功耗权重系数

    返回值
    -------
    results : dict
        包含所有仿真结果
    """
    np.random.seed(seed)

    # 转换参数
    warmup_rpm_array = convert_warmup_rpm(warmup_rpm)
    delta_rpm_bounds_list = convert_bounds(delta_rpm_bounds)
    rpm_bounds_list = convert_bounds(rpm_bounds)

    # 初始化系统和控制器
    sys = MPCThermalSystem(
        n=3,
        m=4,
        dt=1.0,
        rpm_bounds=rpm_bounds_list,
        delta_rpm_bounds=delta_rpm_bounds_list
    )

    if cost_q is None:
        cost_q = COST_Q
    if cost_r is None:
        cost_r = COST_R

    cost_fn = ThermalMPCCost(
        T_cabin_set=TEMP_TARGETS['cabin'],
        T_bat_set=TEMP_TARGETS['battery'],
        T_motor_set=TEMP_TARGETS['motor'],
        Q=cost_q,
        R=cost_r,
        alpha_power=cost_alpha_power
    )
    controller = ThermalMPCControllerDelta(
        sys,
        horizon=horizon,
        cost_fn=cost_fn
    )

    # 初始化FMU环境
    init_dict = {
        'MY_socinit': 0.5,
        'MY_battT0': 303.15,
        'MY_motorT0': 333.15
    }

    env = EnhancedFMUITMS(
        fmu_path=fmu_path,
        step_size=1.0,
        init_dict=init_dict,
        max_steps=warmup_steps + sim_steps
    )

    # 总步数
    total_steps = warmup_steps + sim_steps

    # 运行仿真
    print(f"\n开始 {controller_name} 仿真")
    print(f"预热阶段: {warmup_steps} 步")
    print(f"  - RPM设定: blower={warmup_rpm_array[0]:.0f}, comp={warmup_rpm_array[1]:.0f}, "
          f"batt={warmup_rpm_array[2]:.0f}, motor={warmup_rpm_array[3]:.0f}")
    print(f"MPC控制: {sim_steps} 步")
    print(f"  - 预测视野: {horizon}")
    print(f"  - 增量约束与绝对转速约束已配置")
    print("=" * 150)

    obs = env.reset(init_dict, seed=seed)
    x_k = env.obs_to_array(obs)

    x_history = np.zeros((3, total_steps + 1))
    u_history = np.zeros((4, total_steps))
    delta_u_history = np.zeros((4, total_steps))
    cost_history = np.zeros(total_steps)
    power_history = np.zeros((total_steps, 4))
    extra_vars_history = np.zeros((total_steps, 5))  # 增加一个维度用于qflow
    timestamps = []

    x_history[:, 0] = x_k

    # 初始化控制值
    u_current = warmup_rpm_array.copy()
    u_history[:, 0] = u_current

    # 使用tqdm显示进度条
    pbar = tqdm(range(total_steps), desc=f"{controller_name} 运行中", unit="步")

    for k in pbar:
        timestamps.append(k)

        if k < warmup_steps:
            u_k = warmup_rpm_array.copy()
            delta_u_k = np.zeros(4)
            phase_label = "预热"
        else:
            try:
                delta_u_seq, _, _ = controller.solve(x_k, u_current)
                delta_u_k = delta_u_seq[:, 0]
            except Exception as e:
                warnings.warn(f"MPC求解在步骤 {k} 失败: {e}")
                delta_u_k = np.zeros(4)

            phase_label = "控制"

        # 更新绝对控制值
        u_k = u_current + delta_u_k
        for j in range(4):
            u_k[j] = np.clip(u_k[j], rpm_bounds_list[j][0], rpm_bounds_list[j][1])

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
        extra_vars_history[k, 4] = extra_vars.get('qflow', 0.0)  # m^3/s

        temp_errors = x_next - np.array([TEMP_TARGETS['cabin'], TEMP_TARGETS['battery'], TEMP_TARGETS['motor']])

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
            'ΔU_blower': f'{delta_u_k[0]:+7.0f}',
            'ΔU_comp': f'{delta_u_k[1]:+7.0f}',
            'ΔU_batt': f'{delta_u_k[2]:+7.0f}',
            'ΔU_motor': f'{delta_u_k[3]:+7.0f}',
            'P_blower': f'{power_history[k, 0]:6.3f}kW',
            'P_comp': f'{power_history[k, 1]:6.3f}kW',
            'P_batt': f'{power_history[k, 2]:6.3f}kW',
            'P_motor': f'{power_history[k, 3]:6.3f}kW',
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
    print(f"预热阶段摘要: {min(warmup_steps, len(timestamps))} 步")
    print(f"MPC控制阶段摘要: {max(0, len(timestamps) - warmup_steps)} 步")

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
        'warmup_steps': warmup_steps,
        'mpc_steps': len(timestamps) - warmup_steps,
        'action_names': env.action_names,
        'warmup_rpm': warmup_rpm_array,
        'delta_rpm_bounds': delta_rpm_bounds_list,
        'rpm_bounds': rpm_bounds_list,
        'cost_q': cost_q if cost_q is not None else COST_Q,
        'cost_r': cost_r if cost_r is not None else COST_R,
        'cost_alpha_power': cost_alpha_power,
        'temp_targets': TEMP_TARGETS
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
    action_names = results.get('action_names', ['RPM_blower', 'RPM_comp', 'RPM_batt', 'RPM_motor'])

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
    ax1.axhline(25.0, color='b', linestyle='--', alpha=0.3)
    ax1.axhline(35.0, color='r', linestyle='--', alpha=0.3)
    ax1.axhline(40.0, color='g', linestyle='--', alpha=0.3)
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
    targets = np.array([TEMP_TARGETS['cabin'], TEMP_TARGETS['battery'], TEMP_TARGETS['motor']])
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
    delta_bounds = results.get('delta_rpm_bounds', [(-500, 500) for _ in range(4)])
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
    action_names = results.get('action_names', ['RPM_blower', 'RPM_comp', 'RPM_batt', 'RPM_motor'])
    warmup_rpm = results.get('warmup_rpm', np.array([1000, 1000, 1000, 1000]))
    delta_rpm_bounds = results.get('delta_rpm_bounds', [(-500, 500) for _ in range(4)])
    rpm_bounds = results.get('rpm_bounds', [(0, 5000) for _ in range(4)])

    mode_str = "FMU" if fmu_available else "仿真模式"
    temp_targets = results.get('temp_targets', TEMP_TARGETS)
    targets = np.array([temp_targets['cabin'], temp_targets['battery'], temp_targets['motor']])

    print("\n" + "=" * 150)
    print(f"仿真摘要统计 - {controller_name} [{mode_str}]")
    print("=" * 150)

    print(f"\n仿真配置:")
    print(f"  预热阶段: {warmup_steps} 步")
    print(f"    - RPM设定: blower={warmup_rpm[0]:.0f}, comp={warmup_rpm[1]:.0f}, "
          f"batt={warmup_rpm[2]:.0f}, motor={warmup_rpm[3]:.0f}")
    print(f"  MPC控制阶段: {mpc_steps} 步")
    print(f"    - 增量约束:")
    for i, (name, bounds) in enumerate(zip(['blower', 'comp', 'batt', 'motor'], delta_rpm_bounds)):
        print(f"      {name:>6}: [{bounds[0]:+6.0f}, {bounds[1]:+6.0f}] RPM/step")
    print(f"    - 绝对转速约束:")
    for i, (name, bounds) in enumerate(zip(['blower', 'comp', 'batt', 'motor'], rpm_bounds)):
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
    power_names_short = ['鼓风机', '压缩机', '电池泵', '电机泵']
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

    # MPC 超参数
    print(f"\nMPC 超参数:")
    print(f"  成本权重矩阵 Q (状态权重):")
    cost_q = results.get('cost_q', COST_Q)
    for i in range(3):
        print(f"    {cost_q[i, i]:.2f}")
    print(f"  成本权重矩阵 R (控制增量权重):")
    cost_r = results.get('cost_r', COST_R)
    for i in range(4):
        print(f"    {cost_r[i, i]:.3f}")
    print(f"  功耗权重系数 (alpha_power): {results.get('cost_alpha_power', COST_ALPHA_POWER):.2e}")

    print("=" * 150 + "\n")


# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 150)
    print("短视野增量型MPC热管理系统控制（FMU）- 带预热阶段")
    print("=" * 150 + "\n")

    # 运行仿真
    results = run_thermal_mpc_simulation(
        fmu_path=FMU_PATH,
        sim_steps=100,
        horizon=5,
        seed=42,
        controller_name="增量型MPC (N=5)",
        warmup_steps=WARMUP_STEPS,
        warmup_rpm=WARMUP_RPM,
        delta_rpm_bounds=DELTA_RPM_BOUNDS,
        rpm_bounds=RPM_BOUNDS,
        cost_q=COST_Q,
        cost_r=COST_R,
        cost_alpha_power=COST_ALPHA_POWER
    )

    # 打印统计信息
    print_summary_statistics(results)

    # 生成绘图
    plot_thermal_mpc_results(results)

    print("\n所有结果已生成成功！")
