import numpy as np


def generate_init_dict(spec, seed=None):
    if seed is not None: np.random.seed(seed)
    out = {}
    for k, (lo, hi, st) in spec.items():
        lo = float(lo)
        hi = float(hi)
        st = float(st)
        if lo > hi or st <= 0:
            out[k] = lo
            continue
        n = int(np.floor((hi - lo) / st)) + 1
        if n <= 0: raise ValueError
        idx = np.random.randint(0, n)
        v = lo + idx * st
        if np.isclose(st, round(st)): v = float(int(round(v)))
        out[k] = v
    return out

def fill_observation(obs_dict, obs):
    filled = []
    for row in obs_dict:
        new_row = []
        for item in row:
            if isinstance(item, str):
                # 如果是字符串路径，尝试从 obs 中取值
                if item in obs:
                    new_row.append(obs[item])
                else:
                    raise KeyError(f"Key not found in obs: {item}")
            else:
                # 非字符串（如 T_cabin_set），保留原值
                new_row.append(item)
        filled.append(new_row)
    return filled


class ThermalMPCCost:

    def __init__(
        self,
        T_cabin_set,
        T_bat_set,
        T_motor_set,
        Q=None,
        alpha_power=1e-3
    ):
        """
        Q: 状态误差权重矩阵
        alpha_power: 功率惩罚系数
        """

        # 目标温度
        self.x_ref = np.array([
            T_cabin_set,
            T_bat_set,
            T_motor_set
        ], dtype=float)

        # 状态权重矩阵
        if Q is None:
            self.Q = np.diag([
                10.0,  # cabin 权重大
                5.0,   # battery
                5.0    # motor
            ])
        else:
            self.Q = Q

        # 功率权重
        self.alpha_power = alpha_power

    def stage_cost(self, x, power_cabin, power_refrigerant, power_coolant):
        """
        x: [T_cabin, T_bat, T_motor]
        """

        x = np.asarray(x)

        # -------- 状态误差 --------
        error = x - self.x_ref
        state_cost = error.T @ self.Q @ error

        # -------- 总功率 --------
        total_power = (
            power_cabin +
            power_refrigerant +
            power_coolant
        )

        power_cost = self.alpha_power * total_power

        return float(state_cost + power_cost)



import numpy as np
import warnings


class SimulateSystemFMUITMS:

    def __init__(
        self,
        env,
        controller,
        observation_list,
        action_names,
        cost_function=None
    ):
        self.env = env
        self.controller = controller
        self.observation_list = observation_list
        self.action_names = action_names
        self.cost_function = cost_function

        self.n = len(observation_list)
        self.m = len(action_names)

    def _obs_dict_to_array(self, obs_dict):
        return np.array(
            [obs_dict[name] for name in self.observation_list],
            dtype=float
        )

    def _u_array_to_action_dict(self, u):
        return {
            name: u[i]
            for i, name in enumerate(self.action_names)
        }

    def simulate(self, init_dict, sim_steps):

        obs_dict = self.env.reset(init_dict)

        x0 = self._obs_dict_to_array(obs_dict)

        x_history = np.zeros((self.n, sim_steps + 1))
        u_history = np.zeros((self.m, sim_steps))
        cost_history = np.zeros(sim_steps)

        x_history[:, 0] = x0

        last_valid_u = np.zeros(self.m)
        consecutive_failures = 0

        for k in range(sim_steps):

            x_k = x_history[:, k]

            # ---------- MPC 求解 ----------
            try:
                u_seq, _, _ = self.controller.solve(x_k)

                if u_seq.shape[1] >= 1:
                    u_k = u_seq[:, 0]
                    last_valid_u = u_k.copy()
                    consecutive_failures = 0
                else:
                    u_k = last_valid_u.copy()
                    consecutive_failures += 1

            except Exception as e:
                u_k = last_valid_u.copy()
                consecutive_failures += 1
                warnings.warn(f"MPC solve failed at step {k}: {e}")

            if consecutive_failures > 3:
                u_k *= 0.8

            # ---------- 写入 FMU ----------
            action_dict = self._u_array_to_action_dict(u_k)

            obs_dict, terminated, truncated = self.env.step(action_dict)

            x_next = self._obs_dict_to_array(obs_dict)

            x_history[:, k + 1] = x_next
            u_history[:, k] = u_k

            # ---------- 计算真实 stage cost ----------
            if self.cost_function is not None:

                power_cabin = obs_dict["power_cabin"]
                power_refrigerant = obs_dict["power_refrigerant"]
                power_coolant = obs_dict["power_coolant"]

                cost_history[k] = self.cost_function.stage_cost(
                    x_k,
                    power_cabin,
                    power_refrigerant,
                    power_coolant
                )

            if terminated or truncated:
                print(f"Simulation ended at step {k}")
                break

        return x_history, u_history, cost_history

if __name__ == '__main__':
    from config.base_config import config
    from env.fmu_env_itms import FMUITMS
    env = FMUITMS(fmu_path=config["fmu_path"], step_size=config["fmu_step_size"])
    init_dict = generate_init_dict(config["env_reset_dict"])
    obs_raw = env.reset(init_dict)
    obs = fill_observation(config["obs_dict"], obs_raw)
    mpc_obs_dict = [
        "cabinVolume.summary.T",
        "battery.Batt_top[1].T",
        "machine.heatCapacitor.T",
    ]
    mpc_action_dict = [
        "RPM_blower",
        "RPM_comp",
        "RPM_batt",
        "RPM_motor",
    ]
    mpc = MPC(sys, N, None, None)
    cost_fn = ThermalMPCCost(
        T_cabin_set=25,
        T_bat_set=35,
        T_motor_set=40,
        alpha_power=1e-3
    )

    sim = SimulateSystemFMUITMS(
        env=env,
        controller=mpc,
        observation_list=mpc_obs_dict,
        action_names=mpc_action_dict,
        cost_function=cost_fn
    )

    x_hist, u_hist, cost_hist = sim.simulate( # x_mpc_wtc, u_mpc_wtc, cost_mpc_wtc_traj
        init_dict=init_dict,
        sim_steps=300
    )

