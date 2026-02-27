"E:\Program Files\Python311\python.exe" C:\Users\li.zhengen\Desktop\main-dis\mympc\main_mympc.py 

======================================================================================================================================================
短地平线增量型MPC热管理系统控制（FMU）- 带预热阶段
======================================================================================================================================================

✅ 找到FMU文件: ..\env\MyITMS.fmu
✅ FMU已成功加载（Co-Simulation模式）
   将获取功耗和其他监测变量

开始 增量型MPC (N=5) 仿真
预热阶段: 20 步
  - RPM设定: blower=30, comp=1000, batt=1000, motor=1000
MPC控制: 20 步
  - 增量约束: blower=[-5, +5], comp=[-10, +10], batt=[-10, +10], motor=[-10, +10]
======================================================================================================================================================
增量型MPC (N=5) 运行中:  52%|█████▎    | 21/40 [00:20<00:07,  2.69步/s, 阶段=控制, T_cabin=27.85°C, T_battery=30.08°C, T_motor=60.12°C, err_cabin=+2.85°C, err_battery=-4.92°C, err_motor=+20.12°C, RPM_blower=35, RPM_comp=1007, RPM_batt=990, RPM_motor=990, ΔU_blower=+5, ΔU_comp=+7, ΔU_batt=-10, ΔU_motor=-10, P_blower=0.020kW, P_comp=1.792kW, P_batt=0.885kW, P_motor=0.885kW]C:\Users\li.zhengen\Desktop\main-dis\mympc\main_mympc.py:460: UserWarning: MPC求解器失败: Error in Function::call for 'mpc_solver' [IpoptInterface] at .../casadi/core/function.cpp:1466:
Error in Function::call for 'mpc_solver' [IpoptInterface] at .../casadi/core/function.cpp:362:
.../casadi/core/function_internal.hpp:1728: Input 0 (x0) has mismatching shape. Got 16-by-1. Allowed dimensions, in general, are:
 - The input dimension N-by-M (here 20-by-1)
 - A scalar, i.e. 1-by-1
 - M-by-N if N=1 or M=1 (i.e. a transposed vector)
 - N-by-M1 if K*M1=M for some K (argument repeated horizontally)
 - N-by-P*M, indicating evaluation with multiple arguments (P must be a multiple of 1 for consistency with previous inputs)。使用初始猜测。
  warnings.warn(f"MPC求解器失败: {str(e)}。使用初始猜测。")
C:\Users\li.zhengen\Desktop\main-dis\mympc\main_mympc.py:1008: UserWarning: MPC求解在步骤 21 失败: cannot reshape array of size 16 into shape (5,4)
  warnings.warn(f"MPC求解在步骤 {k} 失败: {e}")
增量型MPC (N=5) 运行中:  57%|█████▊    | 23/40 [00:22<00:12,  1.34步/s, 阶段=控制, T_cabin=27.26°C, T_battery=30.08°C, T_motor=60.15°C, err_cabin=+2.26°C, err_battery=-4.92°C, err_motor=+20.15°C, RPM_blower=40, RPM_comp=1014, RPM_batt=980, RPM_motor=980, ΔU_blower=+5, ΔU_comp=+7, ΔU_batt=-10, ΔU_motor=-10, P_blower=0.026kW, P_comp=1.822kW, P_batt=0.884kW, P_motor=0.884kW]C:\Users\li.zhengen\Desktop\main-dis\mympc\main_mympc.py:1008: UserWarning: MPC求解在步骤 23 失败: cannot reshape array of size 16 into shape (5,4)
  warnings.warn(f"MPC求解在步骤 {k} 失败: {e}")
增量型MPC (N=5) 运行中:  62%|██████▎   | 25/40 [00:25<00:14,  1.04步/s, 阶段=控制, T_cabin=26.59°C, T_battery=30.09°C, T_motor=60.19°C, err_cabin=+1.59°C, err_battery=-4.91°C, err_motor=+20.19°C, RPM_blower=45, RPM_comp=1021, RPM_batt=970, RPM_motor=970, ΔU_blower=+5, ΔU_comp=+7, ΔU_batt=-10, ΔU_motor=-10, P_blower=0.032kW, P_comp=1.851kW, P_batt=0.884kW, P_motor=0.884kW]C:\Users\li.zhengen\Desktop\main-dis\mympc\main_mympc.py:1008: UserWarning: MPC求解在步骤 25 失败: cannot reshape array of size 16 into shape (5,4)
  warnings.warn(f"MPC求解在步骤 {k} 失败: {e}")
增量型MPC (N=5) 运行中:  68%|██████▊   | 27/40 [00:26<00:11,  1.11步/s, 阶段=控制, T_cabin=25.85°C, T_battery=30.09°C, T_motor=60.30°C, err_cabin=+0.85°C, err_battery=-4.91°C, err_motor=+20.30°C, RPM_blower=50, RPM_comp=1027, RPM_batt=960, RPM_motor=960, ΔU_blower=+5, ΔU_comp=+7, ΔU_batt=-10, ΔU_motor=-10, P_blower=0.041kW, P_comp=1.879kW, P_batt=0.884kW, P_motor=0.884kW]C:\Users\li.zhengen\Desktop\main-dis\mympc\main_mympc.py:1008: UserWarning: MPC求解在步骤 27 失败: cannot reshape array of size 16 into shape (5,4)
  warnings.warn(f"MPC求解在步骤 {k} 失败: {e}")
增量型MPC (N=5) 运行中:  72%|███████▎  | 29/40 [00:27<00:08,  1.37步/s, 阶段=控制, T_cabin=25.07°C, T_battery=30.10°C, T_motor=60.38°C, err_cabin=+0.07°C, err_battery=-4.90°C, err_motor=+20.38°C, RPM_blower=55, RPM_comp=1034, RPM_batt=950, RPM_motor=950, ΔU_blower=+5, ΔU_comp=+7, ΔU_batt=-10, ΔU_motor=-10, P_blower=0.051kW, P_comp=1.905kW, P_batt=0.884kW, P_motor=0.884kW]C:\Users\li.zhengen\Desktop\main-dis\mympc\main_mympc.py:1008: UserWarning: MPC求解在步骤 29 失败: cannot reshape array of size 16 into shape (5,4)
  warnings.warn(f"MPC求解在步骤 {k} 失败: {e}")
增量型MPC (N=5) 运行中:  78%|███████▊  | 31/40 [00:28<00:05,  1.57步/s, 阶段=控制, T_cabin=24.28°C, T_battery=30.10°C, T_motor=60.45°C, err_cabin=-0.72°C, err_battery=-4.90°C, err_motor=+20.45°C, RPM_blower=60, RPM_comp=1040, RPM_batt=940, RPM_motor=940, ΔU_blower=+5, ΔU_comp=+6, ΔU_batt=-10, ΔU_motor=-10, P_blower=0.064kW, P_comp=1.929kW, P_batt=0.884kW, P_motor=0.884kW]C:\Users\li.zhengen\Desktop\main-dis\mympc\main_mympc.py:1008: UserWarning: MPC求解在步骤 31 失败: cannot reshape array of size 16 into shape (5,4)
  warnings.warn(f"MPC求解在步骤 {k} 失败: {e}")
增量型MPC (N=5) 运行中:  82%|████████▎ | 33/40 [00:30<00:06,  1.12步/s, 阶段=控制, T_cabin=23.48°C, T_battery=30.10°C, T_motor=60.49°C, err_cabin=-1.52°C, err_battery=-4.90°C, err_motor=+20.49°C, RPM_blower=65, RPM_comp=1047, RPM_batt=930, RPM_motor=930, ΔU_blower=+5, ΔU_comp=+6, ΔU_batt=-10, ΔU_motor=-10, P_blower=0.081kW, P_comp=1.952kW, P_batt=0.884kW, P_motor=0.884kW]C:\Users\li.zhengen\Desktop\main-dis\mympc\main_mympc.py:1008: UserWarning: MPC求解在步骤 33 失败: cannot reshape array of size 16 into shape (5,4)
  warnings.warn(f"MPC求解在步骤 {k} 失败: {e}")
增量型MPC (N=5) 运行中:  88%|████████▊ | 35/40 [00:31<00:03,  1.34步/s, 阶段=控制, T_cabin=22.71°C, T_battery=30.11°C, T_motor=60.56°C, err_cabin=-2.29°C, err_battery=-4.89°C, err_motor=+20.56°C, RPM_blower=70, RPM_comp=1053, RPM_batt=920, RPM_motor=920, ΔU_blower=+5, ΔU_comp=+6, ΔU_batt=-10, ΔU_motor=-10, P_blower=0.107kW, P_comp=1.974kW, P_batt=0.884kW, P_motor=0.884kW]C:\Users\li.zhengen\Desktop\main-dis\mympc\main_mympc.py:1008: UserWarning: MPC求解在步骤 35 失败: cannot reshape array of size 16 into shape (5,4)
  warnings.warn(f"MPC求解在步骤 {k} 失败: {e}")
增量型MPC (N=5) 运行中:  92%|█████████▎| 37/40 [00:33<00:02,  1.48步/s, 阶段=控制, T_cabin=21.95°C, T_battery=30.11°C, T_motor=60.62°C, err_cabin=-3.05°C, err_battery=-4.89°C, err_motor=+20.62°C, RPM_blower=75, RPM_comp=1059, RPM_batt=910, RPM_motor=910, ΔU_blower=+5, ΔU_comp=+6, ΔU_batt=-10, ΔU_motor=-10, P_blower=0.221kW, P_comp=1.995kW, P_batt=0.884kW, P_motor=0.884kW]C:\Users\li.zhengen\Desktop\main-dis\mympc\main_mympc.py:1008: UserWarning: MPC求解在步骤 37 失败: cannot reshape array of size 16 into shape (5,4)
  warnings.warn(f"MPC求解在步骤 {k} 失败: {e}")
增量型MPC (N=5) 运行中:  98%|█████████▊| 39/40 [00:33<00:00,  1.74步/s, 阶段=控制, T_cabin=21.25°C, T_battery=30.11°C, T_motor=60.64°C, err_cabin=-3.75°C, err_battery=-4.89°C, err_motor=+20.64°C, RPM_blower=80, RPM_comp=1065, RPM_batt=900, RPM_motor=900, ΔU_blower=+5, ΔU_comp=+6, ΔU_batt=-10, ΔU_motor=-10, P_blower=0.829kW, P_comp=2.014kW, P_batt=0.884kW, P_motor=0.884kW]C:\Users\li.zhengen\Desktop\main-dis\mympc\main_mympc.py:1008: UserWarning: MPC求解在步骤 39 失败: cannot reshape array of size 16 into shape (5,4)
  warnings.warn(f"MPC求解在步骤 {k} 失败: {e}")

仿真在步骤 40 结束
======================================================================================================================================================
增量型MPC (N=5) 仿真完成！
预热阶段摘要: 20 步
MPC控制阶段摘要: 20 步
增量型MPC (N=5) 运行中:  98%|█████████▊| 39/40 [00:34<00:00,  1.15步/s, 阶段=控制, T_cabin=20.93°C, T_battery=30.11°C, T_motor=60.65°C, err_cabin=-4.07°C, err_battery=-4.89°C, err_motor=+20.65°C, RPM_blower=80, RPM_comp=1065, RPM_batt=900, RPM_motor=900, ΔU_blower=+0, ΔU_comp=+0, ΔU_batt=+0, ΔU_motor=+0, P_blower=1.223kW, P_comp=2.024kW, P_batt=0.884kW, P_motor=0.884kW]

======================================================================================================================================================
仿真摘要统计 - 增量型MPC (N=5) [FMU]
======================================================================================================================================================

仿真配置:
  预热阶段: 20 步
    - RPM设定: blower=30, comp=1000, batt=1000, motor=1000
  MPC控制阶段: 20 步
    - 增量约束:
      blower: [    -5,     +5] RPM/step
        comp: [   -10,    +10] RPM/step
        batt: [   -10,    +10] RPM/step
       motor: [   -10,    +10] RPM/step
  总步数: 40

车舱温度:
  预热阶段:
    平均温度: 29.61°C
    标准差: 0.68°C
    平均误差: +4.61°C
  控制阶段:
    目标温度: 25.0°C
    平均温度: 24.62°C
    标准差: 2.26°C
    最小值: 20.93°C，最大值: 28.12°C
    平均误差: -0.38°C
    误差范围: -4.07°C ~ +3.12°C

电池温度:
  预热阶段:
    平均温度: 30.06°C
    标准差: 0.02°C
    平均误差: -4.94°C
  控制阶段:
    目标温度: 35.0°C
    平均温度: 30.10°C
    标准差: 0.01°C
    最小值: 30.08°C，最大值: 30.11°C
    平均误差: -4.90°C
    误差范围: -4.92°C ~ -4.89°C

电机温度:
  预热阶段:
    平均温度: 59.96°C
    标准差: 0.04°C
    平均误差: +19.96°C
  控制阶段:
    目标温度: 40.0°C
    平均温度: 60.39°C
    标准差: 0.19°C
    最小值: 60.07°C，最大值: 60.65°C
    平均误差: +20.39°C
    误差范围: +20.07°C ~ +20.65°C

控制输入（控制阶段）:
    RPM_blower:
    转速范围:     35 ~     80 RPM (平均=    57)
    增量统计: 平均=    +2, 最大幅度=     5 RPM/step
      RPM_comp:
    转速范围:   1007 ~   1065 RPM (平均=  1037)
    增量统计: 平均=    +3, 最大幅度=     7 RPM/step
      RPM_batt:
    转速范围:    900 ~    990 RPM (平均=   945)
    增量统计: 平均=    -5, 最大幅度=    10 RPM/step
     RPM_motor:
    转速范围:    900 ~    990 RPM (平均=   945)
    增量统计: 平均=    -5, 最大幅度=    10 RPM/step

功耗分析（从FMU获取，控制阶段）:
     鼓风机: 平均=  0.179kW, 最大=  1.223kW
     压缩机: 平均=  1.918kW, 最大=  2.024kW
     电池泵: 平均=  0.884kW, 最大=  0.885kW
     电机泵: 平均=  0.884kW, 最大=  0.885kW
  总功率: 平均=3.865kW, 最大=5.016kW, 总计=77.30kWh

驾驶工况（控制阶段）:
  油门踏板: 平均=0.1%, 最大=0.3%
  制动踏板: 平均=0.0%, 最大=0.0%
  车速: 平均=7.8km/h, 最大=9.4km/h
  电池SOC: 平均=49.7%, 范围=[49.7%, 49.7%]

控制性能:
  总成本: 44885.99
  平均单步成本: 2244.2997
  控制步数: 20
======================================================================================================================================================


图表已保存: results/mpc_results_20260227_114413.png

所有结果已生成成功！

进程已结束，退出代码为 0


