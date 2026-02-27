1. mpc的控制参数也得设置上下界了,并且能分个设置不同的上下界
2. 获取fmu中的fan2Table.qflow并绘制到RPM的subplot中，在legend中命名为QFLOW_blower，注意：获取到的单位为m^3/h帮我换算为m^3/s
3. mpc控制过程中，如果打印warnings.warn("MPC优化未完全收敛（但仍返回结果）。"），这是什么意思呢？
4. mpc控制的可调超参在哪里呢？
5. 预热周期iternum改成50
6. 
