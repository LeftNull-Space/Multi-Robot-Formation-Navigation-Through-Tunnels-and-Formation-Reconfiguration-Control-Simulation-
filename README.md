# 通信延迟条件下多机器人编队导航与隧道重构控制及仿真
**时间**：2025.11-2025.12

## 项目背景
本项目针对「多机器人二阶动力学系统+固定通信延迟」场景，实现8个全向移动机器人穿越2m宽狭窄通道并重构圆形队形的任务，核心挑战包括：<br>
1）**通信延迟补偿**：机器人间状态交互存在固定延迟 τ=0.01s，需设计预测机制抵消信息滞后影响；<br>
2）**安全约束满足**：机器人直径0.5m，需确保中心间距≥0.5m（物理防碰），通道内含静态障碍物（x∈[2,4], y∈[0,0.8]）；<br>
3）**队形精度要求**：初始/终点为直径5m的均匀圆形队形，出口区域RMS误差需<0.03m。

## 核心技术方案
### 1. 动态队形调度机制
1）基于平滑权重 α（队形类型）、β（队形缩放）实现「圆形→线性→圆形」连续过渡；<br>
2）通道内压缩至最小半径 R_min=0.65m（适配2m通道），出口处自动展开至 R0=2.5m（初始半径）。

### 2. 通信延迟补偿策略
1）针对 τ=0.01s 延迟，采用一阶泰勒预测模型：$\hat{p}_i(t) = p_i(t-\tau) + \tau \cdot v_i(t-\tau)$；<br>
2）仅基于实时通信拓扑（邻居距离≤2.5m）计算排斥力，降低无效计算。

### 3. 鲁棒分布式控制律
融合多目标优化项，严格满足物理约束（v_max=1.0m/s，a_max=5.0m/s²）：<br>
1）跟踪控制：位置增益 kp=2.4 + 速度阻尼 kd=2.0，确保轨迹收敛；<br>
2）安全避让：机器人排斥力（kd_rep=4.0）、边界排斥力（kwall_rep=1.2）、障碍物排斥力（kobs_rep=1.4）。

### 4. 出口角色重分配
1）按机器人x坐标降序（前→后）分配目标角度（0°~180°），避免顺序混乱导致的队形畸变；<br>
2）重分配后5s内收敛至新目标，无振荡。

## 仿真结果与性能
### 1. 队形演化关键阶段
| 时间点 | 队形状态 | 图片 |
|--------|----------|------|
| t=0s   | 初始圆形队形 | ![初始队形](docs/Simulation_Result_Visualization/formation_evolution_1.png) |
| t=20s  | 通道内线性排队 | ![线性队形](docs/Simulation_Result_Visualization/formation_evolution_2.png) |
| t=30s  | 出口圆形重构 | ![重构队形](docs/Simulation_Result_Visualization/formation_evolution_3.png) |
| t=40s  | 队形重构完成 | ![最终队形](docs/Simulation_Result_Visualization/formation_evolution_4.png) |

### 2. 关键性能指标（优于设计要求）
| 指标 | 测量值 | 设计要求 |
|------|--------|----------|
| 最小机器人间距 | 0.522m | ≥0.5m（物理防碰） |
| 最终RMS误差 | 0.005m | <0.03m（重构精度） |
| 最大速度 | 0.995m/s | <1.0m/s（速度约束） |
| 总任务时间 | 35.6s | <200s（超时阈值） |

### 3. 性能趋势图
![性能指标](docs/Simulation_Result_Visualization/performance_curves.png)
蓝色：RMS误差（最终0.005m）；绿色：最小间距（最低0.522m，无碰撞）；红色：最大速度（≤0.995m/s）。

### 4. 视频演示
点击图片查看完整仿真视频：
[![仿真视频封面](docs/Simulation_Result_Visualization/formation_custom_reassign_with_topology.png)](https://www.bilibili.com/video/BV12VkcBDEXG/?vd_source=408c8ac6c7b1898983e992b2e3fef192)