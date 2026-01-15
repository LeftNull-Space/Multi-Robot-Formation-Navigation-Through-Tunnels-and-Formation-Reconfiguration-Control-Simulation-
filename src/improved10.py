import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 防止 GUI 闪退
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from collections import deque
import os

# -----------------------------
# Global Parameters
# -----------------------------
N = 8
R0 = 2.5
robot_diam = 0.5
r_safe = robot_diam + 0.05
v_max = 1.0
a_max = 5.0

corr = {'x0': 0.0, 'x1': 20.0, 'ymin': 0.0, 'ymax': 2.0}
obs = {'xmin': 2.0, 'xmax': 4.0, 'ymin': 0.0, 'ymax': 0.8}

x_queue_start = -3.0
x_queue_done = -1.2
x_exit = corr['x1']
x_expand_start = x_exit
x_expand_done = x_exit + 5.0

R_min = 0.65
line_spacing = 0.65

k_p = 2.4
k_d = 2.0
k_rep_rr = 4.0
k_rep_wall = 1.2
k_rep_obs = 1.4

d0_rr = 0.8
d0_wall = 0.6
d0_obs = 0.7

# ⭐ 通信半径（用于构建通信拓扑）
R_comm = 2.5

tau = 0.01
dt_sim = 0.02
Tmax = 200.0
goal_freeze_radius = 0.6

waypoints = np.array([
    [-3.6, 1.0],
    [-2.6, 1.0],
    [-1.0, 1.0],
    [0.5, 1.5],
    [3.0, 1.5],
    [10.0, 1.0],
    [20.5, 1.0],
    [23.0, 1.0],
    [26.0, 1.0]
])
pL_final = waypoints[-1].copy()

theta0 = np.linspace(0, 2 * np.pi, N, endpoint=False)
pC0 = np.array([-4.0, 1.0])
p_init = np.stack([
    pC0[0] + R0 * np.cos(theta0),
    pC0[1] + R0 * np.sin(theta0)
], axis=1)

right_to_left_order = np.argsort(-p_init[:, 0])
rank_in_queue = np.empty(N, dtype=int)
for rank, idx in enumerate(right_to_left_order):
    rank_in_queue[idx] = rank


# -----------------------------
# Helper Functions
# -----------------------------
def smoothstep(x, a, b):
    if x <= a:
        return 0.0
    elif x >= b:
        return 1.0
    else:
        s = (x - a) / (b - a)
        return s * s * (3 - 2 * s)


def min_pair_dist(p):
    dmin = np.inf
    for i in range(p.shape[0]):
        for j in range(i + 1, p.shape[0]):
            d = np.linalg.norm(p[i] - p[j])
            if d < dmin:
                dmin = d
    return dmin


def obs_repulse(pi, obs, d0, krep):
    cx = np.clip(pi[0], obs['xmin'], obs['xmax'])
    cy = np.clip(pi[1], obs['ymin'], obs['ymax'])
    closest = np.array([cx, cy])
    r_vec = pi - closest
    d = np.linalg.norm(r_vec)

    inside = (obs['xmin'] <= pi[0] <= obs['xmax']) and (obs['ymin'] <= pi[1] <= obs['ymax'])
    if inside:
        dxL = pi[0] - obs['xmin']
        dxR = obs['xmax'] - pi[0]
        dyB = pi[1] - obs['ymin']
        dyT = obs['ymax'] - pi[1]
        dists = [dxL, dxR, dyB, dyT]
        idx = np.argmin(dists)
        dirs = [np.array([-1, 0]), np.array([1, 0]), np.array([0, -1]), np.array([0, 1])]
        return 5 * krep * dirs[idx]

    if d < 1e-6 or d >= d0:
        return np.zeros(2)
    mag = krep * (1/d - 1/d0) / (d**2)
    return mag * (r_vec / d)


class DelayBuffer:
    def __init__(self, max_delay, dt):
        self.max_steps = int(np.ceil(max_delay / dt)) + 10
        self.buffer = deque(maxlen=self.max_steps)
        self.times = deque(maxlen=self.max_steps)

    def add(self, t, state):
        self.times.append(t)
        self.buffer.append(state.copy())

    def get(self, t_query):
        if len(self.times) == 0:
            return np.zeros(4 * N)
        if t_query <= self.times[0]:
            return self.buffer[0].copy()
        if t_query >= self.times[-1]:
            return self.buffer[-1].copy()
        t_arr = np.array(self.times)
        idx = np.searchsorted(t_arr, t_query)
        t0, t1 = t_arr[idx - 1], t_arr[idx]
        s0, s1 = self.buffer[idx - 1], self.buffer[idx]
        alpha = (t_query - t0) / (t1 - t0 + 1e-9)
        return s0 + alpha * (s1 - s0)


# -----------------------------
# Main Simulation (improved10)
# -----------------------------
def main():
    p = p_init.copy()
    v = np.zeros_like(p)
    Y0 = np.hstack([p.flatten(), v.flatten()])

    pL = np.mean(p, axis=0)
    t_freeze = None
    wp_idx = 0

    delay_buffer = DelayBuffer(tau, dt_sim)
    delay_buffer.add(0.0, Y0)

    t_vals = [0.0]
    Y_vals = [Y0.copy()]
    min_dist_log = []
    err_rms_log = []

    final_theta_assigned = False
    theta_final = np.copy(theta0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-8, 30)
    ax.set_ylim(-2, 5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Exit Reassignment + Communication Topology (R_comm=2.5m)')

    ax.plot([corr['x0'], corr['x1']], [corr['ymin'], corr['ymin']], 'k', lw=2)
    ax.plot([corr['x0'], corr['x1']], [corr['ymax'], corr['ymax']], 'k', lw=2)
    ax.add_patch(Rectangle(
        (obs['xmin'], obs['ymin']),
        obs['xmax'] - obs['xmin'],
        obs['ymax'] - obs['ymin'],
        facecolor=[0.85, 0.85, 0.85],
        edgecolor='k'
    ))

    colors = plt.cm.tab10(np.linspace(0, 1, N))
    circles = [Circle((p[i, 0], p[i, 1]), robot_diam / 2, color=colors[i], alpha=0.7) for i in range(N)]
    for c in circles:
        ax.add_patch(c)
    hL, = ax.plot([pL[0]], [pL[1]], 'r+', ms=12)
    status_text = ax.text(-7.5, 4.6, '', fontsize=12, fontweight='bold', va='top')
    plt.pause(0.01)

    t = 0.0
    done = False
    step = 0
    while t < Tmax and not done:
        # Leader update
        if np.linalg.norm(pL - pL_final) < goal_freeze_radius:
            if t_freeze is None:
                t_freeze = t
            pL = pL_final.copy()
        else:
            if wp_idx < len(waypoints):
                wp = waypoints[wp_idx]
                if np.linalg.norm(pL - wp) < 0.35:
                    wp_idx += 1
                if wp_idx < len(waypoints):
                    wp = waypoints[wp_idx]
                else:
                    wp = pL_final
            else:
                wp = pL_final

            dirL = wp - pL
            if np.linalg.norm(dirL) < 1e-6:
                vL = np.zeros(2)
            else:
                vL = v_max * dirL / max(np.linalg.norm(dirL), 0.5)

            if corr['x0'] <= pL[0] <= corr['x1']:
                vL[0] += 0.25

            if np.linalg.norm(vL) > v_max:
                vL = vL / np.linalg.norm(vL) * v_max

            pL += dt_sim * vL

        # State retrieval
        p_curr = Y_vals[-1][0:2 * N].reshape((N, 2))
        v_curr = Y_vals[-1][2 * N:4 * N].reshape((N, 2))

        if t < tau:
            p_pred, v_pred = p_curr.copy(), v_curr.copy()
        else:
            Y_delay = delay_buffer.get(t - tau)
            p_delay = Y_delay[0:2 * N].reshape((N, 2))
            v_delay = Y_delay[2 * N:4 * N].reshape((N, 2))
            p_pred = p_delay + tau * v_delay
            v_pred = v_delay

        # Role reassignment at exit: custom mapping
        if not final_theta_assigned and pL[0] >= x_expand_start + 0.5:
            if np.all(p_curr[:, 0] > obs['xmax'] + 0.5):
                xs = p_curr[:, 0]
                order_desc = np.argsort(-xs)  # x descending: front to back

                custom_angles_deg = [0, 45, 315, 90, 270, 135, 225, 180]
                assert len(custom_angles_deg) == N, "Custom angle list must have exactly N entries"
                standard_thetas = np.radians(custom_angles_deg)

                theta_final = np.zeros(N)
                for rank, robot_idx in enumerate(order_desc):
                    theta_final[robot_idx] = standard_thetas[rank]

                final_theta_assigned = True
                angle_deg = np.degrees(theta_final)
                angle_str = ', '.join(f"{a:.1f}" for a in angle_deg)
                print(f"[OK] Role reassigned at t={t:.2f}s! Angles (deg): [{angle_str}]")

        # Formation scheduling
        xL = pL[0]
        alpha = smoothstep(xL, x_queue_start, x_queue_done) if xL < x_queue_done else (
            1.0 if xL < x_exit else 1.0 - smoothstep(xL, x_exit, x_exit + 3.0))
        beta = smoothstep(xL, x_queue_start, x_queue_done) if xL < x_queue_done else (
            1.0 if xL < x_expand_start else (1.0 - smoothstep(xL, x_expand_start, x_expand_done) if xL < x_expand_done else 0.0))
        Ruse = (1 - beta) * R0 + beta * R_min

        rel_circle = np.stack([Ruse * np.cos(theta_final), Ruse * np.sin(theta_final)], axis=1)

        rel_line = np.zeros((N, 2))
        for i in range(N):
            x_offset = -(rank_in_queue[i] - (N - 1) / 2) * line_spacing
            rel_line[i, 0] = x_offset
            rel_line[i, 1] = 0.01 * ((-1) ** rank_in_queue[i])

        rel_set = (1 - alpha) * rel_circle + alpha * rel_line
        des_pts = pL + rel_set

        # ⭐⭐⭐ 构建通信拓扑（关键新增！）⭐⭐⭐
        neighbors = [[] for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if np.linalg.norm(p_curr[i] - p_curr[j]) <= R_comm:
                    neighbors[i].append(j)

        # Control law with communication topology
        u = np.zeros((N, 2))
        for i in range(N):
            u[i] = k_p * (des_pts[i] - p_pred[i]) - k_d * v_pred[i]

            # ⭐ 仅对通信范围内的邻居施加排斥力
            for j in neighbors[i]:
                rij = p_curr[i] - p_curr[j]
                dij = np.linalg.norm(rij)
                if 1e-6 < dij < d0_rr:
                    mag = k_rep_rr * (1 / dij - 1 / d0_rr) / (dij ** 2)
                    mag = np.clip(mag, -a_max, a_max)
                    u[i] += mag * (rij / dij)

            if corr['x0'] - 0.5 < p_curr[i, 0] < corr['x1'] + 0.5:
                dy_bottom = p_curr[i, 1] - corr['ymin']
                if 0 < dy_bottom < d0_wall:
                    u[i, 1] += k_rep_wall * (1 / dy_bottom - 1 / d0_wall) / (dy_bottom ** 2)
                dy_top = corr['ymax'] - p_curr[i, 1]
                if 0 < dy_top < d0_wall:
                    u[i, 1] -= k_rep_wall * (1 / dy_top - 1 / d0_wall) / (dy_top ** 2)

            u[i] += obs_repulse(p_curr[i], obs, d0_obs, k_rep_obs)

            if np.linalg.norm(u[i]) > a_max:
                u[i] = u[i] / np.linalg.norm(u[i]) * a_max

        # State update
        v_new = v_curr + u * dt_sim
        for i in range(N):
            speed = np.linalg.norm(v_new[i])
            if speed > v_max:
                v_new[i] = v_new[i] / speed * v_max
        p_new = p_curr + v_new * dt_sim

        # Safety check
        if not np.all(np.isfinite(p_new)):
            print(f"ERROR: Non-finite state at t={t:.2f}s! Simulation aborted.")
            break

        Y_new = np.hstack([p_new.flatten(), v_new.flatten()])
        t += dt_sim
        t_vals.append(t)
        Y_vals.append(Y_new)
        delay_buffer.add(t, Y_new)

        min_dist_log.append(min_pair_dist(p_new))
        err2 = sum(np.sum((des_pts[i] - p_new[i]) ** 2) for i in range(N))
        err_rms = np.sqrt(err2 / N)
        err_rms_log.append(err_rms)

        max_speed = np.max(np.linalg.norm(v_new, axis=1))
        formation_good = (alpha < 0.02) and (beta < 0.02) and (err_rms < 0.03)
        near_goal = np.linalg.norm(pL - pL_final) < 0.3
        low_velocity = max_speed < 0.03
        if formation_good and near_goal and low_velocity and (t > 30):
            done = True

        # Plot every 20 steps
        if step % 20 == 0:
            for i in range(N):
                circles[i].center = (p_new[i, 0], p_new[i, 1])
            hL.set_data([pL[0]], [pL[1]])
            vmax_now = np.max(np.linalg.norm(v_new, axis=1))
            status_text.set_text(
                f't = {t:.2f}s | alpha={alpha:.2f} beta={beta:.2f} | '
                f'RMS={err_rms:.3f} | vmax={vmax_now:.3f} | '
                f'Assigned: {"Yes" if final_theta_assigned else "No"}'
            )
            plt.pause(0.001)
        step += 1

    # Final report
    print("\n" + "=" * 60)
    print("Simulation completed (with communication topology).")
    print(f"Total time: {t:.2f} seconds")
    final_min_dist = min(min_dist_log) if min_dist_log else float('inf')
    print(f"Minimum inter-robot distance: {final_min_dist:.3f} m (safe threshold: {r_safe:.2f})")
    print("✅ Safe!" if final_min_dist >= r_safe else "⚠️  Collision risk detected!")
    print(f"Final RMS error: {err_rms_log[-1]:.4f} m" if err_rms_log else "N/A")
    speeds = np.linalg.norm(np.array(Y_vals)[:, 2 * N:4 * N].reshape((-1, N, 2)), axis=2)
    print(f"Maximum speed observed: {np.max(speeds):.3f} m/s")
    if final_theta_assigned:
        angle_deg = np.degrees(theta_final)
        angle_str = ', '.join(f"{a:.1f}" for a in angle_deg)
        print(f"Final angles (deg): [{angle_str}]")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "formation_custom_reassign_with_topology.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()

def save_communication_topology():
    """保存通信拓扑图（展示机器人间连接关系）"""
    # 获取最终时刻状态
    final_state = Y_vals[-1]
    p_final = final_state[:2*N].reshape((N, 2))
    
    # 构建通信拓扑（与主程序中逻辑一致）
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j and np.linalg.norm(p_final[i] - p_final[j]) <= R_comm:
                neighbors[i].append(j)
    
    # 创建新图
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-8, 30)
    ax.set_ylim(-2, 5)
    ax.set_aspect('equal')
    ax.set_title('Communication Topology (R_comm=2.5m) - Final State', fontsize=14)
    
    # 绘制走廊和障碍物
    ax.plot([corr['x0'], corr['x1']], [corr['ymin'], corr['ymin']], 'k', lw=2)
    ax.plot([corr['x0'], corr['x1']], [corr['ymax'], corr['ymax']], 'k', lw=2)
    ax.add_patch(Rectangle(
        (obs['xmin'], obs['ymin']),
        obs['xmax'] - obs['xmin'],
        obs['ymax'] - obs['ymin'],
        facecolor=[0.85, 0.85, 0.85],
        edgecolor='k'
    ))
    
    # 绘制机器人位置
    colors = plt.cm.tab10(np.linspace(0, 1, N))
    for i in range(N):
        circle = Circle((p_final[i, 0], p_final[i, 1]), robot_diam/2, 
                       color=colors[i], alpha=0.8, zorder=3)
        ax.add_patch(circle)
        ax.text(p_final[i, 0] + 0.2, p_final[i, 1] + 0.2, f'Robot {i+1}', 
                fontsize=9, color=colors[i])
    
    # 绘制通信连接
    for i in range(N):
        for j in neighbors[i]:
            if i < j:  # 避免重复绘制
                ax.plot([p_final[i, 0], p_final[j, 0]], 
                        [p_final[i, 1], p_final[j, 1]], 
                        'k-', alpha=0.3, lw=1.5, zorder=2)
    
    # 添加图例
    ax.plot([], [], 'k-', alpha=0.3, lw=1.5, label='Communication Link')
    ax.plot([], [], 'o', color='gray', markersize=8, alpha=0.8, label='Robot Position')
    ax.legend(loc='upper left', fontsize=10)
    
    # 保存并显示
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "communication_topology.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Communication topology saved to: {save_path}")

def save_formation_evolution():
    """保存队形演变过程图（展示三个关键阶段）"""
    # 选择三个关键时间点（初始、走廊中、出口后）
    t_points = [0.0, 50.0, 150.0]  # t=0 (初始), t=50 (走廊), t=150 (出口)
    idxs = [0]  # t=0
    for t in t_points[1:]:
        idx = np.argmin(np.abs(np.array(t_vals) - t))
        idxs.append(idx)
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    for ax, idx, t in zip(axes, idxs, t_points):
        # 获取状态
        state = Y_vals[idx]
        p = state[:2*N].reshape((N, 2))
        
        # 绘制走廊和障碍物
        ax.plot([corr['x0'], corr['x1']], [corr['ymin'], corr['ymin']], 'k', lw=2)
        ax.plot([corr['x0'], corr['x1']], [corr['ymax'], corr['ymax']], 'k', lw=2)
        ax.add_patch(Rectangle(
            (obs['xmin'], obs['ymin']),
            obs['xmax'] - obs['xmin'],
            obs['ymax'] - obs['ymin'],
            facecolor=[0.85, 0.85, 0.85],
            edgecolor='k'
        ))
        
        # 绘制机器人
        colors = plt.cm.tab10(np.linspace(0, 1, N))
        for i in range(N):
            circle = Circle((p[i, 0], p[i, 1]), robot_diam/2, 
                           color=colors[i], alpha=0.7)
            ax.add_patch(circle)
            ax.text(p[i, 0] + 0.2, p[i, 1] + 0.2, f'Robot {i+1}', 
                   fontsize=8, color=colors[i])
        
        # 添加标题和网格
        ax.set_title(f'Formation Evolution (t={t:.1f}s)', fontsize=12)
        ax.set_xlim(-8, 30)
        ax.set_ylim(-2, 5)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal')
    
    # 添加说明
    plt.figtext(0.5, 0.01, 
                "Left: Initial Circular Formation | Middle: Queue in Corridor | Right: Custom Reassigned Formation",
                ha="center", fontsize=11, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "formation_evolution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Formation evolution saved to: {save_path}")

def save_performance_curves():
    """保存性能指标曲线图（RMS误差、最小距离、速度）"""
    # 准备数据
    t_data = np.array(t_vals)
    min_dist = np.array(min_dist_log)
    err_rms = np.array(err_rms_log)
    
    # 计算最大速度
    speeds = np.linalg.norm(np.array(Y_vals)[:, 2*N:4*N].reshape((-1, N, 2)), axis=2)
    max_speed = np.max(speeds, axis=1)
    
    # 创建图形
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 绘制RMS误差（主Y轴）
    color = 'tab:blue'
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('RMS Error (m)', color=color, fontsize=12)
    ax1.plot(t_data, err_rms, color=color, linewidth=2.5, label='RMS Error')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(linestyle='--', alpha=0.7)
    
    # 创建次Y轴（用于最小距离和速度）
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Distance/Speed (m/m/s)', color=color, fontsize=12)
    
    # 绘制最小距离
    ax2.plot(t_data, min_dist, color='tab:green', linewidth=2.0, 
             label='Min Distance (m)')
    ax2.axhline(y=r_safe, color='k', linestyle='--', alpha=0.7, 
                label=f'Safe Threshold ({r_safe:.2f}m)')
    
    # 绘制最大速度
    ax2.plot(t_data, max_speed, color=color, linewidth=2.0, 
             label='Max Speed (m/s)')
    
    # 设置图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    # 添加关键阶段标记
    ax1.axvline(x=x_queue_done, color='gray', linestyle='--', alpha=0.5)
    ax1.text(x_queue_done + 0.5, 0.05, 'Queue Exit', rotation=90, fontsize=9)
    
    ax1.axvline(x=x_expand_start, color='gray', linestyle='--', alpha=0.5)
    ax1.text(x_expand_start + 0.5, 0.05, 'Expansion Start', rotation=90, fontsize=9)
    
    # 标题和布局
    plt.title('Formation Control Performance Metrics', fontsize=14)
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "performance_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Performance curves saved to: {save_path}")


if __name__ == "__main__":
    main()
    save_communication_topology()
    save_formation_evolution()
    save_performance_curves()