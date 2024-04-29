import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import KF_tracking as KF

def projectile_motion_with_resistance(state, t, g, k, m):
    x, y, v_x, v_y = state
    n = 3
    dx_dt = v_x
    dy_dt = v_y
    dv_x_dt = -(k/m) * v_x**n 
    dv_y_dt = -g - (k/m) * v_y**n
    
    return [dx_dt, dy_dt, dv_x_dt, dv_y_dt]

def projectile_motion_without_resistance(state, t, g, k, m):
    x, y, v_x, v_y = state
    dx_dt = v_x
    dy_dt = v_y
    dv_x_dt = 0
    dv_y_dt = -g
    
    return [dx_dt, dy_dt, dv_x_dt, dv_y_dt]

# 定义参数
g = 9.8  # 重力加速度
k = 0.1  # 空气阻力系数
m = 1.0  # 物体质量

# 定义初始条件
x0 = 0.0  # 初始水平位置
y0 = 10  # 初始垂直位置
v_x0 = 2.0  # 初始水平速度
v_y0 = 0  # 初始垂直速度

# 定义时间点
t = np.linspace(0, 2, 100)  # 从0到2秒，共取100个时间点

# 定义初始状态向量
initial_state = [x0, y0, v_x0, v_y0]

# 求解ODE方程组
solution_with_resistance = odeint(projectile_motion_with_resistance, initial_state, t, args=(g, k, m))
solution_without_resistance = odeint(projectile_motion_without_resistance, initial_state, t, args=(g, k, m))

# 提取位置和速度信息
x_with_resistance = solution_with_resistance[:, 0]
y_with_resistance = solution_with_resistance[:, 1]
v_x_with_resistance = solution_with_resistance[:, 2]
v_y_with_resistance = solution_with_resistance[:, 3]

x_without_resistance = solution_without_resistance[:, 0]
y_without_resistance = solution_without_resistance[:, 1]
v_x_without_resistance = solution_without_resistance[:, 2]
v_y_without_resistance = solution_without_resistance[:, 3]

# 打印结果
for i in range(len(t)):
    print(f"t = {t[i]:.2f}: (x, y) = ({x_with_resistance[i]:.2f}, {y_with_resistance[i]:.2f}), (v_x, v_y) = ({v_x_with_resistance[i]:.2f}, {v_y_with_resistance[i]:.2f})")

state_list = np.array(KF.KF_position_tracking([[x0],[y0],[v_x0],[v_y0]],v_x_with_resistance,v_y_with_resistance))
print(state_list)

# 绘制轨迹
plt.plot(x_with_resistance, y_with_resistance, label = 'with resistance', marker='o', markersize = 3)
plt.plot(x_without_resistance, y_without_resistance, label = 'without resistance', marker='o', markersize = 3)
#plt.plot(state_list[:, 0], state_list[:, 1], label = 'Kalman Filter tracking', marker='o', markersize = 3)
plt.xlabel('x (meter)')
plt.ylabel('y (meter)')
plt.title('Projectile Motion trajectory')
plt.grid(True)
plt.legend()
plt.show()

# 绘制速度变化曲线
plt.plot(t, v_x_with_resistance, label='v_x_with_resistance', marker='o', markersize = 3)
plt.plot(t, v_y_with_resistance, label='v_y_with_resistance', marker='o', markersize = 3)
plt.plot(t, v_x_without_resistance, label='v_x_without_resistance', marker='o', markersize = 3)
plt.plot(t, v_y_without_resistance, label='v_y_without_resistance', marker='o', markersize = 3)
plt.xlabel('t')
plt.ylabel('Velocity')
plt.title('Velocity vs Time')
plt.legend()
plt.grid(True)
plt.show()