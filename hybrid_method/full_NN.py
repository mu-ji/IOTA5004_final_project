import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import plotly.graph_objects as go
import scipy.io as scio
import random
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d

badminton_trajectory = np.load("final_project_video_process/badminton_trajectory.npy")
print(badminton_trajectory.shape)

#FPS = 30 so each frame = 1/30 s
#badminton_trajectory.shape = 36
#video shape = (1080, 1920)

x_with_resistance = badminton_trajectory[:, 0]
y_with_resistance = badminton_trajectory[:, 1]
x_trajectory = x_with_resistance
y_trajectory = []
for i in range(len(y_with_resistance)):
    y_trajectory.append(1920 - 1 - y_with_resistance[i])

ts = np.linspace(0, 0.033*36, 36)

# 绘制轨迹
plt.plot(x_trajectory, y_trajectory, label = 'badminton trajectory', marker='o', markersize = 3, c = 'r')
#plt.plot(state_list[:, 0], state_list[:, 1], label = 'Kalman Filter tracking', marker='o', markersize = 3)
plt.xlabel('x (meter)')
plt.ylabel('y (meter)')
plt.title('Projectile Motion trajectory')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()

#using interpolation method to generate more data point
x_trajectory_half = x_trajectory[:12]
y_trajectory_half = y_trajectory[:12]

t_train = np.linspace(0, 0.033*12, 12)
y_train = np.array(y_trajectory_half[::-1])
y_train = y_train.copy() - 1700

plt.plot(t_train, y_train)
plt.show()

