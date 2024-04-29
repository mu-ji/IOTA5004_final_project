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
x_train = np.array(x_trajectory_half[::-1])
x_train = x_train.copy() - 1700

plt.plot(t_train, x_train)
plt.show()

n_f = 10000
f_batch_size = 32

reg_in = torch.from_numpy(t_train).type(torch.float32)
reg_in = reg_in[:,None]

reg_ylabel = torch.from_numpy(y_train).type(torch.float32)
reg_ylabel = reg_ylabel[:,None]

print(reg_in)

f_y = np.random.uniform(1700, 2000, n_f)
f_yt = np.random.uniform(-1, 1, n_f)

f_data_y = np.vstack([f_y, f_yt]).T

f_y = Variable(torch.from_numpy(f_data_y[:, 0:1]).type(torch.FloatTensor), requires_grad=True)
f_yt = Variable(torch.from_numpy(f_data_y[:, 1:2]).type(torch.FloatTensor), requires_grad=True)

f_dataset_y = torch.utils.data.TensorDataset(f_y, f_yt)

f_data_loader_y = torch.utils.data.DataLoader(f_dataset_y, batch_size = f_batch_size, shuffle=True)


# Define a MLP and function f
class MLP(nn.Module):
    def __init__(self, in_dim = 1,  hidden_dim = 128, out_dim = 1):

        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
                                nn.Linear(in_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, out_dim),
                                        )

    def forward(self, data_in):
        return self.mlp(data_in)

model = MLP()

class f_t(nn.Module):

    def __init__(self):

        super(f_t, self).__init__()
        self.k = nn.Parameter(1*torch.ones(1, ), requires_grad=True)
        self.g = nn.Parameter(1*torch.ones(1, ), requires_grad=True)

    def forward(self, t):
        u = model(t)
        u_t_y = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]
        u_tt_y = torch.autograd.grad(u_t_y, t, grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]
        return u_tt_y  + self.g + (self.k)*u_t_y
    
# Training the PINN
n_epoch = 50
f_model = f_t()
paras = list(model.parameters()) + list(f_model.parameters())
optimizer_y = optim.Adam(paras, lr=1e-4)
Alpha_k = np.zeros(n_epoch,)
Alpha_g = np.zeros(n_epoch,)


for epoch in range(n_epoch):
    for x, t in tqdm(f_data_loader_y):

        optimizer_y.zero_grad()
        pred = model(reg_in)
        reg_loss = torch.mean((reg_ylabel - pred) ** 2)
        
        f_loss = torch.mean(f_model(t) ** 2)

        loss = reg_loss + f_loss   # adjust the coefficients between two losses
        loss.backward(retain_graph=True)
        optimizer_y.step()

    print("epoch = {}, loss = {}".format(epoch, loss))
    print("epoch = {}, f_loss = {}".format(epoch, f_loss))
    print("epoch = {}, reg_loss = {}".format(epoch, reg_loss))

    Alpha_k[epoch] = f_model.k.detach().numpy().item()
    Alpha_g[epoch] = f_model.g.detach().numpy().item()


ts_torch = torch.from_numpy(t_train).type(torch.float32)
ts_torch = ts_torch[:,None]



plt.figure(figsize=(10, 4))
plt.plot(t_train, y_train,color = "silver", lw = 4)
plt.plot(t_train, model( ts_torch )[:,0:1].detach().numpy(),lw = 2)
plt.scatter(t_train, y_train,alpha=0.5, c = "red")
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.legend(['Ground Truth', 'Prediction (PINN)', "measured data"])
plt.grid(True)
plt.show()

plt.plot(np.asarray(Alpha_k), label = r"estimate $k$")
plt.plot(np.asarray(Alpha_g), label = r"estimate $g$")
#plt.plot(np.linspace(0,n_epoch, n_epoch), k*np.ones(n_epoch),"--", color = "gray", label = r"true $k$")
plt.ylabel(r"$k/g$")
plt.xlabel("epoch")
plt.legend()
plt.show()
