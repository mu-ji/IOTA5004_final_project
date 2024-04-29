import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

class trainable_Euler_method(nn.Module):
    # define a simple neural network
    def __init__(self, hidden_dim = 128):
        
        super(trainable_Euler_method, self).__init__()
        
        self.NN = nn.Sequential(
                                nn.Linear(1, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, 1),                                
                                )
        
        
    def forward(self,x0, nt, dt):
        
        x_pred = torch.zeros(nt,)
        x_pred[0] = x0
        x_k = x0
        
        for i in range(1,nt):
            
            x_k_1_pred = x_k + dt*self.NN(x_k)  # Euler method
            
            x_pred[i] = x_k_1_pred
    
            x_k = x_k_1_pred
            
        return x_pred

#%% Data generation

m = 100
t_m = torch.linspace(0, 2,m)
nt = len(t_m)
x_m =  torch.exp(t_m) + 0.1*torch.randn(m,)
dt = t_m[1] - t_m[0]


#%% Autograd

lr = 1e-3
num_epochs = 1500
global_step = 0
epoch_loss = 0

model = trainable_Euler_method()
params = model.parameters()
optimizer = optim.Adam(params, lr= lr)


lr = 0.001
x0 = x_m[0:1]


for epoch in range(num_epochs):
    
    global_step += 1            
    optimizer.zero_grad()
    
    x_pred = model(x0, nt,dt)
    loss = torch.mean((x_pred - x_m)**2)   
    loss.backward()
    optimizer.step() 

    if not epoch % 100:
           
        print('epoch: {}, loss_train: {:.4f}'.format(epoch, loss)) 

plt.figure()
plt.scatter(t_m,x_m, color = "r", label = "Data")        
plt.plot(t_m, x_pred.detach().numpy(), label = "Prediction"  )   
plt.legend() 

#%%
t_test = torch.linspace(4, 6, m)
x_test =  torch.exp(t_test)
nt_test = len(t_test)
dt_test = t_test[1] - t_test[0]
x0_test = x_test[0:1] 
x_pred_test = model(x0_test, len(t_test), dt_test)

plt.figure()
plt.scatter(t_test[0], x_test[0], color = "r" )
plt.plot(t_test, x_test, color = "silver", lw = 3, label = "reference" )
plt.plot(t_test, x_pred_test.detach().numpy(), label = "Prediction")
plt.legend()

#%% visualize NN(x)

xx = torch.linspace(0,15,100).unsqueeze(-1)
NN_xx = model.NN(xx)

plt.figure()
plt.plot(xx.detach().numpy(),NN_xx.detach().numpy())
plt.xlabel("x")
plt.ylabel("NN(x)")
plt.grid()




