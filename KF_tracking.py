
import numpy as np
import matplotlib.pyplot as plt

def KF_position_tracking(x_initial,v_x_with_resistance,v_y_with_resistance):
    dt = 0.02
    g = 9.8
    v_x_noise_std = 5e-1
    v_y_noise_std = 5e-1
    v_x_noise = np.random.normal(0, v_x_noise_std, len(v_x_with_resistance))
    v_y_noise = np.random.normal(0, v_y_noise_std, len(v_x_with_resistance))
    v_x_with_resistance = v_x_with_resistance + v_x_noise
    v_y_with_resistance = v_y_with_resistance + v_y_noise
    A = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
    B = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,dt]])
    u = np.array([[0],[0],[0],[-g]])
    H = np.array([[0,0,1,0],[0,0,0,1]])
    #过程噪声
    Q = np.diag([1e-5, 1e-5, 1e-5, 1e-5])    
    #测量噪声
    R_stable = np.diag([v_x_noise_std, v_y_noise_std])
    R = R_stable
    # 估计误差协方差
    P = np.eye(4)

    x = [[0],[2],[10],[0]]
    x = x_initial
    state_list = []
    for i in range(len(v_x_with_resistance)):
        x = np.dot(A, x) + np.dot(B, u)  # 状态预测
        P_pred = np.dot(np.dot(A, P), A.T) + Q  # 估计误差协方差预测 

        y = [[v_x_with_resistance[i]],[v_y_with_resistance[i]]] - np.dot(H, x)    # 观测残差

        S = np.dot(np.dot(H, P_pred), H.T) + R  # 观测残差协方差
        K = np.dot(np.dot(P_pred, H.T), np.linalg.inv(S))  # 卡尔曼增益
        x = x + np.dot(K, y)  # 状态更新
        state_list.append(x)
        P = np.dot((np.eye(4) - np.dot(K, H)), P_pred)  # 估计误差协方差更新
    
    return state_list