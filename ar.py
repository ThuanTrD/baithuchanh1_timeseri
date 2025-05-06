import pandas as pd
import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv(r"D:\time_seris_btap1\playground-series-s5e4\train.csv")
df1 = df[df["Publication_Day"] == "Saturday"]
df1 = df1.drop("id", axis=1)
df2 = df1[df1["Publication_Time"] == "Morning"]
observations = df2['Listening_Time_minutes'].values

# Cấu hình Kalman Filter cho mô hình tự hồi quy (AR(1))
phi = 0.9  #
kf = KalmanFilter(
    transition_matrices=[[phi]],
    observation_matrices=[[1]],
    initial_state_mean=[observations[0]],
    initial_state_covariance=[[1]],
    observation_covariance=1,
    transition_covariance=[[0.1]]
)

# Áp dụng Kalman Filter
state_means, _ = kf.filter(observations)

# Vẽ biểu đồ
plt.figure(figsize=(14, 5))
plt.plot(observations, label="Original")
plt.plot(state_means.flatten(), label="Kalman Autoregressive")
plt.title("Kalman Filter with Autoregressive Model on Saturday Morning Listening Time")
plt.legend()
