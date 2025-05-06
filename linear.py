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

# Cấu hình Kalman Filter cho mô hình xu hướng tuyến tính
kf = KalmanFilter(
    transition_matrices=[[1, 1], [0, 1]],  # Ma trận chuyển tiếp: [level, slope]
    observation_matrices=[[1, 0]],         # Chỉ quan sát level, không quan sát slope
    initial_state_mean=[observations[0], 0],  # Giá trị ban đầu: [level, slope]
    initial_state_covariance=[[1, 0], [0, 1]],  # Độ không chắc chắn ban đầu
    observation_covariance=1,              # Nhiễu phép đo
    transition_covariance=[[0.01, 0], [0, 0.01]]  # Nhiễu chuyển tiếp
)

# Áp dụng Kalman Filter
state_means, _ = kf.filter(observations)

# Vẽ biểu đồ
plt.figure(figsize=(14, 5))
plt.plot(observations, label="Original")
plt.plot(state_means[:, 0], label="Kalman Linear Trend")
plt.title("Kalman Filter with Linear Trend on Saturday Morning Listening Time")
plt.legend()
