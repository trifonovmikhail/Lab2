import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_data = np.loadtxt("train/train_data.csv", delimiter=",")
train_data_noise = np.loadtxt("train/train_data_noise.csv", delimiter=",")

train_data_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
train_data_noise_scaled = scaler.transform(train_data_noise.reshape(-1, 1))

np.savetxt("train/train_data_scaled.csv", train_data_scaled, delimiter=",")
np.savetxt("train/train_data_noise_scaled.csv", train_data_noise_scaled, delimiter=",")