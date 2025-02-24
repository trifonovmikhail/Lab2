import numpy as np
import os

os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)

np.random.seed(42)
train_data = np.random.normal(loc=20, scale=5, size=100)
train_data_noise = train_data + np.random.normal(loc=0, scale=2, size=100)

np.savetxt("train/train_data.csv", train_data, delimiter=",")
np.savetxt("train/train_data_noise.csv", train_data_noise, delimiter=",")

test_data = np.random.normal(loc=20, scale=5, size=50)
test_data_noise = test_data + np.random.normal(loc=0, scale=2, size=50)

np.savetxt("test/test_data.csv", test_data, delimiter=",")
np.savetxt("test/test_data_noise.csv", test_data_noise, delimiter=",")