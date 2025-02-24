import numpy as np
from sklearn.linear_model import LinearRegression

X_train = np.loadtxt("train/train_data_scaled.csv", delimiter=",")
y_train = np.loadtxt("train/train_data_noise_scaled.csv", delimiter=",")

model = LinearRegression()
model.fit(X_train, y_train)

import joblib
joblib.dump(model, "model.pkl")