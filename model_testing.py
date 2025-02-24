import numpy as np
import joblib

model = joblib.load("model.pkl")

X_test = np.loadtxt("test/test_data.csv", delimiter=",")
y_test = np.loadtxt("test/test_data_noise.csv", delimiter=",")

predictions = model.predict(X_test.reshape(-1, 1))

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions)
print(f"MSE: {mse}")