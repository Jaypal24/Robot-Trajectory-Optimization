import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import joblib

X = np.load("X.npy")
Y = np.load("Y.npy")

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = MLPRegressor(
    hidden_layer_sizes=(128, 128),
    activation='relu',
    max_iter=500
)

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)

print("Test MSE:", mse)

joblib.dump(model, "trajectory_model.pkl")
