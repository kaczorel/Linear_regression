import numpy as np
import argparse

def load_data(filepath):
    data = np.loadtxt(filepath)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def split_data(X, y, train_ratio=0.8):
    n = len(X)
    split_index = int(n * train_ratio)
    return X[:split_index], y[:split_index], X[split_index:], y[split_index:]

# Model  liniowy
def train_linear_model_iterative(X, y, epochs=1000, lr=0.01):
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    theta = np.zeros(X_bias.shape[1])

    for _ in range(epochs):
        predictions = X_bias @ theta
        errors = predictions - y
        gradient = (X_bias.T @ errors) / len(y)
        theta -= lr * gradient

    return theta

def predict_linear(X, theta):
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    return X_bias @ theta

# Model wielomianowy
def train_polynomial_model(X, y, degree=2, epochs=1000, lr=0.01):
    X_poly = np.hstack([X**i for i in range(degree + 1)])
    theta = np.zeros(X_poly.shape[1])

    for _ in range(epochs):
        predictions = X_poly @ theta
        errors = predictions - y
        gradient = (X_poly.T @ errors) / len(y)
        theta -= lr * gradient

    return theta

def predict_polynomial(X, theta, degree=2):
    X_poly = np.hstack([X**i for i in range(degree + 1)])
    return X_poly @ theta

def mean_squared_error_manual(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Ścieżka do pliku z danymi")
    args = parser.parse_args()

    X, y = load_data(args.filename)
    X_train, y_train, X_test, y_test = split_data(X, y)

    theta1 = train_linear_model_iterative(X_train, y_train)
    y_pred1 = predict_linear(X_test, theta1)
    mse1 = mean_squared_error_manual(y_test, y_pred1)

    theta2 = train_polynomial_model(X_train, y_train, degree=2)
    y_pred2 = predict_polynomial(X_test, theta2, degree=2)
    mse2 = mean_squared_error_manual(y_test, y_pred2)

    print("Model 1 (liniowy): MSE =", mse1)
    print("Model 2 (wielomianowy): MSE =", mse2)



if __name__ == "__main__":
    main()
