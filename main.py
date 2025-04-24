import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def load_data(filename):
    data = np.loadtxt(filename)
    X, y = data[:, :-1], data[:, -1]
    return X, y

def train_linear_model_iterative(X, y, epochs=1000, lr=0.01):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # dodanie biasu
    theta = np.random.randn(X_b.shape[1])
    for _ in range(epochs):
        gradients = 2 / X_b.shape[0] * X_b.T @ (X_b @ theta - y)
        theta -= lr * gradients
    return theta

def predict(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return X_b @ theta

def train_polynomial_model(X, y, degree=2):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    theta = np.linalg.pinv(X_poly.T @ X_poly) @ X_poly.T @ y
    return theta, poly

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Ścieżka do pliku z danymi (np. dane23.txt)")
    args = parser.parse_args()

    X, y = load_data(args.filename)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model liniowy
    theta1 = train_linear_model_iterative(X_train, y_train)
    y_pred1 = predict(X_test, theta1)
    mse1 = mean_squared_error(y_test, y_pred1)
    print(f"Model Liniowy - MSE: {mse1:.4f}")

    # Model wielomianowy
    theta2, poly = train_polynomial_model(X_train, y_train, degree=2)
    X_test_poly = poly.transform(X_test)
    y_pred2 = X_test_poly @ theta2
    mse2 = mean_squared_error(y_test, y_pred2)
    print(f"Model Wielomianowy - MSE: {mse2:.4f}")

    # Porównanie modeli
    if mse1 < mse2:
        print("Model liniowy lepszy.")
    elif mse1 > mse2:
        print("Model wielomianowy lepszy.")
    else:
        print("Oba modele mają podobną skuteczność.")

    if X.shape[1] == 1:
        plt.scatter(X_test, y_test, color="black", label="Prawdziwe")
        sort_idx = np.argsort(X_test[:, 0])
        plt.plot(X_test[sort_idx, 0], y_pred1[sort_idx], label="Model 1", color="blue")
        plt.plot(X_test[sort_idx, 0], y_pred2[sort_idx], label="Model 2", color="red")
        plt.legend()
        plt.title("Porównanie modeli")
        plt.show()

if __name__ == "__main__":
    main()
