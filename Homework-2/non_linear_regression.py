# Module imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gradient_descent(X, Y, a, b, c, d, L, epochs):
    """
    Gradient descent to find local minimum
    """
    N = len(X)
    for e in range(epochs):
        # Model with cubic function in the format: Y = aX^3 + bX^2 + cX + d
        Y_pred = a * X**3 + b * X**2 + c * X + d
        # Monitor MSE(Model should aim to minimize this)
        mse = ((Y_pred - Y) ** 2).mean()
        if e % 1000 == 0:
            print(f"Epoch: {e}, MSE: {mse}")

        # Calculate gradients
        da = (-2/N) * np.sum((Y - Y_pred) * X**3)
        db = (-2/N) * np.sum((Y - Y_pred) * X**2)
        dc = (-2/N) * np.sum((Y - Y_pred) * X)
        dd = (-2/N) * np.sum(Y - Y_pred)

        # Update parameters using gradients
        a -= L * da
        b -= L * db
        c -= L * dc
        d -= L * dd

    return a, b, c, d


if __name__ == "__main__":
    # Read the CSV file containing non-linear data
    df = pd.read_csv("HW2_nonlinear_data.csv")
    X = df['X'].values
    Y = df['Y'].values

    # Initialize parameters (a, b, c, d)
    a = b = c = d = 0
    L = 1e-6
    epochs = 10000

    # Train the model
    a, b, c, d = gradient_descent(X, Y, a, b, c, d, L, epochs)

    # Plot the results
    Y_pred = a * X**3 + b * X**2 + c * X + d

    # Save the plotted results
    plt.scatter(X, Y, color='blue')
    plt.scatter(X, Y_pred, color='red')
    plt.title('Non-Linear Regression')
    plt.savefig("non_linear_regression.png")
    print("Plot saved at: ./non_linear_regression.png")
