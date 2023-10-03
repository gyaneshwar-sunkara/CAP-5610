# Module imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Gradient descent to find local minimum
def gradient_descent(X, Y, m, c, L, epochs):
    N = len(X)
    for e in range(epochs):
        # Linear model in the format Y = mX + c
        Y_pred = m * X + c
        # Monitor MSE(Model should aim to minimize this)
        mse = ((Y - Y_pred) ** 2).mean()
        if e % 100 == 0:
            print(f"Epoch: {e}, MSE: {mse}")
        # Derivative with respective to m
        dm = (-2/N) * np.sum(X * (Y - Y_pred))
        # Derivative with respective to c
        dc = (-2/N) * np.sum(Y - Y_pred)

        # Update weights(m and c)
        m = m - L * dm
        c = c - L * dc
    return m, c


if __name__ == "__main__":
    # Read the CSV file containing linear data
    df = pd.read_csv("HW2_linear_data.csv", names=['X', 'Y'])

    # Extract X and Y values from the dataframe
    X = df['X'].values
    Y = df['Y'].values

    # Initialize parameters
    L = 0.0001
    epochs = 1000
    m = 0
    c = 0

    # Train the model
    m, c = gradient_descent(X, Y, m, c, L, epochs)

    # Plot the results
    Y_pred = X * m + c

    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')

    # Save image
    plt.title('Linear Regression')
    plt.savefig("linear_regression.png")

    print("========================")

    # Display model MSE
    mse = ((Y - Y_pred) ** 2).mean()
    print(f"Mean Square Error: {mse}")

    # Display computed weights after gradient descent
    print(f"Slope (m): {m}")
    print(f"Intercept (c): {c}")
