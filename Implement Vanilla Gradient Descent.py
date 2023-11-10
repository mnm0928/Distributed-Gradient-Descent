import numpy as np
import matplotlib.pyplot as plt


# Define the function and its gradient
def f(x):
    return x ** 2


def delta_f(x):
    return 2 * x


x = 3.5
lr = 0.1
iters = 20
eps = 1e-6

# List to store x values
x_hist = [x]

for i in range(iters):
    x = x - (lr * delta_f(x))
    # Add the updated x to the history
    x_hist.append(x)
    # Check for convergence
    if np.abs(x_hist[i + 1] - x_hist[i]) < eps:
        break

print(x_hist)

# Plot the function and the points visited by x
x_grid = np.linspace(-4, 4, 1000)
plt.plot(x_grid, f(x_grid), label='f(x)')
plt.scatter(x_hist, [f(x) for x in x_hist], color='r', label='x')
plt.title("A simple function f(x)=x^2")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
