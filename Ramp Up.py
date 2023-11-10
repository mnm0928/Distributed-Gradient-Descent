import numpy as np
import matplotlib.pyplot as plt

X = np.arange(0, 1, 0.01)
Y = X + np.random.normal(0, 0.2, len(X))

theta1 = 1.0
theta2 = 0.0


def y_hat(x, theta1, theta2):
    return theta1 * X + theta2


# Plot data and model predictions
plt.scatter(X, Y)
plt.plot(X, y_hat(X, theta1, theta2), color='red')
plt.show()


# Part 2

# Define the loss function L(y, y_hat)
def loss(y, y_hat):
    return np.sum((y - y_hat) ** 2)


# Define the gradients of the loss and model functions
def grad_loss_y_hat(y, y_hat):
    return -2 * (y - y_hat)


def grad_y_hat_theta1(x):
    return x


def grad_y_hat_theta2():
    return 1


t1 = 0.5
t2 = 0.2
eta = 0.006
iters = 2000

loss_values = []  # empty list to store loss values


#fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(20, 16))

for epoch in range(iters):
    # Compute y_hat
    y_pred = y_hat(X, t1, t2)

    # Compute derivatives
    d_l_wrt_yhat = grad_loss_y_hat(Y, y_pred)
    d_yhat_wrt_theta1 = grad_y_hat_theta1(X)
    d_yhat_wrt_theta2 = grad_y_hat_theta2()

    # Update parameters
    t1 = t1 - eta * (np.sum(d_l_wrt_yhat * d_yhat_wrt_theta1) / len(Y))
    t2 = t2 - eta * (np.sum(d_l_wrt_yhat * d_yhat_wrt_theta2) / len(Y))

    # Compute loss
    loss_epoch = loss(Y, y_pred)
    print("Epoch: ", epoch, "Loss :", loss_epoch)

    loss_values.append(loss_epoch)  # append loss for this epoch

    # Plot the data and model predictions in each subplot
    #row, col = epoch // 10, epoch % 10
    #axes[row, col].clear()
    #axes[row, col].scatter(X, Y)
    #axes[row, col].plot(X, y_hat(X, t1, t2), color='red')

print("Optimal parameter values: ", "theta1", t1, "theta2", t2)

# plot loss against epochs
plt.figure()
plt.plot(range(1, iters+1), loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.show()
