import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import logging

logging.basicConfig(format='Rank %(name)s: %(message)s')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
logger = logging.getLogger(str(rank))

my_X = None
my_Y = None
if rank == 0:
    X = np.arange(0, 1, 0.01)
    Y = X + np.random.normal(0, 0.2, len(X))
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    my_X = np.array_split(X, size)
    my_Y = np.array_split(Y, size)

my_X = comm.scatter(my_X, root=0)
my_Y = comm.scatter(my_Y, root=0)

t1 = 0.5
t2 = 0.2
eta = 0.006
iters = 1000

loss_values = []  # empty list to store loss values


# Define predicted value of y
def y_hat(x, theta1, theta2):
    return theta1 * x + theta2


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


for epoch in range(iters):
    # Compute y_hat
    sub_y_pred = y_hat(my_X, t1, t2)

    # Compute derivatives
    sub_d_l_wrt_yhat = grad_loss_y_hat(my_Y, sub_y_pred)
    sub_d_yhat_wrt_theta1 = grad_y_hat_theta1(my_X)
    d_yhat_wrt_theta2 = grad_y_hat_theta2()

    # Collect subgradients at coordinator
    d_l_wrt_yhat = comm.gather(sub_d_l_wrt_yhat, root=0)
    d_yhat_wrt_theta1 = comm.gather(sub_d_yhat_wrt_theta1, root=0)
    ypred = comm.gather(sub_y_pred, root=0)

    if rank == 0:
        d_l_yhat = sum(d_l_wrt_yhat)
        d_yhat_t1 = sum(d_yhat_wrt_theta1)
        Y_pred = np.vstack(ypred)

        # Update parameters
        t1 = t1 - eta * (np.sum(d_l_yhat * d_yhat_t1) / len(Y))
        t2 = t2 - eta * (np.sum(d_l_yhat * d_yhat_wrt_theta2) / len(Y))
        # Compute loss
        loss_epoch = loss(Y, Y_pred)
        print("Epoch: ", epoch, "Loss :", loss_epoch)

        loss_values.append(loss_epoch)  # append loss for this epoch


if rank == 0:
    print("Optimal parameter values: ", "theta1", t1, "theta2", t2)

    # plot loss against epochs
    plt.figure()
    plt.plot(range(1, iters + 1), loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.show()
