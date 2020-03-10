import numpy as np
import matplotlib.pyplot as plt
import scipy as sip
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def temp_beam(Tleft, Tright, T0, L, deltaT, deltaX, endTime):
    D = deltaT / deltaX ** 2

    # Checking if the FTCS will be stable or not:
    print("Checking stability")
    print("------------------")

    if (D < 0.5):
        print("Stability check okey")
    else:
        print("WARNING --> unstable for given parameters")
        return

    T = np.int(endTime / deltaT)  # setting number of iterations
    X = np.int(L / deltaX + 1)  # setting number of iterations along the beam

    temp_prev = np.zeros(X, float)  # initial condition is that all the temperatures along the beam is zero
    temp_next = np.zeros(X, float)

    for j in range(0, T):
        for i in range(0, X):
            if i == 0:
                temp_next[i] = 100  # left boundary condition
            elif i == X - 1:
                temp_next[i] = 0  # right boundary condition
            elif i == 1:
                temp_next[i] = D * (temp_prev[i + 1] + 100) + (1 - 2 * D) * temp_prev[i]
            elif i == X - 1:
                temp_next[i] = D * (temp_prev[i - 1]) + (1 - 2 * D) * temp_prev[i]
            else:
                temp_next[i] = D * (temp_prev[i - 1] + temp_prev[i + 1]) + (1 - 2 * D) * temp_prev[i]
        temp_prev = temp_next

    return temp_next


# --End-of-function

Temp1 = temp_beam(100, 0, 0, 1, 10 ** (-5), 0.01, 0.001)
Temp2 = temp_beam(100, 0, 0, 1, 10 ** (-5), 0.01, 0.025)
Temp3 = temp_beam(100, 0, 0, 1, 10 ** (-5), 0.01, 0.4)

x = np.linspace(0, 1, 101)

plt.plot(x, Temp1, label="t=0.001")
plt.plot(x, Temp2, label="t=0.025")
plt.plot(x, Temp3, label="t=0.4")
plt.legend()
plt.ylabel("Temperature")
plt.xlabel("x-position")
plt.grid()
plt.show()


def temp_field(Tleft, Ttop, l, N):
    A = np.zeros((N * N, N * N), int)
    b = np.zeros((N * N), int)

    for j in range(0, N):
        for i in range(1, N + 1):
            A[j * N + i - 1, j * N + i - 1] = -4  # setting the diagonals to -4

            if i > 1:
                A[j * N + i - 1, j * N + i - 2] = 1
            if i < N:
                A[j * N + i - 1, j * N + i] = 1
            if j > 0:
                A[j * N + i - 1, j * N + i - (N + 1)] = 1
            if j < N - 1:
                A[j * N + i - 1, j * N + i + (N - 1)] = 1

            if j == 0:  # if we are at the bottom of the plate, we need to use ghost nodes
                A[j * N + i - 1, i + (N - 1)] = 2

            if i == N:
                A[j * N + i - 1, j * N + i - 2] = 2

            if i == 1 and j == N - 1:  # if we are at the node next to the top left corner
                b[j * N + i - 1] = -(Tleft + Ttop)
                A[j * N + i - 1, j * N + i] = 1
                A[j * N + i - 1, j * N + i - 1 - N] = 1
            elif i == 1 and j == 0:
                b[j * N + i - 1] = -Tleft
                A[j * N + i - 1, j * N + i] = 1
                A[j * N + i - 1, j * N + i - 1 + N] = 2
            elif i == 1:  # if we are next to the left boundary
                b[j * N + i - 1] = -Tleft
                A[j * N + i - 1, j * N + i] = 1
                A[j * N + i - 1, j * N + i - 1 + N] = 1
                A[j * N + i - 1, j * N + i - 1 - N] = 1
            elif j == N - 1 and i == N:
                b[j * N + i - 1] = -Ttop  # if we are next to the top boundary
                A[j * N + i - 1, j * N + i - 2] = 2
                A[j * N + i - 1, j * N + i - 1 - N] = 1
            elif j == N - 1:
                b[j * N + i - 1] = -Ttop
                A[j * N + i - 1, j * N + i - 2] = 1
                A[j * N + i - 1, j * N + i] = 1
                A[j * N + i - 1, j * N + i - 1 - N] = 1

    return A, b


# --End-of-function

def plot_surface_neumann_dirichlet(Temp, Ttop, Tleft, l, N, nxTicks=4, nyTicks=4):
    """ Surface plot of the stationary temperature in quadratic beam cross-section.
        Note that the components of T has to be started in the
        lower left part of the grid with increasing indexes in the
        x-direction first.


         Args:
             Temp(array):  the unknown temperatures, i.e. [T_1 .... T_(NxN)]
             Ttop(float):  temperature at the top boundary
             Tleft(float): temperature at the left boundary
             l(float):     height/width of the sides
             N(int):       number of nodes with unknown temperature in x/y direction
             nxTicks(int): number of ticks on x-label (default=4)
             nyTicks(int): number of ticks on y-label (default=4)
    """
    x = np.linspace(0, l, N + 1)
    y = np.linspace(0, l, N + 1)

    X, Y = np.meshgrid(x, y)

    T = np.zeros_like(X)

    T[-1, :] = Ttop
    T[:, 0] = Tleft
    k = 1
    for j in range(N):
        T[j, 1:] = Temp[N * (k - 1):N * k]
        k += 1

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, T, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_zlim(0, Ttop)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('T [$^o$C]')

    xticks = np.linspace(0.0, l, nxTicks + 1)
    ax.set_xticks(xticks)

    yticks = np.linspace(0.0, l, nyTicks + 1)
    ax.set_yticks(yticks)
    plt.show()


# -End-of-function

A, b = temp_field(50, 100, 1, 50)
A_invers = np.linalg.inv(A)

Temp_temp = A_invers * b
Temp = np.zeros(np.size(b), float)
for i in range(np.size(Temp)):
    Temp[i] = np.sum(Temp_temp[i, :])

# print(Temp)
plot_surface_neumann_dirichlet(Temp, 100, 50, 1, 50)

# plotting a 50x50 temperature mesh

