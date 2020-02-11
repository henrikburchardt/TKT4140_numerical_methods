import scipy.special
import scipy as sp
import matplotlib.pylab as plt
import numpy as np


'Problem 1'


def euler(x, y):

    return 1 - 3*x + y + x**2 + x*y


h = 0.01
x = np.arange(0, 1.5, h)
y = np.zeros(x.size)
y[0] = 0

for i in range(1, x.size):

    y[i] = y[i-1] + h*(euler(x[i-1], y[i-1]))


def newton(x):

    return x - x**2 + x**3/3 - x**4/6 + x**5/30 - x**6/45


k = np.zeros(x.size)

for i in range(1, x.size):
    k[i] = newton(x[i])


def analytical(x):

    return 3*np.sqrt(2*np.pi*np.e)*np.exp(x*(1+x/2))*(sp.special.erf(np.sqrt(2)/2*(1+x))-sp.special.erf(np.sqrt(2)/2))+4*(1-np.exp(x*(1+x/2)))-x


l = np.zeros(len(x))

for i in range(1, len(x)):
    l[i] = analytical(x[i])

plt.figure()
plt.plot(x, y)
plt.plot(x, k)
plt.plot(x, l)
plt.legend(['Euler', 'Newton', 'Analytical'])
plt.xlabel('x')
plt.ylabel('y')
plt.show()


'Problem 2'


# theta' = u
# u' = -my/m*u-g/l*sin(theta)


def euler_method(tend, dt, theta0, theta_dot0):
    g = 9.81
    my = 1
    le = 1
    m = 1
    t = np.arange(0, tend, dt)
    theta = np.zeros(t.size)
    u = np.zeros(t.size)
    theta[0] = theta0
    u[0] = theta_dot0
    for i in range(t.size-1):
        u[i+1] = u[i] + dt*(-my/m*u[i]-g/le*np.sin(theta[i]))
        theta[i+1] = theta[i] + dt*u[i]
    return t, theta


theta_0 = np.deg2rad(85)
theta_dot = 0
t1, theta1 = euler_method(10, 0.01, theta_0, theta_dot)
plt.plot(t1, theta1)
plt.xlabel('time')
plt.ylabel('radians')
plt.legend('Euler Amp')
plt.show()


'Problem 3'


# 3 a)


def heuns_method(tend, dt, theta0, theta_dot0):
    g = 9.81
    my = 1
    le = 1
    m = 1
    t = np.arange(0, tend, dt)
    theta = np.zeros(t.size)
    u = np.zeros(t.size)
    theta[0] = theta0
    u[0] = theta_dot0
    for i in range(t.size - 1):
        theta_s = theta[i] + dt * u[i]
        u_s = u[i] + dt*(-my / m * u[i] - g / le * np.sin(theta[i]))
        theta[i+1] = theta[i] + dt/2*(u[i]+u_s)
        u[i + 1] = u[i] + dt / 2 * ((-my / m * u[i] - g / le * np.sin(theta[i]))
        + (-my / m * u_s - g / le * np.sin(theta_s)))
    return t, theta


# 3 b)
def amplitude(t, theta):
    for i in range(t.size-2):

        if theta[i] > theta[i+1] < theta[i+2]:
            print(theta[i+1])
        elif theta[i] < theta[i+1] > theta[i+2]:
            print(theta[i+1])


theta_0 = np.deg2rad(85)
theta_dot = 0
t, theta = heuns_method(10, 0.1, theta_0, theta_dot)
amplitude(t, theta)
plt.plot(t, theta)
plt.legend('Amplitude')
plt.xlabel('time')
plt.ylabel('radians')
plt.show()


if __name__ == '__main__':
    euler(x, y)
    newton(x)
    analytical(x)
    euler_method(10, 0.1, theta_0, theta_dot)
    heuns_method(10, 0.1, theta_0, theta_dot)
    amplitude(t, theta)


