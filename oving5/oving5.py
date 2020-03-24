import numpy as np
from matplotlib import animation
from scipy import interpolate
from numpy import where
from math import sin
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pylab as plt
plt.get_current_fig_manager().window.raise_()


LNWDT=2; FNT=15
plt.rcParams['lines.linewidth'] = LNWDT; plt.rcParams['font.size'] = FNT


# Declaring constants
l = 1
t_start = 0
t_end = 0.8
a_constant = 1
c = 1
N = 100
global delta_x
delta_x = (1 / N)
global delta_t
delta_t = delta_x * c / a_constant
# delta t found from the CFL-condition.
# Also asuming that we are using a constant delta t during the nonlinear numerical scheme
x = np.linspace(0, l, N + 1)


# t = np.linspace(t_start,t_end, N*c*a + 1)

def aNonLinear(u):
    return 0.9 + 0.1 * u


"""def D(a, delta_t, delta_x):
    return a*delta_t/delta_x"""


# initial advection equation, u0 when time is zero, is given by:
def u0(x):
    u0 = np.zeros(np.size(x), float)
    for i in range(np.size(x)):
        if x[i] < 0.2 and x[i] > 0:
            u0[i] = (np.sin(np.pi * (x[i] / 0.2))) ** 2
        else:
            u0[i] = 0
    return u0


# Analytical solution is given by:
def u_analytical(u0, x, a, t):
    return u0(x - a * t)


# Computing the numerical solution of the next time step in the advection equation
# This is equivalent to one iteration through the numerical scheme
def linear(u_prev, x, a):
    D = a * delta_t / delta_x
    u_next = np.zeros(np.size(x), dtype=np.float)
    for j in range(1, np.size(x)):
        u_next[j] = u_prev[j] - D * (u_prev[j] - u_prev[j - 1])
    return u_next

    # The non-linear numerical scheme, asuming a is a vector with different values. Resulting in D becoming a vector.


def nonLinear(u_prev, x, a):
    D = a * delta_t / delta_x
    u_next = np.zeros(np.size(x), dtype=np.float64)
    for j in range(1, np.size(x)):
        u_next[j] = u_prev[j] - D[j] * (u_prev[j] - u_prev[j - 1])
    return u_next


def conservative(u_prev, x):  # The expression of F is: 0.9*u + 0.1*u**2

    F = np.zeros(np.size(u_prev), dtype=np.float32)
    for i in range(np.size(u_prev)):
        F[i] = np.float32(0.9 * u_prev[i] + 0.1 * (np.float32(u_prev[i] ** 2)))
    F = np.array(F, dtype=np.float32)
    u_next = np.zeros(np.size(x), dtype=np.float64)
    for j in range(1, np.size(x)):
        u_next[j] = u_prev[j] - (delta_t / delta_x) * (np.float64(F[j] - F[j - 1]))
    return u_next


# Preallocating vectors for analytical solution and numerical solution.
# Need to keep the previous step to calculate the next one
time = np.arange(t_start, t_end + delta_t, delta_t)

u_linear0 = u_analytical(u0, x, a_constant, 0)
u_nonlinear0 = u_analytical(u0, x, aNonLinear(u0(x)), 0)
u_conservative0 = u_analytical(u0, x, a_constant, 0)

u_Analytical = u_analytical(u0, x, a_constant, 0)

# Incrementing through the numerical scheme
for i in range(1, np.size(time)):
    u_linear1 = linear(u_linear0, x, a_constant)
    u_linear1[np.size(u_linear1) - 1] = u_analytical(u0, x - a_constant * delta_t, a_constant, time[i])[
        np.size(u_linear1) - 1]  # setting right boundary condition

    u_nonlinear1 = nonLinear(u_nonlinear0, x, aNonLinear(u_nonlinear0))
    u_nonlinear1[np.size(u_nonlinear1) - 1] = \
    u_analytical(u0, x - aNonLinear(u_nonlinear0) * delta_t, a_constant, time[i])[
        np.size(u_nonlinear1) - 1]  # setting right boundary condition

    u_conservative1 = conservative(u_conservative0, x)
    # u_Analytical = u_analytical(u0,x,a_constant,time[i])
    u_linear0 = u_linear1
    u_nonlinear0 = u_nonlinear1
    u_conservative0 = u_conservative1

# This will plot the result for the last time-step
plt.figure('Oppgave a, b, c - ftbs')
plt.plot(x, u_analytical(u0, x, a_constant, 0.8), label="Analytical")
plt.plot(x, u_linear1, linestyle='dashed', label="Numerical linear solution")
plt.plot(x, u_nonlinear1, linestyle='dashed', label="Numerical non-linear solution")
plt.plot(x, u_conservative1, linestyle='dashed', label="Conservative")
plt.legend()
plt.title("Solution at time = %s" % t_end)
plt.ylabel("Deviation")
plt.xlabel("x-position")
plt.grid()
plt.show()

# -----

plt.get_current_fig_manager().window.raise_()


LNWDT=2; FNT=15
plt.rcParams['lines.linewidth'] = LNWDT; plt.rcParams['font.size'] = FNT


l = 1.0
a0 = 1.0
t = 0.8
dx = 0.02
#set the value for dt such that the conservative solution dont become unstable
dt = 0.002
m = int(l/dx)
n = int(t/dt)
c = a0*dt/dx
def f(x):
    f = np.zeros_like(x)
    x_left = 0.0
    x_right = 0.2
    f = where((x > x_left) & (x < x_right), np.sin(np.pi * x / 0.2) ** 2, f)
    return f


def a(u):
    return 0.9 + 0.1*u


x = np.linspace(0, l, m+1)
u_0 = np.zeros_like(x)
u_0[0:int(0.2/dx)] = f(x[0:int(0.2/dx)])




# time vector
time = np.linspace(0,0.8,n+1)

u_a = np.zeros((len(time), len(x)))
u_a[0,:] = u_0

# analytical u
u_analytic = np.zeros_like(u_a)

# timestep t = 0
u_analytic[0,:] = f(x)
u_p = np.zeros_like(u_0)

#task d_a
for n, t in enumerate(time[1:]):
    u_bc = interpolate.interp1d(x[-2:], u_a[n,-2:])
    u_p[:-1] = u_a[n,:-1] + c*(u_a[n,:-1]-u_a[n,1:])
    u_a[n+1,1:-1] = 1/2*(u_a[n,1:-1]+u_p[1:-1]) + c/2*(u_p[:-2]-u_p[1:-1])
    u_a[n+1,-1] = u_bc(x[-1] - a0*dt)
    u_analytic[n+1, :] = f(x-a0*t)



# the solution will be a wave that propogates forward in time, the approximate solution match the analytical for all
# time steps

# task d_b


u_b = np.zeros((len(time),len(x)))
u_b[0,:] = u_0

u_p = np.zeros_like(u_0)
for n, t in enumerate(time[1:]):
    u_bc = interpolate.interp1d(x[-2:], u_b[n,-2:])
    c = a(u_b[n,:-1])*dt/dx
    u_p[:-1] = u_b[n, :-1] + c * (u_b[n, :-1] - u_b[n, 1:])
    u_b[n + 1, 1:-1] = 1 / 2 * (u_b[n, 1:-1] + u_p[1:-1]) + c[1:] / 2 * (u_p[:-2] - u_p[1:-1])
    u_b[n+1,-1] = u_bc(x[-1] - a(u_b[n,-1])*dt)


# task d_c
# F becomes F = u*(0.9+0.1*u)

def F(u):
    return u*0.9+0.1*u**2


u_f = np.zeros_like(u_a)
u_f[0,:] = u_0

for n, t in enumerate(time[1:]):
    u_bc = interpolate.interp1d(x[-2:], u_f[n, -2:])
    u_p[:-1] = u_f[n, :-1] + dt/dx * (F(u_f[n, :-1]) - F(u_f[n, 1:]))
    u_f[n + 1, 1:-1] = 1 / 2 * (u_f[n, 1:-1] + u_p[1:-1]) + dt/(dx*2) * (F(u_p[:-2]) - F(u_p[1:-1]))
    u_f[n+1,-1] = u_bc(x[-1] - a(u_f[n, -1])*dt)

# Becomes unstable with these values

# Animation

u_solutions=np.zeros((4,len(time),len(x)))
u_solutions[0, :, :] = u_a
u_solutions[1, :, :] = u_b
u_solutions[2, :, :] = u_f
u_solutions[3, :, :] = u_analytic

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure('Oppgave d - MacCormack')
ax = plt.axes(xlim=(0, l), ylim=(np.min(u_a), np.max(u_a) * 1.1))

lines = []  # list for plot lines for solvers and analytical solutions
legends = []  # list for legends for solvers and analytical solutions

solvers = ['Oppgave d-a', 'Oppgave d-b', 'Conservative', 'Analytical']

for solver in solvers:
    line, = ax.plot([], [])
    lines.append(line)
    legends.append(solver)

plt.xlabel('x-coordinate [-]')
plt.ylabel('Amplitude [-]')
plt.legend(legends, loc=3, frameon=False)

# initialization function: plot the background of each frame
def init():
    for line in lines:
        line.set_data([], [])
    return lines,


def animate_alt(i):
    for k, l in enumerate(lines):
        l.set_data(x, u_solutions[k, i, :])
    return lines,


# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate_alt, init_func=init, frames=n+1, interval=20, blit=False)
plt.show()
