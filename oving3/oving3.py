import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pylab as plt
import numpy as np
import scipy.sparse.linalg
plt.get_current_fig_manager().window.raise_()

N = 90
x = np.linspace(0, 0.9, N + 1)


def y_analytic(x):
    return 0.5*(np.log(np.abs(x-1))-np.log(np.abs(x+1))) + 2


def error_func(phi0, phi1, s0, s1):
    if np.any(phi1 - phi0) > 0.0:
        return -phi1 * (s1 - s0) / float(phi1 - phi0)
    else:
        return 0.0


def f(z, t):
    zout = np.zeros_like(z)
    zout[:] = [z[1], -2.0*t*z[0]**2]
    return zout


beta = y_analytic(0.9)


def dydx(Y, x):
    y0 = Y[0]
    y1 = Y[1]

    return np.array([y1, -2*x*y1**2])


def rk4(func, z0, time):
    """The Runge-Kutta 4 scheme for solution of systems of ODEs.
    z0 is a vector for the initial conditions,
    the right hand side of the system is represented by func which returns
    a vector with the same size as z0 ."""

    z = np.zeros((np.size(time), np.size(z0)))
    z[0, :] = z0
    zp = np.zeros_like(z0)

    for i, t in enumerate(time[0:-1]):
        dt = time[i + 1] - time[i]
        dt2 = dt / 2.0
        k1 = np.asarray(func(z[i, :], t))  # predictor step 1
        k2 = np.asarray(func(z[i, :] + k1 * dt2, t + dt2))  # predictor step 2
        k3 = np.asarray(func(z[i, :] + k2 * dt2, t + dt2))  # predictor step 3
        k4 = np.asarray(func(z[i, :] + k3 * dt, t + dt))  # predictor step 4
        z[i+1,:] = z[i,:] + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)  # Corrector step

    return z


#Oppgave 1 a)

s = [-0.5, -0.6]
z0 = np.zeros(2)
z0[0] = 2
z0[1] = s[0]

z = rk4(f, z0, x)
print(z)
phi0 = z[N, 0]-beta
n_max = 10
eps = 1.0e-3

for n in range(n_max):
    z0[1] = s[1]
    z = rk4(f, z0, x)
    phi1 = z[N, 0] - beta
    ds = error_func(phi0, phi1, s[0], s[1])
    s[0] = s[1]
    s[1] += ds
    phi0 = phi1
    print(phi0)
    print('n = {} s1 = {} and ds = {}'.format(n, s[1], ds))
    plt.plot(x, z[:, 0])
    if abs(ds) <= eps:
        print('Solution converged for eps = {} and s1 ={} and ds = {}. \n'.format(eps, s[1], ds))
        break
plt.interactive(False)


#Oppgave 1 b)

s_start, s_end = -0.5, -1.23
sList = np.linspace(s_start, s_end, 51)
phiList = np.zeros_like(sList)
for n, s in enumerate(sList):
    Y_0 = [2, s]
    Y_shoot = rk4(dydx, Y_0, x)
    y_shoot = Y_shoot[:, 0]
    # extract y, and not y'
    phiList[n] = y_shoot[-1] - beta
plt.figure()
plt.plot(sList, phiList)
plt.plot(sList, np.zeros_like(sList), 'k--')
plt.xlim([s_start, s_end])
plt.ylim([np.min(phiList), np.max(phiList)])
plt.xlabel('s')
plt.ylabel(r'$\phi$')
plt.show()
plt.interactive(False)

s1 = -0.6
plt.figure()
grey = '0.75'
linestyles = [grey, 'r--', 'g--', 'b--', 'y--', 'm--', 'c--']
plt.plot(x, y_analytic, 'k')

#Oppgave 2

N = 3
h = 1./(N + 1)
Beta2 = 4.
x = np.linspace(0, 1, N + 2)

i_list = np.linspace(0, N + 1, N + 2)

MainDiag = -(2*i_list[1:-1] + Beta2*h)
subDiag = i_list[2:-1] - 1
supDiag = i_list[1:-2] + 1
d = np.zeros(N)
d[-1] = -(N + 1)
A = scipy.sparse.diags([subDiag, MainDiag, supDiag], [-1, 0, 1], format='csc')
ThetaSol = scipy.sparse.linalg.spsolve(A, d)
theta1 = ThetaSol[0]
theta0 = theta1/(1 + (Beta2/2)*h + (Beta2**2/12)*h**2)
thetaEnd = 1
Theta = np.append(theta0, ThetaSol)
Theta = np.append(Theta, thetaEnd)
print(Theta)
plt.figure()
plt.plot(x, Theta)
plt.xlabel('x')
plt.ylabel(r'$\theta$')
plt.show()
plt.interactive(False)


#Oppgave 3 a)

N = 8
# number of unknowns
x = np.linspace(0, 0.9, N + 2)
h = 0.9/(N + 1)
y_analytic = 0.5*(np.log(np.abs(x-1))-np.log(np.abs(x+1))) + 2

x_unknown = x[1:-1]

y_0, y_End = y_analytic[0], y_analytic[-1]
# boundaries

Y0 = np.linspace(y_0, y_End, N + 2)
# first value of Y

for n in range(2):
    Y0Plus = Y0[2:]
    Y0Minus = Y0[0:-2]
    d = -0.5*x_unknown*(Y0Plus-Y0Minus)**2
    d[0] = d[0] - y_0
    d[-1] = d[-1] - y_End
    A = scipy.sparse.diags([1, -2, 1], [-1, 0, 1], shape=(N, N), format='csc')
    Y1 = scipy.sparse.linalg.spsolve(A, d)
    Y1Full = np.append(y_0, Y1)
    Y1Full = np.append(Y1Full, y_End)
    Y0 = Y1Full


#Oppgave 3 b)

N = 8
x = np.linspace(0, 0.9, N + 2)
h = 0.9/(N + 1)
y_analytic = 0.5*(np.log(np.abs(x-1))-np.log(np.abs(x+1))) + 2
x_unknown = x[1:-1]
y0, yEnd = y_analytic[0], y_analytic[-1]
Y0 = np.linspace(y0, yEnd, N + 2)

for n in range(2):
    Y0Plus = Y0[2:]
    Y0Minus = Y0[0:-2]
    alpha = Y0Plus - Y0Minus
    MainDiag = -2*np.ones(N)
    supDiag = 1 + x_unknown[:-1]*alpha[:-1]
    subDiag = 1 - x_unknown[1:]*alpha[1:]
    d = 0.5*x_unknown*(alpha)**2
    d[0]= d[0] - y0*(1 - x_unknown[0]*alpha[0])
    d[-1]= d[-1] - yEnd*(1 + x_unknown[-1]*alpha[-1])
    A = scipy.sparse.diags([subDiag, MainDiag, supDiag], [-1, 0, 1], format='csc')
    Y1 = scipy.sparse.linalg.spsolve(A, d)
    Y1Full = np.append(y0, Y1)
    Y1Full = np.append(Y1Full, yEnd)
    Y0 = Y1Full

print(len(x), len(Y0), len(Y1Full))
plt.figure()
plt.plot(x, y_analytic, 'k')
plt.plot(x, Y1Full, 'r--')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['analytic', 'numeric'], frameon=False)
plt.show()


'''''må inn over oppgave 1 for å lage korrekt plott'''''
'''''s0 = -0.5
Y_0 = [2, s0]
Y_shoot = rk4(dydx, Y_0, x)
y_shoot = Y_shoot[:, 0]
phi0 = y_shoot[-1] - beta

s1 = -0.6
plt.figure()
grey = '0.75'
linestyles = [grey, 'r--', 'g--', 'b--', 'y--', 'm--', 'c--']
plt.plot(x, y_analytic, 'k')
plt.plot(x, y_shoot, linestyles[0])
legendList = ['analytic', 'it = 0']'''''