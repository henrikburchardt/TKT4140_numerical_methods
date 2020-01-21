import numpy as np
import matplotlib.pyplot as plt


def hello_world():
    print('hello world!')


#A


a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
A = np.array([[1, 1, 2], [2, 3, 3], [4, 4, 5]])
B = np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]])


#B


print('a+b=', a+b)
print('a*b=', a*b)
print('A@B=', A@B)
print('A^T=', A.T)
print('A^-1=', np.linalg.inv(A))
print('Ax = b', np.linalg.inv(A)*b)


#C


def fib(n):
    f1 = 0
    f2 = 1
    print(f1, f2, end=' ')
    for i in range(n-2):
        temp = f2
        f1 = f2
        f2 = f2 + temp
        print(f2, end=' ')


#D


def plot_sin():
    plt.figure()
    plt.subplot(211)
    x1 = np.linspace(0, 2*np.pi, 10)
    y1 = np.sin(x1)
    x2 = np.linspace(0, 2*np.pi, 50)
    y2 = np.sin(x2)
    plt.plot(x1, y1, x2, y2)
    p1, p2 = plt.plot(x1, y1, x2, y2)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('sin(x) plot')
    plt.legend((p1, p2,), ('!0 points', '50 points'))
    plt.show()


#E


def fib_array(n):
    f1 = 0
    f2 = 1
    list1 = [0, 1]
    t = np.array(list1)
    for i in range(n - 2):
        temp = f2
        f1 = f2
        f2 = f2 + temp
        t = np.append(t,f2)
    plt.figure()
    n = np.linspace(0, n, n)
    plt.plot(n, t, '--')
    plt.yscale('log')
    plt.title('Fibonacci plot')
    plt.grid(True)
    plt.show()





if __name__ == '__main__':
    hello_world()
    fib(30)
    plot_sin()
    fib_array(30)











