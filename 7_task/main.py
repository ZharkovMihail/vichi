import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad


def ritz(n, f, k, k_d, q, u_0, N):

    d = np.zeros(n)
    C = np.zeros((n, n))
    p = []
    p.append(lambda x: 1.0)
    p.append(lambda x: x)
    for i in range(1, n - 1):
        p.append(rec(p[i], p[i - 1], i))

    p_d = [lambda x: -2 * x]
    for i in range(1, n):
        p_d.append(dif(p[i], p[i - 1], i))

    for i in range(n):
        d[i] = quad(lambda x, i: f(x) * p[i](x) * (1 - x * x), -1, 1, args=(i))[0]
    for i in range(n):
        for j in range(n):
            C[i][j] = \
            quad(lambda x, i, j: k(x) * p_d[i](x) * p_d[j](x) + q(x) * p[i](x) * p[j](x) * (1 - x * x) ** 2, -1, 1,
                 args=(i, j))[0]
    b = np.linalg.solve(C, d)

    u = np.zeros(N)

    x_ = np.linspace(-1, 1, N)
    for i in range(N):
        for j in range(n):
            u[i] += b[j] * p[j](x_[i]) * (1 - x_[i] * x_[i])

    y = [u_0(x) for x in x_]
    err = np.abs(y - u)
    plt.plot(x_, err, marker='.', lw=0)

    plt.xlabel('x')
    plt.ylabel('error')
    plt.legend()
    plt.show()


def rec(p_1, p_2, n):
    return lambda x: (2 * n + 1) / (n + 1) * x * p_1(x) - n / (n + 1) * p_2(x)


def dif(p_1, p_2, i):
    return lambda x: i * (p_2(x) - x * p_1(x)) - 2 * x * p_1(x)


def dif_2(p_1, p_2, p_3, i):
    return lambda x: -(i + 2) * p_1(x) + i / (1 - x * x) * (
                (i - 1) * (p_3(x) - x * p_2(x)) - (i + 2) * x * (p_2(x) - x * p_1(x)))


def f(x): return math.log10(2 + x) * (x-1) - (3*x**2 + 7*x + 2*(x+2)**2*math.log10(x+2) - 4)/(x+2)**2


def k(x): return x


def k_d(x): return 0


def q(x): return 1


def u(x): return math.log10(2 + x) * (x-1)


n = [3, 5, 7, 10]
N = 1000

for i in n:
    ritz(i, f, k, k_d, q, u, 1000)
