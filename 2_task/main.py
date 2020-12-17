import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt


def lu(a):
    """
    разложение LU, метод Дулиттла

    a - матрица
    """

    n = a.shape[0]
    L = np.zeros([a.shape[0], a.shape[1]])
    U = np.zeros([a.shape[0], a.shape[1]])

    for i in range(n):
        for k in range(i, n):
            s = 0
            for j in range(i):
                s += (L[i][j] * U[j][k])
            U[i][k] = a[i][k] - s
        for k in range(i, n):
            if i == k:
                L[i][i] = 1
            else:
                s = 0
                for j in range(i):
                    s += (L[k][j] * U[j][i])
                L[k][i] = (a[k][i] - s) / U[i][i]
    return L, U


def solve_LU(a, b):
    """
    решение для разложения LU

    a - матрица, b - вектор
    """
    LU = lu(a)
    L, U = LU[0], LU[1]
    y = np.array([0.]*L.shape[0])
    for i in range(y.shape[0]):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    x = np.array([0.]*U.shape[0])
    for i in range(1, x.shape[0] + 1):
        x[-i] = (y[-i] - np.dot(U[-i, -i:], x[-i:])) / U[-i, -i]

    return x


def qr(a):
    """
    разложение QR методом вращений

    a - матрица
    """
    n = a.shape[0]
    Q = np.eye(n)
    R = a

    for j in range(0, n):
        for i in range(j+1, n):
            T = np.eye(n)
            teta = math.atan(-R[i][j]/R[j][j])
            T[i, i] = math.cos(teta)
            T[j, j] = T[i, i]
            T[i, j] = math.sin(teta)
            T[j, i] = -T[i, j]
            R = np.dot(T, R)
            Q = np.dot(Q, np.transpose(T))

    return Q, R


def solve_QR(a, b):
    """
    решение для разложения QR

    a - матрица, b - вектор
    """
    QR = qr(a)
    b = np.dot(np.transpose(QR[0]), b)
    x = np.array([0.]*QR[1].shape[0])
    for i in range(1, x.shape[0] + 1):
        x[-i] = (b[-i] - np.dot(QR[1][-i, -i:], x[-i:])) / QR[1][-i, -i]
    return x


def hilbert(n):
    """
    возвращает матницу Гильберта размерности n
    """
    return np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])


if __name__ == "__main__":

    tests = [15, 20, 25]
    methods = [solve_LU, solve_QR]

    for n in tests:
        e = np.array([1]*n)
        H = hilbert(n)
        b = np.dot(H, e)
        print("\n расчёт для матрицы Гильберта размерности {0} \n".format(n))

        for method in methods:
            print("погрешность {0}:     {1}".format(str(method.__name__), str(LA.norm(method(H, b) - e))))

            alfa = 1e-12
            D = []
            m = [1, 0]
            x = []
            for i in range(2, 500):
                d = LA.norm(method(H + np.dot(alfa + i*alfa, np.eye(n)), b) - e)
                D.append(d)
                x.append(alfa + i*alfa)
                if d < m[0]:
                    m = [d, alfa + i*alfa]
            print("наименьшая погрешность d={0} достигается при alfa={1}".format(m[0], m[1]))
            fig, ax = plt.subplots()
            plt.xlabel('alfa', fontsize=10)
            plt.ylabel("d", fontsize=10)
            ax.plot(x, D)
            plt.show()
