import numpy as np
from numpy import linalg as LA
import math


def max_element(a):
    """
    ищет максимальный элемент в матрице a, но не на диагонали
    возврвщает само значение, i и j координвты в матрице
    """
    n = np.shape(a)[0]
    m = a[0][0]
    max_i = 0
    max_j = 0
    for j in range(n):
        for i in range(n):
            if i != j and m < a[i][j]:
                m = a[i][j]
                max_i = i
                max_j = j
    return m, max_i, max_j


def jacobi_max_stratagy(a, epsilin):
    """
    Метод Якоби с выбором максимального элемента

    a - матрица, epsilon - необходимая точность
    """

    n = np.shape(a)[0]
    m = max_element(np.abs(np.triu(a, k=1)))
    count = 0

    while m[0] > epsilin:
        F = np.eye(n)
        if a[m[1]][m[1]] != a[m[2]][m[2]]:
            phi = math.atan(-2 * a[m[1]][m[2]] / (a[m[2]][m[2]] - a[m[1]][m[1]])) / 2
        else:
            phi = math.pi / 4
        c = math.cos(phi)
        s = math.sin(phi)

        F[m[1], m[1]] = c
        F[m[2], m[2]] = c
        F[m[1], m[2]] = -s
        F[m[2], m[1]] = s

        a = np.dot(np.dot(np.transpose(F), a), F)
        m = max_element(np.abs(np.triu(a, k=1)))
        count += 1
    return np.diag(a), count


def jacobi_cyclically_stratagy(a, epsilin):
    """
    Метод Якоби с циклическим выбором элемента

    a - матрица, epsilon - необходимая точность
    """

    n = np.shape(a)[0]
    m = max_element(np.abs(np.triu(a, k=1)))
    count = 0

    while m[0] > epsilin:
        for i in range(n):
            for j in range(i+1, n):
                F = np.eye(n)
                if a[i][i] != a[j][j]:
                    phi = math.atan(-2 * a[i][j] / (a[j][j] - a[i][i])) / 2
                else:
                    phi = math.pi / 4
                c = math.cos(phi)
                s = math.sin(phi)

                F[i, i] = c
                F[j, j] = c
                F[i, j] = -s
                F[j, i] = s

                a = np.dot(np.dot(np.transpose(F), a), F)
                m = max_element(np.abs(np.triu(a, k=1)))
                count += 1
    return np.diag(a), count


def hilbert(n):
    """
    возвращает матницу Гильберта размерности n
    """
    return np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])


if __name__ == "__main__":
    n = 20
    a = hilbert(n)
    m = jacobi_max_stratagy(a, 1e-12)
    print("собственные числа метод Якоби с выбором максимального элемента:\n{0} \nколичество итераций: {1}".format(
        sorted(m[0]), m[1]))
    c = jacobi_cyclically_stratagy(a, 1e-12)
    print("собственные числа метод Якоби с циклическим выбором элемента:\n{0} \nколичество итераций: {1}".format(
        sorted(c[0]), c[1]))
    print("собственные числа посчитанные numpy:\n{0}".format(sorted(LA.eig(a)[0])))
