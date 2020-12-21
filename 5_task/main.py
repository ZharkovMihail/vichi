import numpy as np
from numpy import linalg as LA
import math


def power_method(a, epsilon):
    """
    Степенной метод

    a - матрица, epsilon - необходимая точность
    """
    x = np.random.random(np.shape(a)[0])
    x[0] = (np.max(x) + 1) / 2
    X = np.dot(a, x)
    l = LA.norm(X[0])/LA.norm(x[0])
    count = 1
    delta = 1
    while delta > epsilon:
        delta = l
        x = X / LA.norm(np.dot(a, x))
        X = np.dot(a, x)
        l = LA.norm(X[0]) / LA.norm(x[0])
        delta = abs(delta - l)
        count += 1
    return l, count


def scalar_product_method(a, epsilon):
    """
    Метод скалярных произведений

    a - матрица, epsilon - необходимая точность
    """
    x = np.random.random(np.shape(a)[0])
    x[0] = (np.max(x) + 1) / 2
    X = np.dot(a, x) / LA.norm(np.dot(a, x))
    y = x
    Y = np.dot(np.transpose(a), x) / LA.norm(np.dot(np.transpose(a), y))
    l = np.dot(np.dot(a, X), np.dot(np.transpose(a), Y)) / np.dot(X, np.dot(np.transpose(a), Y))
    count = 1
    delta = 1
    while delta > epsilon:
        delta = l
        x = X
        X = np.dot(a, x) / LA.norm(np.dot(a, x))
        y = Y
        Y = np.dot(np.transpose(a), x) / LA.norm(np.dot(np.transpose(a), y))
        l = np.dot(np.dot(a, X), np.dot(np.transpose(a), Y)) / np.dot(X, np.dot(np.transpose(a), Y))
        delta = abs(delta - l)
        count += 1
    return l, count


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
    epsilon = 1e-12
    a = hilbert(n)
    p = power_method(a, epsilon)
    s = scalar_product_method(a, epsilon)
    print("максимальное собственное число посчитанное степенным методом: {0}\n число итераций: {1}".format(p[0], p[1]))
    print("максимальное собственное число посчитанное методом скалярных произведений: {0} \n число итераций: {1}".
          format(s[0], s[1]))
    j = jacobi_cyclically_stratagy(a, epsilon)
    print("максимальное собственное число посчитанное методом Якоби: {0}\n число итераций: {1}".format(max(j[0]), j[1]))
    print("максимальное собственное число посчитанное numpy: {0}".format(max(LA.eig(a)[0])))