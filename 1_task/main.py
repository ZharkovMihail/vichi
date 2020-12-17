import warnings
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA


def runge_kutta(A, y0, x0, h, x_end):
    """
    медод Рунге-Кутты

    A: матрица; y0: нач условия; h: шаг; x0: первый узел; x_end: последний узел;
    """

    N = int(np.ceil((x_end - x0) / h))
    y = np.zeros((LA.matrix_rank(A), N + 1))
    y[:, 0] = y0

    for k in range(N):
        k1 = h*np.dot(A, y[:, k])
        # with warnings.catch_warnings(record=True) as w:
        #     c = y[:, k] + k1/2
        #     if len(w) > 0:
        #         print(A, y[:, k], k1/2, k)
        # k2 = h*np.dot(A, c)
        k2 = h * np.dot(A, y[:, k] + k1/2)
        k3 = h*np.dot(A, (y[:, k] + k2/2))
        k4 = h*np.dot(A, (y[:, k] + k3))
        y[:, k + 1] = y[:, k] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return y


def adams(A, y0, x0, h, x_end):
    """
    неявный метод Адамса третьего порядка

    A: матрица; y0: нач условия; h: шаг; x0: первый узел; x_end: последний узел;
    """

    N = int(np.ceil((x_end - x0) / h))
    y = np.zeros((LA.matrix_rank(A), N + 1))
    y[:, 0] = y0
    rk0 = runge_kutta(A, y0, x0, h, x0 + 2 * h)
    y[:, 1], y[:, 2] = rk0[:, 1], rk0[:, 2]

    for i in range(2, N):
        pred = y[:, i] + np.dot(h / 12 * A, (23*y[:, i] - 16 * y[:, i-1] + 5 * y[:, i-2]))
        y[:, i + 1] = y[:, i] + np.dot(h / 12 * A, (5*pred + 8 * y[:, i] - y[:, i-1]))
    return y


def rozenbrok(A, y0, x0, h, x_end):
    """
    метод CROS1

    A: матрица; y0: нач условия; h: шаг; x0: первый узел; x_end: последний узел;
    """

    N = int(np.ceil((x_end - x0) / h))
    y = np.zeros((LA.matrix_rank(A), N + 1))
    y[:, 0] = y0

    for i in range(N):
        w = LA.solve(np.eye(LA.matrix_rank(A)) - np.dot((1 + 1j) / 2 * h, A), np.dot(A, y[:, i]))
        y[:, i + 1] = y[:, i] + np.dot(h, [x.real for x in w])
    return y


def richardson(func, epsilon, task):
    """
    оценка погрешности по правилу Ричардсона

    func: оценивааемая функция; p: теор порядок точности метода; epsilon - точность до которой вычисляем
    A: матрица; y0: нач условия; h: шаг; x0: первый узел; x_end: последний узел;
    """
    A, y0, x0, h, x_end = task["A"], task["y0"], task["x0"], task["h"], task["x_end"]
    coef_p = {"runge_kutta": 4, "adams": 3, "rozenbrok": 2}
    p = coef_p[func.__name__]
    rez = []
    N = []
    v1 = func(A, y0, x0, h, x_end)
    for i in range(1, 10):
        N.append((x_end-x0)/h)
        h /= 2
        v2 = func(A, y0, x0, h, x_end)
        delta0 = (v2[:, 0] - v1[:, 0])/(pow(2, p) - 1)
        m = -1
        for j in range(int(np.ceil((x_end - x0) / h))):
            try:
                delta2 = (v2[:, 2*j] - v1[:, j])/(pow(2, p) - 1)
                delta1 = (delta0 + delta2) / 2
                m = max(max([abs(x) for x in delta0]),
                        max([abs(x) for x in delta1]),
                        max([abs(x) for x in delta2]),
                        m
                        )
                delta0 = delta2
            except IndexError:
                pass
        rez.append(m)
        v1 = v2
        if m < epsilon:
            return [rez, N]
    return [rez, N]


def kalitkina_matrix_1(m0, m1, v1, m2, v2):
    """
    1 матрица для тесирования из статьи Н.Н.Калиткиной
    m0, m1, v1, m2, v2 - коэффиценты
    """
    return np.array([[m0, 0, 0, 0, 0],
                     [m0 - m1, m1 + v1, -v1, 0, 0],
                     [m0 - m1 - v1, 2 * v1, m1 - v1, 0, 0],
                     [m0 - m1 - v1, 2 * v1, m1 - v1 - m2, m2 + v2, -v2],
                     [m0 - m1 - v1, 2 * v1, m1 - v1 - m2 - v2, 2 * v2, m2 - v2]
                     ])


def kalitkina_matrix_2(m1, m2):
    """
    2 матрица для тесирования из статьи Н.Н.Калиткиной
    m1, m2 - коэффиценты
    """
    return np.array([[m1, 0, 0, 0, 0, 0],
                     [1, m1, 0, 0, 0, 0],
                     [0, 0, m2, 0, 0, 0],
                     [0, 0, 1, m2, 0, 0],
                     [0, 0, 0, 2, m2, 0],
                     [0, 0, 0, 0, 3, m2]
                     ])


if __name__ == "__main__":

    methods = [runge_kutta, adams, rozenbrok]

    colors = {"runge_kutta": "red", "adams": "blue", "rozenbrok": "green"}

    test1 = {"A": np.array([[-125, 123.1],
                            [123.1, -123]]),
             "y0": np.array([1, 1]),
             "x0": 0,
             "h": 1e-2,
             "x_end": 0.5}

    test2 = {"A": kalitkina_matrix_1(m0=-100, m1=-1, v1=1, m2=-10000, v2=10),
             "y0": np.array([10, 11, 11, 111, 111]),
             "x0": 0,
             "h": 1e-3,
             "x_end": 0.5}

    test3 = {"A": kalitkina_matrix_1(m0=-10000, m1=1, v1=1, m2=-100, v2=1000),
             "y0": np.array([100, 101, 101, 201, 201]),
             "x0": 0,
             "h": 1e-3,
             "x_end": 0.5}

    test4 = {"A": kalitkina_matrix_2(m1=-1, m2=-10000),
             "y0": np.array([1, 1, 1000, 1000, 1000, 1000]),
             "x0": 0,
             "h": 1e-3,
             "x_end": 1}

    tests = [test1, test2, test3, test4]
    # tests = [test1]

    for test in tests:
        print(test["A"], test["y0"])
        fig, ax = plt.subplots()
        plt.xlabel('N', fontsize=15)
        plt.ylabel("степень в которую нужно возвести 10 чтобы получить |\u0394|", fontsize=9)
        for method in methods:
            print("\n" + "-" * 10 + method.__name__ + "-" * 10)
            rez = richardson(method, 1e-10, test)
            print(rez[1], rez[0])
            y = []
            for i in rez[0]:
                e = 1
                while i * 10**e < 1:
                    e += 1
                y.append(-e)
            ax.plot(rez[1], y, 'ro', color=colors[method.__name__], label=method.__name__)
        plt.legend()
        plt.show()


