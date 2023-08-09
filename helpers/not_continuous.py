import numpy as np
import math
import matplotlib.pyplot as plt


def discretize(c, sample_rate=1000):
    x = []
    y = []
    z = []

    s, e = c["s"], c["e"]
    if c["isCircle"]:
        x0, y0 = c["c"]
        r = 1 / max_k
        i1 = math.atan2(s[1] - y0, s[0] - x0)
        i2 = math.atan2(e[1] - y0, e[0] - x0)
        tl = np.linspace(i1, i2, sample_rate).tolist()
        xt = list(map(lambda t: x0 + r * math.cos(t), tl))
        yt = list(map(lambda t: y0 + r * math.sin(t), tl))
        x = x + xt
        y = y + yt
        z = z + tl

    else:
        a1, a2, a3, a4 = c["c"]
        tl = np.linspace(s[0], e[0], sample_rate).tolist()
        yt = list(map(lambda t: a1 + a2 * t + a3 * t**2 + a4 * t**3, tl))
        zt = list(map(lambda t: a2 + 2 * a3 * t + 3 * a4 * t**2, tl))
        x = x + tl
        y = y + yt
        z = z + zt

    return x, y, z


def getPols():
    for i in range(len(optimized) - 1):
        s, e = optimized[i], optimized[i + 1]
        if len(s) == 4:
            c.append({"c": s[3], "s": s, "e": e, "isCircle": True})

        else:
            A = np.array(
                [
                    [1, s[0], s[0] ** 2, s[0] ** 3],
                    [1, e[0], e[0] ** 2, e[0] ** 3],
                    [0, 1, 2 * s[0], 3 * s[0] ** 2],
                    [0, 1, 2 * e[0], 3 * e[0] ** 2],
                ]
            )

            B = np.array([s[1], e[1], np.tan(s[2]), np.tan(e[2])]).T

            X = np.linalg.solve(A, B).T

            coef = X.tolist()
            c.append({"c": coef, "s": s, "e": e, "isCircle": False})


def pruneVertex():
    global optimized
    for i in range(len(optimized) - 2):
        s, m, e = optimized[i], optimized[i + 1], optimized[i + 2]
        if np.sign(m[0] - s[0]) != np.sign(e[0] - m[0]):
            biss = (m[2] + math.radians(180) + e[2]) / 2
            dteta = e[2] - m[2]
            desloc = (1 / max_k) / (math.cos(dteta / 2))
            dsegs = (1 / max_k) * math.tan(dteta / 2)
            c = [m[0] + desloc * np.cos(biss), m[1] + desloc * np.sin(biss)]
            new_seg_start = [
                m[0] - dsegs * math.cos(m[2]),
                m[1] - dsegs * math.sin(m[2]),
                m[2],
                c,
            ]
            new_seg_end = [
                m[0] + dsegs * math.cos(e[2]),
                m[1] + dsegs * math.sin(e[2]),
                e[2],
            ]
            optimized = (
                optimized[: i + 1] + [new_seg_start, new_seg_end] + optimized[i + 2 :]
            )


max_k = 5

optimized = [
    [0, 0, np.radians(30)],
    [1, 1, math.atan2(1, 1)],
    [0, 2, math.atan2(1, -1)],
]

c = []

pruneVertex()

getPols()

x, y = [], []

for co in c:
    xt, yt, _ = discretize(co)
    x += xt
    y += yt

plt.plot([optimized[1][0], 1], [optimized[1][1], 1], "-r")
plt.plot([optimized[2][0], 1], [optimized[2][1], 1], "-r")
# for i in range(len(optimized)):
#     plt.plot(optnmp[i, 0], optnmp[i, 1], 'xr')

# plt.plot(optnmp[:, :2], '-r')
# plt.plot(optimized[1][3][0], optimized[1][3][1], 'xr')
plt.plot(x, y, "-g")
plt.xlim(0, 2)

plt.show()
