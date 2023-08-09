import math
import matplotlib.pyplot as plt
import numpy as np

p1 = [0, 0, None]

p2 = [1, 2, math.atan2(2, 1)]

p3 = [0, 3, math.atan2(1, -1)]

plt.plot(np.array([p1[0], p2[0]]), np.array([p1[1], p2[1]]))

plt.plot(np.array([p2[0], p3[0]]), np.array([p2[1], p3[1]]))

b = (p2[2] + math.radians(180) + p3[2]) / 2

r = 5

dteta = p3[2] - p2[2]

desloc = r / (math.cos(dteta / 2))

dsegs = r * np.tan(dteta / 2)

tg1 = [p2[0] - dsegs * np.cos(p2[2]), p2[1] - dsegs * np.sin(p2[2])]

tg2 = [p2[0] + dsegs * np.cos(p3[2]), p2[1] + dsegs * np.sin(p3[2])]

plt.plot(*tg1, "x")
plt.plot(*tg2, "x")

print(desloc)

i_b = b - math.radians(180)

p4 = [p2[0] + 5 * np.cos(b), p2[1] + 5 * np.sin(b)]

plt.plot(np.array([p2[0], p4[0]]), np.array([p2[1], p4[1]]))

c = [p2[0] + desloc * np.cos(b), p2[1] + desloc * np.sin(b)]

print(c)

t = np.linspace(0, 2 * math.pi, 100)

x_c = lambda x: c[0] + r * np.cos(t)

y_c = lambda y: c[1] + r * np.sin(t)

xt = x_c(t)

yt = y_c(t)

plt.plot(xt, yt)

plt.plot(c[0], c[1], "x")

xl, xr = plt.xlim()

yl, yr = plt.ylim()

plt.xlim((min((xl, yl))), max((xr, yr)))

plt.ylim(plt.xlim())

plt.show()
