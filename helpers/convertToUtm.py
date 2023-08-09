import pyproj
import json
import shapely
from matplotlib import pyplot as plt
from math import pi, sin, cos, radians, atan2


def get_heading(c, h):
    ip = [c[0] + (10 * cos(h) / 2), c[1] + (10 * sin(h) / 2)]

    return ip


def lat_long_to_utm(latitude, longitude):
    # Definir o sistema de coordenadas de origem (lat-long)
    src_proj = pyproj.Proj(proj="latlong", datum="WGS84")

    # Definir o sistema de coordenadas de destino (UTM)
    dst_proj = pyproj.Proj(proj="utm", zone=30, datum="WGS84")

    # Converter as coordenadas
    easting, northing = pyproj.transform(src_proj, dst_proj, longitude, latitude)

    return easting, northing


f = open("workplaces.json", "r")

bounds = json.loads(f.read())

f.close()

w1 = bounds[0]["boundary"]["coordinates"]
w2 = bounds[1]["boundary"]["coordinates"]

f = open("spots.json", "r")

spots = json.loads(f.read())

f.close()

s1 = list(
    map(lambda x: list(map(lambda y: lat_long_to_utm(*y[:2]), x)), spots[0]["spots"])
)
s2 = list(
    map(lambda x: list(map(lambda y: lat_long_to_utm(*y[:2]), x)), spots[1]["spots"])
)

f = open("realSituations", "w")

s = [
    {"name": "Mine Parking 1", "bound": w1, "obstacles": s1},
    {"name": "Mine Parking 2", "bound": w2, "obstacles": s2},
]

f.write(json.dumps(s))

f.close()

sj1 = list(map(shapely.Polygon, s1))
sj2 = list(map(shapely.Polygon, s2))

w1utm = []
w2utm = []

for p in w1:
    e, n = lat_long_to_utm(p[0], p[1])
    w1utm.append([e, n])

for p in w2:
    e, n = lat_long_to_utm(p[0], p[1])
    w2utm.append([e, n])

f = open("realSituations.json", "w")

s = [
    {"name": "Mine Parking 1", "bound": w1utm, "obstacles": s1},
    {"name": "Mine Parking 2", "bound": w2utm, "obstacles": s2},
]

f.write(json.dumps(s))

f.close()

plt.plot(*shapely.Polygon(w2utm).exterior.xy)

st1 = [4229305, 6233640, -2.144017077201519]
ed1 = [4229413.188211228, 6233461.66019543, -1.2]

st2 = [4229030, 6233760, 0]
ed2 = [4228960.758833968, 6233886.183977293, -2.144017077201519]

plt.plot(*st2[:2], "xg")
plt.plot(*ed2[:2], "xm")

ls = shapely.LineString((ed2[:2], get_heading(ed2[:2], ed2[2])))

plt.plot(*ls.xy, "m")

c = 0
# p = shapely.Polygon([
#         [4228953.154331809, 6233900.83304679],
#         [4228948.207729958, 6233893.170097482],
#         [4228940.363353019, 6233901.534906868],
#         [4228945.309940724, 6233909.197841923]
#       ])
# plt.plot(*p.exterior.xy, 'r')

# d1 = [4228953.154331809, 6233900.83304679]
# d2 = [4228948.207729958, 6233893.170097482]
# d3 = [4228940.363353019, 6233901.534906868]
# d4 = [4228945.309940724, 6233909.197841923]

# def calcReta(p1, p2):

#     dy = p2[1] - p1[1]
#     dx = p2[0] - p1[0]

#     m = dy/dx

#     b = p1[1] - (m * p1[0])

#     return m, b

# m1, b1 = calcReta(d1, d3)
# m2, b2 = calcReta(d2, d4)

# x = (b2 - b1)/(m1 - m2)
# y = m1*x + b1

# print(x, y)

# dely = d2[1] - d1[1]
# delx = d2[0] - d1[0]

# ang = atan2(dely, delx)

# print(ang)

# ls = shapely.LineString((d1, get_heading(d1, ang)))

# plt.plot(*ls.xy, 'm')

counter = 0
for p in sj2:
    color = "b" if counter == 14 else "r"
    plt.plot(*p.exterior.xy, color)
    counter += 1

plt.show()
