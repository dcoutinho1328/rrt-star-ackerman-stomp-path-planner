from localPathPlanner import Vehicle, Node, Planner
import numpy as np
import matplotlib.pyplot as plt
import shapely
import json
from random import randint
import math
import time

f = open("situations.json", "r")

situations = json.loads(f.read())

def normalizeSituation(s):

    if s["rand_area"] == None and s["bound"]:
        [x_offset, _, y_offset, _] = get_bounds(s["bound"])
    else:
        [x_offset, _, y_offset, _] = s["rand_area"]
        s["rand_area"][0] -= x_offset
        s["rand_area"][1] -= x_offset
        s["rand_area"][2] -= y_offset
        s["rand_area"][3] -= y_offset

    if s["bound"]:
        for v in s["bound"]:
            v[0] -= x_offset
            v[1] -= y_offset

    for o in s["obstacles"]:
        for v in o:
            v[0] -= x_offset
            v[1] -= y_offset    

    s["start"][0] -= x_offset
    s["start"][1] -= y_offset

    s["end"][0] -= x_offset
    s["end"][1] -= y_offset

    return x_offset, y_offset


def get_bounds(bounds):
    xs = list(map(lambda x: x[0], bounds))
    ys = list(map(lambda x: x[1], bounds))

    return [min(xs), max(xs), min(ys), max(ys)]


def setSituation(index):
    global name, rand_area, obstacles, start, end, obstacles, bounds, offset

    s = situations[index]

    offset = list(normalizeSituation(s))

    name = s["name"]
    bounds = s["bound"]
    if s["rand_area"] == None and bounds:
        rand_area = get_bounds(bounds)
    else:
        rand_area = s["rand_area"]
    obstacles = s["obstacles"]
    start = Node(*s["start"])
    end = Node(*s["end"])
    bounds = s["bound"]


if __name__ == "__main__":

    sit_id = int(input("Set situation: \n"))

    setSituation(sit_id)

    # vehicle = Vehicle(2, 3, 2.5, 30)

    # vehicle = Vehicle(4, 8, 6, 30)

    vehicle = Vehicle(7,14, 10, 30)

    obstacles = list(map(lambda x: shapely.Polygon(x), obstacles))

    # plt.plot(start.x, start.y, 'x')
    # plt.plot(end.x, end.y, 'x')

    # for o in obstacles:
    #     plt.plot(*o.exterior.xy)

    # plt.show()

    if bounds:
        bounds = shapely.Polygon(bounds)

    ts = time.time()

    t1 = []
    t2 = []
    t3 = []
    t4 = []

    lpp = Planner(
        start=start,
        goal=end,
        obstacle_list=obstacles,
        vehicle=vehicle,
        rand_area=rand_area,
        bounds=bounds,
        pol_divide=20,
        max_iter=10000,
        goal_sample_rate=50,
        expand_dis=30,
        search_radius_param=200,
    )

    if sit_id in [3,4]:
        lpp.tight_end()

    lpp.run_planner()
    t1.append(lpp.t1)
    t2.append(lpp.t2)
    t3.append(lpp.t3)
    t4.append(lpp.t4)

    retries = 0

    valid = lpp.validatePath()

    while not valid:
        for _ in range(5):
            lpp.pol_divide = randint(2*vehicle.length, 5*vehicle.length)
            lpp.getPolynomials(max_length=lpp.pol_divide, divide=True)
            valid = lpp.validatePath()                
        retries += 1
        lpp.run_planner()
        t1.append(lpp.t1)
        t2.append(lpp.t2)
        t3.append(lpp.t3)
        t4.append(lpp.t4)
        valid = lpp.validatePath()

    if sit_id in [3, 4]:
        lpp.add_point_to_path(end)

    te = time.time()

    plt.subplots()

    for o, o1 in zip(lpp.obstacle_list, lpp.grown_obstacles):
        plt.plot(*o.exterior.xy)
        # plt.plot(*o1.exterior.xy)

    if bounds:
        plt.plot(*bounds.exterior.xy)

    plt.plot(lpp.start.x, lpp.start.y, "xr")
    plt.plot(lpp.goal.x, lpp.goal.y, "xb")

    path = np.array(lpp.final_course)
    stomp = np.array(lpp.smooth_path)

    plt.plot(path[:, 0], path[:, 1], "-r", label="RRT*")
    plt.plot(stomp[:, 0], stomp[:, 1], "-b", label="STOMP")

    plt.legend()

    plt.subplots()

    min_path = np.array(list(map(lambda x: x[:3], lpp.min_course)))
    final_x, final_y, _ = lpp.discretize()
    final_x, final_y = np.array(final_x), np.array(final_y)

    for o, o1 in zip(lpp.obstacle_list, lpp.grown_obstacles):
        plt.plot(*o.exterior.xy)
        # plt.plot(*o1.exterior.xy)

    plt.plot(lpp.start.x, lpp.start.y, "xr")
    plt.plot(lpp.goal.x, lpp.goal.y, "xb")

    if bounds:
        plt.plot(*bounds.exterior.xy)

    plt.plot(min_path[:, 0], min_path[:, 1], "-r", label="Post Processing")
    plt.plot(final_x, final_y, "-b", label="Polynomials")

    plt.legend()

    plt.show()

    f = open(f"results_{sit_id}.txt", 'w')

    f.write(f"Total execution time: {te - ts} \n")

    f.write(f"Average RRT Execution Time: {np.average(t1)} \n")
    f.write(f"Average Total time after stomp: {np.average(t2)} \n")
    f.write(f"Average Total time after post processing {np.average(t3)} \n")
    f.write(f"Average Total time after polynom generator {np.average(t4)} \n")

    f.write(f"Nodes in the three {len(lpp.node_list)} \n")

    f.write(f"Number of retries {retries}")

    f.close()

