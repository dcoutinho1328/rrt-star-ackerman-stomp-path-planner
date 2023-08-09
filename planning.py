from localPathPlanner import Vehicle, Node, Planner
import numpy as np
import matplotlib.pyplot as plt
import shapely
import json
from random import randint
import time

f = open("situations.json", "r")

situations = json.loads(f.read())


# Normalize coordinates values
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


# Creates rand_area values
def get_bounds(bounds):
    xs = list(map(lambda x: x[0], bounds))
    ys = list(map(lambda x: x[1], bounds))

    return [min(xs), max(xs), min(ys), max(ys)]


# Selects the situation
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


vehicles = [Vehicle(2, 3, 2.5, 30), Vehicle(4, 8, 6, 30), Vehicle(7, 14, 10, 30)]


if __name__ == "__main__":
    # Selects the situation
    sit_id = int(input("Set situation: \n"))
    setSituation(sit_id)

    # Selects the vehicle size
    vehicle_size = int(
        input(
            """Set vehicle size: 
                             1 - Small (2x3)
                             2 - Medium (4x8)
                             3 - Large (7x14)
                             """
        )
    )

    vehicle = vehicles[vehicle_size - 1]

    obstacles = list(map(lambda x: shapely.Polygon(x), obstacles))

    if bounds:
        bounds = shapely.Polygon(bounds)

    ts = time.time()

    # RRT* times
    t1 = []
    # Smooth STOMP times
    t2 = []
    # Post-Processing times
    t3 = []
    # Polynoms times
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
        goal_sample_rate=95,
        expand_dis=10,
        search_radius_param=200,
    )

    # Goal moving
    if sit_id in [3, 4]:
        lpp.tight_end()

    # Run the whole alg
    lpp.run_planner()

    # RRT* times
    t1.append(lpp.t1)
    # Smooth STOMP times
    t2.append(lpp.t2)
    # Post-Processing times
    t3.append(lpp.t3)
    # Polynoms times
    t4.append(lpp.t4)

    retries = 0

    valid = lpp.validatePath()

    while not valid:
        # Tries to divide the segments to generate better polynoms
        for _ in range(5):
            lpp.pol_divide = randint(2 * vehicle.length, 5 * vehicle.length)
            lpp.getPolynomials(max_length=lpp.pol_divide, divide=True)
            valid = lpp.validatePath()
        lpp.run_planner()
        retries += 1
        t1.append(lpp.t1)
        t2.append(lpp.t2)
        t3.append(lpp.t3)
        t4.append(lpp.t4)
        valid = lpp.validatePath()

    # Return to goal
    if sit_id in [3, 4]:
        lpp.add_point_to_path(end)

    te = time.time()

    # Plots the RRT* and the Smooth STOMP
    plt.subplots()

    for o in lpp.obstacle_list:
        plt.plot(*o.exterior.xy)

    if bounds:
        plt.plot(*bounds.exterior.xy)

    plt.plot(lpp.start.x, lpp.start.y, "xr")
    plt.plot(lpp.goal.x, lpp.goal.y, "xb")

    path = np.array(lpp.final_course)
    stomp = np.array(lpp.smooth_path)

    plt.plot(path[:, 0], path[:, 1], "-r", label="RRT*")
    plt.plot(stomp[:, 0], stomp[:, 1], "-b", label="STOMP")

    plt.legend()

    # Plots the Post Processing and the Polynomials
    plt.subplots()

    min_path = np.array(list(map(lambda x: x[:3], lpp.min_course)))
    final_x, final_y, _ = lpp.discretize()
    final_x, final_y = np.array(final_x), np.array(final_y)

    for o in lpp.obstacle_list:
        plt.plot(*o.exterior.xy)

    plt.plot(lpp.start.x, lpp.start.y, "xr")
    plt.plot(lpp.goal.x, lpp.goal.y, "xb")

    if bounds:
        plt.plot(*bounds.exterior.xy)

    plt.plot(min_path[:, 0], min_path[:, 1], "-r", label="Post Processing")
    plt.plot(final_x, final_y, "-b", label="Polynomials")

    # Plots the RRT* and the Post Processing
    plt.subplots()

    plt.plot(path[:, 0], path[:, 1], "-r", label="RRT*")
    plt.plot(min_path[:, 0], min_path[:, 1], "-b", label="Post Processing")

    for o in lpp.obstacle_list:
        plt.plot(*o.exterior.xy)

    plt.plot(lpp.start.x, lpp.start.y, "xr")
    plt.plot(lpp.goal.x, lpp.goal.y, "xb")

    if bounds:
        plt.plot(*bounds.exterior.xy)

    plt.legend()

    plt.show()

    results = input("Save results? (y/n)\n").lower()

    # Calculates and saves analytical results
    if results == "y":
        f = open(f"results/results_{sit_id}.txt", "w")

        f.write(f"Total execution time: {te - ts} \n")

        f.write(f"Average RRT Execution Time: {np.average(t1)} \n")
        f.write(f"Average Total time after stomp: {np.average(t2)} \n")
        f.write(f"Average Total time after post processing {np.average(t3)} \n")
        f.write(f"Average Total time after polynom generator {np.average(t4)} \n")

        f.write(f"Nodes in the three {len(lpp.node_list)} \n")

        f.write(f"Number of retries {retries}")

        f.close()
