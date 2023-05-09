from localPathPlanner import Vehicle, Node, Planner
import numpy as np
import matplotlib.pyplot as plt
import shapely

situations = [
    {
        "name": "First Test Case",
        "rand_area": [0, 80, 0, 80],
        "obstacles": [
            ((20, 20), (12, 12), (10, 12)),
            ((40, 40), (60, 40), (60, 60), (40, 60)),
            ((10, 10), (20, 10), (20, 15), (10, 15)),
            ((30, 15), (40, 15), (40, 20), (30, 20)),
        ],
        "start": Node(0, 0, 0.5, None, 0),
        "end": Node(80, 80, 1.2, None, 0),
        "bound": None,
    },
    {
        "name": "Simple boundary",
        "rand_area": [0, 80, 0, 80],
        "obstacles": [],
        "start": Node(12, 12, 0.5, None, 0),
        "end": Node(62, 62, 1.2, None, 0),
        "bound": ((10, 10), (10, 70), (70, 70), (70, 10)),
    },
]


def setSituation(index):
    global name, rand_area, obstacles, start, end, obstacles, bounds

    s = situations[index]

    name = s["name"]
    rand_area = s["rand_area"]
    obstacles = s["obstacles"]
    start = s["start"]
    end = s["end"]
    bounds = s["bound"]


if __name__ == "__main__":
    setSituation(0)

    vehicle = Vehicle(2, 3, 30 * np.pi / 180)

    obstacles = list(map(lambda x: shapely.Polygon(x), obstacles))

    if bounds:
        bounds = shapely.Polygon(bounds)

    lpp = Planner(
        start=start,
        goal=end,
        obstacle_list=obstacles,
        vehicle=vehicle,
        rand_area=rand_area,
        max_iter=10000,
        goal_sample_rate=5,
        expand_dis=2,
        search_radius_param=50,
    )

    lpp.run_planner()

    if not lpp.validatePath():
        lpp.run_planner()

    plt.subplots()

    for o, o1 in zip(lpp.obstacle_list, lpp.grown_obstacles):
        plt.plot(*o.exterior.xy)
        plt.plot(*o1.exterior.xy)

    if bounds:
        plt.plot(*bounds.exterior.xy)

    plt.plot(lpp.start.x, lpp.start.y, "xr")
    plt.plot(lpp.goal.x, lpp.goal.y, "xb")

    path = np.array(lpp.final_course)
    stomp = np.array(lpp.smooth_path)

    plt.plot(path[:, 0], path[:, 1], "-r")
    plt.plot(stomp[:, 0], stomp[:, 1], "-b")

    plt.subplots()

    min_path = np.array(lpp.min_course)
    final_x, final_y = lpp.discretize()
    final_x, final_y = np.array(final_x), np.array(final_y)

    for o, o1 in zip(lpp.obstacle_list, lpp.grown_obstacles):
        plt.plot(*o.exterior.xy)
        plt.plot(*o1.exterior.xy)

    plt.plot(lpp.start.x, lpp.start.y, "xr")
    plt.plot(lpp.goal.x, lpp.goal.y, "xb")

    if bounds:
        plt.plot(*bounds.exterior.xy)

    plt.plot(min_path[:, 0], min_path[:, 1], "-r")
    plt.plot(final_x, final_y, "-b")

    plt.show()
