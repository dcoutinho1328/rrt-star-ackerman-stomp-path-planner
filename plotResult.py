from rrt import RRTStar, Node, Vehicle
from stomp import smooth_stomp
import matplotlib.pyplot as plt
import numpy as np
import shapely

if __name__ == "__main__":
    vehicle = Vehicle(2, 3, 30*np.pi/180)
    rand_area = [0, 80, 0, 80]
    obstacle_list = [
        # ((20, 20), (12, 12), (10, 12)),
        ((40, 40), (60, 40), (60, 60), (40, 60)),
        ((10, 10), (20, 10), (20, 15), (10, 15)),
        # ((30, 15), (40, 15), (40, 20), (30, 20)),
    ]
    obstacle_list = list(map(lambda x: shapely.Polygon(x), obstacle_list))
    start = Node(0, 0, 0.5, None, 0)
    end = Node(80, 80, 1.2, None, 0)
    rrt_star = RRTStar(
        start=start,
        goal=end,
        obstacle_list=obstacle_list,
        vehicle=vehicle,
        rand_area=rand_area,
        max_iter=10000,
        goal_sample_rate=5,
        expand_dis=2,
        search_radius_param=50
    )

    path = rrt_star.run_rrt_star()
    path2 = smooth_stomp(path=path, weight_data=0.1, weight_smooth=0.1, tolerance=0.00001)

    if path is None:
        print("Couldn't find the path")
    else:
        print("Path found")
        path = np.array(path)
        path2 = np.array(path2)
        plt.plot(path[:, 0], path[:, 1], "-r")
        plt.plot(path2[:, 0], path2[:, 1], "-b")

    for o, o1 in zip(rrt_star.obstacle_list, rrt_star.grown_obstacles):
        plt.plot(*o.exterior.xy)
        plt.plot(*o1.exterior.xy)

    rewire_count = 0
    for node in rrt_star.node_list:
        if node.rewired:
            rewire_count += 1

    print("Rewire count:", rewire_count)
    plt.plot(rrt_star.start.x, rrt_star.start.y, "xr")
    plt.plot(rrt_star.goal.x, rrt_star.goal.y, "xb")
    plt.grid(True)
    plt.axis("equal")
    plt.show()
