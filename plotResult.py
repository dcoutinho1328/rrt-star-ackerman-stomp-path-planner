from rrt import RRTStar, Node
import matplotlib.pyplot as plt
import numpy as np
import shapely

if __name__ == "__main__":
    rand_area = [0, 50, 0, 50]
    obstacle_list = [
        ((1, 1), (2, 2), (1, 2)),
        ((4, 4), (6, 4), (6, 6), (4, 6)),
        ((10, 10), (20, 10), (20, 15), (10, 15)),
        ((30, 15), (40, 15), (40, 20), (30, 20)),
    ]
    obstacle_list = list(map(lambda x: shapely.Polygon(x), obstacle_list))
    start = Node(0, 0, 0.5, None, 0)
    end = Node(50, 50, 1.2, None, 0)
    rrt_star = RRTStar(
        start=start,
        goal=end,
        obstacle_list=obstacle_list,
        rand_area=rand_area,
        max_iter=5000,
    )
    path = rrt_star.run_rrt_star()
    if path is None:
        print("Não foi possível encontrar o caminho")
    else:
        print("Caminho encontrado")
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], "-r")

    for o in rrt_star.obstacle_list:
        plt.plot(*o.exterior.xy)

    rewire_count = 0
    for node in rrt_star.node_list:
        if node.rewired:
            rewire_count += 1

    print("Número de rewires:", rewire_count)
    plt.plot(rrt_star.start.x, rrt_star.start.y, "xr")
    plt.plot(rrt_star.goal.x, rrt_star.goal.y, "xb")
    plt.grid(True)
    plt.axis("equal")
    plt.show()
