import numpy as np
import random
from shapely import LineString, Point, Polygon
import math


class Vehicle:
    def __init__(self, width, length, max_steering_angle, collision_padding=1):
        self.width = width
        self.length = length
        self.max_steering_angle = max_steering_angle
        self.collision_padding = collision_padding


class Node:
    def __init__(self, x, y, yaw, parent_idx=None, cost=0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.parent_idx = parent_idx
        self.cost = cost
        self.rewired = False


class Planner:
    def __init__(
        self,
        start,
        goal,
        obstacle_list,
        vehicle,
        rand_area,
        expand_dis=0.3,
        goal_sample_rate=5,
        max_iter=500,
        search_radius_param=5,
        bounds=None,
        retries=3,
    ):
        self.start = start
        self.goal = goal
        self.obstacle_list = obstacle_list
        self.vehicle = vehicle
        self.growObstacles()
        self.rand_area = rand_area
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = []
        self.node_list.append(self.start)
        self.search_radius_param = search_radius_param
        self.final_course = None
        self.bounds = bounds
        self.retries = retries

    def growObstacles(self):
        self.grown_obstacles = []

        for obs in self.obstacle_list:
            gp = Polygon(
                obs.buffer(
                    math.sqrt(self.vehicle.width**2 + self.vehicle.length**2)
                    + self.vehicle.collision_padding
                )
            )
            self.grown_obstacles.append(gp)

    def calc_dist_to_goal(self, x, y):
        return np.linalg.norm([x - self.goal.x, y - self.goal.y])

    def line_distance(self, n1, n2):
        return np.linalg.norm([n1.x - n2.x, n1.y - n2.y])

    def get_random_point(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = [
                random.uniform(self.rand_area[0], self.rand_area[1]),
                random.uniform(self.rand_area[2], self.rand_area[3]),
            ]
        else:  # goal point sampling
            rnd = [self.goal.x, self.goal.y]

        return rnd

    def steer(self, from_node, to_theta, distance):
        x = from_node.x + distance * np.cos(to_theta)
        y = from_node.y + distance * np.sin(to_theta)
        yaw = to_theta
        new_node = Node(x, y, yaw, None, 0)

        return new_node

    def find_near_nodes(self, new_node):
        nnode = len(self.node_list)
        if nnode == 1:
            return [0]
        # The formula for 'r' is proposed by the original RRT* algorithm
        # The radius decrease as the number of nodes grow
        r = self.search_radius_param * np.sqrt((np.log(nnode) / nnode))
        dlist = [
            (node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2
            for node in self.node_list
        ]
        near_indices = [dlist.index(i) for i in dlist if i <= r**2]
        return near_indices

    def get_nearest_list_index(self, node_list, rnd):
        dlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    def choose_parent(self, new_node, near_indices):
        if not near_indices:
            return None

        dlist = [
            self.calc_dist_to_goal(self.node_list[i].x, self.node_list[i].y)
            for i in near_indices
        ]
        minind = near_indices[dlist.index(min(dlist))]
        new_node.parent_idx = minind
        p_node = self.node_list[minind]
        added_cost = self.line_distance(new_node, p_node)
        new_node.cost = p_node.cost + added_cost
        return new_node

    def check_collision(self, n1, n2=None):
        if n2 is not None:
            line = LineString([[n1.x, n1.y], [n2.x, n2.y]])

            for o in self.grown_obstacles:
                if o.intersects(line):
                    return False  # collision
                if self.bounds and not self.bounds.contains(line):
                    return False  # Out of boundaries

        else:
            point = Point((n1.x, n1.y))

            for o in self.grown_obstacles:
                if o.intersects(point):
                    return False  # collision
                if self.bounds and not self.bounds.contains(point):
                    return False  # Out of boundaries

        return True

    def check_collision_coords(self, p1, p2):
        line = LineString([p1, p2])

        for o in self.grown_obstacles:
            if o.intersects(line):
                return False  # collision

        if self.bounds and not self.bounds.contains(line):
            return False  # Out of boundaries

        return True

    def rewire(self, new_node, near_nodes):
        for i in near_nodes:
            near_node = self.node_list[i]
            line_dist = self.line_distance(near_node, new_node)

            if near_node.cost > new_node.cost + line_dist:
                # if self.check_collision(near_node, new_node) and self.checkSteering(new_node, near_node):
                if self.check_collision(near_node, new_node):
                    prev_cost = near_node.cost
                    near_node.parent_idx = len(self.node_list) - 1
                    near_node.cost = new_node.cost + line_dist
                    new_node.yaw = np.arctan2(
                        near_node.y - new_node.y, near_node.x - new_node.x
                    )
                    near_node.rewired = True

                    # Propagate cost changes to descendants
                    self.propagate_cost_to_leaves(near_node, prev_cost)

                # Check if new trajectory is not more efficient than previous one
                goal_dist = self.calc_dist_to_goal(near_node.x, near_node.y)
                if near_node.cost + goal_dist > new_node.cost + goal_dist:
                    return

    def propagate_cost_to_leaves(self, from_node, previous_cost):
        start = self.node_list.index(from_node)
        for n in self.node_list[start:]:
            if self.node_list[n.parent_idx] == from_node:
                prev_cost = n.cost
                n.cost = n.cost - previous_cost + from_node.cost
                self.propagate_cost_to_leaves(n, prev_cost)

    def generate_final_course(self, goal_ind):
        path = [[self.goal.x, self.goal.y, self.goal.yaw]]
        node = self.node_list[goal_ind]
        count = 0
        while node.parent_idx is not None:
            count += 1
            path.append([node.x, node.y, node.yaw])
            node = self.node_list[node.parent_idx]
        path.append([node.x, node.y, node.yaw])
        self.final_course = path[::-1]

    def run_planner(self):
        r = 0
        self.node_list = []
        self.node_list.append(self.start)
        while True:
            rnd = self.get_random_point()
            nearest_ind = self.get_nearest_list_index(self.node_list, rnd)
            nearest_node = self.node_list[nearest_ind]
            theta = np.arctan2(rnd[1] - nearest_node.y, rnd[0] - nearest_node.x)

            new_node = self.steer(nearest_node, theta, self.expand_dis)
            if self.check_collision(nearest_node, new_node):
                near_indices = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_indices)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_indices)

            if (
                self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y)
                <= self.expand_dis
            ):
                final_node = self.node_list[-1]
                goal_ind = self.get_nearest_list_index(
                    self.node_list, [self.goal.x, self.goal.y]
                )
                if self.check_collision(final_node, self.node_list[goal_ind]):
                    self.generate_final_course(goal_ind)
                    if self.final_course:
                        self.smooth_stomp()
                        self.post_processing()
                        self.getPolynomials(divide=True)
                        break
                    elif self.retries > r:
                        r += 1
                        self.node_list = []
                        self.node_list.append(self.start)
                        continue

            if len(self.node_list) > self.max_iter:
                print("Reached Maximum Iterations")
                # return None

    def post_processing(self):
        optimized = [self.final_course[0]]
        last_index = 0
        while last_index < len(self.final_course) - 1:
            last_n = None
            for n in self.final_course[last_index + 1 :]:
                if self.check_collision_coords(n[:2], optimized[-1][:2]):
                    last_n = n
            optimized.append(last_n)
            last_index = self.final_course.index(last_n)
        for i in range(len(optimized) - 2):
            s, e = optimized[i], optimized[i + 1]
            theta = np.arctan2(e[1] - s[1], e[0] - s[0])
            optimized[i + 1][2] = theta

        self.min_course = optimized

    def smooth_stomp(self, weight_data=0.1, weight_smooth=0.1, tolerance=0.00001):
        # Copiar o caminho original para um novo caminho de saída
        new_path = [p for p in self.final_course]

        change = tolerance
        while change >= tolerance:
            change = 0.0
            for i in range(1, len(self.final_course) - 1):
                for j in range(2):
                    old_val = new_path[i][j]
                    new_val = (
                        new_path[i][j]
                        + weight_data * (self.final_course[i][j] - new_path[i][j])
                        + weight_smooth
                        * (
                            new_path[i - 1][j]
                            + new_path[i + 1][j]
                            - (2.0 * new_path[i][j])
                        )
                    )
                    new_path[i] = (
                        new_path[i][0] if j == 1 else new_val,
                        new_path[i][1] if j == 0 else new_val,
                        new_path[i][2],
                    )
                    change += abs(old_val - new_val)

        self.smooth_path = new_path

    def getPolynomials(self, max_length=20, divide=False):
        c = []
        new_path = self.min_course

        if divide:
            new_path = []
            for i in range(len(self.min_course) - 1):
                s, e = self.min_course[i], self.min_course[i + 1]
                d = np.sqrt((s[0] - e[0]) ** 2 + (s[1] - e[1]) ** 2)
                new_path.append(s)
                if i == len(self.min_course) - 2:
                    e[2] = np.arctan2(e[1] - s[1], e[0] - s[0])
                if d > max_length:
                    n = int(np.ceil(d / max_length))
                    new_x = np.linspace(s[0], e[0], n + 1).tolist()[1:-1]
                    new_y = np.linspace(s[1], e[1], n + 1).tolist()[1:-1]
                    for i in range(len(new_x)):
                        new_path.append([new_x[i], new_y[i], e[2]])

            new_path.append(self.min_course[-1])
            new_path[-1][2] = self.goal.yaw

        for i in range(len(new_path) - 1):
            s, e = new_path[i], new_path[i + 1]
            d = np.sqrt((s[0] - e[0]) ** 2 + (s[1] - e[1]) ** 2)
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
            c.append({"c": coef, "s": s, "e": e})

        self.coefs = c

        # return c, new_path

    def discretize(self, sample_rate=100):
        x = []
        y = []

        for c in self.coefs:
            s, e = c["s"], c["e"]
            a1, a2, a3, a4 = c["c"]
            tl = np.linspace(s[0], e[0], sample_rate).tolist()
            yt = list(map(lambda t: a1 + a2 * t + a3 * t**2 + a4 * t**3, tl))
            x = x + tl
            y = y + yt

        return x, y

    def validatePath(self):
        x, y = self.discretize()

        for i in range(len(x) - 2):
            s = (x[i], y[i])
            e = (x[i + 1], y[i + 1])

            if not self.check_collision_coords(s, e):
                return False

        return True  # Valid
