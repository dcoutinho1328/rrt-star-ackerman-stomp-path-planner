import numpy as np
import random
from shapely import LineString, Point, Polygon
import math
from time import time
import matplotlib.pyplot as plt


# Vehicle Constructor
class Vehicle:
    def __init__(
        self,
        width,
        length,
        axis_distance,
        max_steering_angle,
        collision_padding=1,
        back_traction=True,
    ):
        self.width = width
        self.length = length
        self.axis_distance = axis_distance
        self.max_steering_angle = max_steering_angle
        self.collision_padding = collision_padding
        self.back_traction = back_traction


# Node Constructor
class Node:
    def __init__(self, x, y, yaw, parent_idx=None, cost=0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.parent_idx = parent_idx
        self.cost = cost
        self.rewired = False


# Path Planner
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
        pol_divide=20,
        conservative_collision=False,
    ):
        self.start = start
        self.goal = goal
        self.obstacle_list = obstacle_list
        self.vehicle = vehicle

        # Adds padding to obstacles
        self.growObstacles()

        self.rand_area = rand_area
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter

        # Initializes the node tree
        self.node_list = []
        self.node_list.append(self.start)

        self.search_radius_param = search_radius_param
        self.final_course = None
        self.bounds = bounds
        self.max_retries = retries
        self.retries = 0
        self.max_k = 0
        self.pol_divide = pol_divide

        # Selects collision-check method
        self.conservative_collision = conservative_collision

        self.check_problem()

    # Steer the goal point to allow entrance in tight spaces
    def tight_end(self):
        new_x = self.goal.x - self.vehicle.length * math.cos(self.goal.yaw)
        new_y = self.goal.y - self.vehicle.length * math.sin(self.goal.yaw)
        self.goal = Node(new_x, new_y, self.goal.yaw)
        self.check_problem()

    # Adds one more point to the end of the path
    def add_point_to_path(self, point):
        if self.final_course:
            s, e = self.final_course[-1], [point.x, point.y, point.yaw]
            self.final_course.append(e)
            self.min_course.append(e)
            self.smooth_path.append(e)
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

            self.coefs.append({"c": coef, "s": s, "e": e, "isCircle": False})

            self.goal = point

    # Resets the tree
    def __clean(self):
        self.node_list = [self.start]
        self.final_course = None

    # Check if the start and goal are valid
    def check_problem(self):
        s = Point((self.start.x, self.start.y))
        e = Point((self.goal.x, self.goal.y))

        if self.bounds and (not self.bounds.contains(s) or not self.bounds.contains(e)):
            print("Start or Goal out of boundaries")
            quit()  # Out of boundaries

        for o in self.obstacle_list:
            if o.intersects(s) or o.intersects(e):
                print("Start or Goal colliding with obstacles")
                quit()  # Out of boundaries

        if self.conservative_collision:
            for o in self.grown_obstacles:
                if o.intersects(s) or o.intersects(e):
                    return False  # collision

        return True

    # Adds padding to obstacles
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

    # Calculates euclidean distance from a point to the goal
    def calc_dist_to_goal(self, x, y):
        return np.linalg.norm([x - self.goal.x, y - self.goal.y])

    # Calculates euclidean distance between two nodes
    def line_distance(self, n1, n2):
        return np.linalg.norm([n1.x - n2.x, n1.y - n2.y])

    # Generates random pointo to steer
    def get_random_point(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = [
                random.uniform(self.rand_area[0], self.rand_area[1]),
                random.uniform(self.rand_area[2], self.rand_area[3]),
            ]
        else:  # goal point sampling
            rnd = [self.goal.x, self.goal.y]

        return rnd

    # Steer from a point to a direction
    def steer(self, from_node, to_theta, distance):
        x = from_node.x + distance * np.cos(to_theta)
        y = from_node.y + distance * np.sin(to_theta)
        yaw = to_theta
        new_node = Node(x, y, yaw, None, 0)

        return new_node

    # Find other nodes near a node
    def find_near_nodes(self, new_node):
        nnode = len(self.node_list)
        if nnode == 1:
            return [0]
        r = self.search_radius_param * np.sqrt((np.log(nnode) / nnode))
        dlist = [
            (node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2
            for node in self.node_list
        ]
        near_indices = [dlist.index(i) for i in dlist if i <= r**2]
        return near_indices

    # Gets the index from the closest node
    def get_nearest_list_index(self, node_list, rnd):
        dlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    # Selects the best parent node for a node
    def choose_parent(self, new_node, near_indices):
        if not near_indices:
            return None

        dlist = [
            self.calc_dist_to_goal(self.node_list[i].x, self.node_list[i].y)
            for i in near_indices
        ]
        i = dlist.index(min(dlist))
        while True:
            minind = near_indices[i]
            if self.check_collision(self.node_list[minind], new_node):
                break
            dlist.pop(i)
            near_indices.pop(i)
            try:
                i = dlist.index(min(dlist))
            except:
                minind = None
                break

        if minind is not None:
            new_node.parent_idx = minind
            p_node = self.node_list[minind]
            added_cost = self.line_distance(new_node, p_node)
            new_node.cost = p_node.cost + added_cost
            return new_node

        return None

    # Apply vehicle 2D dimensions to a point or segment
    def create_padding_box(self, b, f, s, st, e):
        t = math.atan2(e[1] - st[1], e[0] - st[0])

        p1 = [
            st[0] - b * math.cos(t) + s * math.sin(t),
            st[1] - b * math.sin(t) - s * math.cos(t),
        ]
        p2 = [
            st[0] - b * math.cos(t) - s * math.sin(t),
            st[1] - b * math.sin(t) + s * math.cos(t),
        ]
        p3 = [
            e[0] + f * math.cos(t) - s * math.sin(t),
            e[1] + f * math.sin(t) + s * math.cos(t),
        ]
        p4 = [
            e[0] + f * math.cos(t) + s * math.sin(t),
            e[1] + f * math.sin(t) - s * math.cos(t),
        ]

        pol = Polygon((p1, p2, p3, p4))

        return pol

    # Checks collision in a node or a segment
    def check_collision(self, n1, n2=None):
        if self.conservative_collision:
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

        else:
            a = (self.vehicle.length + self.vehicle.axis_distance) / 2
            b = (self.vehicle.length - self.vehicle.axis_distance) / 2

            if self.vehicle.back_traction:
                front_padding, back_padding = a, b
            else:
                front_padding, back_padding = b, a

            side_padding = self.vehicle.width / 2

            st, e = n1, n2 if n2 is not None else n1

            col = self.create_padding_box(
                back_padding,
                front_padding,
                side_padding,
                [st.x, st.y, st.yaw],
                [e.x, e.y, e.yaw],
            )

            if self.bounds and not self.bounds.contains(col):
                return False  # Out of boundaries

            for o in self.obstacle_list:
                if o.intersects(col):
                    return False  # collision

        return True

    # Checks collision without using the Node class
    def check_collision_coords(self, p1, p2):
        if self.conservative_collision:
            line = LineString([p1, p2])

            for o in self.grown_obstacles:
                if o.intersects(line):
                    return False  # collision

            if self.bounds and not self.bounds.contains(line):
                return False  # Out of boundaries

        else:
            a = (self.vehicle.length + self.vehicle.axis_distance) / 2
            b = (self.vehicle.length - self.vehicle.axis_distance) / 2

            if self.vehicle.back_traction:
                front_padding, back_padding = a, b
            else:
                front_padding, back_padding = b, a

            side_padding = self.vehicle.width / 2

            col = self.create_padding_box(
                back_padding, front_padding, side_padding, p1, p2
            )

            if self.bounds and not self.bounds.contains(col):
                return False  # Out of boundaries

            for o in self.obstacle_list:
                if o.intersects(col):
                    return False  # collision

        return True

    # Rewires the tree finding costless paths
    def rewire(self, new_node, near_nodes):
        for i in near_nodes:
            near_node = self.node_list[i]
            line_dist = self.line_distance(near_node, new_node)

            # Check if there is a better path to near_node through new_node
            if near_node.cost > new_node.cost + line_dist:
                if self.check_collision(new_node, near_node):
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

    # Updates the tree after rewires
    def propagate_cost_to_leaves(self, from_node, previous_cost):
        start = self.node_list.index(from_node)
        for n in self.node_list[start:]:
            if self.node_list[n.parent_idx] == from_node:
                prev_cost = n.cost
                n.cost = n.cost - previous_cost + from_node.cost
                self.propagate_cost_to_leaves(n, prev_cost)

    # Extracts the nodes that are in the final path
    def generate_final_course(self, goal_ind=-1):
        path = [[self.goal.x, self.goal.y, self.goal.yaw]]
        node = self.node_list[goal_ind]
        while node.parent_idx is not None:
            path.append([node.x, node.y, node.yaw])
            node = self.node_list[node.parent_idx]
        path.append([node.x, node.y, node.yaw])
        self.final_course = path[::-1]

    # Builds the tree
    def run_planner(self):
        # Start timer
        self.ts = time()
        self.__clean
        self.node_list = [self.start]

        # Main loop
        while True:
            # Generates random point to steer
            rnd = self.get_random_point()
            nearest_ind = self.get_nearest_list_index(self.node_list, rnd)
            nearest_node = self.node_list[nearest_ind]
            theta = np.arctan2(rnd[1] - nearest_node.y, rnd[0] - nearest_node.x)

            # Steer and find node candidate and near nodes
            new_node = self.steer(nearest_node, theta, self.expand_dis)
            near_indices = self.find_near_nodes(new_node)
            new_node = self.choose_parent(new_node, near_indices)
            if new_node:
                # Adds the node to the three and check for rewires
                self.node_list.append(new_node)
                self.rewire(new_node, near_indices)
                # self.plotPathSoFar()

            # Checks if goal was reached
            if (
                self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y)
                <= self.expand_dis
            ):
                if self.check_collision(self.node_list[-1], self.goal):
                    self.generate_final_course()
                    if self.final_course:
                        # Calls post processing methods
                        self.t1 = time() - self.ts
                        self.smooth_stomp()
                        self.t2 = time() - self.ts
                        self.post_processing()
                        self.t3 = time() - self.ts - self.t2
                        self.getPolynomials(max_length=self.pol_divide, divide=True)
                        self.t4 = time() - self.ts - self.t2
                    break

            # Limits the number of iterations
            if len(self.node_list) > self.max_iter:
                print("Reached Maximum Iterations")
                quit()

    # Removes unnecessary segments and applies vertice cut
    def post_processing(self):
        optimized = [self.final_course[0]]
        last_index = 0
        while last_index < len(self.final_course) - 1:
            last_n = None
            for n in self.final_course[last_index + 1 :]:
                if self.check_collision_coords(optimized[-1], n):
                    last_n = n
            optimized.append(last_n)
            last_index = self.final_course.index(last_n)
        for i in range(len(optimized) - 2):
            s, e = optimized[i], optimized[i + 1]
            theta = np.arctan2(e[1] - s[1], e[0] - s[0])
            optimized[i + 1][2] = theta

        # Find vertices to be cut
        for i in range(len(optimized) - 3):
            s, m, e = optimized[i], optimized[i + 1], optimized[i + 2]
            if np.sign(m[0] - s[0]) != np.sign(e[0] - m[0]):
                self.calculate_k()
                biss = (m[2] + math.radians(180) + e[2]) / 2
                dteta = e[2] - m[2]
                desloc = (1 / self.max_k) / (math.cos(dteta / 2))
                dsegs = (1 / self.max_k) * math.tan(dteta / 2)
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
                if self.check_collision_coords(new_seg_start, new_seg_end):
                    optimized = (
                        optimized[: i + 1]
                        + [new_seg_start, new_seg_end]
                        + optimized[i + 2 :]
                    )

        self.min_course = optimized

    # Calculates vehicle max curvature
    def calculate_k(self):
        r_min = self.vehicle.axis_distance / math.tan(
            math.radians(self.vehicle.max_steering_angle)
        )

        self.max_k = 1 / r_min

    # Checks polynom curvature
    def check_curvature(self, coefs, x):
        a1, a2, a3, _ = coefs["c"]

        k = lambda x: abs(6 * a1 * x + 2 * a2) / (
            (1 + (3 * a1 * (x**2) + 2 * a2 * x + a3) ** 2) ** (3 / 2)
        )

        for p in x:
            if k(p) > self.max_k:
                return False  # Not valid

        return True

    # Applies STOMP algorithm to path
    def smooth_stomp(self, weight_data=0.1, weight_smooth=0.1, tolerance=0.00001):
        # Copy original path to a new output path
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

    # Plots the progress
    def plotPathSoFar(self):
        segments = []

        for n in self.node_list:
            if n.parent_idx:
                parent = self.node_list[n.parent_idx]
                segments.append([[parent.x, n.x], [parent.y, n.y]])

        path = []
        dists = list(map(lambda n: self.calc_dist_to_goal(n.x, n.y), self.node_list))
        min_id = dists.index(min(dists))
        node = self.node_list[min_id]
        while node.parent_idx is not None:
            path.append([node.x, node.y])
            node = self.node_list[node.parent_idx]
        path.append([node.x, node.y])
        path = np.array(path)

        for s in segments:
            X = np.array(s[0])
            Y = np.array(s[1])
            plt.plot(X, Y, "-r")

        plt.plot(path[:, 0], path[:, 1], "-g", label="Main Path")

        if self.bounds:
            plt.plot(*self.bounds.exterior.xy, "-b")

        for o in self.obstacle_list:
            plt.plot(*o.exterior.xy, "-b")

        plt.plot(self.start.x, self.start.y, "xg", label="Start")
        plt.plot(self.goal.x, self.goal.y, "xr", label="Goal")

        plt.legend()

        plt.show()

    # Finds the polinomials that fit into segments
    def getPolynomials(self, max_length=35, divide=False):
        c = []
        new_path = self.min_course

        # Divides the segment into smaller segments
        if divide:
            new_path = []
            for i in range(len(self.min_course) - 1):
                s, e = self.min_course[i], self.min_course[i + 1]
                if len(s) == 4:
                    new_path.append(s)
                    continue
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

        self.coefs = c

    # Discretizes the polynomials
    def discretize(self, c=None, sample_rate=1000):
        x = []
        y = []
        z = []

        if not c:
            for c in self.coefs:
                s, e = c["s"], c["e"]
                if c["isCircle"]:
                    x0, y0 = c["c"]
                    r = 1 / self.max_k
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
                    yt = list(
                        map(lambda t: a1 + a2 * t + a3 * t**2 + a4 * t**3, tl)
                    )
                    zt = list(map(lambda t: a2 + 2 * a3 * t + 3 * a4 * t**2, tl))
                    x = x + tl
                    y = y + yt
                    z = z + zt

        else:
            s, e = c["s"], c["e"]
            if c["isCircle"]:
                x0, y0 = c["c"]
                r = 1 / self.max_k
                tl = np.linspace(s[2], e[2], sample_rate).tolist()
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

    # Validates the generated polynomials
    def validatePath(self):
        if not self.final_course:
            return False

        self.calculate_k()

        for c in self.coefs:
            x, y, z = self.discretize(c)

            if not c["isCircle"] and not self.check_curvature(c, x):
                return False  # Vehicle cant perform the curvature

            for i in range(len(x) - 2):
                s = (x[i], y[i], z[i])
                e = (x[i + 1], y[i + 1], z[i + 1])

                if not self.check_collision_coords(s, e):
                    return False

        return True  # Valid
