import numpy as np
import random
import matplotlib.pyplot as plt
from shapely import LineString, Point, Polygon
import math
        
class Vehicle:

    def __init__ (self, width, length, max_steering_angle, collision_padding=1):
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


class RRTStar:
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
        search_radius_param=5
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

    def steer(self, from_node, to_theta, distance):
        x = from_node.x + distance * np.cos(to_theta)
        y = from_node.y + distance * np.sin(to_theta)
        yaw = to_theta
        new_node = Node(x, y, yaw, None, 0)

        return new_node

    def check_collision(self, n1, n2=None):
        if n2 is not None:
            line = LineString([[n1.x, n1.y], [n2.x, n2.y]])

            for o in self.grown_obstacles:
                if o.intersects(line):
                    return False  # collision

        else:
            point = Point((n1.x, n1.y))

            for o in self.grown_obstacles:
                if o.intersects(point):
                    return False

        return True

    def checkSteering(self, n1, n2):
        theta = np.arctan2(n2.y - n1.y, n2.x - n1.x)
        return abs(theta - n1.yaw) <= self.vehicle.max_steering_angle

    def get_random_point(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = [
                random.uniform(self.rand_area[0], self.rand_area[1]),
                random.uniform(self.rand_area[2], self.rand_area[3]),
            ]
        else:  # goal point sampling
            rnd = [self.goal.x, self.goal.y]

        return rnd

    def growObstacles(self):

        self.grown_obstacles = []

        for obs in self.obstacle_list:
            gp = Polygon(obs.buffer(math.sqrt(self.vehicle.width**2 + self.vehicle.length**2) + self.vehicle.collision_padding))
            self.grown_obstacles.append(gp)
            

    def get_nearest_list_index(self, node_list, rnd):
        dlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    def generate_final_course(self, goal_ind):
        path = [[self.goal.x, self.goal.y]]
        node = self.node_list[goal_ind]
        count = 0
        while node.parent_idx is not None:
            count += 1
            path.append([node.x, node.y])
            node = self.node_list[node.parent_idx]
        path.append([node.x, node.y])
        return path

    def calc_dist_to_goal(self, x, y):
        return np.linalg.norm([x - self.goal.x, y - self.goal.y])

    def line_distance(self, n1, n2):
        return np.linalg.norm([n1.x - n2.x, n1.y - n2.y])

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
        # print(near_indices)
        return near_indices

    def choose_parent(self, new_node, near_indices):
        if not near_indices:
            return None

        dlist = [
            self.calc_dist_to_goal(self.node_list[i].x, self.node_list[i].y)
            for i in near_indices
        ]
        minind = near_indices[dlist.index(min(dlist))]
        # if not self.checkSteering(self.node_list[minind], new_node):
        #     return None
        new_node.parent_idx = minind
        p_node = self.node_list[minind]
        added_cost = self.line_distance(new_node, p_node)
        new_node.cost = p_node.cost + added_cost
        return new_node

    def propagate_cost_to_leaves(self, from_node, previous_cost):
        start = self.node_list.index(from_node)
        for n in self.node_list[start:]:
            if self.node_list[n.parent_idx] == from_node:
                prev_cost = n.cost
                n.cost = n.cost - previous_cost + from_node.cost
                self.propagate_cost_to_leaves(n, prev_cost)

    def rewire(self, new_node, near_nodes):
        for i in near_nodes:
            near_node = self.node_list[i]
            line_dist = self.line_distance(near_node, new_node)

            if near_node.cost > new_node.cost + line_dist:
                if self.check_collision(near_node, new_node) and self.checkSteering(new_node, near_node):
                    prev_cost = near_node.cost
                    near_node.parent_idx = len(self.node_list) - 1
                    near_node.cost = new_node.cost + line_dist
                    new_node.yaw = np.arctan2(near_node.y - new_node.y, near_node.x - new_node.x)
                    near_node.rewired = True

                    # Propagate cost changes to descendants
                    self.propagate_cost_to_leaves(near_node, prev_cost)

                # Check if new trajectory is not more efficient than previous one
                goal_dist = self.calc_dist_to_goal(near_node.x, near_node.y)
                if near_node.cost + goal_dist > new_node.cost + goal_dist:
                    return

    def run_rrt_star(self):
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
                    return self.generate_final_course(goal_ind)


            if True:
                final_node = self.node_list[-1]
                goal_ind = self.get_nearest_list_index(
                    self.node_list, [self.goal.x, self.goal.y]
                )
                if self.check_collision(final_node, self.node_list[goal_ind]):
                    var = self.generate_final_course(goal_ind)
                    path = np.array(var)



            if len(self.node_list) > self.max_iter:
                print("Reached Maximum Iterations")
                return None

