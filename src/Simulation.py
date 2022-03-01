import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations

class Simulation:
    def __init__(self, env, dt, max_steps):
        self.env = env
        self.dt = dt
        self.t = 0
        self.t_step = 0
        self.max_steps = max_steps


    def simulate(self):
        self.env.visited.add(0)
        for id in range(len(self.env.robots)):
            self.assign(id, self.env.robots[id].prev)
        while len(self.env.visited) < self.env.num_nodes and self.t_step < self.max_steps:
            self.move()

            plt.plot([self.env.robots[0].pos_prev[0], self.env.robots[0].pos[0]], [self.env.robots[0].pos_prev[1], self.env.robots[0].pos[1]], 'r-')
            plt.plot([self.env.robots[1].pos_prev[0], self.env.robots[1].pos[0]], [self.env.robots[1].pos_prev[1], self.env.robots[1].pos[1]], 'b-')
            plt.plot([self.env.robots[2].pos_prev[0], self.env.robots[2].pos[0]], [self.env.robots[2].pos_prev[1], self.env.robots[2].pos[1]], 'm-')
            plt.pause(0.005)

            for r in self.env.robots:
                r.pos_prev = [r.pos[0], r.pos[1]]

            self.t += self.dt
            self.t_step += 1

        robot_paths = []
        for robot in self.env.robots:
            robot_paths.append(robot.visited)
        print("Finished at t = {}".format(self.t))
        return robot_paths

    def assign(self, id, curr_node):
        C = self.env.adj[curr_node,:]
        robot = self.env.robots[id]
        min_node_list = np.argsort(C)
        min_node = -1
        for i in min_node_list:
            if C[i] > 0 and i not in self.env.visited and i not in self.env.frontier:
                min_node = i
                break
        # No A* path
        if min_node == -1:
            print('USING EUCLIDEAN DISTANCE')
            E = []
            idx = []
            for i, node in enumerate(self.env.nodes):
                if i not in self.env.visited and i not in self.env.frontier:
                    idx.append(i)
                    E.append(np.sqrt((node[0] - robot.pos[0]) ** 2 + (node[1] - robot.pos[1]) ** 2))
            if E:
                min_node_i = np.argmin(np.array(E))
                min_node = idx[min_node_i]

        print("Assigned robot {}: node {} at {}".format(id, min_node, self.env.nodes[min_node]))
        plt.plot(self.env.nodes[min_node][0], self.env.nodes[min_node][1], 'go', zorder=101)
        robot.next = min_node
        self.env.frontier.add(min_node)

    def reached(self, id, curr_node):
        r = self.env.robots[id]
        if r.prev is not curr_node:
            r.prev = curr_node
            r.visited.append(curr_node)
            self.env.frontier.remove(curr_node)
            self.env.visited.add(curr_node)
            print("Robot {} reached node {}".format(id, curr_node))
            if len(self.env.visited) + len(self.env.frontier) < self.env.num_nodes:
                self.assign(id, curr_node)

    def move(self):
        for id, r in enumerate(self.env.robots):
            goal = self.env.nodes[r.next]
            dir = np.array([goal[0]-r.pos[0], goal[1]-r.pos[1]])
            dist = np.linalg.norm(dir)
            if dist > 0:
                dir = dir/dist
                dx = r.v*self.dt
                if dx > dist:
                    r.pos[0] = goal[0]
                    r.pos[1] = goal[1]
                    self.reached(id, r.next)
                else:
                    x_next = r.pos + dir*dx
                    r.pos[0] = x_next[0]
                    r.pos[1] = x_next[1]
            else:
                self.reached(id, r.next)