from Environment import Environment
import numpy as np
import matplotlib.pyplot as plt

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
            self.assign(id)
        while len(self.env.visited) < self.env.num_nodes and self.t_step < self.max_steps:
            self.move()

            plt.plot(self.env.robots[0].pos[0], self.env.robots[0].pos[1], 'ro')
            plt.plot(self.env.robots[1].pos[0], self.env.robots[1].pos[1], 'bo')
            plt.plot(self.env.robots[2].pos[0], self.env.robots[2].pos[1], 'go')
            plt.plot(self.env.robots[3].pos[0], self.env.robots[3].pos[1], 'mo')
            plt.pause(0.005)

            self.t += self.dt
            self.t_step += 1
        print("Finished at t = {}".format(self.t))

    def assign(self, id):
        r = self.env.robots[id]
        idx = []
        C = []
        robot = self.env.robots[id]
        for i, node in enumerate(self.env.nodes):
            if i not in self.env.visited and i not in self.env.frontier:
                idx.append(i)
                C.append(np.sqrt((node[0]-robot.pos[0])**2 + (node[1]-robot.pos[1])**2))
        if C:
            min_node = np.argmin(np.array(C))
            min_node_i = idx[min_node]
            print("Assigned robot {}: node {} at {}".format(id, min_node_i, self.env.nodes[min_node_i]))
            robot.next = min_node_i
            self.env.frontier.add(min_node_i)

    def reached(self, id, curr_node):
        r = self.env.robots[id]
        if r.prev is not curr_node:
            r.prev = curr_node
            r.visited.append(curr_node)
            self.env.frontier.remove(curr_node)
            self.env.visited.add(curr_node)
            print("Robot {} reached node {}".format(id, curr_node))
            if len(self.env.visited) + len(self.env.frontier) < self.env.num_nodes:
                self.assign(id)

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