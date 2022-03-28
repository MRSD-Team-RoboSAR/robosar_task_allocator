import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import sys

class Simulation:
    def __init__(self, env, solver, dt, max_steps):
        self.env = env
        self.solver = solver
        solver.init(self.env)
        self.dt = dt
        self.t = 0
        self.t_step = 0
        self.max_steps = max_steps

    def simulate(self):
        for id in range(len(self.env.robots)):
            self.solver.assign(id, self.env.robots[id].prev)

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
        print("Objective value = {}".format(max(self.solver.objective_value)))
        return robot_paths

    def move(self):
        for id, r in enumerate(self.env.robots):
            assert r.next
            goal = self.env.nodes[r.next]
            dir = np.array([goal[0]-r.pos[0], goal[1]-r.pos[1]])
            dist = np.linalg.norm(dir)
            if dist > 0:
                dir = dir/dist
                dx = r.v*self.dt
                if dx > dist:
                    r.pos[0] = goal[0]
                    r.pos[1] = goal[1]
                    self.solver.reached(id, r.next)
                else:
                    x_next = r.pos + dir*dx
                    r.pos[0] = x_next[0]
                    r.pos[1] = x_next[1]
            else:
                self.solver.reached(id, r.next)
