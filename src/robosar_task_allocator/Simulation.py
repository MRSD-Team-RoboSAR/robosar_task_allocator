"""
Simulation class
- Simulates environment and robots moving suing euclidean distance between nodes
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import sys


class Simulation:
    def __init__(self, env, solver, dt, max_steps, colors):
        """
        env: Environment object
        solver: TA object
        dt: float time step
        max_steps: number of time steps before simulation ends
        colors: list of colors for plotting
        """
        self.env = env
        self.solver = solver
        self.solver.init(self.env)
        self.dt = dt
        self.t = 0
        self.t_step = 0
        self.max_steps = max_steps
        self.colors = colors


    def simulate(self):
        """
        Simulation loop
        """
        for id in self.env.robots:
            self.solver.assign(id, self.env.robots[id].prev)

        # simulate until all tasks are completed
        while len(self.env.visited) < self.env.num_nodes and self.t_step < self.max_steps:
            # deactivate robot
            if self.t_step == 10:
                self.deactivate_robot(3)
                self.solver.calculate_mtsp(False)
                for id in self.env.robots:
                    self.solver.assign(id, self.env.robots[id].prev)

            # move robots
            self.move()

            # plotting
            for idx, r in enumerate(self.env.robots.values()):
                plt.plot([r.pos_prev[0], r.pos[0]], [r.pos_prev[1], r.pos[1]], self.colors[idx]+'-')
            plt.pause(0.005)
            for r in self.env.robots.values():
                r.pos_prev = [r.pos[0], r.pos[1]]

            # increase time step
            self.t += self.dt
            self.t_step += 1

        # print paths
        robot_paths = []
        for robot in self.env.robots.values():
            robot_paths.append(robot.visited)
        print("Finished at t = {}".format(self.t))
        print("Objective value = {}".format(max(self.solver.objective_value)))
        return robot_paths


    def move(self):
        """
        Move robot
        """
        for id, r in self.env.robots.items():
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


    def deactivate_robot(self, id):
        """
        Deactivate robot
        """
        active_agent_status = {}
        for r in self.env.robots:
            if r != id:
                active_agent_status[r] = True
            else:
                active_agent_status[r] = False
        self.env.fleet_update(active_agent_status)


