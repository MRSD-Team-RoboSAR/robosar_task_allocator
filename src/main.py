import numpy as np
import matplotlib.pyplot as plt
from Robot import Robot
from Environment import Environment
from Simulation import Simulation

if __name__ == '__main__':
    n = 10
    robot0 = Robot(0, [0, 0])
    robot1 = Robot(1, [0, 0])
    robot2 = Robot(2, [0, 0])
    robot3 = Robot(3, [0, 0])
    robots = [robot0, robot1, robot2, robot3]
    env = Environment(n, robots)

    node_x = []
    node_y = []
    for node in env.nodes:
        node_x.append(node[0])
        node_y.append(node[1])
    plt.plot(node_x, node_y, 'ko', zorder=100)
    plt.plot(robot0.pos[0], robot0.pos[1], 'ro')
    plt.plot(robot1.pos[0], robot1.pos[1], 'bo')
    plt.plot(robot2.pos[0], robot2.pos[1], 'go')
    plt.plot(robot3.pos[0], robot3.pos[1], 'mo')

    sim = Simulation(env, 0.1, 1000)
    sim.simulate()

    plt.show()
