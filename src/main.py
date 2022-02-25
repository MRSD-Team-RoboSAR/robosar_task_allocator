import random

import numpy as np
import matplotlib.pyplot as plt
from Robot import Robot
from Environment import Environment
from Simulation import Simulation


def distance(c1, c2):
    diff = (c1[0] - c2[0], c1[1] - c2[1])
    return np.sqrt(diff[0] * diff[0] + diff[1] * diff[1])


def createGraph(n, env_size):
    nodes = []
    adj = np.zeros((n, n))
    for i in range(n):
        nodes.append([random.uniform(0,1)*env_size, random.uniform(0,1)*env_size])
    nodes = np.array(nodes)
    for i in range(n):
        for j in range(i,n):
            adj[i, j] = distance(nodes[i,:], nodes[j,:])
            adj[j, i] = adj[i,j]

    return nodes, adj

if __name__ == '__main__':
    # Create graph
    n = 50
    env_size = 7
    nodes, adj = createGraph(n, env_size)
    # Create robots
    robot0 = Robot(0, nodes[10].tolist(), 10)
    robot1 = Robot(1, nodes[0].tolist(), 0)
    robot2 = Robot(2, nodes[10].tolist(), 10)
    robot3 = Robot(3, nodes[0].tolist(), 0)
    robots = [robot0, robot1, robot2, robot3]
    # Create environment
    env = Environment(nodes, adj, robots)

    # Plotting
    node_x = []
    node_y = []
    for node in nodes:
        node_x.append(node[0])
        node_y.append(node[1])
    plt.plot(node_x, node_y, 'ko', zorder=100)
    plt.plot(robot0.pos[0], robot0.pos[1], 'ro')
    plt.plot(robot1.pos[0], robot1.pos[1], 'bo')
    plt.plot(robot2.pos[0], robot2.pos[1], 'go')
    plt.plot(robot3.pos[0], robot3.pos[1], 'mo')

    sim = Simulation(env, 0.1, 1000)
    robot_paths = sim.simulate()

    for path in robot_paths:
        print(path)

    plt.show()
