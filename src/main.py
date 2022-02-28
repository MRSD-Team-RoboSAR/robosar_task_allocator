import random
import numpy as np
import matplotlib.pyplot as plt
from Robot import Robot
from Environment import Environment
from Simulation import Simulation
from generate_graph import occupancy_map_8n
from PIL import Image, ImageOps
from generate_graph.gridmap import OccupancyGridMap


# def distance(c1, c2):
#     diff = (c1[0] - c2[0], c1[1] - c2[1])
#     return np.sqrt(diff[0] * diff[0] + diff[1] * diff[1])
#
#
# def createGraph(n, env_size):
#     nodes = []
#     adj = np.zeros((n, n))
#     for i in range(n):
#         nodes.append([random.uniform(0,1)*env_size, random.uniform(0,1)*env_size])
#     nodes = np.array(nodes)
#     for i in range(n):
#         for j in range(i,n):
#             adj[i, j] = distance(nodes[i,:], nodes[j,:])
#             adj[j, i] = adj[i,j]
#
#     return nodes, adj

if __name__ == '__main__':
    # Create graph
    filename = 'generate_graph/maps/localization_map_vicon.pgm'
    new_file = "{}.png".format(filename)
    with Image.open(filename) as im:
        im = ImageOps.invert(im)
        im.save(new_file)
    gmap = OccupancyGridMap.from_png(new_file, 1)
    nodes = ((65, 40), (65, 60), (70, 60), (80, 40), (90, 40), (100, 40), (90, 50), (95, 55), (100, 60), (100, 50))
    adj = occupancy_map_8n.createGraph(10, nodes, gmap)

    # Create robots
    robot0 = Robot(0, list(nodes[0]), 0)
    robot1 = Robot(1, list(nodes[0]), 0)
    robots = [robot0, robot1]
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

    sim = Simulation(env, 0.1, 1000)
    robot_paths = sim.simulate()

    for path in robot_paths:
        print(path)

    plt.show()
