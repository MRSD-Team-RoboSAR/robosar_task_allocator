import random
import numpy as np
import matplotlib.pyplot as plt
from Robot import Robot
from TA import *
import utils
from Environment import Environment
from Simulation import Simulation
from generate_graph import occupancy_map_8n
from generate_graph.gridmap import OccupancyGridMap
import pickle
import cv2

# Random graph
def distance(c1, c2):
    diff = (c1[0] - c2[0], c1[1] - c2[1])
    return np.sqrt(diff[0] * diff[0] + diff[1] * diff[1])

def create_random_graph(n, env_size):
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
    n = 60
    make_graph = False
    downsample = 5
    # nodes = np.load("/home/rachelzheng/robosar_ws/src/robosar_task_allocator/src/robosar_task_allocator/vicon_lab_points.npy")
    nodes = np.load("/home/rachelzheng/robosar_ws/src/robosar_task_generator/outputs/willow-full_lean.npy")
    # filename = '/home/rachelzheng/robosar_ws/src/robosar_task_allocator/src/robosar_task_allocator/generate_graph/maps/localization_map_lab.pgm'
    filename = '/home/rachelzheng/robosar_ws/src/robosar_task_generator/maps/willow-full.pgm'
    if make_graph:
        print('creating graph')
        adj = utils.create_graph_from_file(filename, nodes, n, downsample, make_graph)
        print('done')

    utils.plot_pgm(filename)

    # Create robots
    robot0 = Robot(0, nodes[0], 0)
    robot1 = Robot(1, nodes[0], 0)
    robot2 = Robot(2, nodes[0], 0)
    robots = [robot0, robot1, robot2]
    # Create environment
    adj = np.load('/home/rachelzheng/robosar_ws/src/robosar_task_allocator/src/robosar_task_allocator/willow_{}_graph.npy'.format(n))
    env = Environment(nodes[:n,:], adj, robots)

    # Plotting
    plt.plot(nodes[:n,0], nodes[:n,1], 'ko', zorder=100)
    plt.plot(robot0.pos[0], robot0.pos[1], 'ro')
    plt.plot(robot1.pos[0], robot1.pos[1], 'bo')
    plt.plot(robot2.pos[0], robot2.pos[1], 'mo')

    sim = Simulation(env, TA_mTSP(), 1, 300)
    robot_paths = sim.simulate()

    for i, path in enumerate(robot_paths):
        print("Robot {} path: {}".format(i, path))

    plt.show()
