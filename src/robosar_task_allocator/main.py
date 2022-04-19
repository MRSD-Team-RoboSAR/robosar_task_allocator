"""
Main script to be executed. Used for testing without ROS.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from Robot import Robot
from TA import *
import utils
from Environment import Environment
from Simulation import Simulation
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
    # n = 40
    make_graph = False
    downsample = 5
    # nodes = np.load("/home/rachelzheng/robosar_ws/src/robosar_task_allocator/src/robosar_task_allocator/vicon_lab_points.npy")
    nodes = np.load("/home/rachelzheng/robosar_ws/src/robosar_task_allocator/src/robosar_task_allocator/saved_graphs/scott_SVD_points.npy")
    # filename = '/home/rachelzheng/robosar_ws/src/robosar_task_allocator/src/robosar_task_allocator/generate_graph/maps/localization_map_lab.pgm'
    filename = '/home/rachelzheng/robosar_ws/src/robosar_navigation/maps/scott_hall_SVD.pgm'
    if make_graph:
        print('creating graph')
        adj = utils.create_graph_from_file(filename, nodes, n, downsample, False)
        print('done')

    utils.plot_pgm(filename)

    # Create environment
    adj = np.load('/home/rachelzheng/robosar_ws/src/robosar_task_allocator/src/robosar_task_allocator/saved_graphs/scott_SVD_graph.npy')
    n = len(nodes)
    env = Environment(nodes[:n,:], adj)

    # Create robots
    id_list = [0,1,2]
    for id in id_list:
        env.add_robot(id, "robot_"+str(id), id)

    # Plotting
    colors = ['r', 'b', 'm', 'c', 'y', 'g', 'k']
    plt.plot(nodes[:n,0], nodes[:n,1], 'ko', zorder=100)
    for idx, r in enumerate(env.robots.values()):
        plt.plot([r.pos_prev[0], r.pos[0]], [r.pos_prev[1], r.pos[1]], colors[idx] + 'o')

    solver = TA_mTSP()
    solver.init(env, 5)
    sim = Simulation(env, solver, 1, 300, colors)
    robot_paths = sim.simulate()

    for i, path in enumerate(robot_paths):
        print("Robot {} path: {}".format(i, path))

    plt.show()
