import random
import numpy as np
import matplotlib.pyplot as plt
from Robot import Robot
from TA import *
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

def create_graph_from_file(filename, nodes, n, downsample = 1):
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    dim = ((im.shape[1]+1) // downsample, (im.shape[0]+1) // downsample)
    im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
    new_file = "{}_downsample.png".format(filename)
    cv2.imwrite(new_file, im)
    gmap = OccupancyGridMap.from_png(new_file, 1)
    nodes_flip = (np.flip(nodes, axis=1)/downsample).tolist()
    adj = occupancy_map_8n.createGraph(n, nodes_flip, gmap)*downsample
    np.save('willow_{}_graph.npy'.format(n), adj[:n, :n])

def plot_pgm(filename):
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    im = cv2.flip(im, 0)
    plt.imshow(im, cmap='gray', vmin=0, vmax=255, origin='lower', interpolation='none', alpha=1)
    plt.draw()

if __name__ == '__main__':
    # filename = '/home/rachelzheng/robosar_ws/src/robosar_task_allocator/src/robosar_task_allocator/generate_graph/maps/localization_map_lab.pgm'
    # new_file = "{}.png".format(filename)
    # im = Image.open(filename).convert("L")
    # im = ImageOps.invert(im)
    # im.save(new_file)
    # gmap = OccupancyGridMap.from_png(new_file, 1)
    # gmap.plot()
    #
    # vicon_lab_points = np.array([[57, 60], [73, 61], [86, 65], [98, 61], [106, 61], [126, 61],
    #                              [61, 51], [80, 50], [104, 44], [115, 52], [126, 43],
    #                              [65, 36], [80, 34], [102, 35], [113, 38], [125, 33]])
    # plt.plot(vicon_lab_points[:,0], vicon_lab_points[:,1], 'ro')
    # plt.show()
    # np.save('vicon_lab_points.npy', vicon_lab_points)
    
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
        create_graph_from_file(filename, nodes, n, downsample)
        print('done')

    # with open('willow_map_data.pickle', 'rb') as f:
    #     gmap = pickle.load(f)
    # gmap.plot()
    plot_pgm(filename)

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
