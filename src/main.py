import random
import numpy as np
import matplotlib.pyplot as plt
from Robot import Robot
from Environment import Environment
from Simulation import Simulation
from generate_graph import occupancy_map_8n
from PIL import Image, ImageOps
from generate_graph.gridmap import OccupancyGridMap
import pickle

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

def create_graph_from_file(filename, nodes, n):
    new_file = "{}.png".format(filename)
    im = Image.open(filename).convert("L")
    im = ImageOps.invert(im)
    im.save(new_file)
    gmap = OccupancyGridMap.from_png(new_file, 1)
    # plt.plot(nodes[:,0],nodes[:,1], 'go')
    # plt.show()
    nodes_flip = np.flip(nodes, axis=1).tolist()
    adj = occupancy_map_8n.createGraph(n, nodes_flip, gmap)
    np.save('willow_20_graph.npy', adj[:n, :n])
    with open('map_data.pickle', 'wb') as f:
        pickle.dump(gmap, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    # Create graph
    n = 60
    make_graph = False
    nodes = np.load("../../robosar_task_generator/outputs/willow-full_lean.npy")
    filename = '../../robosar_task_generator/maps/willow-full.pgm'
    if make_graph:
        print('creating graph')
        create_graph_from_file(filename, nodes, n)

    with open('map_data.pickle', 'rb') as f:
        gmap = pickle.load(f)
    gmap.plot()

    # Create robots
    robot0 = Robot(0, nodes[0], 0)
    robot1 = Robot(1, nodes[0], 0)
    robot2 = Robot(2, nodes[0], 0)
    robots = [robot0, robot1, robot2]
    # Create environment
    adj = np.load('willow_{}_graph.npy'.format(n))
    env = Environment(nodes[:n,:], adj, robots)

    # Plotting
    plt.plot(nodes[:n,0], nodes[:n,1], 'ko', zorder=100)
    plt.plot(robot0.pos[0], robot0.pos[1], 'ro')
    plt.plot(robot1.pos[0], robot1.pos[1], 'bo')
    plt.plot(robot2.pos[0], robot2.pos[1], 'mo')

    sim = Simulation(env, 1, 1000)
    robot_paths = sim.simulate()

    for i, path in enumerate(robot_paths):
        print("Robot {} path: {}".format(i, path))

    plt.show()
