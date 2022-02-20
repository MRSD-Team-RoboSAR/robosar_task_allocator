import numpy as np
import matplotlib.pyplot as plt
from Robot import Robot
from Environment import Environment

def simulation(dt, max_steps):
    t_step = 0
    t = 0
    env.visited.add(0)
    for id in range(len(env.robots)):
        env.assign(id)
    while len(env.visited) < env.num_nodes and t_step < max_steps:
        env.move(dt)

        plt.plot(env.robots[0].pos[0], env.robots[0].pos[1], 'ro')
        plt.plot(env.robots[1].pos[0], env.robots[1].pos[1], 'bo')
        plt.plot(env.robots[2].pos[0], env.robots[2].pos[1], 'go')
        plt.plot(env.robots[3].pos[0], env.robots[3].pos[1], 'mo')
        plt.pause(0.005)

        t += dt
        t_step += 1
    print("Finished at t = {}".format(t))

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

    simulation(0.1, 1000)

    plt.show()
