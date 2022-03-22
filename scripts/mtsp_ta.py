#!/usr/bin/env python3

# Created by Indraneel on 01/03/22

import rospy
from robosar_task_allocator.Environment import Environment
from robosar_task_allocator.Robot import Robot
from robosar_task_allocator.TA import *
from robosar_task_allocator.task_transmitter.task_tx_move_base import TaskTxMoveBase
import numpy as np
import pickle
import rospkg
from actionlib_msgs.msg import GoalStatus
from nav_msgs.msg import OccupancyGrid
from robosar_messages.srv import *
from robosar_task_allocator.generate_graph import occupancy_map_8n
from robosar_task_allocator.generate_graph.gridmap import OccupancyGridMap
from PIL import Image, ImageOps


rospack = rospkg.RosPack()
maps_path = rospack.get_path('robosar_task_generator')
package_path = rospack.get_path('robosar_task_allocator')

def pixels_to_m(pixels):
    return [pixels[0]*0.1,pixels[1]*0.1]

def create_graph_from_file(filename, nodes, n):
    new_file = "{}.png".format(filename)
    im = Image.open(filename).convert("L")
    im = ImageOps.invert(im)
    im.save(new_file)
    gmap = OccupancyGridMap.from_png(new_file, 1)
    nodes_flip = np.flip(nodes, axis=1).tolist()
    adj = occupancy_map_8n.createGraph(n, nodes_flip, gmap)
    np.save(package_path+'/src/robosar_task_allocator/custom_{}_graph.npy'.format(n), adj[:n, :n])
    with open('willow_map_data.pickle', 'wb') as f:
        pickle.dump(gmap, f, pickle.HIGHEST_PROTOCOL)

def mtsp_allocator():
    rospy.init_node('task_allocator_mtsp', anonymous=True)

    # Get waypoints client
    map_msg = rospy.wait_for_message("/map", OccupancyGrid)
    rospy.wait_for_service('taskgen_getwaypts')
    try:
        print("calling service")
        get_waypoints = rospy.ServiceProxy('taskgen_getwaypts', taskgen_getwaypts)
        resp1 = get_waypoints(map_msg, 1, 20)
        nodes = resp1.waypoints
        nodes = np.reshape(nodes, (-1,2))
        print(nodes)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

    # Create robots
    robot0 = Robot(0, nodes[0], 0)
    robot1 = Robot(1, nodes[0], 0)
    robot2 = Robot(2, nodes[0], 0)
    robots = [robot0, robot1, robot2]

    # Create graph
    n = nodes.shape[0]
    make_graph = True
    # nodes = np.load(maps_path+"/outputs/vicon_lab_points.npy")
    filename = maps_path+'/maps/willow-full.pgm'
    if make_graph:
        print('creating graph')
        create_graph_from_file(filename, nodes, n)
        print('done')
    
    # Create environment
    adj = np.load(package_path+'/src/robosar_task_allocator/custom_{}_graph.npy'.format(n))
    env = Environment(nodes[:n,:], adj, robots)

    print('routing')
    solver = TA_mTSP()
    solver.init(env)
    print('done')

    # plot
    with open('willow_map_data.pickle', 'rb') as f:
        gmap = pickle.load(f)
    gmap.plot()
    plt.plot(nodes[:n, 0], nodes[:n, 1], 'ko', zorder=100)
    for r in range(len(robots)):
        plt.plot(nodes[solver.tours[r], 0], nodes[solver.tours[r], 1], '-')
    plt.show()

    # Create transmitter object
    transmitter = TaskTxMoveBase(robots)

    rate = rospy.Rate(10) # 10hz
    rospy.loginfo('[Task_Alloc_mTSP] Buckle up! Running mTSP allocator!')


    while not rospy.is_shutdown():
        
        for robot in robots:
            status = transmitter.getStatus(robot.id)
            if(status==GoalStatus.SUCCEEDED and robot.next is not robot.prev):
                solver.reached(robot.id, robot.next)
                transmitter.setGoal(robot.id, pixels_to_m(env.nodes[robot.next]))
                print(env.visited)
            elif(status==GoalStatus.LOST):
                solver.assign(robot.id, robot.prev)
                transmitter.setGoal(robot.id, pixels_to_m(env.nodes[robot.next]))
        rate.sleep()

if __name__ == '__main__':
    try:
        mtsp_allocator()
    except rospy.ROSInterruptException:
        pass