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
import robosar_task_allocator.utils as utils


rospack = rospkg.RosPack()
maps_path = rospack.get_path('robosar_task_generator')
package_path = rospack.get_path('robosar_task_allocator')

def mtsp_allocator():
    rospy.init_node('task_allocator_mtsp', anonymous=True)

    # Get waypoints client
    map_msg = rospy.wait_for_message("/map", OccupancyGrid)
    rospy.wait_for_service('taskgen_getwaypts')
    scale = map_msg.info.resolution
    origin = [map_msg.info.origin.position.x, map_msg.info.origin.position.y]
    data = np.reshape(map_msg.data, (map_msg.info.height, map_msg.info.width))
    print(data)
    try:
        print("calling service")
        get_waypoints = rospy.ServiceProxy('taskgen_getwaypts', taskgen_getwaypts)
        resp1 = get_waypoints(map_msg, 1, 20)
        nodes = resp1.waypoints
        nodes = np.reshape(nodes, (-1,2))
        # np.save(package_path+"/src/robosar_task_allocator/custom_{}_points.npy".format(nodes.shape[0]), nodes)
        print(nodes)
    except rospy.ServiceException as e:
        ROS_ERROR("Service call failed: %s" % e)


    # Create graph
    n = nodes.shape[0]
    downsample = 5
    make_graph = True
    filename = maps_path+'/maps/willow-full.pgm'
    if make_graph:
        print('creating graph')
        adj = utils.create_graph_from_file(filename, nodes, n, downsample, False)
        print('done')

    # Create robots
    robot0 = Robot(0, nodes[0], 0)
    robot1 = Robot(1, nodes[0], 0)
    robot2 = Robot(2, nodes[0], 0)
    robots = [robot0, robot1, robot2]
    
    # Create environment
    # adj = np.load(package_path+'/custom_{}_graph.npy'.format(n))
    env = Environment(nodes[:n,:], adj, robots)

    print('routing')
    solver = TA_mTSP()
    solver.init(env)
    print('done')

    # plot
    # utils.plot_pgm(filename)
    utils.plot_pgm_data(data)
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
                transmitter.setGoal(robot.id, utils.pixels_to_m(env.nodes[robot.next], scale, origin))
                print(env.visited)
            elif(status==GoalStatus.LOST):
                solver.assign(robot.id, robot.prev)
                transmitter.setGoal(robot.id, utils.pixels_to_m(env.nodes[robot.next], scale, origin))
        if len(solver.env.visited) == len(nodes):
            print('finished')
            break
        rate.sleep()

if __name__ == '__main__':
    try:
        mtsp_allocator()
    except rospy.ROSInterruptException:
        pass