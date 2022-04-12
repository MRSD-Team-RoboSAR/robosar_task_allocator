#!/usr/bin/env python3

# Created by Indraneel and Rachel on 01/03/22
"""
Task Allocation node: main script to be executed.
- Gets map Occupancy Grid as a message
- Gets tasks by calling task generation service
- Creates graph of tasks in map
- Gets active agents by calling status service
- Solves task allocation problem
- Assigns tasks to robots
"""

import rospy
from robosar_task_allocator.Environment import Environment
from robosar_task_allocator.Robot import Robot
from robosar_task_allocator.TA import *
from robosar_task_allocator.task_transmitter.task_listener_robosar_control import TaskListenerRobosarControl
import numpy as np
import pickle
import rospkg
from actionlib_msgs.msg import GoalStatus
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool
from robosar_messages.srv import *
from robosar_messages.msg import *
from robosar_task_allocator.generate_graph import occupancy_map_8n
from robosar_task_allocator.generate_graph.gridmap import OccupancyGridMap
import robosar_task_allocator.utils as utils
import time
import tf

rospack = rospkg.RosPack()
maps_path = rospack.get_path('robosar_task_generator')
package_path = rospack.get_path('robosar_task_allocator')
agent_active_status = {}


def status_callback(msg):
    rospy.wait_for_service('/robosar_agent_bringup_node/agent_status')
    try:
        print("calling service")
        get_status = rospy.ServiceProxy('/robosar_agent_bringup_node/agent_status', agent_status)
        resp1 = get_status()
        active_agents = resp1.agents_active
        for a in agent_active_status:
            agent_active_status[a] = False
        for a in active_agents:
            agent_active_status[int(a[-1])] = True
        # update fleet
        env.fleet_update(agent_active_status)

    except rospy.ServiceException as e:
        ROS_ERROR("Agent status service call failed: %s" % e)


def mtsp_allocator():
    rospy.init_node('task_allocator_mtsp', anonymous=True)

    # Get active agents
    print("calling agent status service")
    # rospy.Subscriber("/robosar_agent_bringup_node/status", Bool, status_callback)
    rospy.wait_for_service('/robosar_agent_bringup_node/agent_status')
    try:
        get_status = rospy.ServiceProxy('/robosar_agent_bringup_node/agent_status', agent_status)
        resp1 = get_status()
        active_agents = resp1.agents_active
        for a in active_agents:
            agent_active_status[a] = True
        print("{} agents active".format(len(agent_active_status)))
        assert len(agent_active_status) > 0
    except rospy.ServiceException as e:
        ROS_ERROR("Agent status service call failed: %s" % e)
        raise Exception("Agent status service call failed")

    # Get map
    print("Waiting for map")
    map_msg = rospy.wait_for_message("/map", OccupancyGrid)
    print("Map received")

    # Get waypoints
    print("calling task generation service")
    rospy.wait_for_service('taskgen_getwaypts')
    scale = map_msg.info.resolution
    origin = [map_msg.info.origin.position.x, map_msg.info.origin.position.y]
    print("map origin: {}".format(origin))
    data = np.reshape(map_msg.data, (map_msg.info.height, map_msg.info.width))
    try:
        get_waypoints = rospy.ServiceProxy('taskgen_getwaypts', taskgen_getwaypts)
        resp1 = get_waypoints(map_msg, 1, 20)
        nodes = resp1.waypoints
        nodes = np.reshape(nodes, (-1, 2))
        # np.save(package_path+"/src/robosar_task_allocator/saved_graphs/custom_{}_points.npy".format(nodes.shape[0]), nodes)
    except rospy.ServiceException as e:
        ROS_ERROR("Task generation service call failed: %s" % e)
        raise Exception("Task generation service call failed")

    listener = tf.TransformListener()
    robot_init = []
    init_order = []
    listener.waitForTransform('map', list(agent_active_status.keys())[0] + '/base_link', rospy.Time(), rospy.Duration(1.0))
    for name in agent_active_status:
        now = rospy.Time.now()
        listener.waitForTransform('map', name + '/base_link', now, rospy.Duration(1.0))
        (trans, rot) = listener.lookupTransform('map', name + '/base_link', now)
        robot_init.append(utils.m_to_pixels([trans[0], trans[1]], scale, origin))
        init_order.append(name)
    robot_init = np.reshape(robot_init, (-1, 2))
    nodes = np.vstack((robot_init, nodes))

    # Create graph
    n = nodes.shape[0]
    downsample = 5
    # filename = maps_path+'/maps/willow-full.pgm'
    print('creating graph')
    adj = utils.create_graph_from_data(data, nodes, n, downsample, False)
    print('done')

    # Create environment
    # adj = np.load(package_path+'/saved_graphs/custom_{}_graph.npy'.format(n))
    env = Environment(nodes[:n, :], adj)

    # Create robots
    for name in agent_active_status:
        env.add_robot(int(name[-1]), name, init_order.index(name))

    print('routing')
    solver = TA_mTSP()
    solver.init(env, 5)
    print('done')

    # plot
    utils.plot_pgm_data(data)
    plt.plot(nodes[:n, 0], nodes[:n, 1], 'ko', zorder=100)
    for r in range(len(env.robots)):
        plt.plot(nodes[solver.tours[r], 0], nodes[solver.tours[r], 1], '-')
    plt.show()

    # Create listener object
    listener = TaskListenerRobosarControl(env.robots)

    rate = rospy.Rate(10)  # 10hz
    rospy.loginfo('[Task_Alloc_mTSP] Buckle up! Running mTSP allocator!')

    # task publisher
    task_pub = rospy.Publisher('task_allocation', task_allocation, queue_size=10)

    finished = [0 for i in robots]
    names = []
    starts = []
    goals = []

    while not rospy.is_shutdown():

        for robot in env.robots.values():
            status = listener.getStatus(robot.name)
            if status == 2 and (robot.next != robot.prev):
                solver.reached(robot.id, robot.next)
                finished[env.id_dict[robot.id]] = 1
                if robot.next != robot.prev:
                    listener.setBusyStatus(robot.name)
                    names.append(robot.name)
                    starts.append(utils.pixels_to_m(env.nodes[robot.prev], scale, origin))
                    goals.append(utils.pixels_to_m(env.nodes[robot.next], scale, origin))
                    print(env.visited)
        if len(solver.env.visited) == len(nodes):
            print('finished')
            break

        # publish tasks
        if sum(finished) == len(env.robots):
            print("publishing")
            task_msg = task_allocation()
            task_msg.id = names
            task_msg.startx = [s[0] for s in starts]
            task_msg.starty = [s[1] for s in starts]
            task_msg.goalx = [g[0] for g in goals]
            task_msg.goaly = [g[1] for g in goals]
            time.sleep(1)
            task_pub.publish(task_msg)
            time.sleep(1)
            for robot in env.robots.values():
                if robot.next != robot.prev:
                    finished[env.id_dict[robot.id]] = 0
            names = []
            starts = []
            goals = []

        rate.sleep()


if __name__ == '__main__':
    try:
        mtsp_allocator()
    except rospy.ROSInterruptException:
        pass