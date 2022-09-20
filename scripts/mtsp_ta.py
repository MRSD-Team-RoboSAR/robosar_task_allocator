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
import numpy as np
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool, Int32
from robosar_messages.srv import *
from robosar_messages.msg import *
from task_commander import TaskCommander
from robosar_task_allocator.Environment import Environment
from robosar_task_allocator.TA import *
from robosar_task_allocator.task_transmitter.task_listener_robosar_control import TaskListenerRobosarControl
import robosar_task_allocator.utils as utils
import argparse
import tf


class MtspCommander(TaskCommander):

    def __init__(self):
        super().__init__()
        self._make_graph = rospy.get_param("make_graph")
        self._graph_name = rospy.get_param("graph_name")

    def refineNodes(self, r, nodes, map):
        """
        Get rid of tasks that are too close to obstacles
        """
        idx = []
        print(map.shape)
        for i in range(len(nodes)):
            x = nodes[i][1]
            y = nodes[i][0]
            if not self.checkCollision(x, y, r, map):
                idx.append(i)
        return nodes[idx]

    def checkCollision(self, x, y, r, map):
        for i in range(-r, r+1):
            for j in range(-r, r+1):
                if 0 <= x+i < map.shape[0] and 0 <= y+j < map.shape[1]:
                    if map[x + i][y + j] == 100:
                        return True
        return False

    def status_callback(self, msg):
        """
        Agent status callback
        """
        rospy.wait_for_service('/robosar_agent_bringup_node/agent_status')
        try:
            print("calling service")
            get_status = rospy.ServiceProxy(
                '/robosar_agent_bringup_node/agent_status', agent_status)
            resp1 = get_status()
            active_agents = resp1.agents_active
            for a in self.agent_active_status:
                self.agent_active_status[a] = False
            for a in active_agents:
                self.agent_active_status[a] = True
            print(self.agent_active_status)
            self.callback_triggered = True

        except rospy.ServiceException as e:
            print("Agent status service call failed: %s" % e)

    def get_agent_position(self, listener, scale, origin):
        """
        Get robot positions
        """
        robot_init = []
        init_order = []
        for name in self.agent_active_status:
            now = rospy.Time.now()
            listener.waitForTransform(
                'map', name + '/base_link', now, rospy.Duration(1.0))
            (trans, rot) = listener.lookupTransform(
                'map', name + '/base_link', now)
            robot_init.append(utils.m_to_pixels(
                [trans[0], trans[1]], scale, origin))
            init_order.append(name)
        robot_init = np.reshape(robot_init, (-1, 2))
        return robot_init, init_order

    def publish_image(self, image_pub):
        canvas = plt.gca().figure.canvas
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        image = data.reshape(canvas.get_width_height()[::-1] + (3,))
        br = CvBridge()
        image_pub.publish(br.cv2_to_imgmsg(image, "rgb8"))

    def execute(self):
        rospy.loginfo("Starting mTSP allocator")
        # Get active agents
        print("calling agent status service")
        rospy.wait_for_service('/robosar_agent_bringup_node/agent_status')
        try:
            get_status = rospy.ServiceProxy(
                '/robosar_agent_bringup_node/agent_status', agent_status)
            resp1 = get_status()
            active_agents = resp1.agents_active
            for a in active_agents:
                self.agent_active_status[a] = True
            print("{} agents active".format(len(self.agent_active_status)))
            assert len(self.agent_active_status) > 0
        except rospy.ServiceException as e:
            print("Agent status service call failed: %s" % e)
            raise Exception("Agent status service call failed")

        # Get map
        print("Waiting for map")
        map_msg = rospy.wait_for_message("/map", OccupancyGrid)
        print("Map received")

        # Get waypoints
        print("calling task generation service")
        rospy.wait_for_service('taskgen_getwaypts')
        scale = map_msg.info.resolution
        origin = [map_msg.info.origin.position.x,
                  map_msg.info.origin.position.y]
        print("map origin: {}".format(origin))
        data = np.reshape(
            map_msg.data, (map_msg.info.height, map_msg.info.width))
        free_space = 0
        for cell in map_msg.data:
            if 0 <= cell < 100:
                free_space += 1
        print("Map Area: {}".format(free_space * scale * scale))
        try:
            get_waypoints = rospy.ServiceProxy(
                'taskgen_getwaypts', taskgen_getwaypts)
            resp1 = get_waypoints(map_msg, 1, 20)
            nodes = resp1.waypoints
            nodes = np.reshape(nodes, (-1, 2))
        except rospy.ServiceException as e:
            print("Task generation service call failed: %s" % e)
            raise Exception("Task generation service call failed")
        nodes = self.refineNodes(3, nodes, data)
        # masking
        # idx = []
        # for i in range(len(nodes)):
        #     if 90 <= nodes[i][0] <= 565:
        #         idx.append(i)
        # nodes = nodes[idx]
        print("{} nodes received".format(len(nodes)))

        # get robot positions
        tflistener = tf.TransformListener()
        tflistener.waitForTransform('map', list(self.agent_active_status.keys())[
                                    0] + '/base_link', rospy.Time(), rospy.Duration(1.0))
        robot_init, init_order = self.get_agent_position(
            tflistener, scale, origin)
        nodes = np.vstack((robot_init, nodes))

        # Create graph
        n = nodes.shape[0]
        downsample = 1
        if self._make_graph:
            print('creating graph')
            adj = utils.create_graph_from_data(
                data, nodes, n, downsample, False)
            np.save(self.package_path + "/src/robosar_task_allocator/saved_graphs/" +
                    self._graph_name+"_points.npy", nodes)
            np.save(self.package_path + "/src/robosar_task_allocator/saved_graphs/" +
                    self._graph_name+"_graph.npy", adj)
            print('done')

        # Create environment
        if not self._make_graph:
            adj = np.load(self.package_path + '/src/robosar_task_allocator/saved_graphs/' +
                          self._graph_name+'_graph.npy')
        if len(nodes) != len(adj):
            raise Exception(
                "ERROR: length of nodes not equal to number in graph")
        env = Environment(nodes[:, :], adj)

        # Create robots
        for name in self.agent_active_status:
            env.add_robot(name, init_order.index(name))

        print('routing')
        solver = TA_mTSP()
        solver.init(env, 5)
        print('done')

        # plot
        image_pub = rospy.Publisher(
            'task_allocation_image', Image, queue_size=10)
        utils.plot_pgm_data(data)
        plt.plot(nodes[:n, 0], nodes[:n, 1], 'ko', zorder=100)
        for r in range(len(env.robots)):
            plt.plot(nodes[solver.tours[r], 0], nodes[solver.tours[r], 1], '-')
        self.publish_image(image_pub)

        # Create listener object
        listener = TaskListenerRobosarControl(env.robots)

        rate = rospy.Rate(10)  # 10hz
        rospy.loginfo('[Task_Alloc_mTSP] Buckle up! Running mTSP allocator!')

        # task publisher
        task_pub = rospy.Publisher(
            'task_allocation', task_allocation, queue_size=10)
        task_num_pub = rospy.Publisher('tasks_completed', Int32, queue_size=10)

        # agent status update subscriber
        rospy.Subscriber("/robosar_agent_bringup_node/status",
                         Bool, self.status_callback)

        while not rospy.is_shutdown():
            names = []
            starts = []
            goals = []

            # update fleet
            if self.callback_triggered:
                env.fleet_update(self.agent_active_status)
                print("replanning")
                solver.calculate_mtsp(False)
                print("done")
                self.callback_triggered = False

                # plot
                plt.clf()
                utils.plot_pgm_data(data)
                plt.plot(nodes[:, 0], nodes[:, 1], 'ko', zorder=100)
                for node in env.visited:
                    plt.plot(nodes[node, 0], nodes[node, 1], 'go', zorder=200)
                for r in range(len(env.robots)):
                    plt.plot(nodes[solver.tours[r], 0],
                             nodes[solver.tours[r], 1], '-')
                self.publish_image(image_pub)

            for robot in env.robots.values():
                status = listener.getStatus(robot.name)
                if status == 2 and not robot.done:
                    solver.reached(robot.name, robot.next)
                    task_num_pub.publish(len(env.visited)-env.num_robots)
                    if robot.next and robot.prev != robot.next:
                        listener.setBusyStatus(robot.name)
                        names.append(robot.name)
                        now = rospy.Time.now()
                        tflistener.waitForTransform(
                            'map', robot.name + '/base_link', now, rospy.Duration(1.0))
                        (trans, rot) = tflistener.lookupTransform(
                            'map', robot.name + '/base_link', now)
                        starts.append([trans[0], trans[1]])
                        goals.append(utils.pixels_to_m(
                            env.nodes[robot.next], scale, origin))
                        print(env.visited)
            if len(solver.env.visited) == len(nodes):
                print('FINISHED')
                break

            # publish tasks
            if names:
                print("publishing")
                task_msg = task_allocation()
                task_msg.id = names
                task_msg.startx = [s[0] for s in starts]
                task_msg.starty = [s[1] for s in starts]
                task_msg.goalx = [g[0] for g in goals]
                task_msg.goaly = [g[1] for g in goals]
                while task_pub.get_num_connections() == 0:
                    rospy.loginfo("Waiting for subscriber :")
                    rospy.sleep(1)
                task_pub.publish(task_msg)
                self.publish_image(image_pub)
                rospy.sleep(1)

            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('task_commander', log_level=rospy.DEBUG)

    try:
        tc = MtspCommander()
        tc.execute()
    except rospy.ROSInterruptException:
        pass
    plt.close('all')
