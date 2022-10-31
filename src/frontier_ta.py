#!/usr/bin/env python3

import numpy as np
import rospy
import tf
from robosar_messages.msg import *
from robosar_messages.srv import *
from sensor_msgs.msg import Image

import task_allocator.utils as utils
from task_allocator.Environment import UnknownEnvironment
from task_allocator.TA import *
from task_commander import TaskCommander
from task_transmitter.task_listener_robosar_control import TaskListenerRobosarControl


class FrontierAssignmentCommander(TaskCommander):
    def __init__(self):
        super().__init__()
        rateHz = rospy.get_param("~rate", 1.0 / 10)
        self.rate = rospy.Rate(rateHz)
        self.frontiers = []
        self.map_data = None
        rospy.Subscriber(
            "/frontier_filter/filtered_frontiers", PointArray, self.frontier_callback
        )

    def frontier_callback(self, msg):
        points = []
        for point in msg.points:
            points.append([point.x, point.y])
        self.frontiers = np.array(points)

    def get_agent_position(self, listener, scale, origin):
        """
        Get robot positions
        """
        robot_pos = {}
        for name in self.agent_active_status:
            now = rospy.Time.now()
            listener.waitForTransform(
                "map", name + "/base_link", now, rospy.Duration(1.0)
            )
            (trans, rot) = listener.lookupTransform("map", name + "/base_link", now)
            robot_pos[name] = [trans[0], trans[1]]
        return robot_pos

    def arr_m_to_pixels(self, arr, scale, origin):
        output = []
        for i in arr:
            output.append(utils.m_to_pixels([i[0], i[1]], scale, origin))
        return np.array(output)

    def rrt_path_cost_client(robot_x, robot_y, goal_x, goal_y):
        rospy.wait_for_service("rrt_path_cost")
        try:
            rrt_path_service = rospy.ServiceProxy("rrt_path_cost", rrt_path_cost)
            resp1 = rrt_path_service(robot_x, robot_y, goal_x, goal_y)
            return resp1.cost
        except rospy.ServiceException as e:
            print("RRT path service call failed: %s" % e)

    def execute(self):
        """
        Uses frontiers as tasks
        """
        # Get active agents
        self.get_active_agents()

        # Get map
        map_msg, self.map_data, scale, origin = self.get_map_info()

        # Get frontiers
        rospy.loginfo("Waiting for frontiers")
        while len(self.frontiers) < 1:
            rospy.sleep(0.1)
            pass

        # get robot positions
        tflistener = tf.TransformListener()
        tflistener.waitForTransform(
            "map",
            list(self.agent_active_status.keys())[0] + "/base_link",
            rospy.Time(),
            rospy.Duration(1.0),
        )
        robot_pos = self.get_agent_position(tflistener, scale, origin)

        # Create env
        env = UnknownEnvironment(nodes=self.frontiers, scale=scale, origin=origin)
        for name in self.agent_active_status:
            env.add_robot(name, robot_pos[name])

        # Create TA
        solver = TA_frontier_greedy(env)

        # plot
        image_pub = rospy.Publisher("task_allocation_image", Image, queue_size=10)
        utils.plot_pgm_data(self.map_data)
        plt.plot(self.frontiers[:, 0], self.frontiers[:, 1], "go", zorder=100)
        for pos in robot_pos.values():
            plt.plot(pos[0], pos[1], "ko", zorder=100)
        self.publish_image(image_pub)

        # every time robot reaches a frontier, or every 5 seconds, reassign
        # get robot positions, updates env
        # cost: astar path, utility: based on paper below
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1435481&tag=1
        # 1. greedy assignment
        # 2. hungarian assignment
        # 3. mtsp
        rospy.loginfo("Starting task allocator")
        while not rospy.is_shutdown():
            robot_pos = self.get_agent_position(tflistener, scale, origin)
            for rp in robot_pos.values():
                cost = self.rrt_path_cost_client(
                    rp[0], rp[1], self.frontiers[0, 0], self.frontiers[0, 1]
                )
                print(cost)
            env.update(self.frontiers, robot_pos)
            names, starts, goals = solver.assign()

            # publish tasks
            rospy.loginfo("publishing")
            task_msg = task_allocation()
            task_msg.id = [n for n in names]
            task_msg.startx = [s[0] for s in starts]
            task_msg.starty = [s[1] for s in starts]
            task_msg.goalx = [g[0] for g in goals]
            task_msg.goaly = [g[1] for g in goals]
            while self.task_pub.get_num_connections() == 0:
                rospy.loginfo("Waiting for subscriber to task topic:")
                rospy.sleep(1)
            self.task_pub.publish(task_msg)

            # plot
            plt.clf()
            _, self.map_data, scale, origin = self.get_map_info()
            utils.plot_pgm_data(self.map_data)
            pix_frontier = self.arr_m_to_pixels(self.frontiers, scale, origin)
            plt.plot(pix_frontier[:, 0], pix_frontier[:, 1], "go", zorder=100)
            for i in range(len(names)):
                pix_rob = utils.m_to_pixels(starts[i], scale, origin)
                pix_goal = utils.m_to_pixels(goals[i], scale, origin)
                plt.plot(pix_rob[0], pix_rob[1], "ko", zorder=100)
                plt.plot(
                    [pix_rob[0], pix_goal[0]],
                    [pix_rob[1], pix_goal[1]],
                    "k-",
                    zorder=90,
                )
            self.publish_image(image_pub)

            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node("task_commander", anonymous=False, log_level=rospy.INFO)

    try:
        tc = FrontierAssignmentCommander()
        tc.execute()
    except rospy.ROSInterruptException:
        pass
    plt.close("all")
