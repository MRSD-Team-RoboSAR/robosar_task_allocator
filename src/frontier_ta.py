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


class RobotInfo:
    def __init__(self, pos=[], n_frontiers=[], costs=[]) -> None:
        self.name = ""
        self.pos = pos
        self.n_frontiers = n_frontiers
        self.costs = costs
        self.utility = []


class FrontierAssignmentCommander(TaskCommander):
    def __init__(self):
        super().__init__()
        reassign_period = rospy.get_param("~reassign_period", 10.0)
        self.max_range = rospy.get_param("~discount_range", 3.0)
        self.beta = rospy.get_param("~beta", 2.0)
        timer = rospy.Timer(rospy.Duration(reassign_period), self.timer_flag_callback)
        self.rate = rospy.Rate(0.5)
        self.image_pub = rospy.Publisher("task_allocation_image", Image, queue_size=10)
        self.tflistener = tf.TransformListener()
        self.timer_flag = False
        self.frontiers = []
        self.map_data = None
        self.robot_info_dict = {}  # type: dict[str, RobotInfo]
        self.env = None  # type: UnknownEnvironment

    def timer_flag_callback(self, event=None):
        self.timer_flag = True

    def frontier_callback(self, msg):
        # TODO: add locks or just change to subscribe once per loop
        points = []
        for point in msg.points:
            points.append([point.x, point.y])
        self.frontiers = np.array(points)

    def get_agent_position(self):
        """
        Get robot positions
        """
        robot_pos = {}
        for name in self.agent_active_status:
            now = rospy.Time.now()
            self.tflistener.waitForTransform(
                "map", name + "/base_link", now, rospy.Duration(1.0)
            )
            (trans, rot) = self.tflistener.lookupTransform(
                "map", name + "/base_link", now
            )
            robot_pos[name] = [trans[0], trans[1]]
        return robot_pos

    def arr_m_to_pixels(self, arr, scale, origin):
        output = []
        for i in arr:
            output.append(utils.m_to_pixels([i[0], i[1]], scale, origin))
        return np.array(output)

    def rrt_path_cost_client(self, robot_x, robot_y, goal_x, goal_y):
        print("calling rrt_path_cost service")
        rospy.wait_for_service("/frontier_rrt_search_node/rrt_path_cost")
        try:
            rrt_path_service = rospy.ServiceProxy(
                "/frontier_rrt_search_node/rrt_path_cost", rrt_path_cost
            )
            resp1 = rrt_path_service(robot_x, robot_y, goal_x, goal_y)
            return resp1.cost
        except rospy.ServiceException as e:
            print("RRT path service call failed: %s" % e)

    def get_n_closest_frontiers(self, n, robot_pos):
        C = np.linalg.norm(self.frontiers - robot_pos, axis=1)
        min_node_list = np.argsort(C)
        return min_node_list[:n]

    def utility_discount_fn(self, dist):
        p = 0.0
        if dist < self.max_range:
            p = 1.0 - (dist / self.max_range)
        return p

    def reassign(self, solver):
        print("Reassigning")
        # get frontiers
        try:
            msg = rospy.wait_for_message(
                "/frontier_filter/filtered_frontiers", PointArray, timeout=5
            )
        except:
            print("no frontier messages received.")
            return False
        self.frontier_callback(msg)

        if len(self.frontiers) == 0:
            print("no frontiers received.")
            return False

        # get costs
        robot_pos = self.get_agent_position()
        for r, rp in robot_pos.items():
            # only calculate rrt cost for n euclidean closest frontiers
            n_frontiers = self.get_n_closest_frontiers(n=5, robot_pos=rp)
            costs = []
            for f in n_frontiers:
                # cost = self.rrt_path_cost_client(
                #     rp[0], rp[1], self.frontiers[f, 0], self.frontiers[f, 1]
                # )
                cost = np.linalg.norm(rp - self.frontiers[f])
                costs.append(cost)
            # update robot infos
            self.robot_info_dict[r].name = r
            self.robot_info_dict[r].pos = rp
            self.robot_info_dict[r].n_frontiers = n_frontiers
            self.robot_info_dict[r].costs = costs

        # update env
        self.env.update(self.frontiers, self.robot_info_dict)

        # get assignment
        names = []
        starts = []
        goals = []
        for name in self.agent_active_status:
            goal = solver.assign(name)
            if goal != -1:
                names.append(name)
                starts.append(robot_pos[name])
                goals.append(self.frontiers[goal])
                # update utility
                self.env.update_utility(goal, self.utility_discount_fn)

        if len(names) > 0:
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
            self.publish_image(self.image_pub)
            return True

        return False

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
        msg = rospy.wait_for_message(
            "/frontier_filter/filtered_frontiers", PointArray, timeout=None
        )
        self.frontier_callback(msg)

        # get robot position
        self.tflistener.waitForTransform(
            "map",
            list(self.agent_active_status.keys())[0] + "/base_link",
            rospy.Time(),
            rospy.Duration(1.0),
        )
        robot_pos = self.get_agent_position()

        # Create TA
        for name in self.agent_active_status:
            robot_info = RobotInfo(pos=robot_pos[name])
            self.robot_info_dict[name] = robot_info
        self.env = UnknownEnvironment(
            nodes=self.frontiers, robot_info=self.robot_info_dict
        )
        solver = TA_frontier_greedy(self.env, self.beta)

        # Create listener object
        listener = TaskListenerRobosarControl(
            [name for name in self.agent_active_status]
        )

        # plot
        utils.plot_pgm_data(self.map_data)
        pix_frontier = self.arr_m_to_pixels(self.frontiers, scale, origin)
        plt.plot(pix_frontier[:, 0], pix_frontier[:, 1], "go", zorder=100)
        for pos in robot_pos.values():
            pix_rob = utils.m_to_pixels(pos, scale, origin)
            plt.plot(pix_rob[0], pix_rob[1], "ko", zorder=100)
        self.publish_image(self.image_pub)

        # every time robot reaches a frontier, or every 5 seconds, reassign
        # get robot positions, updates env
        # cost: rrt path, utility: based on paper below
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1435481&tag=1
        # 1. greedy assignment
        # 2. hungarian assignment
        # 3. mtsp
        rospy.loginfo("Starting task allocator")

        for name in self.agent_active_status:
            listener.setBusyStatus(name)
        self.reassign(solver)
        self.timer_flag = False

        while not rospy.is_shutdown():
            if len(self.frontiers) == 0:
                rospy.loginfo("No more frontiers. Exiting.")
                break

            agent_reached = 0
            for name in self.agent_active_status:
                # TODO: change so that only one robot is reassigned when reached
                agent_reached = listener.getStatus(name)
                if agent_reached == 2:
                    # print("agent {} reached".format(name))
                    break

            if self.timer_flag:
                for name in self.agent_active_status:
                    listener.setBusyStatus(name)
                self.reassign(solver)
                self.timer_flag = False

            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node("task_commander", anonymous=False, log_level=rospy.INFO)

    try:
        tc = FrontierAssignmentCommander()
        tc.execute()
    except rospy.ROSInterruptException:
        pass
    plt.close("all")
