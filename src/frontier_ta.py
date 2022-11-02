#!/usr/bin/env python3

from queue import Queue

import numpy as np
import rospy
import skimage.measure
import tf
from robosar_messages.msg import *
from robosar_messages.srv import *
from sensor_msgs.msg import Image

import task_allocator.utils as utils
from generate_graph.gridmap import OccupancyGridMap
from generate_graph.a_star import a_star
from task_allocator.Environment import UnknownEnvironment
from task_allocator.TA import *
from task_commander import TaskCommander
from task_transmitter.task_listener_robosar_control import TaskListenerRobosarControl


class RobotInfo:
    def __init__(self, pos=[], n_frontiers=[], costs=[]) -> None:
        self.name = ""
        self.pos = pos
        self.curr = None
        self.prev = None
        self.n_frontiers = n_frontiers
        self.costs = costs
        self.utility = []
        self.obstacle_costs = []
        self.proximity_bonus = []


class FrontierAssignmentCommander(TaskCommander):
    def __init__(self):
        super().__init__()
        reassign_period = rospy.get_param("~reassign_period", 15.0)
        self.utility_range = rospy.get_param("~utility_range", 2.0)
        self.beta = rospy.get_param("~beta", 5.0)
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

    def arr_m_to_pixels(self, arr):
        output = []
        for i in arr:
            output.append(utils.m_to_pixels([i[0], i[1]], self.scale, self.origin))
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
        if dist < self.utility_range:
            p = 1.0 - (float(dist) / self.utility_range)
        return p

    def obstacle_cost(self, node, r):
        pix_node = utils.m_to_pixels(node, self.scale, self.origin)
        pix_range = int(r / self.scale)
        x_min = int(max(pix_node[0] - pix_range, 0))
        x_max = int(min(pix_node[0] + pix_range, self.map_data.shape[1]))
        y_min = int(max(pix_node[1] - pix_range, 0))
        y_max = int(min(pix_node[1] + pix_range, self.map_data.shape[0]))
        min_pix_to_occ = 2.0 * pix_range
        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                if self.map_data[j][i] == 100:
                    min_pix_to_occ = min(
                        min_pix_to_occ,
                        np.linalg.norm([i - pix_node[0], j - pix_node[1]]),
                    )
        pc = 0.0
        min_range_to_occ = min_pix_to_occ * self.scale
        if min_range_to_occ < r:
            pc = 1.0 - (min_range_to_occ / r)
        return pc

    def a_star_cost(self, start, goal, gmap, downsample):
        start = utils.m_to_pixels(start, self.scale, self.origin)
        goal = utils.m_to_pixels(goal, self.scale, self.origin)
        start_flip = [start[1] / downsample, start[0] / downsample]
        goal_flip = [goal[1] / downsample, goal[0] / downsample]
        _, _, cost = a_star(start_flip, goal_flip, gmap, movement="8N")
        return cost

    def proximity_bonus(self, node, prev, r):
        if prev is None:
            return 0.0
        dist = np.linalg.norm([node[0] - prev[0], node[1] - prev[1]])
        p = 0.0
        if dist < r:
            p = 1.0 - (dist / r)
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

        # get map
        _, self.map_data, self.scale, self.origin = self.get_map_info()

        # get costs
        downsample = 1
        robot_pos = self.get_agent_position()
        # resized_image = skimage.measure.block_reduce(self.map_data, (1, 1), np.max)
        gmap = OccupancyGridMap.from_data(self.map_data)
        for r, rp in robot_pos.items():
            # only calculate rrt cost for n euclidean closest frontiers
            n_frontiers = self.get_n_closest_frontiers(n=5, robot_pos=rp)
            costs = []
            obstacle_costs = []
            prox_bonus = []
            print("robot {} calcs".format(r))
            for f in n_frontiers:
                # cost = self.rrt_path_cost_client(
                #     rp[0], rp[1], self.frontiers[f, 0], self.frontiers[f, 1]
                # )
                cost = np.linalg.norm(rp - self.frontiers[f])
                # cost = self.a_star_cost(rp, self.frontiers[f], gmap, downsample)
                pc = self.obstacle_cost(self.frontiers[f], 1.0)
                pb = self.proximity_bonus(
                    self.frontiers[f], self.robot_info_dict[r].prev, 2.0
                )
                costs.append(cost)
                obstacle_costs.append(pc)
                prox_bonus.append(pb)
            print("done")
            # update robot infos
            self.robot_info_dict[r].name = r
            self.robot_info_dict[r].prev = self.robot_info_dict[r].curr
            self.robot_info_dict[r].pos = rp
            self.robot_info_dict[r].n_frontiers = n_frontiers
            self.robot_info_dict[r].costs = np.array(costs)
            self.robot_info_dict[r].obstacle_costs = np.array(obstacle_costs)
            self.robot_info_dict[r].proximity_bonus = np.array(prox_bonus)

        # update env
        self.env.update(self.frontiers, self.robot_info_dict)

        # get assignment
        names = []
        starts = []
        goals = []
        for name in self.agent_active_status:
            goal = solver.assign(name)
            if goal != -1:
                self.robot_info_dict[r].curr = self.frontiers[goal]
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
            utils.plot_pgm_data(self.map_data)
            pix_frontier = self.arr_m_to_pixels(self.frontiers)
            plt.plot(pix_frontier[:, 0], pix_frontier[:, 1], "go", zorder=100)
            for i in range(len(names)):
                pix_rob = utils.m_to_pixels(starts[i], self.scale, self.origin)
                pix_goal = utils.m_to_pixels(goals[i], self.scale, self.origin)
                plt.plot(pix_rob[0], pix_rob[1], "ro", zorder=100)
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
        map_msg, self.map_data, self.scale, self.origin = self.get_map_info()

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
        pix_frontier = self.arr_m_to_pixels(self.frontiers)
        plt.plot(pix_frontier[:, 0], pix_frontier[:, 1], "go", zorder=100)
        for pos in robot_pos.values():
            pix_rob = utils.m_to_pixels(pos, self.scale, self.origin)
            plt.plot(pix_rob[0], pix_rob[1], "ro", zorder=100)
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

        no_frontiers_count = 0
        while not rospy.is_shutdown():
            if len(self.frontiers) == 0:
                no_frontiers_count += 1
                if no_frontiers_count > 40:
                    rospy.loginfo("No more frontiers. Exiting.")
                    break
                continue

            no_frontiers_count = 0
            agent_reached = 0
            for name in self.agent_active_status:
                agent_reached = listener.getStatus(name)
                if agent_reached == 2:
                    # print("agent {} reached".format(name))
                    break

            if self.timer_flag or agent_reached == 2:
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
