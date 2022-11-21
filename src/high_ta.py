#!/usr/bin/env python3

import numpy as np
import rospy
import skimage.measure
import task_allocator.utils as utils
from frontier_ta import FrontierAssignmentCommander, RobotInfo
from generate_graph.gridmap import OccupancyGridMap
from std_msgs.msg import Float32
from task_allocator.CostCalculator import CostCalculator
from task_allocator.Environment import UnknownEnvironment
from task_allocator.TA import *
from task_transmitter.task_listener_robosar_control import TaskListenerRobosarControl
from collections import deque
from nav_msgs.msg import OccupancyGrid

from robosar_messages.msg import *
from robosar_messages.srv import *


class HIGHAssignmentCommander(FrontierAssignmentCommander):
    def __init__(self):
        super().__init__()
        self.fronters_info_gain = []
        self.geofence = rospy.get_param("/robosar_task_graph_node/geofence", [-0.5, 12.0, -10.0, 2.0])
        self.tot_area = (self.geofence[1] - self.geofence[0]) * (
            self.geofence[3] - self.geofence[2]
        )
        self.covered_area = 0.0
        self.area_explored_pub = rospy.Publisher("/percent_completed", Float32, queue_size=1)
        self.info_map_pub = rospy.Publisher("/information_map", OccupancyGrid, queue_size=1)
        self.cost_calculator = CostCalculator(self.utility_range, self.downsample)

    def frontier_callback(self, msg):
        points = []
        for point in msg.points:
            points.append([point.x, point.y])
        self.frontiers = np.array(points)
        self.fronters_info_gain = np.array(msg.infoGain)
        if len(self.fronters_info_gain) > 0:
            self.fronters_info_gain = self.fronters_info_gain / np.max(self.fronters_info_gain)

    def task_graph_client(self):
        task_ids = []
        points = []
        task_types = []
        info_gains = []
        # print("calling task graph getter service")
        rospy.wait_for_service("/robosar_task_generator/task_graph_getter")
        try:
            task_graph_getter_service = rospy.ServiceProxy(
                "/robosar_task_generator/task_graph_getter", task_graph_getter
            )
            resp1 = task_graph_getter_service()
            task_ids = resp1.task_ids
            points = resp1.points
            task_types = resp1.task_types
            info_gains = resp1.info_gains
        except rospy.ServiceException as e:
            print("task graph getter service call failed: %s" % e)
            return False

        if len(task_ids) == 0:
            return False
        for i in range(len(task_ids)):
            if task_types[i] == resp1.COVERAGE:
                self.coverage_tasks[task_ids[i]] = [points[i].x, points[i].y, info_gains[i]]
        return True

    def calculate_e2_weights(self):
        percent_explored = min(self.covered_area / float(self.tot_area), 1.0)
        explore_weight = 1.0 - percent_explored**2
        return explore_weight

    def prepare_high_env(self):
        # update env
        self.env.exploration_weight = self.calculate_e2_weights()
        assert len(self.frontiers) == len(self.fronters_info_gain)
        self.env.update_tasks(
            self.frontiers, self.coverage_tasks, self.fronters_info_gain
        )
        return True

    def reassign(self, avail_robots, solver):
        self.env.update_robot_info(self.robot_info_dict)

        # get assignment
        names, goal_tasks = solver.assign(avail_robots, self.cost_calculator)
        starts = []
        goals = []
        goal_types = []
        goal_ids = []
        if len(names) > 0:
            for i, name in enumerate(names):
                goal = goal_tasks[i]
                self.robot_info_dict[name].curr = goal
                task_type = 1
                if goal.task_type == "coverage":
                    task_type = 2
                starts.append(self.robot_info_dict[name].pos)
                goals.append(goal.pos)
                goal_types.append(task_type)
                goal_ids.append(goal.id)

        return names, starts, goals, goal_types, goal_ids

    def execute(self):
        """
        Uses frontiers as tasks
        """
        # Get active agents
        self.get_active_agents()

        # Get map
        (
            self.map_msg,
            self.map_data,
            self.scale,
            self.origin,
            self.covered_area,
        ) = self.get_map_info()

        # Get frontiers
        rospy.loginfo("Waiting for frontiers")
        while len(self.frontiers) == 0:
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
            robot_info = RobotInfo(name=name, pos=robot_pos[name])
            self.robot_info_dict[name] = robot_info
        self.env = UnknownEnvironment(
            frontier_tasks=self.frontiers, robot_info=self.robot_info_dict
        )
        solver = TA_HIGH(self.env)

        # Create listener object
        task_listener = TaskListenerRobosarControl(
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

        rospy.loginfo("Starting frontier task allocator")

        self.prepare_env()
        self.prepare_high_env()
        avail_robots = [name for name in self.agent_active_status]
        names, starts, goals, goal_types, goal_ids = self.reassign(avail_robots, solver)
        for name in names:
            task_listener.setBusyStatus(name)
        if len(names) > 0:
            self.publish_visualize(names, starts, goals, goal_types, goal_ids)
            pe = Float32()
            pe.data = min(self.covered_area / self.tot_area, 1.0)
            self.area_explored_pub.publish(pe)
        self.timer_flag = False

        while not rospy.is_shutdown():
            # get frontiers
            if not self.prepare_env() or not self.prepare_high_env():
                continue

            agent_reached = {name: False for name in self.agent_active_status}
            agent_reached_flag = False
            for name in self.agent_active_status:
                status = task_listener.getStatus(name)
                curr_goal_id = task_listener.getGoalID(name)
                if status == 2 and solver.reached(self.robot_info_dict[name], curr_goal_id):
                    agent_reached[name] = True
                    agent_reached_flag = True

            if self.timer_flag or agent_reached_flag:
                unvisited_coverage = self.env.get_unvisited_coverage_tasks_pos()
                visited_coverage = self.env.get_visited_coverage_tasks_pos()

                # reassign tasks
                # TODO: aggregate valid robots and reassign
                avail_robots = []
                for name in self.agent_active_status:
                    if (
                        self.robot_info_dict[name].curr
                        and self.robot_info_dict[name].curr.task_type == "coverage"
                        and not agent_reached[name]
                    ):
                        self.env.update_utility(
                            self.robot_info_dict[name].curr, self.cost_calculator.utility_discount_fn
                        )
                        continue
                    avail_robots.append(name)
                names, starts, goals, goal_types, goal_ids = self.reassign(avail_robots, solver)
                for name in names:
                    task_listener.setBusyStatus(name)
                self.send_visited_to_task_graph()
                self.timer_flag = False

                if len(self.env.available_tasks) > 0:
                    self.visualise_utility()

                if len(names) > 0:
                    self.publish_visualize(
                        names,
                        starts,
                        goals,
                        goal_types,
                        goal_ids,
                        unvisited_coverage,
                        visited_coverage,
                    )
                    pe = Float32()
                    pe.data = min(self.covered_area / self.tot_area, 1.0)
                    self.area_explored_pub.publish(pe)

            self.rate.sleep()

    def inflate_cell_with_bfs(self, information_map, pos, info_gain):
        """
        Inflate cell with BFS
        """
        # local variables
        cost_scaling_factor = 0.05
        neighbours = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]]
        inflation_radius_cells = 100

        # Get pixel position
        pixel_pos = utils.m_to_pixels(pos, self.scale, self.origin)

        queue = deque()
        queue.append((pixel_pos[0], pixel_pos[1]))
        visited = set()
        visited.add((pixel_pos[0], pixel_pos[1]))
        while len(queue) > 0:
            curr = queue.popleft()

            # Get distance from center
            dist = abs(curr[0] - pixel_pos[0]) + abs(curr[1] - pixel_pos[1])

            # check if distance is within inflation radius
            if dist>inflation_radius_cells:
                break

            factor = np.exp(-1.0 * cost_scaling_factor *float(dist))
            information = factor * info_gain
            # Update information map
            information_map[curr[1]][curr[0]] += information

            for neighbor in neighbours:
                new_pos = (curr[0] + neighbor[0], curr[1] + neighbor[1])
                if new_pos not in visited and new_pos[0] >= 0 and new_pos[1] >= 0 and new_pos[0] < information_map.shape[1] and new_pos[1] < information_map.shape[0]:
                    visited.add(new_pos)
                    queue.append(new_pos)


    def visualise_utility(self):
        
        info_offset = 5.0 

         # Create information map
        info_map = np.zeros(self.map_data.shape)

        # Iterate through all available tasks
        for task in self.env.available_tasks:
            info_utility = 1
            # Get info gain 
            info_gain = task.info_gain
            # get e2 weight
            e2_weight = 1
            if task.task_type == "frontier":
                e2_weight = self.env.exploration_weight
            elif task.task_type == "coverage":
                e2_weight = 1-self.env.exploration_weight
            
            # Total utility
            info_utility = info_gain*e2_weight
            # Inflate the position with a gaussian
            self.inflate_cell_with_bfs(info_map, task.pos, info_utility+info_offset)

        # Scale the map from 0 to 100
        if(np.max(info_map) - np.min(info_map)) > 0:
            info_map = (info_map - np.min(info_map)) / (np.max(info_map) - np.min(info_map)) * 100
        
        info_map = info_map.astype(np.uint8)
        
        # Publish the map
        cost_map = OccupancyGrid()
        cost_map.header.frame_id = "map"
        cost_map.header.stamp = rospy.Time.now()
        cost_map.info.resolution = self.scale
        cost_map.info.width = self.map_msg.info.width
        cost_map.info.height = self.map_msg.info.height
        cost_map.info.origin.position.x = self.origin[0]
        cost_map.info.origin.position.y = self.origin[1]
        cost_map.info.origin.position.z = 0
        cost_map.info.origin.orientation.x = 0
        cost_map.info.origin.orientation.y = 0
        cost_map.info.origin.orientation.z = 0
        cost_map.info.origin.orientation.w = 1
        cost_map.data = info_map.flatten().tolist()
        self.info_map_pub.publish(cost_map)


if __name__ == "__main__":
    rospy.init_node("task_commander", anonymous=False, log_level=rospy.INFO)

    try:
        tc = HIGHAssignmentCommander()
        tc.execute()
    except rospy.ROSInterruptException:
        pass
    plt.close("all")
