#!/usr/bin/env python3

import numpy as np
import rospy
from robosar_messages.msg import *
from robosar_messages.srv import *

import task_allocator.utils as utils
from frontier_ta import FrontierAssignmentCommander, RobotInfo
from task_allocator.Environment import UnknownEnvironment
from task_allocator.TA import *
from task_transmitter.task_listener_robosar_control import TaskListenerRobosarControl


class HeterogeneousAssignmentCommander(FrontierAssignmentCommander):
    def __init__(self):
        super().__init__()

    def assign_roles(self):
        ratio = 0.5
        num_robots = len(self.robot_info_dict)
        num_frontier_robots = np.ceil(num_robots * ratio)
        it = 0
        for robot_info in self.robot_info_dict.values():
            if it < num_frontier_robots:
                robot_info.role = "frontier"
            else:
                robot_info.role = "coverage"
            rospy.loginfo("Assigned {} as {} role.".format(robot_info.name, robot_info.role))
            it += 1

    def prepare_costs(self, robot_id):
        rp = self.robot_info_dict[robot_id].pos
        role = self.robot_info_dict[robot_id].role
        print(role)
        # only calculate rrt cost for n euclidean closest frontiers
        n_tasks = self.env.get_n_closest_tasks(n=self.n, robot_pos=rp, goal_type=role)
        costs = []
        obstacle_costs = []
        # prox_bonus = []
        for task in n_tasks:
            task_pos = task.pos
            # A* path
            cost = self.a_star_cost(rp, task_pos)
            pc = self.obstacle_cost(task_pos, 1.0)
            costs.append(cost)
            obstacle_costs.append(pc)
            # prox_bonus.append(pb)
        # update robot infos
        self.robot_info_dict[robot_id].prev = self.robot_info_dict[robot_id].curr
        self.robot_info_dict[robot_id].n_tasks = n_tasks
        self.robot_info_dict[robot_id].costs = np.array(costs)
        self.robot_info_dict[robot_id].obstacle_costs = np.array(obstacle_costs)

    def reassign(self, name, solver):
        # get costs
        self.prepare_costs(name)
        self.env.update_robot_info(self.robot_info_dict)

        # get assignment
        goal = solver.assign(name)
        if goal is not None:
            self.robot_info_dict[name].curr = goal
            # update utility
            self.env.update_utility(goal, self.utility_discount_fn)
            task_type = 1
            if goal.task_type == "coverage":
                task_type = 2
            return self.robot_info_dict[name].pos, goal.pos, task_type

        return [], [], None

    def execute(self):
        """
        Uses frontiers as tasks
        """
        # Get active agents
        self.get_active_agents()

        # Get map
        self.map_msg, self.map_data, self.scale, self.origin = self.get_map_info()

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
        solver = TA_frontier_greedy(self.env)

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

        # reassign roles
        self.assign_roles()

        self.task_graph_client()
        self.prepare_env()
        names = []
        starts = []
        goals = []
        goal_types = []
        for name in self.agent_active_status:
            start, goal, goal_type = self.reassign(name, solver)
            if len(start) > 0:
                names.append(name)
                starts.append(start)
                goals.append(goal)
                goal_types.append(goal_type)
                task_listener.setBusyStatus(name)
        if len(names) > 0:
            self.publish_visualize(names, starts, goals, goal_types)
        self.timer_flag = False

        while not rospy.is_shutdown():
            # get frontiers
            if not self.prepare_env():
                continue

            agent_reached = {name: False for name in self.agent_active_status}
            agent_reached_flag = False
            for name in self.agent_active_status:
                status = task_listener.getStatus(name)
                if status == 2:
                    agent_reached[name] = True
                    agent_reached_flag = True

            if self.timer_flag or agent_reached_flag:
                # get new coverage tasks
                self.task_graph_client()
                unvisited_coverage = self.env.get_unvisited_coverage_tasks_pos()
                visited_coverage = self.env.get_visited_coverage_tasks_pos()

                # reassign tasks
                names = []
                starts = []
                goals = []
                goal_types = []
                for name in self.agent_active_status:
                    if (
                        self.robot_info_dict[name].curr
                        and self.robot_info_dict[name].curr.task_type == "coverage"
                        and not agent_reached[name]
                    ):
                        self.env.update_utility(
                            self.robot_info_dict[name].curr, self.utility_discount_fn
                        )
                        continue
                    start, goal, goal_type = self.reassign(name, solver)
                    if len(start) > 0:
                        names.append(name)
                        starts.append(start)
                        goals.append(goal)
                        goal_types.append(goal_type)
                        task_listener.setBusyStatus(name)
                    # print("{}: status {}".format(name, task_listener.getStatus(name)))
                self.send_visited_to_task_graph()
                self.timer_flag = False

                if len(names) > 0:
                    self.publish_visualize(
                        names,
                        starts,
                        goals,
                        goal_types,
                        unvisited_coverage,
                        visited_coverage,
                    )

            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node("task_commander", anonymous=False, log_level=rospy.INFO)

    try:
        tc = HeterogeneousAssignmentCommander()
        tc.execute()
    except rospy.ROSInterruptException:
        pass
    plt.close("all")
