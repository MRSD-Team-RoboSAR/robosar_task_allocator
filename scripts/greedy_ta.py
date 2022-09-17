#!/usr/bin/env python3

# Created by Indraneel on 01/03/22

from task_commander import TaskCommander
import rospy
from robosar_task_allocator.Environment import Environment
from robosar_task_allocator.Robot import Robot
from robosar_task_allocator.TA import *
from robosar_task_allocator.task_transmitter.task_tx_move_base import TaskTxMoveBase
import numpy as np
import argparse
from actionlib_msgs.msg import GoalStatus


class GreedyCommander(TaskCommander):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def pixels_to_m(self, pixels):
        return [pixels[0]*0.1, pixels[1]*0.1]

    def execute(self):
        # Set up
        nodes = np.load(self.maps_path+"/outputs/willow-full_lean.npy")
        #filename = maps_path+'/maps/willow-full.pgm'

        # Create robots
        print(nodes[0])
        robot0 = Robot(0, nodes[0], 0)
        robot1 = Robot(1, nodes[0], 0)
        robot2 = Robot(2, nodes[0], 0)
        robots = [robot0, robot1, robot2]
        # Create environment
        adj = np.load(
            self.package_path+'/src/robosar_task_allocator/willow_{}_graph.npy'.format(self.args.graph_name))
        env = Environment(nodes[:n, :], adj, robots)

        solver = TA_greedy()
        solver.init(env)

        # Create transmitter object
        transmitter = TaskTxMoveBase(robots)

        rate = rospy.Rate(10)  # 10hz
        rospy.loginfo(
            'Task_Alloc_Greedy] Buckle up! Running greedy allocator!')

        while not rospy.is_shutdown() and not self.e_stop:

            for robot in robots:
                status = transmitter.getStatus(robot.id)
                if (status == GoalStatus.SUCCEEDED):
                    solver.reached(robot.id, robot.next)
                    transmitter.setGoal(
                        robot.id, self.pixels_to_m(env.nodes[robot.next]))
                    print(env.visited)
                elif (status == GoalStatus.LOST):
                    solver.assign(robot.id, robot.prev)
                    transmitter.setGoal(
                        robot.id, self.pixels_to_m(env.nodes[robot.next]))
            rate.sleep()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--make_graph",
                        help="Make graph", type=bool, default=False)
    parser.add_argument("-g", "--graph_name",
                        help="Graph name", type=str, default="temp")
    args = parser.parse_args()

    try:
        tc = GreedyCommander(args)
        tc.execute()
    except rospy.ROSInterruptException:
        pass
