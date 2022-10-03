#!/usr/bin/env python3

import rospy
import argparse
import roslaunch
import tf
from robosar_messages.srv import *
from robosar_messages.msg import *


class MissionCommander:

    def __init__(self, args):
        rospy.init_node('mission_commander', log_level=rospy.DEBUG)
        rospy.logdebug("Initializing Mission Commander ...")
        rospy.Subscriber('/system_mission_command',
                         mission_command, self.handle_commands)
        self.args = args
        rospy.set_param('/make_graph', self.args.make_graph)
        rospy.set_param('/graph_name', self.args.graph_name)
        # TODO: change to be loaded from rosparam
        rospy.set_param('/home_positions', [[45, 10], [49, 11], [50, 10]])
        self.launch_mission = False
        self.stop_mission = False
        self.home_mission = False
        self._tc_node = None
        self._launch = roslaunch.scriptapi.ROSLaunch()
        self._tc_process = None

    def mission_main(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.launch_mission and self._tc_process is None:
                self.execute()
                self.launch_mission = False
            elif self.stop_mission:
                self.stop()
                self.stop_mission = False
            elif self.home_mission:
                self.homing()
                self.home_mission = False

            if self._tc_process and not self._tc_process.is_alive():
                self._tc_process.stop()
                self._tc_process = None
            rate.sleep()

    def handle_commands(self, msg):
        if msg.data == mission_command.START:
            self.launch_mission = True
        elif msg.data == mission_command.STOP:
            self.stop_mission = True
        elif msg.data == mission_command.HOME:
            self.home_mission = True
        else:
            rospy.loginfo_throttle(1, "Invalid command published")

    def execute(self):
        rospy.logdebug("Executing")
        if self.args.task_allocator == "mtsp":
            self._tc_node = roslaunch.core.Node(
                "robosar_task_allocator", "mtsp_ta.py", name="task_commander", output="screen")
        elif self.args.task_allocator == "greedy":
            self._tc_node = roslaunch.core.Node(
                "robosar_task_allocator", "greedy_ta.py")
        else:
            raise Exception("Invalid TA type")
        self._launch.start()
        self._tc_process = self._launch.launch(self._tc_node)

    def homing(self):
        self.stop()

        starts = []
        goals = []
        names = []
        agent_active_status = {}

        # Get active agents
        print("calling agent status service")
        rospy.wait_for_service('/robosar_agent_bringup_node/agent_status')
        try:
            get_status = rospy.ServiceProxy(
                '/robosar_agent_bringup_node/agent_status', agent_status)
            resp1 = get_status()
            active_agents = resp1.agents_active
            for a in active_agents:
                agent_active_status[a] = True
            print("{} agents active".format(len(agent_active_status)))
            assert len(agent_active_status) > 0
        except rospy.ServiceException as e:
            print("Agent status service call failed: %s" % e)
            raise Exception("Agent status service call failed")

        # get robot positions
        robot_init = []
        listener = tf.TransformListener()
        listener.waitForTransform('map', list(agent_active_status.keys())[
                                  0] + '/base_link', rospy.Time(), rospy.Duration(1.0))
        for name in agent_active_status:
            now = rospy.Time.now()
            listener.waitForTransform(
                'map', name + '/base_link', now, rospy.Duration(1.0))
            (trans, rot) = listener.lookupTransform(
                'map', name + '/base_link', now)
            robot_init.append([trans[0], trans[1]])

        # fill in message
        goals = rospy.get_param('/home_positions')
        for i, name in enumerate(agent_active_status.keys()):
            names.append(name)
            starts.append(robot_init[i])

        # publish
        task_pub = rospy.Publisher(
            'task_allocation', task_allocation, queue_size=10)
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
        rospy.sleep(1)
        print("sent homing positions {}".format(goals))

    def stop(self):
        if self._tc_node:
            self._tc_process.stop()
        rospy.logwarn("Stopping task allocation.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_allocator",
                        help="Task allocator type", type=str, default="mtsp")
    parser.add_argument("-m", "--make_graph",
                        help="Make graph", type=bool, default=True)
    parser.add_argument("-g", "--graph_name",
                        help="Graph name", type=str, default="temp")
    args = parser.parse_args()

    try:
        mc = MissionCommander(args)
        mc.mission_main()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()
