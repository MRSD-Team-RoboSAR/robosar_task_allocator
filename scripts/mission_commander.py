#!/usr/bin/env python3

import rospy
import argparse
import tf
from std_msgs.msg import Int32
from robosar_messages.srv import *
from robosar_messages.msg import *
from mtsp_ta import MtspCommander
from greedy_ta import GreedyCommander


class MissionCommander:

    def __init__(self, args):
        rospy.init_node('mission_commander', anonymous=True,
                        log_level=rospy.DEBUG)
        rospy.logdebug("Initializing Mission Commander ...")
        rospy.Subscriber('/mission_status', Int32, self.handle_commands)
        self.args = args
        self.tc_ = None

    def handle_commands(self, msg):
        if msg.data == 1:
            self.execute()
        elif msg.data == 2:
            self.stop()
        elif msg.data == 3:
            self.homing()
        else:
            rospy.loginfo_throttle(1, "Invalid command published")

    def execute(self):
        rospy.logdebug("Executing")
        if self.args.task_allocator == "mtsp":
            self.tc_ = MtspCommander(self.args)
        elif self.args.task_allocator == "greedy":
            self.tc_ = GreedyCommander(self.args)
        else:
            raise Exception("Invalid TA type")
        self.tc_.execute()

    def homing(self):
        self.tc_.stop()
        
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
        for i, name in enumerate(agent_active_status.keys()):
            names.append(name)
            starts.append(robot_init[i])
            if i % 2 == 0:
                goals.append([0+0.6*((i+1)//2), 0])
            else:
                goals.append([0 - 0.6 * ((i+1)//2), 0])

        # publish
        task_pub = rospy.Publisher(
            'task_allocation', task_allocation, queue_size=10)
        print("publishing")
        task_msg = task_allocation()
        print(goals)
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
        print("sent")

    def stop(self):
        self.tc_.stop()
        rospy.signal_shutdown("E-Stop")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_allocator",
                        help="Task allocator type", type=str, default="mtsp")
    parser.add_argument("-m", "--make_graph",
                        help="Make graph", type=bool, default=False)
    parser.add_argument("-g", "--graph_name",
                        help="Graph name", type=str, default="temp")
    args = parser.parse_args()

    try:
        MissionCommander(args)
    except rospy.ROSInterruptException:
        pass
    rospy.spin()
