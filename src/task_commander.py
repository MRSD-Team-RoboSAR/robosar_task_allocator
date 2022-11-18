#!/usr/bin/env python3

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import rospkg
import rospy
import task_allocator.utils as utils
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Int32

from robosar_messages.msg import *
from robosar_messages.srv import *


class TaskCommander(ABC):
    def __init__(self):
        self.rospack = rospkg.RosPack()
        self.maps_path = self.rospack.get_path("robosar_task_generator")
        self.package_path = self.rospack.get_path("robosar_task_allocator")
        self.agent_active_status = {}
        self.geofence = []
        self.callback_triggered = False
        # task publisher
        self.task_pub = rospy.Publisher(
            "task_allocation", task_allocation, queue_size=10
        )
        self.task_num_pub = rospy.Publisher("tasks_completed", Int32, queue_size=10)

    def status_callback(self, msg):
        """
        Agent status callback
        """
        rospy.wait_for_service("/robosar_agent_bringup_node/agent_status")
        try:
            rospy.loginfo("calling service")
            get_status = rospy.ServiceProxy(
                "/robosar_agent_bringup_node/agent_status", agent_status
            )
            resp1 = get_status()
            active_agents = resp1.agents_active
            for a in self.agent_active_status:
                self.agent_active_status[a] = False
            for a in active_agents:
                self.agent_active_status[a] = True
            rospy.loginfo(self.agent_active_status)
            self.callback_triggered = True

        except rospy.ServiceException as e:
            rospy.loginfo("Agent status service call failed: %s" % e)

    def get_active_agents(self):
        """
        Updates agent_active_status
        """
        rospy.loginfo("Starting task allocator")
        rospy.loginfo("calling agent status service")
        rospy.wait_for_service("/robosar_agent_bringup_node/agent_status")
        try:
            get_status = rospy.ServiceProxy(
                "/robosar_agent_bringup_node/agent_status", agent_status
            )
            resp1 = get_status()
            active_agents = resp1.agents_active
            for a in active_agents:
                self.agent_active_status[a] = True
            rospy.loginfo("{} agents active".format(len(self.agent_active_status)))
            assert len(self.agent_active_status) > 0
        except rospy.ServiceException as e:
            rospy.loginfo("Agent status service call failed: %s" % e)
            raise Exception("Agent status service call failed")

    def get_map_info(self):
        """
        Gets map message
        """
        rospy.logdebug("Waiting for map")
        map_msg = rospy.wait_for_message("/slam_toolbox/map", OccupancyGrid)
        rospy.logdebug("Map received")
        scale = map_msg.info.resolution
        origin = [map_msg.info.origin.position.x, map_msg.info.origin.position.y]
        rospy.logdebug("map origin: {}".format(origin))
        data = np.reshape(map_msg.data, (map_msg.info.height, map_msg.info.width))
        free_space = 0
        for idx, cell_val in enumerate(map_msg.data):
            if 0 <= cell_val:
                free_space += 1
        area = free_space * (scale**2)
        rospy.logdebug("Map Area: {}".format(area))
        return map_msg, data, scale, origin, area

    def publish_image(self, image_pub):
        canvas = plt.gca().figure.canvas
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        image = data.reshape(canvas.get_width_height()[::-1] + (3,))
        br = CvBridge()
        image_pub.publish(br.cv2_to_imgmsg(image, "rgb8"))

    @abstractmethod
    def execute(self):
        pass
