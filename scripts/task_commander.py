#!/usr/bin/env python3

from abc import ABC, abstractmethod
import rospkg
import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool, Int32
from robosar_messages.srv import *
from robosar_messages.msg import *


class TaskCommander(ABC):

    def __init__(self):
        self.rospack = rospkg.RosPack()
        self.maps_path = self.rospack.get_path('robosar_task_generator')
        self.package_path = self.rospack.get_path('robosar_task_allocator')
        self.agent_active_status = {}
        self.callback_triggered = False
        self.e_stop = False

    def stop(self):
        self.e_stop = True

    @abstractmethod
    def execute(self):
        pass
