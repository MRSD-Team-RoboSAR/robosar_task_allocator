#!/usr/bin/env python3
"""
Task Commander

Interface for different task commanders.
"""

from abc import ABC, abstractmethod
import rospkg
from robosar_messages.srv import *
from robosar_messages.msg import *


class TaskCommander(ABC):

    def __init__(self):
        self.rospack = rospkg.RosPack()
        self.maps_path = self.rospack.get_path('robosar_task_generator')
        self.package_path = self.rospack.get_path('robosar_task_allocator')
        self.agent_active_status = {}
        self.callback_triggered = False

    @abstractmethod
    def execute(self):
        pass
