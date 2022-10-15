#!/usr/bin/env python3

import sys
import os
from copy import copy
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PointStamped
import tf
from numpy import array, vstack
from functions import gridValue, informationGain
from sklearn.cluster import MeanShift
from robosar_messages.msg import PointArray
import numpy as np
import matplotlib.pyplot as plt


class FrontierFilter:

    def __init__(self) -> None:
        self.mapData = OccupancyGrid()
        self.received_frontiers = []
        self.filtered_frontiers = []

        # fetching all parameters
        self.map_topic = rospy.get_param('~map_topic', '/map')
        self.occ_threshold = rospy.get_param('~costmap_clearing_threshold', 70)
        self.info_threshold = rospy.get_param('~info_gain_threshold', 0.2)
        self.cluster_bandwidth = rospy.get_param('~cluster_bandwidth', 1.0)
        # this can be smaller than the laser scanner range, >> smaller >>less computation time>> too small is not good, info gain won't be accurate
        self.info_radius = rospy.get_param('~info_radius', 0.5)
        self.goals_topic = rospy.get_param('~goals_topic', '/detected_points')
        self.namespace = rospy.get_param('~namespace', '')
        rateHz = rospy.get_param('~rate', 2)
        self.robot_frame = rospy.get_param('~robot_frame', 'agent1/base_link')

        self.rate = rospy.Rate(rateHz)
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.mapCallback)

        rospy.loginfo('Waiting for the map')
        while (len(self.mapData.data) < 1):
            rospy.sleep(0.1)
            pass

        global_frame = "/"+self.mapData.header.frame_id
        # wait if map is not received yet
        rospy.loginfo('Waiting for robot transform')
        self.tfLisn = tf.TransformListener()
        self.tfLisn.waitForTransform(
            global_frame, '/'+self.robot_frame, rospy.Time(0), rospy.Duration(10.0))
        rospy.Subscriber(self.goals_topic, PointStamped,
                         callback=self.frontiersCallback)
        self.frontier_pub = rospy.Publisher(
            'frontier_centroids', Marker, queue_size=10)
        self.filtered_pub = rospy.Publisher(
            'filtered_points', PointArray, queue_size=10)

        rospy.loginfo("the map and global costmaps are received")

    def frontiersCallback(self, data):
        x = [array([data.point.x, data.point.y])]
        if len(self.received_frontiers) > 0:
            self.received_frontiers = vstack((self.received_frontiers, x))
        else:
            self.received_frontiers = x

    def mapCallback(self, data):
        self.mapData = data

    def init_markers(self):
        points = Marker()
        points_clust = Marker()
        # Set the frame ID and timestamp.  See the TF tutorials for information on these.
        points.header.frame_id = self.mapData.header.frame_id
        points.header.stamp = rospy.Time.now()
        points.ns = "markers2"
        points.id = 0
        points.type = Marker.POINTS
        # Set the marker action for latched self.frontiers.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
        points.action = Marker.ADD
        points.pose.orientation.w = 1.0
        points.scale.x = 0.2
        points.scale.y = 0.2
        points.color.r = 255.0/255.0
        points.color.g = 255.0/255.0
        points.color.b = 0.0/255.0
        points.color.a = 1
        points.lifetime = rospy.Duration()

        points_clust.header.frame_id = self.mapData.header.frame_id
        points_clust.header.stamp = rospy.Time.now()
        points_clust.ns = "markers3"
        points_clust.id = 4

        points_clust.type = Marker.POINTS
        # Set the marker action for centroids.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
        points_clust.action = Marker.ADD
        points_clust.pose.orientation.w = 1.0
        points_clust.scale.x = 0.2
        points_clust.scale.y = 0.2
        points_clust.color.r = 0.0/255.0
        points_clust.color.g = 255.0/255.0
        points_clust.color.b = 0.0/255.0
        points_clust.color.a = 1
        points_clust.lifetime = rospy.Duration()

        return points, points_clust

    def filter(self):
        # wait if no frontier is received yet
        rospy.loginfo("Waiting for frontiers")
        while len(self.received_frontiers) < 1:
            pass

        points, points_clust = self.init_markers()

        temppoint = PointStamped()
        temppoint.header.frame_id = self.mapData.header.frame_id
        temppoint.header.stamp = rospy.Time(0)
        temppoint.point.z = 0.0

        arraypoints = PointArray()
        tempPoint = Point()
        tempPoint.z = 0.0

        p = Point()
        p.z = 0

        rospy.loginfo("Starting filter")
        while not rospy.is_shutdown():
            centroids = []
            front = []
            for f in self.received_frontiers:
                front.append(f)

            self.received_frontiers = copy(front)
            # Filter out by information gain
            for f in self.filtered_frontiers:
                info_gain = informationGain(
                    self.mapData, [f[0], f[1]], self.info_radius)
                if info_gain > self.info_threshold:
                    front.append(f)

            # Clustering frontier points
            if len(front) > 1:
                ms = MeanShift(bandwidth=self.cluster_bandwidth)
                ms.fit(front)
                centroids = ms.cluster_centers_  # centroids array is the centers of each cluster
            if len(front) == 1:
                centroids = front

            self.filtered_frontiers = copy(centroids)

            # make sure centroid is not occupied
            centroids_filtered = []
            for c in centroids:
                temppoint.point.x = c[0]
                temppoint.point.y = c[1]
                x = array([temppoint.point.x, temppoint.point.y])
                if gridValue(self.mapData, x) < self.occ_threshold:
                    centroids_filtered.append(c)

            # publishing
            arraypoints.points = []
            for i in centroids_filtered:
                tempPoint.x = i[0]
                tempPoint.y = i[1]
                arraypoints.points.append(copy(tempPoint))
            self.filtered_pub.publish(arraypoints)
            pp = []
            for q in range(0, len(centroids_filtered)):
                p.x = centroids_filtered[q][0]
                p.y = centroids_filtered[q][1]
                pp.append(copy(p))
            points_clust.points = pp
            self.frontier_pub.publish(points_clust)

            self.received_frontiers = []
            self.rate.sleep()


if __name__ == '__main__':

    rospy.init_node('frontier_filter', anonymous=False)
    try:
        ff = FrontierFilter()
        ff.filter()
    except rospy.ROSInterruptException:
        pass
