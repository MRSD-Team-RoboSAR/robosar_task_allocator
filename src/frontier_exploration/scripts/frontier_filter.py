#!/usr/bin/env python3

from copy import copy
from threading import Lock

import numpy as np
import rospy
import tf
from geometry_msgs.msg import Point, PointStamped
from nav_msgs.msg import OccupancyGrid
from numpy import array, vstack
from robosar_messages.msg import PointArray
from sklearn.cluster import MeanShift
from visualization_msgs.msg import Marker
import functions

from functions import gridValue, informationGain


class FrontierFilter:
    def __init__(self) -> None:
        self.mapData = OccupancyGrid()
        self.received_frontiers = []
        self.filtered_frontiers = []
        self.map_lock = Lock()
        self.frontier_lock = Lock()

        # fetching all parameters
        ns = rospy.get_name()
        self.map_topic = rospy.get_param("~map_topic", "/map")
        self.occ_threshold = rospy.get_param("~costmap_clearing_threshold", 70)
        self.info_threshold = rospy.get_param("~info_gain_threshold", 0.2)
        self.cluster_bandwidth = rospy.get_param("~cluster_bandwidth", 1.0)
        # this can be smaller than the laser scanner range, >> smaller >>less computation time>> too small is not good, info gain won't be accurate
        self.info_radius = rospy.get_param("~info_radius", 0.5)
        self.goals_topic = rospy.get_param("~goals_topic", "/detected_points")
        self.namespace = rospy.get_param("~namespace", "")
        rateHz = rospy.get_param("~rate", 2)
        self.robot_frame = rospy.get_param("~robot_frame", "agent1/base_link")

        self.rate = rospy.Rate(rateHz)
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.mapCallback)

        rospy.loginfo("Waiting for the map")
        while len(self.mapData.data) < 1:
            rospy.sleep(0.1)
            pass

        global_frame = "/" + self.mapData.header.frame_id
        # wait if map is not received yet
        rospy.loginfo("Waiting for robot transform")
        self.tfLisn = tf.TransformListener()
        self.tfLisn.waitForTransform(
            global_frame, "/" + self.robot_frame, rospy.Time(0), rospy.Duration(10.0)
        )
        rospy.Subscriber(
            self.goals_topic, PointStamped, callback=self.frontiersCallback
        )
        self.frontier_marker_pub = rospy.Publisher(
            ns + "/frontier_centroids", Marker, queue_size=10
        )
        self.frontier_array_pub = rospy.Publisher(
            ns + "/filtered_frontiers", PointArray, queue_size=10
        )

        rospy.loginfo("the map and global costmaps are received")

    def frontiersCallback(self, data):
        with self.frontier_lock:
            x = np.array([data.point.x, data.point.y])
            self.received_frontiers.append(x)

    def mapCallback(self, data):
        with self.map_lock:
            self.mapData = data

    def init_markers(self):
        points_clust = Marker()
        points_clust.header.frame_id = self.mapData.header.frame_id
        points_clust.header.stamp = rospy.Time.now()
        points_clust.ns = "cluster_markers"
        points_clust.id = 4

        points_clust.type = Marker.POINTS
        # Set the marker action for centroids.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
        points_clust.action = Marker.ADD
        points_clust.pose.orientation.w = 1.0
        points_clust.scale.x = 0.2
        points_clust.scale.y = 0.2
        points_clust.color.r = 0.0 / 255.0
        points_clust.color.g = 255.0 / 255.0
        points_clust.color.b = 0.0 / 255.0
        points_clust.color.a = 1
        points_clust.lifetime = rospy.Duration()

        return points_clust

    def is_valid_frontier(self, node):
        if node[0] > -0.5 and node[0] < 12.0 and node[1] > -10.0 and node[1] < 2.0:
            return True
        return False

    def obstacle_free(self, xnear, xnew):
        rez = float(self.mapData.info.resolution) * 0.2
        stepz = int(np.ceil(functions.Norm(xnew[0], xnew[1], xnear[0], xnear[1])) / rez)
        xi = xnear
        obs = 0
        unk = 0

        for _ in range(stepz):
            xi = functions.Steer(xi, xnew, rez)
            if functions.unvalid(self.mapData, xi):
                obs = 1
                break
            if gridValue(self.mapData, xi) == 100:
                obs = 1
            if gridValue(self.mapData, xi) == -1:
                unk = 1
                break
        out = 0
        xnew = xi
        if unk == 1:
            out = -1  # unknown
        if obs == 1:
            out = 0  # occupied
        if obs != 1 and unk != 1:
            out = 1  # free

        return out

    def check_obs_intersection(self, centroid, label, cluster_labels, frontiers):
        points = []
        for idx, l in enumerate(cluster_labels):
            if l == label:
                points.append(frontiers[idx])
        for p in points:
            if self.obstacle_free(p, centroid) == 0:
                return True
        return False

    def filter(self):
        # wait if no frontier is received yet
        rospy.loginfo("Waiting for frontiers")
        while len(self.received_frontiers) < 1:
            pass

        centroid_markers = self.init_markers()

        centroid_point = PointStamped()
        centroid_point.header.frame_id = self.mapData.header.frame_id
        centroid_point.header.stamp = rospy.Time(0)
        centroid_point.point.z = 0.0

        arraypoints = PointArray()

        rospy.loginfo("Starting filter")
        while not rospy.is_shutdown():
            with self.map_lock and self.frontier_lock:
                centroids = []
                possible_frontiers = []
                labels = []
                # Add received frontiers
                for f in self.received_frontiers:
                    possible_frontiers.append(f)

                # Filter out previous centroids by information gain
                for f in self.filtered_frontiers:
                    info_gain = informationGain(
                        self.mapData, [f[0], f[1]], self.info_radius
                    )
                    if info_gain > self.info_threshold:
                        possible_frontiers.append(f)

                # Clustering frontier points
                if len(possible_frontiers) > 1:
                    ms = MeanShift(bandwidth=self.cluster_bandwidth)
                    ms.fit(possible_frontiers)
                    centroids = (
                        ms.cluster_centers_
                    )  # centroids array is the centers of each cluster
                    labels = ms.labels_
                if len(possible_frontiers) == 1:
                    centroids = possible_frontiers

                # make sure centroid is not occupied, filter out by information gain
                centroids_filtered = []
                for idx, c in enumerate(centroids):
                    centroid_point.point.x = c[0]
                    centroid_point.point.y = c[1]
                    x = array([centroid_point.point.x, centroid_point.point.y])
                    if (
                        gridValue(self.mapData, x) < self.occ_threshold
                        and informationGain(
                            self.mapData, [x[0], x[1]], self.info_radius
                        )
                        > 0.15
                        and not self.check_obs_intersection(
                            c, idx, labels, possible_frontiers
                        )
                        # and self.is_valid_frontier(x)
                    ):
                        centroids_filtered.append(c)
                self.filtered_frontiers = copy(centroids_filtered)

                # publishing
                arraypoints.points = []
                for i in centroids_filtered:
                    published_point = Point()
                    published_point.z = 0.0
                    published_point.x = i[0]
                    published_point.y = i[1]
                    arraypoints.points.append(published_point)
                self.frontier_array_pub.publish(arraypoints)
                pp = []
                for q in range(0, len(centroids_filtered)):
                    p = Point()
                    p.z = 0
                    p.x = centroids_filtered[q][0]
                    p.y = centroids_filtered[q][1]
                    pp.append(p)
                centroid_markers.points = pp
                self.frontier_marker_pub.publish(centroid_markers)

                self.received_frontiers = []
            self.rate.sleep()


if __name__ == "__main__":

    rospy.init_node("frontier_filter", anonymous=False)
    try:
        ff = FrontierFilter()
        ff.filter()
    except rospy.ROSInterruptException:
        pass
