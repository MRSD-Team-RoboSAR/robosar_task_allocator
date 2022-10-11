#!/usr/bin/env python3

from copy import copy
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PointStamped
import tf
from numpy import array, vstack, delete
from functions import gridValue, informationGain
from sklearn.cluster import MeanShift
from robosar_messages.msg import PointArray


class FrontierFilter:

    def __init__(self) -> None:
        self.mapData = OccupancyGrid()
        self.frontiers = []

        # fetching all parameters
        self.map_topic = rospy.get_param('~map_topic', '/map')
        self.occ_threshold = rospy.get_param('~costmap_clearing_threshold', 50)
        self.info_threshold = rospy.get_param('~info_gain_threshold', 0.7)
        # this can be smaller than the laser scanner range, >> smaller >>less computation time>> too small is not good, info gain won't be accurate
        self.info_radius = rospy.get_param('~info_radius', 1.0)
        self.goals_topic = rospy.get_param('~goals_topic', '/detected_points')
        self.namespace = rospy.get_param('~namespace', '')
        rateHz = rospy.get_param('~rate', 10)
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
        rospy.Subscriber(self.goals_topic, PointStamped, callback=self.frontiersCallback,
                         callback_args=[self.tfLisn, global_frame])
        self.frontier_pub = rospy.Publisher('frontiers', Marker, queue_size=10)
        self.centroids_pub = rospy.Publisher(
            'centroids', Marker, queue_size=10)
        self.filtered_pub = rospy.Publisher(
            'filtered_points', PointArray, queue_size=10)

        rospy.loginfo("the map and global costmaps are received")

    def frontiersCallback(self, data, args):
        transformedPoint = args[0].transformPoint(args[1], data)
        x = [array([transformedPoint.point.x, transformedPoint.point.y])]
        if len(self.frontiers) > 0:
            self.frontiers = vstack((self.frontiers, x))
        else:
            self.frontiers = x

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
        while len(self.frontiers) < 1:
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
            # -------------------------------------------------------------------------
            # Clustering frontier points
            centroids = []
            front = copy(self.frontiers)
            if len(front) > 1:
                ms = MeanShift(bandwidth=0.3)
                ms.fit(front)
                centroids = ms.cluster_centers_  # centroids array is the centers of each cluster

            # if there is only one frontier no need for clustering, i.e. centroids=self.frontiers
            if len(front) == 1:
                centroids = front
            self.frontiers = copy(centroids)
            # -------------------------------------------------------------------------
            # clearing old self.frontiers
            z = 0
            while z < len(centroids):
                temppoint.point.x = centroids[z][0]
                temppoint.point.y = centroids[z][1]

                transformedPoint = self.tfLisn.transformPoint(
                    self.mapData.header.frame_id, temppoint)
                x = array([temppoint.point.x, temppoint.point.y])
                cond = (gridValue(self.mapData, x) > self.occ_threshold)
                if (cond or (informationGain(self.mapData, [centroids[z][0], centroids[z][1]], self.info_radius*2.0)) < self.info_threshold):
                    centroids = delete(centroids, (z), axis=0)
                    z = z-1
                z += 1
            # -------------------------------------------------------------------------
            # publishing
            arraypoints.points = []
            for i in centroids:
                tempPoint.x = i[0]
                tempPoint.y = i[1]
                arraypoints.points.append(copy(tempPoint))
            self.filtered_pub.publish(arraypoints)
            pp = []
            for q in range(0, len(self.frontiers)):
                p.x = self.frontiers[q][0]
                p.y = self.frontiers[q][1]
                pp.append(copy(p))
            points.points = pp
            pp = []
            for q in range(0, len(centroids)):
                p.x = centroids[q][0]
                p.y = centroids[q][1]
                pp.append(copy(p))
            points_clust.points = pp
            self.frontier_pub.publish(points)
            self.centroids_pub.publish(points_clust)
            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node('frontier_filter', anonymous=False)
    try:
        ff = FrontierFilter()
        ff.filter()
    except rospy.ROSInterruptException:
        pass
