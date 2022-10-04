#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <costmap_2d/costmap_2d_ros.h>

#include <robosar_messages/frontier_exploration.h>
#include "frontier_search.h"

bool get_frontiers(robosar_messages::frontier_exploration::Request &req, robosar_messages::frontier_exploration::Response &resp)
{
    tf2_ros::Buffer tf2_buffer_(ros::Duration(10));
    tf2_ros::TransformListener tf2_listener_(tf2_buffer_);
    ROS_INFO("making costmap");
    boost::shared_ptr<costmap_2d::Costmap2DROS> costmap_ros_ = boost::make_shared<costmap_2d::Costmap2DROS>("costmap", tf2_buffer_);
    ROS_INFO("Made costmap object");
    frontier_exploration::FrontierSearch frontierSearch(*(costmap_ros_->getCostmap()), req.min_frontier_size, "centroid");
    ROS_INFO("Got costmap");
    resp.frontiers = frontierSearch.searchFrom(req.start_point);
    return true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "frontier_exploration_server");

    ros::NodeHandle nh;
    ros::param::set("~/costmap/robot_base_frame", "agent_0/base_link"); // TODO: change

    ros::ServiceServer service = nh.advertiseService("frontier_exploration", get_frontiers);

    ros::spin();
    return 0;
}