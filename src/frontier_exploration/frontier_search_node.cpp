#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <costmap_2d/costmap_2d_ros.h>

#include <robosar_messages/frontier_exploration.h>
#include "frontier_search.h"

bool get_frontiers(robosar_messages::frontier_exploration::Request &req, robosar_messages::frontier_exploration::Response &resp)
{
    tf2_ros::Buffer tf2_buffer_;
    boost::shared_ptr<costmap_2d::Costmap2DROS> costmap_ros_ = boost::make_shared<costmap_2d::Costmap2DROS>("map", tf2_buffer_);
    frontier_exploration::FrontierSearch frontierSearch(*(costmap_ros_->getCostmap()), req.min_frontier_size, "centroid");
    resp.frontiers = frontierSearch.searchFrom(req.start_point);
    return true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "frontier_exploration_server");

    ros::NodeHandle nh;

    ros::ServiceServer service = nh.advertiseService("frontier_exploration", get_frontiers);

    ros::spin();
    return 0;
}