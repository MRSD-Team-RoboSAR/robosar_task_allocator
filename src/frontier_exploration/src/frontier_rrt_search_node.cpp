#include <ros/ros.h>

#include "frontier_rrt_search.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "frontier_rrt_search_node");
    ros::NodeHandle nh("~");

    FrontierRRTSearch frontier_search(nh);
    frontier_search.startSearch();

    ros::spin();

    return 0;
}