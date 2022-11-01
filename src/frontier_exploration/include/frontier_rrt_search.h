#ifndef FRONTIER_RRT_SEARCH_H
#define FRONTIER_RRT_SEARCH_H

#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include "stdint.h"
#include "functions.h"
#include "mtrand.h"
#include "RRT.h"

#include "nav_msgs/OccupancyGrid.h"
#include "geometry_msgs/PointStamped.h"
#include "std_msgs/Header.h"
#include "nav_msgs/MapMetaData.h"
#include "geometry_msgs/Point.h"
#include "visualization_msgs/Marker.h"
// #include "robosar_messages/rrt_path_cost.h"
#include "visualization_msgs/MarkerArray.h"

class FrontierRRTSearch
{

public:
    FrontierRRTSearch(ros::NodeHandle &nh);
    ~FrontierRRTSearch(){};
    void startSearch();

protected:
    void mapCallBack(const nav_msgs::OccupancyGrid::ConstPtr &msg);
    void getRobotLeaderPosition();
    void publishPoints();
    void initMarkers();

    // Steer function prototype
    std::pair<float, float> Steer(std::pair<float, float> &x_nearest, std::pair<float, float> &x_new, float rez);

    // ObstacleFree function prototype
    char ObstacleFree(std::pair<float, float> &x_nearest, std::pair<float, float> &x_new);

    int gridValue(std::pair<float, float> &x);
    std::pair<float, float> pixelsToMap(int x_pixel, int y_pixel);

    void pruneRRT();

    // bool getPathCost(robosar_messages::rrt_path_cost::Request &req, robosar_messages::rrt_path_cost::Response &resp);

private:
    bool isValidCoveragePoint(std::pair<float, float> x_new, float info_radius);
    float informationGain(std::pair<float, float> &x);

    ros::NodeHandle &nh_;
    ros::Subscriber map_sub;
    ros::Publisher targets_pub;
    ros::Publisher marker_pub;
    ros::Publisher marker_coverage_area_pub;
    ros::Timer pub_timer;
    ros::ServiceServer rrt_path_service_;

    std::string ns;
    std::string robot_leader;
    nav_msgs::OccupancyGrid mapData;
    boost::mutex map_mutex_, rrt_mutex_;
    RRT rrt_;
    visualization_msgs::Marker marker_points, marker_line;
    visualization_msgs::MarkerArray marker_coverage_area_array;
    float xdim, ydim, resolution, Xstartx, Xstarty, init_map_x, init_map_y;
    float eta, range;
    std::string map_topic, base_frame_topic;
    rdm r;        // for genrating random numbers
    MTRand drand; // double in [0, 1) generator, already init
};

#endif // FRONTIER_RRT_SEARCH_H