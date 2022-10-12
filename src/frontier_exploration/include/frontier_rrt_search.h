#ifndef FRONTIER_RRT_SEARCH_H
#define FRONTIER_RRT_SEARCH_H

#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include "stdint.h"
#include "functions.h"
#include "mtrand.h"

#include "nav_msgs/OccupancyGrid.h"
#include "geometry_msgs/PointStamped.h"
#include "std_msgs/Header.h"
#include "nav_msgs/MapMetaData.h"
#include "geometry_msgs/Point.h"
#include "visualization_msgs/Marker.h"
#include <tf/transform_listener.h>

class FrontierRRTSearch {

public:
    FrontierRRTSearch(ros::NodeHandle& nh);
    ~FrontierRRTSearch() {};
    void startSearch();

protected:
    void mapCallBack(const nav_msgs::OccupancyGrid::ConstPtr &msg);
    void rvizCallBack(const geometry_msgs::PointStamped::ConstPtr &msg);
    void publishPoints();
    std::vector<float> Nearest(std::vector<std::vector<float>> V, std::vector<float> x_rand);

    // Steer function prototype
    std::vector<float> Steer(std::vector<float> x_nearest, std::vector<float> x_new, float rez);

    // ObstacleFree function prototype
    char ObstacleFree(std::vector<float> x_nearest, std::vector<float> & x_new);

    int gridValue(std::vector<float> x);
    std::vector<float> pixelsToMap(int x_pixel, int y_pixel);

private:
    ros::NodeHandle& nh_;
    ros::Subscriber map_sub;
    ros::Subscriber rviz_sub;
    ros::Publisher targets_pub;
    ros::Publisher marker_pub;
    ros::Timer pub_timer;

    std::string ns;
    bool started_ = false;
    nav_msgs::OccupancyGrid mapData;
    geometry_msgs::PointStamped clickedpoint;
    geometry_msgs::PointStamped exploration_goal;
    std::vector<std::vector<float>> V; // list of RRT nodes
    visualization_msgs::Marker points, line;
    float xdim, ydim, resolution, Xstartx, Xstarty, init_map_x, init_map_y;
    float eta, range;
    std::string map_topic, base_frame_topic;
    rdm r; // for genrating random numbers
    MTRand drand;                     // double in [0, 1) generator, already init

};

#endif // FRONTIER_RRT_SEARCH_H