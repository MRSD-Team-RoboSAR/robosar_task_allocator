#include <ros/ros.h>
#include <tf/tf.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>

#include "frontier_rrt_search.h"

FrontierRRTSearch::FrontierRRTSearch(ros::NodeHandle &nh) : nh_(nh)
{

    ns = ros::this_node::getName();
    ros::param::param<float>(ns + "/eta", eta, 10);
    ros::param::param<std::string>(ns + "/map_topic", map_topic, "/map");
    ros::param::param<std::string>(ns + "/robot_leader", robot_leader, "agent1");

    map_sub = nh_.subscribe(map_topic, 100, &FrontierRRTSearch::mapCallBack, this);
    targets_pub = nh_.advertise<geometry_msgs::PointStamped>("/detected_points", 10);
    marker_pub = nh_.advertise<visualization_msgs::Marker>(ns + "_shapes", 10);
    pub_timer = nh_.createTimer(ros::Duration(1 / 10.0), std::bind(&FrontierRRTSearch::publishPoints, this));
}

// Subscribers callback functions---------------------------------------
void FrontierRRTSearch::mapCallBack(const nav_msgs::OccupancyGrid::ConstPtr &msg)
{
    mapData = *msg;
}

void FrontierRRTSearch::getRobotLeaderPosition()
{
    ROS_INFO("Waiting for robot transform.");
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_(tf_buffer_);
    geometry_msgs::TransformStamped tf;
    ros::Duration(5.0).sleep();
    try
    {
        ros::Time now = ros::Time(0);
        ROS_INFO("%s",robot_leader.c_str());
        tf = tf_buffer_.lookupTransform("map", robot_leader + "/base_link", now);

        geometry_msgs::Point p;
        p.x = tf.transform.translation.x;
        p.y = tf.transform.translation.y;
        p.z = 0;

        points.points.push_back(p);
        marker_pub.publish(points);
    }
    catch (tf2::TransformException &ex)
    {
        ROS_ERROR("%s",ex.what());
    }
}

void FrontierRRTSearch::publishPoints()
{
    if (started_)
        targets_pub.publish(exploration_goal);
}

// Nearest function
std::vector<float> FrontierRRTSearch::Nearest(std::vector<std::vector<float>> V, std::vector<float> x)
{

    float min = Norm(V[0], x);
    int min_index;
    float temp;

    for (int j = 0; j < V.size(); j++)
    {
        temp = Norm(V[j], x);
        if (temp <= min)
        {
            min = temp;
            min_index = j;
        }
    }

    return V[min_index];
}

// Steer function
std::vector<float> FrontierRRTSearch::Steer(std::vector<float> x_nearest, std::vector<float> x_rand, float eta)
{
    std::vector<float> x_new;

    if (Norm(x_nearest, x_rand) <= eta)
    {
        x_new = x_rand;
    }
    else
    {
        float m = (x_rand[1] - x_nearest[1]) / (x_rand[0] - x_nearest[0]);
        if (x_rand[0] == x_nearest[0])
        {
            x_new = {x_nearest[0], x_nearest[1] + eta};
        }
        x_new.push_back((sign(x_rand[0] - x_nearest[0])) * (sqrt((pow(eta, 2)) / ((pow(m, 2)) + 1))) + x_nearest[0]);
        x_new.push_back(m * (x_new[0] - x_nearest[0]) + x_nearest[1]);
    }
    return x_new;
}

std::vector<float> FrontierRRTSearch::pixelsToMap(int x_pixel, int y_pixel)
{
    std::vector<float> map_coords;
    float scale = mapData.info.resolution;
    float x_origin = mapData.info.origin.position.x;
    float y_origin = mapData.info.origin.position.y;
    map_coords = {x_pixel * scale + x_origin, y_pixel * scale + y_origin};
    return map_coords;
}

// gridValue function
int FrontierRRTSearch::gridValue(std::vector<float> Xp)
{

    float resolution = mapData.info.resolution;
    float Xstartx = mapData.info.origin.position.x;
    float Xstarty = mapData.info.origin.position.y;

    float width = mapData.info.width;
    std::vector<signed char> Data = mapData.data;

    // returns grid value at "Xp" location
    // map data:  100 occupied      -1 unknown       0 free
    float indx = (floor((Xp[1] - Xstarty) / resolution) * width) + (floor((Xp[0] - Xstartx) / resolution));
    int out;
    out = Data[int(indx)];
    return out;
}

// ObstacleFree function-------------------------------------

char FrontierRRTSearch::ObstacleFree(std::vector<float> xnear, std::vector<float> &xnew)
{
    float rez = float(mapData.info.resolution) * .2;
    float stepz = int(ceil(Norm(xnew, xnear)) / rez);
    std::vector<float> xi = xnear;
    char obs = 0;
    char unk = 0;

    geometry_msgs::Point p;
    for (int c = 0; c < stepz; c++)
    {
        xi = Steer(xi, xnew, rez);

        if (gridValue(xi) == 100)
        {
            obs = 1;
        }

        if (gridValue(xi) == -1)
        {
            unk = 1;
            break;
        }
    }
    char out = 0;
    xnew = xi;
    if (unk == 1)
    {
        out = -1;
    }

    if (obs == 1)
    {
        out = 0;
    }

    if (obs != 1 && unk != 1)
    {
        out = 1;
    }

    return out;
}

void FrontierRRTSearch::initMarkers()
{
    // visualizations  points and lines..
    points.header.frame_id = mapData.header.frame_id;
    line.header.frame_id = mapData.header.frame_id;
    points.header.stamp = ros::Time(0);
    line.header.stamp = ros::Time(0);

    points.ns = line.ns = "markers";
    points.id = 0;
    line.id = 1;

    points.type = points.POINTS;
    line.type = line.LINE_LIST;

    // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
    points.action = points.ADD;
    line.action = line.ADD;
    points.pose.orientation.w = 1.0;
    line.pose.orientation.w = 1.0;
    line.scale.x = 0.03;
    line.scale.y = 0.03;
    points.scale.x = 0.3;
    points.scale.y = 0.3;

    line.color.r = 9.0 / 255.0;
    line.color.g = 91.0 / 255.0;
    line.color.b = 236.0 / 255.0;
    points.color.r = 255.0 / 255.0;
    points.color.g = 0.0 / 255.0;
    points.color.b = 0.0 / 255.0;
    points.color.a = 1.0;
    line.color.a = 1.0;
    points.lifetime = ros::Duration();
    line.lifetime = ros::Duration();

    getRobotLeaderPosition();
    started_ = true;
    ROS_INFO("Received start point.");
}

void FrontierRRTSearch::startSearch()
{
    unsigned long init[4] = {0x123, 0x234, 0x345, 0x456}, length = 7;
    MTRand_int32 irand(init, length); // 32-bit int generator
    // this is an example of initializing by an array
    // you may use MTRand(seed) with any 32bit integer
    // as a seed for a simpler initialization
    MTRand drand; // double in [0, 1) generator, already init

    // wait until map is received, when a map is received, mapData.header.seq will not be < 1
    ROS_INFO("Waiting for map topic %s", map_topic.c_str());
    while (mapData.header.seq < 1 or mapData.data.size() < 1)
    {
        ros::spinOnce();
        ros::Duration(0.1).sleep();
    }

    initMarkers();
    geometry_msgs::Point trans;
    trans = points.points[0];
    std::vector<float> xnew;
    xnew.push_back(trans.x);
    xnew.push_back(trans.y);
    V.push_back(xnew);

    points.points.clear();
    marker_pub.publish(points);

    std::vector<float> frontiers;
    int i = 0;
    float xr, yr;
    std::vector<float> x_rand, x_nearest, x_new;

    // Main loop
    ROS_INFO("Starting RRT");
    ros::Rate rate(100);

    while (ros::ok())
    {
        // Sample free
        x_rand.clear();
        int xp_r = drand() * mapData.info.width;
        int yp_r = drand() * mapData.info.height;
        std::vector<float> map_coords = pixelsToMap(xp_r, yp_r);
        xr = map_coords[0] + drand() * 0.2;
        yr = map_coords[1] + drand() * 0.2;

        x_rand.push_back(xr);
        x_rand.push_back(yr);

        // Nearest
        x_nearest = Nearest(V, x_rand);

        // Steer
        x_new = Steer(x_nearest, x_rand, eta);

        // ObstacleFree    1:free     -1:unkown (frontier region)      0:obstacle
        char checking = ObstacleFree(x_nearest, x_new);

        if (checking == -1)
        {
            exploration_goal.header.stamp = ros::Time(0);
            exploration_goal.header.frame_id = mapData.header.frame_id;
            exploration_goal.point.x = x_new[0];
            exploration_goal.point.y = x_new[1];
            exploration_goal.point.z = 0.0;
            geometry_msgs::Point p;
            p.x = x_new[0];
            p.y = x_new[1];
            p.z = 0.0;
            points.points.push_back(p);
            marker_pub.publish(points);
            points.points.clear();
        }

        else if (checking == 1)
        {
            V.push_back(x_new);
            geometry_msgs::Point p;
            p.x = x_new[0];
            p.y = x_new[1];
            p.z = 0.0;
            line.points.push_back(p);
            p.x = x_nearest[0];
            p.y = x_nearest[1];
            p.z = 0.0;
            line.points.push_back(p);
        }

        marker_pub.publish(line);

        ros::spinOnce();
        rate.sleep();
    }
}