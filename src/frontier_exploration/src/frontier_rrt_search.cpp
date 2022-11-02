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
    targets_pub = nh_.advertise<geometry_msgs::PointStamped>(ns + "/detected_points", 10);
    marker_pub = nh_.advertise<visualization_msgs::Marker>(ns + "_shapes", 10);
    rrt_path_service_ = nh_.advertiseService("rrt_path_cost", &FrontierRRTSearch::getPathCost, this);
}

// Subscribers callback functions---------------------------------------
void FrontierRRTSearch::mapCallBack(const nav_msgs::OccupancyGrid::ConstPtr &msg)
{
    mapData = *msg;
}

// Service
bool FrontierRRTSearch::getPathCost(robosar_messages::rrt_path_cost::Request &req, robosar_messages::rrt_path_cost::Response &resp)
{
    boost::mutex::scoped_lock(rrt_mutex_);
    ROS_INFO("rrt_path_cost request received.");
    int robot_node_id = rrt_.nearest(req.robot_x, req.robot_y);
    int goal_node_id = rrt_.nearest(req.goal_x, req.goal_y);
    if (robot_node_id == -1 || goal_node_id == -1)
        return false;
    ROS_INFO("Finding path from %d to %d", robot_node_id, goal_node_id);
    float cost = rrt_.dijkstra(robot_node_id, goal_node_id);
    ROS_INFO("sending response.");
    resp.cost = cost;
    return true;
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
        ROS_INFO("%s", robot_leader.c_str());
        tf = tf_buffer_.lookupTransform("map", robot_leader + "/base_link", now);

        geometry_msgs::Point p;
        p.x = tf.transform.translation.x;
        p.y = tf.transform.translation.y;
        p.z = 0;

        marker_points.points.push_back(p);
        marker_pub.publish(marker_points);
    }
    catch (tf2::TransformException &ex)
    {
        ROS_ERROR("%s", ex.what());
    }
}

// Steer function
std::pair<float, float> FrontierRRTSearch::Steer(std::pair<float, float> &x_nearest, std::pair<float, float> &x_rand, float eta)
{
    std::pair<float, float> x_new;

    if (Norm(x_nearest.first, x_nearest.second, x_rand.first, x_rand.second) <= eta)
    {
        x_new = x_rand;
    }
    else
    {
        float m = (x_rand.second - x_nearest.second) / (x_rand.first - x_nearest.first);
        if (x_rand.first == x_nearest.first)
        {
            x_new = {x_nearest.first, x_nearest.second + eta};
        }
        x_new.first = (sign(x_rand.first - x_nearest.first)) * (sqrt((pow(eta, 2)) / ((pow(m, 2)) + 1))) + x_nearest.first;
        x_new.second = m * (x_new.first - x_nearest.first) + x_nearest.second;
    }
    return x_new;
}

std::pair<float, float> FrontierRRTSearch::pixelsToMap(int x_pixel, int y_pixel)
{
    std::pair<float, float> map_coords;
    float scale = mapData.info.resolution;
    float x_origin = mapData.info.origin.position.x;
    float y_origin = mapData.info.origin.position.y;
    map_coords = {x_pixel * scale + x_origin, y_pixel * scale + y_origin};
    return map_coords;
}

// gridValue function
int FrontierRRTSearch::gridValue(std::pair<float, float> &Xp)
{
    float resolution = mapData.info.resolution;
    float Xstartx = mapData.info.origin.position.x;
    float Xstarty = mapData.info.origin.position.y;

    float width = mapData.info.width;
    std::vector<signed char> Data = mapData.data;

    // returns grid value at "Xp" location
    // map data:  100 occupied      -1 unknown       0 free
    float indx = (floor((Xp.second - Xstarty) / resolution) * width) + (floor((Xp.first - Xstartx) / resolution));
    int out;
    out = Data[int(indx)];
    return out;
}

char FrontierRRTSearch::ObstacleFree(std::pair<float, float> &xnear, std::pair<float, float> &xnew)
{
    float rez = float(mapData.info.resolution) * .2;
    float stepz = int(ceil(Norm(xnew.first, xnew.second, xnear.first, xnear.second)) / rez);
    std::pair<float, float> xi = xnear;
    char obs = 0;
    char unk = 0;

    geometry_msgs::Point p;
    for (int c = 0; c < stepz; c++)
    {
        xi = Steer(xi, xnew, rez);

        if (gridValue(xi) == 100)
            obs = 1;
        if (gridValue(xi) == -1)
        {
            unk = 1;
            break;
        }
    }
    char out = 0;
    xnew = xi;
    if (unk == 1)
        out = -1;
    if (obs == 1)
        out = 0;
    if (obs != 1 && unk != 1)
        out = 1;

    return out;
}

void FrontierRRTSearch::pruneRRT()
{
    std::vector<int> to_remove;
    for (auto j = rrt_.nodes_.begin(); j != rrt_.nodes_.end(); j++)
    {
        if (j->second->get_parent() == -1)
            continue;
        std::pair<float, float> x_child = j->second->get_coord();
        std::pair<float, float> x_parent = rrt_.get_parent_node(j->second)->get_coord();
        char checking = ObstacleFree(x_parent, x_child);
        if (checking == 0)
        {
            to_remove.push_back(j->first);
        }
    }
    for (int id : to_remove)
    {
        rrt_.remove_node(id);
    }
    // visualization
    marker_line.points.clear();
    for (auto j = rrt_.nodes_.begin(); j != rrt_.nodes_.end(); j++)
    {
        if (j->second->get_parent() == -1)
            continue;
        geometry_msgs::Point p;
        p.x = j->second->get_x();
        p.y = j->second->get_y();
        p.z = 0.0;
        marker_line.points.push_back(p);
        p.x = rrt_.get_parent_node(j->second)->get_x();
        p.y = rrt_.get_parent_node(j->second)->get_y();
        p.z = 0.0;
        marker_line.points.push_back(p);
    }
}

void FrontierRRTSearch::initMarkers()
{
    // visualizations  points and lines..
    marker_points.header.frame_id = mapData.header.frame_id;
    marker_line.header.frame_id = mapData.header.frame_id;
    marker_points.header.stamp = ros::Time(0);
    marker_line.header.stamp = ros::Time(0);

    marker_points.ns = marker_line.ns = "markers";
    marker_points.id = 0;
    marker_line.id = 1;

    marker_points.type = marker_points.POINTS;
    marker_line.type = marker_line.LINE_LIST;

    // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
    marker_points.action = marker_points.ADD;
    marker_line.action = marker_line.ADD;
    marker_points.pose.orientation.w = 1.0;
    marker_line.pose.orientation.w = 1.0;
    marker_line.scale.x = 0.03;
    marker_line.scale.y = 0.03;
    marker_points.scale.x = 0.3;
    marker_points.scale.y = 0.3;

    marker_line.color.r = 9.0 / 255.0;
    marker_line.color.g = 91.0 / 255.0;
    marker_line.color.b = 236.0 / 255.0;
    marker_points.color.r = 255.0 / 255.0;
    marker_points.color.g = 0.0 / 255.0;
    marker_points.color.b = 0.0 / 255.0;
    marker_points.color.a = 1.0;
    marker_line.color.a = 1.0;
    marker_points.lifetime = ros::Duration();
    marker_line.lifetime = ros::Duration();

    getRobotLeaderPosition();
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
    trans = marker_points.points[0];
    rrt_.add_node(trans.x, trans.y, -1);

    marker_points.points.clear();
    marker_pub.publish(marker_points);

    int i = 0;
    float xr, yr;
    std::pair<float, float> x_rand, x_nearest, x_new;

    // Main loop
    ROS_INFO("Starting RRT");
    ros::Rate rate(50);
    geometry_msgs::PointStamped exploration_goal;

    int prune_counter = 0;

    while (ros::ok())
    {
        rrt_mutex_.lock();
        if (prune_counter == 500)
        {
            pruneRRT();
            prune_counter = 0;
        }

        // Sample free
        int xp_r = drand() * mapData.info.width;
        int yp_r = drand() * mapData.info.height;
        std::pair<float, float> map_coords = pixelsToMap(xp_r, yp_r);
        xr = map_coords.first + drand() * 0.2;
        yr = map_coords.second + drand() * 0.2;

        x_rand = {xr, yr};

        // Nearest
        int x_nearest_id = rrt_.nearest(x_rand.first, x_rand.second);
        // ROS_INFO("%d", rrt_.nodes_.size());
        // ROS_INFO("%d", x_nearest_id);
        if (x_nearest_id == -1)
            continue;
        x_nearest = rrt_.get_node(x_nearest_id)->get_coord();

        // Steer
        x_new = Steer(x_nearest, x_rand, eta);

        // ObstacleFree    1:free     -1:unkown (frontier region)      0:obstacle
        char checking = ObstacleFree(x_nearest, x_new);

        if (checking == -1)
        {
            exploration_goal.header.stamp = ros::Time(0);
            exploration_goal.header.frame_id = mapData.header.frame_id;
            exploration_goal.point.x = x_nearest.first;
            exploration_goal.point.y = x_nearest.second;
            exploration_goal.point.z = 0.0;
            geometry_msgs::Point p;
            p.x = x_nearest.first;
            p.y = x_nearest.second;
            p.z = 0.0;
            marker_points.points.push_back(p);
            marker_pub.publish(marker_points);
            targets_pub.publish(exploration_goal);
            marker_points.points.clear();
        }

        else if (checking == 1)
        {
            rrt_.add_node(x_new.first, x_new.second, x_nearest_id);
            geometry_msgs::Point p;
            p.x = x_new.first;
            p.y = x_new.second;
            p.z = 0.0;
            marker_line.points.push_back(p);
            p.x = x_nearest.first;
            p.y = x_nearest.second;
            p.z = 0.0;
            marker_line.points.push_back(p);
        }

        marker_pub.publish(marker_line);
        prune_counter++;
        rrt_mutex_.unlock();

        ros::spinOnce();
        rate.sleep();
    }
}