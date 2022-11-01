#ifndef NODE_H
#define NODE_H

#include <unordered_set>

class Node
{
public:
    Node(float x, float y, int id, int parent_id, float info_gain_radius, bool is_coverage_node) : 
        x_(x), y_(y), id_(id), parent_(parent_id), info_gain_radius_(info_gain_radius), is_coverage_node_(is_coverage_node){};

    ~Node(){};
    float get_x() { return x_; }
    float get_y() { return y_; }
    int get_id() { return id_; }
    float get_info_gain_radius() { return info_gain_radius_; };
    bool is_coverage_node() { return is_coverage_node_; };
    void set_non_coverage_node() { is_coverage_node_ = false; };
    std::pair<float, float> get_coord() { return std::make_pair(x_, y_); }
    void add_child(int child)
    {
        children_.insert(child);
    }
    void remove_child(int child)
    {
        if (children_.find(child) == children_.end())
        {
            ROS_INFO("Child %d does not exist, cannot remove.", child);
            return;
        }
        children_.erase(child);
    }
    std::unordered_set<int> get_children() { return children_; }
    int get_parent() { return parent_; }

private:
    bool is_coverage_node_;
    float info_gain_radius_;
    float x_;
    float y_;
    int id_;
    const int parent_;
    std::unordered_set<int> children_;
};

#endif // NODE_H