#ifndef RRT_H
#define RRT_H

#include <vector>
#include <unordered_map>
#include "ros/ros.h"
#include "Node.h"

class RRT
{
public:
    RRT(){};
    ~RRT(){};
    std::shared_ptr<Node> get_node(int id)
    {
        if (nodes_.find(id) == nodes_.end())
        {
            ROS_INFO("Node ID does not exist.");
            return nullptr;
        }
        return nodes_[id];
    }
    void add_node(float x, float y, int parent)
    {
        nodes_[next_id_] = std::make_shared<Node>(x, y, next_id_, parent);
        next_id_++;
    };
    void remove_node(int id)
    {
        if (nodes_.find(id) == nodes_.end())
        {
            ROS_INFO("Node ID does not exist.");
            return;
        }
        auto curr_node = nodes_[id];
        std::unordered_set<int> children = curr_node->get_children();
        // A leaf node
        if (children.size() == 0)
        {
            if (curr_node->get_parent() != -1)
            {
                auto parent = nodes_[curr_node->get_parent()];
                parent->remove_child(id);
            }
            nodes_.erase(id);
            return;
        }
        // remove children
        for (int child_id : children)
        {
            remove_node(child_id);
        }
    };

    std::unordered_map<int, std::shared_ptr<Node>> nodes_;
    int next_id_ = 0;
};

#endif // RRT_H