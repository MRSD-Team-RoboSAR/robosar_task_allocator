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
            ROS_INFO("Node ID %d does not exist, cannot get.", id);
            return nullptr;
        }
        return nodes_[id];
    }
    void add_node(float x, float y, int parent)
    {
        nodes_[next_id_] = std::make_shared<Node>(x, y, next_id_, parent);
        auto parent_node = get_node(parent);
        if (parent_node)
            parent_node->add_child(next_id_);
        next_id_++;
    };
    void remove_node(int id)
    {
        if (nodes_.find(id) == nodes_.end())
        {
            ROS_INFO("Node ID %d does not exist, cannot remove.", id);
            return;
        }
        auto curr_node = get_node(id);
        std::unordered_set<int> children = curr_node->get_children();
        // A leaf node
        if (children.size() == 0)
        {
            if (curr_node->get_parent() != -1)
            {
                auto parent = get_node(curr_node->get_parent());
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
        remove_node(id);
    };

    std::unordered_map<int, std::shared_ptr<Node>> nodes_;
    int next_id_ = 0;
};

#endif // RRT_H