#ifndef RRT_H
#define RRT_H

#include <vector>
#include <queue>
#include <unordered_map>

#include "ros/ros.h"
#include "functions.h"
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
    }

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
    }

    std::shared_ptr<Node> get_parent_node(std::shared_ptr<Node> child)
    {
        return get_node(child->get_parent());
    }

    int nearest(float x, float y)
    {
        float min = std::numeric_limits<float>::max();
        int min_index = -1;
        float temp;
        // ROS_INFO("x: %f, y: %f", x, y);

        for (auto j = nodes_.begin(); j != nodes_.end(); j++)
        {
            // ROS_INFO("node: %d, x: %f, y: %f", j->second->get_id(), j->second->get_x(), j->second->get_y());
            temp = Norm(j->second->get_x(), j->second->get_y(), x, y);
            if (temp <= min)
            {
                min = temp;
                min_index = j->first;
            }
        }

        return min_index;
    }

    float dijkstra(int src, int dest)
    {
        using iPair = std::pair<float, int>;
        std::priority_queue<iPair, std::vector<iPair>, std::greater<iPair>> pq;
        std::vector<float> dist(nodes_.size(), std::numeric_limits<float>::max());
        pq.push({0.0, src});
        dist[src] = 0.0;

        while (!pq.empty())
        {
            int node_id = pq.top().second;
            auto node = get_node(node_id);
            pq.pop();

            // found dest
            if (node_id == dest)
            {
                return dist[dest];
            }

            for (int neighbor : node->get_children())
            {
                float weight = Norm(node->get_x(), node->get_y(), get_node(neighbor)->get_x(), get_node(neighbor)->get_y());
                if (dist[neighbor] > dist[node_id] + weight)
                {
                    dist[neighbor] = dist[node_id] + weight;
                    pq.push({dist[neighbor], neighbor});
                }
            }
            if (node->get_parent() != -1)
            {
                int parent_id = node->get_parent();
                float weight = Norm(node->get_x(), node->get_y(), get_node(parent_id)->get_x(), get_node(parent_id)->get_y());
                if (dist[parent_id] > dist[node_id] + weight)
                {
                    dist[parent_id] = dist[node_id] + weight;
                    pq.push({dist[parent_id], parent_id});
                }
            }
        }
        return dist[dest];
    }

    std::unordered_map<int, std::shared_ptr<Node>> nodes_;
    int next_id_ = 0;
};

#endif // RRT_H