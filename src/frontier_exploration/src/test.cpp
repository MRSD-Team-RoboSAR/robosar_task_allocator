#include <iostream>

#include "RRT.h"

int main(int argc, char **argv)
{
    RRT rrt = RRT();
    rrt.add_node(0.0, 0.0, -1); // 0
    rrt.add_node(0.0, 1.0, 0);  // 1
    rrt.add_node(1.0, 1.0, 0);  // 2
    rrt.add_node(2.0, 1.0, 1);  // 3
    rrt.add_node(2.0, 2.0, 1);  // 4
    rrt.add_node(2.0, 2.0, 3);  // 5
    rrt.add_node(2.0, 2.0, 4);  // 6
    rrt.add_node(2.0, 2.0, 2);  // 7
    rrt.remove_node(1);

    // 7, 2, 0
    for (auto j = rrt.nodes_.begin(); j != rrt.nodes_.end(); j++)
    {
        std::cout << j->first << "->";
    }
    std::cout << std::endl;
    // 2
    auto children = rrt.get_node(0)->get_children();
    for (auto j : children)
    {
        std::cout << j << "->";
    }
    // 7
    std::cout << std::endl;
    children = rrt.get_node(2)->get_children();
    for (auto j : children)
    {
        std::cout << j << "->";
    }
}