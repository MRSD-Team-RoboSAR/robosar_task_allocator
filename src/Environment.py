import numpy as np
from itertools import combinations

class Environment:

    def __init__(self, nodes, adj, robots):
        self.nodes = nodes
        self.adj = adj
        self.num_nodes = len(nodes)

        self.robots = robots
        self.visited = set()
        for r in self.robots:
            self.visited.add(r.prev)
        self.frontier = set()




