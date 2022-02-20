import numpy as np

class Environment:

    def __init__(self, n, robots):
        self.nodes = []
        for i in range(n):
            for j in range(n):
                self.nodes.append( [i, j] )
        self.num_nodes = n*n

        self.robots = robots
        self.visited = set()
        for r in self.robots:
            self.visited.add(r.prev)
        self.frontier = set()



