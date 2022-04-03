import numpy as np
from itertools import combinations
import heapq

class Environment:

    def __init__(self, nodes, adj, robots):
        self.nodes = nodes
        self.adj = adj
        # fill in adjacency matrix to be fully connected
        for i in range(len(adj)):
            for j in range(i, len(adj)):
                if i == j:
                    self.adj[i][j] = 0
                elif self.adj[i][j] < 0:
                    d = self.dijkstra(i, j)
                    assert d > 0
                    self.adj[i][j] = d
                    self.adj[j][i] = self.adj[i][j]

        self.num_nodes = adj.shape[0]
        self.robots = robots
        self.id_dict = {}
        for idx, id in enumerate(self.robots.keys()):
            self.id_dict[id] = idx
        self.num_robots = len(self.robots)
        self.visited = set()
        for r in self.robots.values():
            self.visited.add(r.prev)
        self.frontier = set()

    def dijkstra(self, start, goal):
        m = self.adj.shape[0]
        dist = [float('inf') for _ in range(m)]
        dist[start] = 0
        minHeap = [(0, start)]  # distance, node
        d = -1
        while minHeap:
            d, s = heapq.heappop(minHeap)
            if d > dist[s]: continue
            if s == goal:
                return d  # Reach to goal
            neighbors = []
            for i, cost in enumerate(self.adj[s]):
                if cost > 0:
                    neighbors.append(i)
            for i in neighbors:
                newDist = d + self.adj[s][i]
                if dist[i] > newDist:
                    dist[i] = newDist
                    heapq.heappush(minHeap, (dist[i], i))
        return d




