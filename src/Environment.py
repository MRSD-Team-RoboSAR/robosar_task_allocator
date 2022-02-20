import Robot
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

    def assign(self, id):
        r = self.robots[id]
        idx = []
        C = []
        robot = self.robots[id]
        for i, node in enumerate(self.nodes):
            if i not in self.visited and i not in self.frontier:
                idx.append(i)
                C.append(np.sqrt((node[0]-robot.pos[0])**2 + (node[1]-robot.pos[1])**2))
        if C:
            min_node = np.argmin(np.array(C))
            min_node_i = idx[min_node]
            print("Assigned robot {}: node {} at {}".format(id, min_node_i, self.nodes[min_node_i]))
            robot.next = min_node_i
            self.frontier.add(min_node_i)

    def reached(self, id, curr_node):
        r = self.robots[id]
        if r.prev is not curr_node:
            r.prev = curr_node
            r.visited.append(curr_node)
            self.frontier.remove(curr_node)
            self.visited.add(curr_node)
            print("Robot {} reached node {}".format(id, curr_node))
            if len(self.visited) + len(self.frontier) < self.num_nodes:
                self.assign(id)

    def move(self, dt):
        for id, r in enumerate(self.robots):
            goal = self.nodes[r.next]
            dir = np.array([goal[0]-r.pos[0], goal[1]-r.pos[1]])
            dist = np.linalg.norm(dir)
            if dist > 0:
                dir = dir/dist
                dx = r.v*dt
                if dx > dist:
                    r.pos[0] = goal[0]
                    r.pos[1] = goal[1]
                    self.reached(id, r.next)
                else:
                    x_next = r.pos + dir*dx
                    r.pos[0] = x_next[0]
                    r.pos[1] = x_next[1]
            else:
                self.reached(id, r.next)

