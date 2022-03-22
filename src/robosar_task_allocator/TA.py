from abc import ABC, abstractmethod
import robosar_task_allocator.mTSP as mTSP
import numpy as np
import matplotlib.pyplot as plt

class TA(ABC):
    def init(self, env):
        self.env = env

    def reached(self, id, curr_node):
        r = self.env.robots[id]
        if r.prev is not curr_node:
            r.prev = curr_node
            r.visited.append(curr_node)
            self.env.frontier.remove(curr_node)
            self.env.visited.add(curr_node)
            print("Robot {} reached node {}".format(id, curr_node))
            if len(self.env.visited) + len(self.env.frontier) < self.env.num_nodes:
                self.assign(id, curr_node)

    @abstractmethod
    def assign(self, id, curr_node):
        pass


class TA_greedy(TA):
    def assign(self, id, curr_node):
        C = self.env.adj[curr_node, :]
        robot = self.env.robots[id]
        min_node_list = np.argsort(C)
        min_node = -1
        for i in min_node_list:
            if C[i] > 0 and i not in self.env.visited and i not in self.env.frontier:
                min_node = i
                break
        # No A* path
        if min_node == -1:
            print('USING EUCLIDEAN DISTANCE')
            E = []
            idx = []
            for i, node in enumerate(self.env.nodes):
                if i not in self.env.visited and i not in self.env.frontier:
                    idx.append(i)
                    E.append(np.sqrt((node[0] - robot.pos[0]) ** 2 + (node[1] - robot.pos[1]) ** 2))
            if E:
                min_node_i = np.argmin(np.array(E))
                min_node = idx[min_node_i]

        print("Assigned robot {}: node {} at {}".format(id, min_node, self.env.nodes[min_node]))
        plt.plot(self.env.nodes[min_node][0], self.env.nodes[min_node][1], 'go', zorder=101)
        robot.next = min_node
        self.env.frontier.add(min_node)


class TA_mTSP(TA):
    def init(self, env):
        self.env = env
        self.tours = self.calculate_mtsp()

    def assign(self, id, curr_node):
        robot = self.env.robots[id]
        next_tour_idx = len(robot.visited)
        if self.tours[id] and next_tour_idx < len(self.tours[id]):
            min_node = self.tours[id][next_tour_idx]
            print("Assigned robot {}: node {} at {}".format(id, min_node, self.env.nodes[min_node]))
            plt.plot(self.env.nodes[min_node][0], self.env.nodes[min_node][1], 'go', zorder=200)
            robot.next = min_node
            self.env.frontier.add(min_node)

    def get_initial_adj(self):
        adj = np.zeros((len(self.env.adj), len(self.env.adj)))
        for i in range(len(self.env.adj)):
            for j in range(len(self.env.adj)):
                if self.env.adj[i][j] < 0:
                    adj[i][j] = 1*np.sqrt((self.env.nodes[i, 0] - self.env.nodes[j, 0]) ** 2 + (self.env.nodes[i, 1] - self.env.nodes[j, 1]) ** 2)
                else:
                    adj[i][j] = self.env.adj[i][j]
        return adj

    def calculate_mtsp(self):
        data = {}
        adj = self.get_initial_adj()
        data['distance_matrix'] = adj
        data['num_vehicles'] = len(self.env.robots)
        data['starts'] = [r.prev for r in self.env.robots]
        data['ends'] = [0 for i in range(len(self.env.robots))]
        tours = mTSP.main(data)

        tours_new = []
        # refine tours
        for tour in tours:
            visited = set()
            visited.add(tour[0])
            tour_new = [tour[0]]
            start = 0
            for i in range(len(tour)-1):
                C = self.env.adj[start, tour]
                min_node_list = np.argsort(C)
                min_node = -1
                for j in min_node_list:
                    if C[j] > 0 and tour[j] not in visited:
                        min_node = tour[j]
                        break
                # No A* path
                if min_node == -1:
                    E = []
                    idx = []
                    for t in tour:
                        if t not in visited:
                            idx.append(t)
                            E.append(np.sqrt((self.env.nodes[t, 0] - self.env.nodes[start, 0]) ** 2 + (self.env.nodes[t, 1] - self.env.nodes[start, 1]) ** 2))
                    if E:
                        min_node_i = np.argmin(np.array(E))
                        min_node = idx[min_node_i]
                start = min_node
                visited.add(min_node)
                tour_new.append(min_node)
            tours_new.append(tour_new)
        return tours_new