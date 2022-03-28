from abc import ABC, abstractmethod
# import robosar_task_allocator.mTSP as mTSP
import mTSP
import numpy as np
import matplotlib.pyplot as plt

class TA(ABC):
    def init(self, env):
        self.env = env
        self.objective_value = [0] * len(self.env.robots)

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
        self.objective_value[id] += self.env.adj[robot.prev][robot.next]
        self.env.frontier.add(min_node)


class TA_mTSP(TA):
    def init(self, env):
        self.env = env
        # self.tours = [[0, 6, 11, 12, 13, 14], [0, 1, 2, 3, 4, 5], [0, 7, 8, 9, 10]]
        self.tours = self.calculate_mtsp(True)
        self.objective_value = [0] * len(self.env.robots)

    def reached(self, id, curr_node):
        r = self.env.robots[id]
        if r.prev is not curr_node:
            r.prev = curr_node
            r.visited.append(curr_node)
            self.env.frontier.remove(curr_node)
            self.env.visited.add(curr_node)
            print("Robot {} reached node {}".format(id, curr_node))
            if len(self.env.visited) + len(self.env.frontier) < self.env.num_nodes:
                # new_subtour = self.calculate_mtsp(False)[id]
                # if len(new_subtour) > 1:
                #     self.tours[id] = new_subtour
                # plt.plot(self.env.nodes[:, 0], self.env.nodes[:, 1], 'ko', zorder=100)
                # for r in range(len(self.env.robots)):
                #     plt.plot(self.env.nodes[self.tours[r], 0], self.env.nodes[self.tours[r], 1], '-')
                # plt.pause(3)
                self.assign(id, curr_node)

    def assign(self, id, curr_node):
        robot = self.env.robots[id]
        if self.tours[id] and len(self.tours[id]) > 1:
            min_node = self.tours[id][1]
            print("Assigned robot {}: node {} at {}".format(id, min_node, self.env.nodes[min_node]))
            plt.plot(self.env.nodes[min_node][0], self.env.nodes[min_node][1], 'go', zorder=200)
            robot.next = min_node
            self.tours[id] = self.tours[id][1:]
            self.objective_value[id] += self.env.adj[robot.prev][robot.next]
            self.env.frontier.add(min_node)

    def get_next_adj(self):
        to_visit = []
        starts = [r.next for r in self.env.robots]
        for i in range(self.env.num_nodes):
            if (i not in self.env.visited) or (i in starts):
                to_visit.append(i)

        adj = np.zeros((len(to_visit), len(to_visit)))
        for idx1, n1 in enumerate(to_visit):
            for idx2, n2 in enumerate(to_visit):
                if self.env.adj[n1][n2] < 0:
                    adj[idx1][idx2] = 5*np.sqrt((self.env.nodes[n1, 0] - self.env.nodes[n2, 0]) ** 2 + (self.env.nodes[n1, 1] - self.env.nodes[n2, 1]) ** 2)
                else:
                    adj[idx1][idx2] = self.env.adj[n1][n2]
        return adj, to_visit

    def calculate_mtsp(self, initial):
        data = {}
        starts = [r.prev for r in self.env.robots]
        adj = self.tsp2hamiltonian(starts)
        if initial:
            data['starts'] = starts
            data['ends'] = [self.env.num_nodes+i for i in range(self.env.num_robots)]
        else:
            adj, to_visit = self.get_next_adj()
            data['starts'] = [to_visit.index(r.next) for r in self.env.robots]
            data['ends'] = [self.env.num_nodes+i for i in range(self.env.num_robots)]
        data['num_vehicles'] = len(self.env.robots)
        data['distance_matrix'] = adj
        tours = mTSP.main(data)
        
        if not initial:
            tours = [[to_visit[i] for i in tour] for tour in tours]
            
        # tours = self.refine_tours(tours)
        return tours

    def refine_tours(self, tours):
        tours_new = []
        # refine tours
        for tour in tours:
            visited = set()
            visited.add(tour[0])
            tour_new = [tour[0]]
            start = 0
            for i in range(len(tour) - 1):
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
                            E.append(np.sqrt((self.env.nodes[t, 0] - self.env.nodes[start, 0]) ** 2 + (
                                        self.env.nodes[t, 1] - self.env.nodes[start, 1]) ** 2))
                    if E:
                        min_node_i = np.argmin(np.array(E))
                        min_node = idx[min_node_i]
                start = min_node
                visited.add(min_node)
                tour_new.append(min_node)
            tours_new.append(tour_new)
        return tours_new
    
    def tsp2hamiltonian(self, starts):
        adj_new = np.zeros((self.env.num_nodes+self.env.num_robots, self.env.num_nodes+self.env.num_robots))
        adj_new[:self.env.num_nodes, :self.env.num_nodes] = self.env.adj
        for i in range(self.env.num_robots):
            for j in starts:
                adj_new[self.env.num_nodes+i, j] = 10e4
                adj_new[j, self.env.num_nodes + i] = 10e4
        return adj_new

            