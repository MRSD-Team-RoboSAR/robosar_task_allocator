from abc import ABC, abstractmethod
import robosar_task_allocator.mTSP_utils as mTSP_utils
# import mTSP_utils
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
                # if len(self.env.visited)%30 == 0:
                #     new_tour = self.calculate_mtsp(False)
                #     self.tours = new_tour
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
            if i not in self.env.visited or i in starts:
                to_visit.append(i)

        adj = np.zeros((len(to_visit), len(to_visit)))
        for idx1, n1 in enumerate(to_visit):
            for idx2, n2 in enumerate(to_visit):
                adj[idx1][idx2] = self.env.adj[n1][n2]
        start_idx = [to_visit.index(s) for s in starts]
        adj = self.tsp2hamiltonian(adj, start_idx)
        return adj, to_visit


    def calculate_mtsp(self, initial):
        data = {}
        starts = [r.prev for r in self.env.robots]
        if initial:
            adj = self.tsp2hamiltonian(self.env.adj, starts)
            data['starts'] = starts
            data['ends'] = [self.env.num_nodes+i for i in range(self.env.num_robots)]
        else:
            adj, to_visit = self.get_next_adj()
            data['starts'] = [to_visit.index(r.next) for r in self.env.robots]
            data['ends'] = [len(to_visit)+i for i in range(self.env.num_robots)]
        data['num_vehicles'] = len(self.env.robots)
        data['distance_matrix'] = adj
        tours = mTSP_utils.solve(data)
        
        if not initial:
            tours = [[to_visit[i] for i in tour] for tour in tours]
            
        return tours


    def tsp2hamiltonian(self, adj, starts):
        adj_new = np.zeros((len(adj)+self.env.num_robots, len(adj)+self.env.num_robots))
        adj_new[:len(adj), :len(adj)] = adj
        for i in range(self.env.num_robots):
            for j in starts:
                adj_new[len(adj)+i, j] = 10e4
                adj_new[j, len(adj) + i] = 10e4
            for j in range(i+1, self.env.num_robots):
                adj_new[len(adj) + i, len(adj)+j] = 10e4
                adj_new[len(adj)+j, len(adj) + i] = 10e4
        return adj_new

            