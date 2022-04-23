"""
Task Allocation classes
- TA_greedy: greedy task allocator
- TA_mTSP: Multiple Traveling Salesman Problem task allocator
"""

from abc import ABC, abstractmethod
import robosar_task_allocator.mTSP_utils as mTSP_utils
# import mTSP_utils
import numpy as np
import matplotlib.pyplot as plt


class TA(ABC):
    """
    Task allocation abstract class
    """

    def init(self, env):
        """ initializes task allocation object
        env: Environment object
        """
        self.env = env
        self.objective_value = [0] * len(self.env.robots)


    def reached(self, id, curr_node):
        """ called after robot completes task
        id: int
        curr_node: int
        """
        r = self.env.robots[id]
        if r.prev is not curr_node:
            r.prev = curr_node
            r.visited.append(curr_node)
            self.env.frontier.remove(curr_node)
            self.env.visited.add(curr_node)
            print("Robot {} reached node {}".format(id, curr_node))
            if len(self.env.visited) + len(self.env.frontier) < self.env.num_nodes:
                self.assign(id, curr_node)
        else:
            r.done = True

    @abstractmethod
    def assign(self, id, curr_node):
        """ called by reached() to assign robot its next task
       id: int
       curr_node: int
       """
        pass

class TA_greedy(TA):
    """
    TA_greedy: greedy task allocator
    """

    def assign(self, id, curr_node):
        # Find next unvisited min cost task
        C = self.env.adj[curr_node, :]
        robot = self.env.robots[id]
        min_node_list = np.argsort(C)
        min_node = -1
        for i in min_node_list:
            if C[i] > 0 and i not in self.env.visited and i not in self.env.frontier:
                min_node = i
                break
        # No feasible path to any task
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
        self.objective_value[self.env.id_dict[id]] += self.env.adj[robot.prev][robot.next]
        self.env.frontier.add(min_node)


class TA_mTSP(TA):
    """
    TA_mTSP: Multiple Traveling Salesman Problem task allocator
    """

    def init(self, env, timeout=5):
        self.timeout = timeout
        self.env = env
        self.tours = self.calculate_mtsp(True)
        self.objective_value = [0] * len(self.env.robots)

    def reached(self, id, curr_node):
        r = self.env.robots[id]
        # print("{}: {}".format(id, self.tours[self.env.id_dict[id]]))
        self.tours[self.env.id_dict[id]].pop(0)
        if not self.tours[self.env.id_dict[id]]: # if robot has finished
            r.done = True
        if r.next is None: # first assignment
            plt.plot(self.env.nodes[r.prev][0], self.env.nodes[r.prev][1], 'go', zorder=200)
            self.assign(id, curr_node)
        elif r.prev is not curr_node: # if robot not done
            r.prev = curr_node
            r.visited.append(curr_node)
            plt.plot(self.env.nodes[r.prev][0], self.env.nodes[r.prev][1], 'go', zorder=200)
            self.env.frontier.remove(curr_node)
            self.env.visited.add(curr_node)
            print("Robot {} reached node {}".format(id, curr_node))
            if len(self.env.visited) + len(self.env.frontier) < self.env.num_nodes:
                self.assign(id, curr_node)
        elif self.tours[self.env.id_dict[id]]: # if robot was done, but is now reassigned
            r.done = False
            if len(self.env.visited) + len(self.env.frontier) < self.env.num_nodes:
                self.assign(id, curr_node)
        plt.pause(0.1)


    def assign(self, id, curr_node):
        robot = self.env.robots[id]
        if self.tours[self.env.id_dict[id]]:
            min_node = self.tours[self.env.id_dict[id]][0]
            print("Assigned robot {}: node {} at {}".format(id, min_node, self.env.nodes[min_node]))
            robot.next = min_node
            self.objective_value[self.env.id_dict[id]] += self.env.adj[robot.prev][robot.next]
            self.env.frontier.add(min_node)


    def get_next_adj(self):
        """
        Get cost matrix for unvisited tasks
        """
        to_visit = []
        starts = [r.next for r in self.env.robots.values()]
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


    def calculate_mtsp(self, initial = True):
        """
        Uses Google OR-Tools VRP Library to solve for optimal tours
        initial: Bool
        """
        data = {}
        starts = [r.prev for r in self.env.robots.values()]
        if initial: # first solution
            adj = self.tsp2hamiltonian(self.env.adj, starts)
            data['starts'] = starts
            data['ends'] = [self.env.num_nodes+i for i in range(self.env.num_robots)]
        else:
            adj, to_visit = self.get_next_adj()
            # prev = {r.id: r.prev for r in self.env.robots.values()}
            data['starts'] = [to_visit.index(r.next) for r in self.env.robots.values()]
            data['ends'] = [len(to_visit)+i for i in range(self.env.num_robots)]
        data['num_vehicles'] = len(self.env.robots)
        data['distance_matrix'] = adj
        tours = mTSP_utils.solve(data, self.timeout)
        
        if not initial:
            self.tours = [[to_visit[i] for i in tour] for tour in tours]
            for id, robot in self.env.robots.items():
                if len(self.tours[self.env.id_dict[id]]) > 1:
                    robot.done = False
                else:
                    robot.done = True

        return tours


    def tsp2hamiltonian(self, adj, starts):
        """
        Converts TSP problem formulation to Hamiltonian Path
        adj: nxn np.array
        starts: n list
        """
        adj_new = np.zeros((len(adj)+self.env.num_robots, len(adj)+self.env.num_robots))
        adj_new[:len(adj), :len(adj)] = adj
        for i in range(self.env.num_robots):
            for j in range(i+1, self.env.num_robots):
                adj_new[len(adj) + i, len(adj)+j] = 10e4
                adj_new[len(adj)+j, len(adj) + i] = 10e4
        return adj_new

            