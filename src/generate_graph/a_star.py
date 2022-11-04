from collections import deque
import math
from heapq import heappush, heappop
from .utils import dist2d


def _get_movements_4n():
    """
    Get all possible 4-connectivity movements.
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    return [(1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0)]


def _get_movements_8n():
    """
    Get all possible 8-connectivity movements. Equivalent to get_movements_in_radius(1).
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    s2 = math.sqrt(2)
    return [
        (1, 0, 1.0),
        (0, 1, 1.0),
        (-1, 0, 1.0),
        (0, -1, 1.0),
        (1, 1, s2),
        (-1, 1, s2),
        (-1, -1, s2),
        (1, -1, s2),
    ]


def get_closest_free_cell(start, gmap):
    pix_range = 10
    x_min = int(max(start[0] - pix_range, 0))
    x_max = int(min(start[0] + pix_range, gmap.dim_cells[0] - 1))
    y_min = int(max(start[1] - pix_range, 0))
    y_max = int(min(start[1] + pix_range, gmap.dim_cells[1] - 1))
    q = deque([])
    visited = [
        [False for _ in range(x_min, x_max + 1)] for _ in range(y_min, y_max + 1)
    ]
    q.append(start)
    visited[start[0] - x_min][start[1] - y_min] = True
    iter = 0
    prop_model = _get_movements_8n()
    while len(q) > 0:
        curr = q.popleft()

        if gmap.get_data_idx(curr) == 0:
            print("Found free node {} after {} iterations".format(curr, iter))
            return tuple(curr)

        for p in prop_model:
            x_new = p[0] + curr[0]
            y_new = p[1] + curr[1]
            if (
                x_new >= x_min
                and x_new <= x_max
                and y_new >= y_min
                and y_new <= y_max
                and not visited[x_new - x_min][y_new - y_min]
            ):
                q.append([x_new, y_new])
                visited[x_new - x_min][y_new - y_min] = True
        iter += 1

    print("free node not found within range")
    return tuple(start)


def a_star(start_m, goal_m, gmap, movement="8N", occupancy_cost_factor=3):
    """
    A* for 2D occupancy grid.
    :param start_m: start node (x, y) in meters
    :param goal_m: goal node (x, y) in meters
    :param gmap: the grid map
    :param movement: select between 4-connectivity ('4N') and 8-connectivity ('8N', default)
    :param occupancy_cost_factor: a number the will be multiplied by the occupancy probability
        of a grid map cell to give the additional movement cost to this cell (default: 3).
    :return: a tuple that contains: (the resulting path in meters, the resulting path in data array indices)
    """
    total_dist = 0.0
    # get array indices of start and goal
    start = gmap.get_index_from_coordinates(start_m[0], start_m[1])
    goal = gmap.get_index_from_coordinates(goal_m[0], goal_m[1])
    print("start: ", start)
    print("goal: ", goal)

    # check if start and goal nodes correspond to free spaces
    if gmap.is_occupied_idx(start):
        print("Start node {} is not traversable".format(start))
        start = get_closest_free_cell(start, gmap)

    if gmap.is_occupied_idx(goal):
        # print(gmap.is_occupied_idx(goal[0], goal[1]))
        print("Goal node {} is not traversable".format(goal))
        goal = get_closest_free_cell(goal, gmap)

    # add start node to front
    # front is a list of (total estimated cost to goal, total cost from start to node, node, previous node)
    start_node_cost = 0
    start_node_estimated_cost_to_goal = dist2d(start, goal) + start_node_cost
    front = [(start_node_estimated_cost_to_goal, start_node_cost, start, None)]

    # use a dictionary to remember where we came from in order to reconstruct the path later on
    came_from = {}

    # get possible movements
    if movement == "4N":
        movements = _get_movements_4n()
    elif movement == "8N":
        movements = _get_movements_8n()
    else:
        raise ValueError("Unknown movement")

    # while there are elements to investigate in our front.
    iter = 0
    while front:
        if iter == 10000:
            print("WARN: astar timeout")
            return [], [], 5 * math.dist(start, goal)
        # get smallest item and remove from front.
        element = heappop(front)

        # if this has been visited already, skip it
        total_cost, cost, pos, previous = element
        if gmap.is_visited_idx(pos):
            continue

        # now it has been visited, mark with cost
        gmap.mark_visited_idx(pos)

        # set its previous node
        came_from[pos] = previous

        # if the goal has been reached, we are done!
        if pos == goal:
            break

        # check all neighbors
        for dx, dy, deltacost in movements:
            # determine new position
            new_x = pos[0] + dx
            new_y = pos[1] + dy
            new_pos = (new_x, new_y)

            # check whether new position is inside the map
            # if not, skip node
            if not gmap.is_inside_idx(new_pos):
                continue

            # add node to front if it was not visited before and is in free space
            if (not gmap.is_visited_idx(new_pos)) and (gmap.get_data_idx(new_pos) == 0):
                potential_function_cost = (
                    gmap.get_data_idx(new_pos) * occupancy_cost_factor
                )
                new_cost = cost + deltacost + potential_function_cost
                new_total_cost_to_goal = (
                    new_cost + dist2d(new_pos, goal) + potential_function_cost
                )

                heappush(front, (new_total_cost_to_goal, new_cost, new_pos, pos))
        iter += 1

    # reconstruct path backwards (only if we reached the goal)
    path = []
    path_idx = []
    if pos == goal:
        pos_m_x = None
        while pos:
            path_idx.append(pos)
            if pos_m_x is None:
                total_dist = 0.0
            else:
                total_dist += math.dist([pos_m_x, pos_m_y], [pos[0], pos[1]])
            # transform array indices to meters
            pos_m_x, pos_m_y = gmap.get_coordinates_from_index(pos[0], pos[1])
            path.append((pos_m_x, pos_m_y))
            pos = came_from[pos]

        # reverse so that path is from start to goal.
        path.reverse()
        path_idx.reverse()

    return path, path_idx, total_dist
