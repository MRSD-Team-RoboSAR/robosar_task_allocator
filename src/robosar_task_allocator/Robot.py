"""
Robot class
- Represents a robot
- Keeps track of robot id, current position, previous and next task.
"""
class Robot:

    def __init__(self, id, name, start, prev):
        self.id = id
        self.name = name
        self.v = 15
        self.pos = [float(start[0]), float(start[1])]
        self.pos_prev = [float(start[0]), float(start[1])]
        self.next = None
        self.prev = prev
        self.visited = [self.prev]