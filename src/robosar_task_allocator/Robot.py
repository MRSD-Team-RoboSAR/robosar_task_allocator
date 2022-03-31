
class Robot:

    def __init__(self, id, start, prev):
        self.id = id
        self.name = 'robot_' + str(self.id)
        self.v = 15
        self.pos = [float(start[0]), float(start[1])]
        self.pos_prev = [float(start[0]), float(start[1])]
        self.next = None
        self.prev = prev
        self.visited = [self.prev]