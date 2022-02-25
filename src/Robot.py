
class Robot:

    def __init__(self, id, start, prev):
        self.id = id
        self.v = 1
        self.pos = start
        self.next = -1
        self.prev = prev
        self.visited = [self.prev]