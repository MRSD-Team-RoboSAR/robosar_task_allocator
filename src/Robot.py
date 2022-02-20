
class Robot:

    def __init__(self, id, start):
        self.id = id
        self.v = 1
        self.pos = start
        self.next = -1
        self.prev = 0
        self.visited = [self.prev]