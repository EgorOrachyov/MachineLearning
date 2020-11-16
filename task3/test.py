class Edge:

    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.s = 0

    def __eq__(self, other):
        return self.i == other.i and self.j == other.j

    def __hash__(self):
        return hash(self.i) + hash(self.j)

    def __str__(self):
        return "(src={0},dst={1},s={2})".format(self.i, self.j, self.s)


a = [Edge(1,2), Edge(2, 1), Edge(0, 0), Edge(1, 2)]
b = set(a)

print(b)