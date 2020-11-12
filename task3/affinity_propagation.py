import random


class Edge:

    def __init__(self, src, dst, s):
        self.src = src
        self.dst = dst
        self.s = s
        self.a = 0
        self.r = 0

    def __str__(self):
        return "(src={0},dst={1},s={2},a={3},r={4})".format(self.src, self.dst, self.s, self.a, self.r)


class Graph:

    def __init__(self):
        self.oute = []
        self.ine = []
        self.edges = []
        self.m = 0
        self.n = 0


def get_data(path):
    graph = Graph()

    with open(path, 'r') as datafile:
        for line in datafile.readlines():
            ids = line.replace('\n', '').split('\t')
            i = int(ids[0])
            j = int(ids[1])
            s = 1.0 + random.uniform(0, 0.5) # random.uniform(0, 1) * 1e-16

            graph.edges.append(Edge(i, j, s))
            graph.m += 1
            graph.n = max(graph.n, i, j)

    graph.n += 1

    for i in range(graph.n):
        graph.edges.append(Edge(i, i, random.uniform(-25, -15)))  # todo
        graph.m += 1

    graph.ine = [[] for _ in range(graph.n)]
    graph.oute = [[] for _ in range(graph.n)]

    for e in graph.edges:
        graph.oute[e.src].append(e)
        graph.ine[e.dst].append(e)

    return graph


def smooth(a, b, smoothing):
    return a * smoothing + (1 - smoothing) * b


def aff_prop(graph, max_iterations, max_stagnation, exp_smoothing):
    print(graph.n)
    print(graph.m)
    stagnation = 0

    examplars = [-1 for _ in range(graph.n)]

    for iter in range(max_iterations):
        print("Iteration: %s" % iter)

        for i in range(graph.n):
            edges = graph.oute[i]
            count = len(edges)
            max1, max2, argmax1 = -1e16, -1e16, -1
            for k in range(count):
                sum = edges[k].s + edges[k].a

                if sum > max1:
                    max1, sum = sum, max1
                    argmax1 = k
                if sum > max2:
                    max2 = sum

            for k in range(count):
                if k != argmax1:
                    edges[k].r = smooth(edges[k].r, edges[k].s - max1, exp_smoothing)
                else:
                    edges[k].r = smooth(edges[k].r, edges[k].s - max2, exp_smoothing)

        for k in range(graph.n):
            edges = graph.ine[k]
            count = len(edges)
            sum = 0.0
            for i in range(count - 1):
                sum += max(0, edges[i].r)

            r_k_k = edges[-1].r
            for i in range(count - 1):
                edges[i].a = smooth(edges[i].a, min(0, r_k_k + sum - max(0.0, edges[i].r)), exp_smoothing)

            edges[-1].a = smooth(edges[-1].a, sum, exp_smoothing)

        something_changed = False
        for i in range(graph.n):
            edges = graph.oute[i]
            count = len(edges)
            max_v = -1e16
            argmax = -1

            for k in range(count):
                v = edges[k].a + edges[k].r

                if v > max_v:
                    max_v = v
                    argmax = edges[k].dst

            if examplars[i] != argmax:
                examplars[i] = argmax
                something_changed = True

        if something_changed:
            print("Something changed: %s" % len(set(examplars)))
            stagnation = 0
        else:
            stagnation += 1

        if stagnation >= max_stagnation:
            break

    return examplars


edges_path = "./dataset/loc-gowalla_edges.txt"
graph = get_data(edges_path)

max_iterations = 100
max_stagnation = 50
exp_smoothing = 0.6

examplars = aff_prop(graph, max_iterations, max_stagnation, exp_smoothing)
as_set = set(examplars)
print("======================>", len(as_set))
