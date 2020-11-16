import random


class Edge:

    def __init__(self, src, dst, s):
        self.src = src
        self.dst = dst
        self.s = s
        self.a = 0
        self.r = 0

    def __eq__(self, other):
        return self.src == other.src and self.dst == other.dst

    def __hash__(self):
        return hash(self.src) + hash(self.dst)

    def __str__(self):
        return "(src={0},dst={1},s={2},a={3},r={4})".format(self.src, self.dst, self.s, self.a, self.r)


class Graph:

    def __init__(self):
        self.oute = []
        self.ine = []
        self.edges = []
        self.m = 0
        self.n = 0

    def finalize_build(self):
        for i in range(self.n):
            self.edges.append(Edge(i, i, -1.5 - random.uniform(0, 1e-4)))  # with noise
            self.m += 1

        self.ine = [[] for _ in range(self.n)]
        self.oute = [[] for _ in range(self.n)]

        for e in self.edges:
            self.oute[e.src].append(e)
            self.ine[e.dst].append(e)


def get_data(path):
    graph = Graph()

    with open(path, 'r') as datafile:
        for line in datafile.readlines():
            ids = line.replace('\n', '').split('\t')
            i = int(ids[0])
            j = int(ids[1])
            s = 1.0 + random.uniform(0, 1e-4) # with noise

            graph.edges.append(Edge(i, j, s))
            graph.m += 1
            graph.n = max(graph.n, i, j)

    graph.n += 1
    graph.finalize_build()

    return graph


def smooth(a, b, smoothing):
    return a * smoothing + (1 - smoothing) * b


def aff_prop(graph, max_iterations, max_stagnation, exp_smoothing):
    print("Vertices count: ", graph.n)
    print("Edges count", graph.m)

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
            print("-- Something changed: %s" % len(set(examplars)))
            stagnation = 0
        else:
            stagnation += 1

        if stagnation >= max_stagnation:
            break

    return examplars


def load_checkins(path):
    checkins = dict()

    with open(path, 'r') as datafile:
        for line in datafile.readlines():
            dt = line.replace('\n', '').split('\t')

            user_id = int(dt[0])
            location_id = int(dt[4])

            if user_id in checkins:
                checkins[user_id].append(location_id)
            else:
                checkins[user_id] = [location_id]

    return checkins


def compute_suggestions(examplars, checkins, count_to_validate, count_to_suggest):
    users = random.sample(checkins.keys(), count_to_validate)

    stat = []

    for user in users:
        cluster = examplars[user]
        users_in_cluster = [friend_id for friend_id in range(len(examplars)) if examplars[friend_id] == cluster and friend_id != user]

        locations = dict()

        for friend in users_in_cluster:
            if friend in checkins:
                for check in checkins[friend]:
                    if check in locations:
                        locations[check] += 1
                    else:
                        locations[check] = 1

        locations = sorted(locations.items(), key=lambda p: p[1])
        locations.reverse()

        top = locations[0:min(count_to_suggest, len(locations))]

        hits = 0

        for (k,v) in top:
            if k in checkins[user]:
                hits += 1.0 / count_to_suggest

        stat.append(hits)

    return stat


def compute_stat(data):
    n = len(data)
    expectation = 0.0

    for d in data:
        expectation += d / float(n)

    sd = 0.0
    for d in data:
        sd += ((d - expectation) ** 2) / float(n)

    return expectation, sd ** 0.5


def link(edges: list, g: Graph, src, i, depth, factor):
    if depth == 0:
        newEdge = Edge(src, i, 1.0 / factor)
        edges.append(newEdge)
        return

    for e in g.oute[i]:
        link(edges, g, src, e.dst, depth - 1, factor / 0.5)


edges_path = "./dataset/loc-gowalla_edges.txt"
graph = get_data(edges_path)

checkins_path = "./dataset/loc-gowalla_totalCheckins.txt"
checkins = load_checkins(checkins_path)

max_iterations = 30
max_stagnation = 40
exp_smoothing = 0.9

examplars = aff_prop(graph, max_iterations, max_stagnation, exp_smoothing)

count_to_validate = 1000
count_to_suggest = 10

suggestions_stat = compute_suggestions(examplars, checkins, count_to_validate, count_to_suggest)
suggestions_exp, suggestions_sd = compute_stat(suggestions_stat)

print("-========================================================================-")
print("Total users count to compute check-ins recommendations: ", count_to_validate)
print("Total check-ins count to recommend: ", count_to_suggest)
print("Hits E: ", suggestions_exp)
print("Hits SD: ", suggestions_sd)
print("Average cluster size: ", graph.n / len(set(examplars)))