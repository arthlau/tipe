import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# =========================
# 1. GENERATION DU RESEAU
# =========================

def generate_fictitious_region(seed=0):

    np.random.seed(seed)

    communes = []

    for i in range(30):

        r = np.random.normal(loc=20, scale=10)
        theta = np.random.uniform(0, 2*np.pi)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        dist = np.sqrt(x**2 + y**2)

        pop = int(20000 * np.exp(-dist/30) + np.random.randint(200,2000))

        P = pop * 0.4  # kW

        communes.append({
            "id": i,
            "type": "commune",
            "lat": y,
            "lon": x,
            "P_kW": P,
            "Q_kVAr": 0.35 * P,
            "gen_values_kW": [0,0,0]
        })

    turbines = []

    for i in range(15):

        r = np.random.uniform(25,50)
        theta = np.random.uniform(0,2*np.pi)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        turbines.append({
            "id": 30+i,
            "type": "turbine",
            "lat": y,
            "lon": x,
            "P_kW": 0,
            "Q_kVAr": 0,
            "gen_values_kW": [200,400,600]
        })

    feeder = {
        "id": 45,
        "type": "feeder",
        "lat": 0.0,
        "lon": 0.0,
        "P_kW": 0,
        "Q_kVAr": 0,
        "gen_values_kW": [0,0,0]
    }

    return turbines, feeder, communes


# =========================
# 2. DISTANCE
# =========================

def distance(a,b):

    return np.sqrt(
        (a["lat"]-b["lat"])**2 +
        (a["lon"]-b["lon"])**2
    )


# =========================
# 3. DISTFLOW 
# =========================

def build_tree_adjacency(tree, n):

    adj = {i: [] for i in range(n)}

    for e in tree:

        u,v = e["u"], e["v"]

        adj[u].append(v)
        adj[v].append(u)

    return adj


def bfs_tree(adj, root, n):

    parent = [-1]*n
    order = []

    visited = [False]*n

    queue = [root]
    visited[root] = True

    while queue:

        u = queue.pop(0)

        order.append(u)

        for v in adj[u]:

            if not visited[v]:

                visited[v] = True
                parent[v] = u

                queue.append(v)

    return parent, order


def distflow(tree, nodes, feeder_id=45):

    n = len(nodes)

    adj = build_tree_adjacency(tree, n)

    parent, order = bfs_tree(adj, feeder_id, n)


    P = np.zeros(n)  # W
    Q = np.zeros(n)  # var

    for i, nd in enumerate(nodes):

        prod = nd["gen_values_kW"][1] * 1e3

        loadP = nd["P_kW"] * 1e3
        loadQ = nd["Q_kVAr"] * 1e3

        P[i] = loadP - prod
        Q[i] = loadQ

    # -------------------------
    # Flux remontants
    # -------------------------

    for u in reversed(order):

        if parent[u] != -1:

            P[parent[u]] += P[u]
            Q[parent[u]] += Q[u]

    # -------------------------
    # Pertes Joule
    # -------------------------

    V = 20e3  # volts

    rho = 0.05  # ohm/km

    echelle_km = 0.2

    losses = 0.0

    for u in range(n):

        p = parent[u]

        if p != -1:

            d = echelle_km * distance(nodes[u], nodes[p])

            R = rho * d

            I2 = (P[u]**2 + Q[u]**2) / (V**2)

            losses += R * I2

    return losses / 1e3  # kW


# =========================
# 4. KRUSKAL
# =========================

def kruskal(n, edges):

    parent = list(range(n))

    def find(x):

        if parent[x] != x:
            parent[x] = find(parent[x])

        return parent[x]

    def union(a,b):

        a,b = find(a),find(b)

        if a == b:
            return False

        parent[a] = b

        return True

    acm = []

    edges = sorted(edges, key=lambda e:e["w"])

    for e in edges:

        if union(e["u"], e["v"]):

            acm.append(e)

        if len(acm) == n-1:
            break

    return acm


# =========================
# 5. CONSTRUCTION GRAPHE
# =========================

def build_network(nodes):

    pts = np.array([
        [n["lat"], n["lon"]]
        for n in nodes
    ])

    tri = Delaunay(pts)

    edges = set()

    for simplex in tri.simplices:

        for i in range(3):

            a = simplex[i]
            b = simplex[(i+1)%3]

            edges.add((min(a,b), max(a,b)))

    edge_list = []

    for a,b in edges:

        d = distance(nodes[a], nodes[b])

        edge_list.append({
            "u": a,
            "v": b,
            "w": d
        })

    return edge_list


# =========================
# 6. COUT
# =========================

def compute_cost(tree, nodes):

    losses = distflow(tree, nodes)

    # pénalisation longueur totale
    total_length = sum(e["w"] for e in tree)

    return losses + 0.05 * total_length


# =========================
# 7. BRKGA CORRIGE
# =========================

def decode(keys, edges, n):

    modified_edges = []

    for i,e in enumerate(edges):

        # IMPORTANT :
        # on conserve la distance physique
        # et on ajoute une petite perturbation

        w = e["w"] * (1 + 0.15*keys[i])

        modified_edges.append({
            "u": e["u"],
            "v": e["v"],
            "w": w
        })

    return kruskal(n, modified_edges)


def run_brkga(nodes, edges,
              n_gen=150,
              pop_size=80,
              elite_size=20):

    ne = len(edges)

    pop = np.random.rand(pop_size, ne)

    def fitness(ind):

        tree = decode(ind, edges, len(nodes))

        return compute_cost(tree, nodes)

    for g in range(n_gen):

        fit = np.array([
            fitness(ind)
            for ind in pop
        ])

        idx = np.argsort(fit)

        elite = pop[idx[:elite_size]]

        new_pop = list(elite)

        while len(new_pop) < pop_size:

            p1 = elite[np.random.randint(elite_size)]
            p2 = pop[np.random.randint(pop_size)]

            mask = np.random.rand(ne) < 0.7

            child = np.where(mask, p1, p2)

            # mutation légère
            mutation = np.random.rand(ne) < 0.02

            child[mutation] = np.random.rand(np.sum(mutation))

            new_pop.append(child)

        pop = np.array(new_pop)

    fit = np.array([
        fitness(ind)
        for ind in pop
    ])

    best = pop[np.argmin(fit)]

    return decode(best, edges, len(nodes))


# =========================
# 8. AFFICHAGE
# =========================

def plot_network(tree, nodes, title):

    plt.figure()

    for e in tree:

        a = nodes[e["u"]]
        b = nodes[e["v"]]

        plt.plot(
            [a["lon"], b["lon"]],
            [a["lat"], b["lat"]],
            'gray'
        )

    for n in nodes:

        if n["type"] == "commune":

            plt.scatter(
                n["lon"],
                n["lat"],
                c='red'
            )

        elif n["type"] == "turbine":

            plt.scatter(
                n["lon"],
                n["lat"],
                c='blue'
            )

        else:

            plt.scatter(
                n["lon"],
                n["lat"],
                c='green',
                s=120
            )

    plt.title(title)

    plt.axis("equal")


# =========================
# 9. MAIN
# =========================

def main():

    turbines, feeder, communes = generate_fictitious_region()

    nodes = communes + turbines + [feeder]

    edges = build_network(nodes)

    # -------------------------
    # arbre initial
    # -------------------------

    initial_tree = kruskal(len(nodes), edges)

    # -------------------------
    # optimisation
    # -------------------------

    optimal_tree = run_brkga(nodes, edges)

    # -------------------------
    # pertes
    # -------------------------

    loss_initial = distflow(initial_tree, nodes)

    loss_optimal = distflow(optimal_tree, nodes)

    delta = loss_initial - loss_optimal

    gain = 100 * delta / loss_initial

    print("\n=== COMPARAISON ===")

    print(f"Pertes initiales : {loss_initial:.2f} kW")
    print(f"Pertes optimales : {loss_optimal:.2f} kW")

    print(f"Gain absolu : {delta:.2f} kW")
    print(f"Gain relatif : {gain:.2f} %")

    # -------------------------
    # plots
    # -------------------------

    plot_network(
        initial_tree,
        nodes,
        "Reseau initial"
    )

    plot_network(
        optimal_tree,
        nodes,
        "Reseau optimise BRKGA"
    )

    plt.show()


if __name__ == "__main__":
    main()