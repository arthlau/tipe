import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# =========================
# 1. GENERATION DU RESEAU FICTIF
# =========================

def generate_fictitious_region(seed=0):  #O(n)
    np.random.seed(seed)

    center = np.array([0.0, 0.0])

    communes = []
    for i in range(30):
        r = np.random.normal(loc=20, scale=10)
        theta = np.random.uniform(0, 2*np.pi)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        dist = np.sqrt(x**2 + y**2)
        pop = int(20000 * np.exp(-dist/30) + np.random.randint(200, 2000))

        P = pop * 0.4

        communes.append({
            "id": i,
            "type": "commune",
            "lat": y,
            "lon": x,
            "P_kW": P,
            "Q_kVAr": P * 0.35,
            "gen_values_kW": [0,0,0]
        })

    turbines = []
    for i in range(15):
        r = np.random.uniform(25, 50)
        theta = np.random.uniform(0, 2*np.pi)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        P = np.random.choice([200,400,600,800])

        turbines.append({
            "id": 30+i,
            "type": "turbine",
            "lat": y,
            "lon": x,
            "P_kW": 0,
            "Q_kVAr": 0,
            "gen_values_kW": [200,700,1400]
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

def distance(a,b): #O(1)
    return np.sqrt((a["lat"]-b["lat"])**2 + (a["lon"]-b["lon"])**2)


# =========================
# 2.5. DISFLOW
# =========================

def build_tree_adjacency(tree, n):
    adj = {i: [] for i in range(n)}
    for e in tree:
        u, v = e["u"], e["v"]
        adj[u].append((v, e["w"]))
        adj[v].append((u, e["w"]))
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
        for v,_ in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                queue.append(v)

    return parent, order


def distflow(tree, nodes, feeder_id=45):
    n = len(nodes)

    adj = build_tree_adjacency(tree, n)
    parent, order = bfs_tree(adj, feeder_id, n)

    # initialisation injections
    P = np.zeros(n)
    Q = np.zeros(n)

    for i, nd in enumerate(nodes):
        prod = nd["gen_values_kW"][1]  # scénario moyen
        P[i] = nd["P_kW"] - prod
        Q[i] = nd["Q_kVAr"]

    # remontée des flux 
    for u in reversed(order):
        if parent[u] != -1:
            P[parent[u]] += P[u]
            Q[parent[u]] += Q[u]

    # calcul pertes
    V = 20.0  # kV
    loss = 0

    for u in range(n):
        if parent[u] != -1:
            r = 0.3 * distance(nodes[u], nodes[parent[u]]) # 0.3 mOhm / km
            loss += r * (P[u]**2 + Q[u]**2) / (V**2 * 1e6) * 1e3 # r * (P^2 + Q^2) / V^2

    return loss


# =========================
# 3. KRUSKAL
# =========================

def kruskal(n, edges): #O(ElogE)
    parent = list(range(n))

    def find(x):
        if parent[x]!=x:
            parent[x]=find(parent[x])
        return parent[x]

    def union(a,b):
        a,b=find(a),find(b)
        if a==b:
            return False
        parent[a]=b
        return True

    acm=[]
    edges=sorted(edges,key=lambda e:e["w"])

    for e in edges:
        if union(e["u"],e["v"]):
            acm.append(e)
        if len(acm)==n-1:
            break

    return acm

# =========================
# 4. CONSTRUCTION GRAPHE
# =========================

def build_network(nodes): #O(nlogn)
    pts=np.array([[n["lat"],n["lon"]] for n in nodes])
    tri=Delaunay(pts)

    edges=set()
    for simplex in tri.simplices:
        for i in range(3):
            a=simplex[i]
            b=simplex[(i+1)%3]
            edges.add((min(a,b),max(a,b)))

    edge_list=[]
    for a,b in edges:
        d=distance(nodes[a],nodes[b])
        edge_list.append({"u":a,"v":b,"w":d})

    return edge_list

# =========================
# 5. COUT
# =========================

def compute_cost(tree, nodes):
    return distflow(tree, nodes)

# =========================
# 6. BRKGA
# =========================

def decode(keys, edges, n):
    new_edges=[{"u":edges[i]["u"],"v":edges[i]["v"],"w":keys[i]} for i in range(len(edges))]
    return kruskal(n,new_edges)


def run_brkga(nodes, edges, n_gen=200, pop_size=100):
    ne=len(edges)

    pop=np.random.rand(pop_size,ne)

    def fitness(ind):
        tree=decode(ind,edges,len(nodes))
        return compute_cost(tree,nodes)

    for g in range(n_gen):
        fit=np.array([fitness(ind) for ind in pop])
        idx=np.argsort(fit)

        elite=pop[idx[:20]]
        new_pop=list(elite)

        while len(new_pop)<pop_size:
            p1=elite[np.random.randint(20)]
            p2=pop[np.random.randint(pop_size)]
            child=np.where(np.random.rand(ne)<0.7,p1,p2)
            new_pop.append(child)

        pop=np.array(new_pop)

    best=pop[np.argmin([fitness(ind) for ind in pop])]
    return decode(best,edges,len(nodes))

# =========================
# 7. MAIN
# =========================

def main():
    turbines, feeder, communes = generate_fictitious_region()
    nodes = communes + turbines + [feeder]

    edges = build_network(nodes)

    # --- arbre initial (MST classique) ---
    initial_tree = kruskal(len(nodes), edges)

    # plot initial
    plt.figure()
    for e in initial_tree:
       a = nodes[e["u"]]
       b = nodes[e["v"]]
       plt.plot([a["lon"], b["lon"]],
             [a["lat"], b["lat"]], 'gray')

    for n in nodes:
        if n["type"]=="commune":
            plt.scatter(n["lon"], n["lat"], c='red')
        elif n["type"]=="turbine":
            plt.scatter(n["lon"], n["lat"], c='blue')
        else:
            plt.scatter(n["lon"], n["lat"], c='green')

    plt.title("Réseau initial (MST géographique)")


    tree = run_brkga(nodes, edges)

    # pertes
    loss_initial = distflow(initial_tree, nodes)
    loss_optimal = distflow(tree, nodes)

    # comparaison
    delta = loss_initial - loss_optimal
    gain = delta / loss_initial * 100

    print("=== COMPARAISON ===")
    print(f"Pertes initiales : {loss_initial:.2f}")
    print(f"Pertes optimales : {loss_optimal:.2f}")
    print(f"Gain absolu : {delta:.2f}")
    print(f"Gain relatif : {gain:.2f} %")

    plt.figure()
    # affichage
    for e in tree:
        a=nodes[e["u"]]
        b=nodes[e["v"]]
        plt.plot([a["lon"],b["lon"]],[a["lat"],b["lat"]],'k-')

    for n in nodes:
        if n["type"]=="commune":
            plt.scatter(n["lon"],n["lat"],c='red')
        elif n["type"]=="turbine":
            plt.scatter(n["lon"],n["lat"],c='blue')
        else:
            plt.scatter(n["lon"],n["lat"],c='green')

    plt.title("Reseau optimise (BRKGA)")
    plt.show()

if __name__ == "__main__":
    main()
