import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import random

import argparse
import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D




##########################################
# 1. trucs
##########################################



def haversine(lat1, lon1, lat2, lon2):
    """Distance en km entre deux points GPS (vectorisé numpy)."""
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))

def load_turbines(shp_path, n=50):
    """
    Charge le shapefile éolien, reprojette en WGS-84,
    retourne les n éoliennes les plus proches de Bourges.
    """
    BOURGES = (47.0810, 2.3988)
    gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")

    coords = []
    for geom in gdf.geometry:
        if geom.geom_type == "MultiPoint":
            for pt in geom.geoms:
                coords.append((pt.y, pt.x))
        elif geom.geom_type == "Point":
            coords.append((geom.y, geom.x))
    coords = np.array(coords)

    dists = haversine(BOURGES[0], BOURGES[1], coords[:, 0], coords[:, 1])
    idx = np.argsort(dists)[:n]
    top = coords[idx]
    d   = dists[idx]

    np.random.seed(42)
    turbines = []
    for i, (lat, lon, dist) in enumerate(zip(top[:, 0], top[:, 1], d)):
        P = float(np.random.choice([200, 400, 600, 800], p=[0.3, 0.4, 0.2, 0.1]))
        turbines.append({
            "id": i,
            "name": f"Éolienne #{i}",
            "type": "turbine",
            "lat": float(lat),
            "lon": float(lon),
            "dist_km": float(dist),
            "P_kW": P,
            "Q_kVAr": round(P * 0.3, 1),
            # 3 scénarios de vent : 10 %, 35 %, 70 % de 2 MW installés
            "gen_values_kW": [200.0, 700.0, 1400.0],
            "gen_probs":     [0.3,   0.5,   0.2],
        })
    return turbines


def build_communes():
    """6 communes les plus proches de Bourges (hors Bourges)."""
    data = [
        {"name": "Saint-Doulchard",      "lat": 47.104, "lon": 2.347, "pop":  9500},
        {"name": "Mehun-sur-Yèvre",      "lat": 47.139, "lon": 2.221, "pop":  7000},
        {"name": "Vierzon",              "lat": 47.222, "lon": 2.069, "pop": 26000},
        {"name": "Saint-Florent",        "lat": 47.000, "lon": 2.255, "pop":  6800},
        {"name": "Dun-sur-Auron",        "lat": 46.883, "lon": 2.567, "pop":  3800},
        {"name": "Châteauneuf-s-Cher",   "lat": 46.857, "lon": 2.942, "pop":  1200},
    ]
    communes = []
    for i, c in enumerate(data):
        P = float(c["pop"] * 0.4)          # ~0.4 kW/habitant
        communes.append({
            "id": 51 + i,
            "name": c["name"],
            "type": "commune",
            "lat": c["lat"],
            "lon": c["lon"],
            "pop": c["pop"],
            "P_kW": P,
            "Q_kVAr": round(P * 0.35, 1),
            "gen_values_kW": [0, 0, 0],
            "gen_probs":     [0.3, 0.5, 0.2],
        })
    return communes


def build_network(turbines, feeder, communes):
    """
    Construit le graphe complet par triangulation de Delaunay
    sur les positions GPS réelles de tous les nœuds.

    La triangulation de Delaunay sur les coordonnées géographiques
    donne une topologie proche des tracés de câbles réels (voisins
    naturels), sans nécessiter l'accès aux données réseau propriétaires
    d'Enedis/RTE.

    Paramètres ligne HTA 20 kV :
        r = 0.30 Ω/km  (résistance)
        x = 0.15 Ω/km  (réactance)
    """
    all_nodes = turbines + [feeder] + communes
    N = len(all_nodes)
    pts = np.array([[n["lat"], n["lon"]] for n in all_nodes])

    # Triangulation de Delaunay → arcs candidats
    tri = Delaunay(pts)
    edge_set = set()
    for s in tri.simplices:
        for i in range(3):
            a, b = s[i], s[(i + 1) % 3]
            if a > b:
                a, b = b, a
            edge_set.add((a, b))

    # Forcer la connexion de chaque commune à ses 3 plus proches voisins
    for ci in range(51, N):
        dists_to = sorted(
            [(haversine(pts[ci, 0], pts[ci, 1], pts[oi, 0], pts[oi, 1]), oi)
             for oi in range(N) if oi != ci]
        )
        for _, oi in dists_to[:3]:
            a, b = min(ci, oi), max(ci, oi)
            edge_set.add((a, b))

    edges = []
    for a, b in edge_set:
        d = haversine(pts[a, 0], pts[a, 1], pts[b, 0], pts[b, 1])
        edges.append({
            "from": int(a), "to": int(b),
            "dist_km": round(d, 3),
            "r": round(d * 0.30, 4),
            "x": round(d * 0.15, 4),
        })

    # Configuration initiale = MST géographique (Kruskal)
    mst = kruskal(N, edges)
    mst_set = ({(e["from"], e["to"]) for e in mst}
               | {(e["to"], e["from"]) for e in mst})
    for e in edges:
        e["switch"] = "NF" if (e["from"], e["to"]) in mst_set else "NO"

    return all_nodes, edges, N


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


# =========================|
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


def run_brkga(nodes, edges, n_gen=150, pop_size=80, elite_size=20):

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

    # seed = random.randrange(1000000)
    seed = 0

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

    optimal_tree = run_brkga(nodes, edges, n_gen=1000)

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
    plt.savefig("proto1" + str(seed) + ".pdf", bbox_inches = "tight")

    plot_network(
        optimal_tree,
        nodes,
        "Reseau optimise BRKGA"
    )

    plt.savefig("proto2" + str(seed) + ".pdf", bbox_inches = "tight")

    plt.show()


if __name__ == "__main__":
    main()