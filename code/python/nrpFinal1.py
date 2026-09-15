import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import random

import argparse
import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D



# ============================================================
# BUILD DE TOUT
# ============================================================

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



# ============================================================
# FEEDER
# ============================================================

def build_feeder():

    return {
        "id": 50,
        "name": "Poste source Bourges",
        "type": "feeder",
        "lat": 47.0810,
        "lon": 2.3988,
        "P_kW": 0,
        "Q_kVAr": 0,
        "gen_values_kW": [0, 0, 0],
        "gen_probs": [1, 0, 0]
    }


# ============================================================
# DISTANCE REELLE
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    """Distance en km entre deux points GPS (vectorisé numpy)."""
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))

def edge_distance(nodes, a, b):

    return haversine(
        nodes[a]["lat"],
        nodes[a]["lon"],
        nodes[b]["lat"],
        nodes[b]["lon"]
    )

# ============================================================
# KRUSKAL
# ============================================================

def kruskal(n, edges):

    parent = list(range(n))

    def find(x):

        if parent[x] != x:
            parent[x] = find(parent[x])

        return parent[x]

    def union(a, b):

        ra = find(a)
        rb = find(b)

        if ra == rb:
            return False

        parent[ra] = rb

        return True

    mst = []

    edges_sorted = sorted(edges, key=lambda e: e["w"])

    for e in edges_sorted:

        if union(e["u"], e["v"]):

            mst.append(e)

        if len(mst) == n - 1:
            break

    return mst


# ============================================================
# RESEAU DELAUNAY
# ============================================================

def build_network(nodes):

    pts = np.array([
        [n["lat"], n["lon"]]
        for n in nodes
    ])

    tri = Delaunay(pts)

    edge_set = set()

    for simplex in tri.simplices:

        for i in range(3):

            a = simplex[i]
            b = simplex[(i + 1) % 3]

            edge_set.add((min(a, b), max(a, b)))

    edges = []

    for a, b in edge_set:

        d = edge_distance(nodes, a, b)

        edges.append({
            "u": a,
            "v": b,
            "w": d
        })

    return edges

# ============================================================
# FEEDER AUTO
# ============================================================

def get_feeder_index(nodes):

    for i, n in enumerate(nodes):

        if n["type"] == "feeder":
            return i

    raise ValueError("Feeder introuvable")


# ============================================================
# DISTFLOW
# ============================================================

def build_tree_adjacency(tree, n):

    adj = {i: [] for i in range(n)}

    for e in tree:

        adj[e["u"]].append(e["v"])
        adj[e["v"]].append(e["u"])

    return adj


def bfs_tree(adj, root):

    parent = [-1] * len(adj)

    order = []

    visited = [False] * len(adj)

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


def distflow(tree, nodes):

    feeder = get_feeder_index(nodes)

    n = len(nodes)

    adj = build_tree_adjacency(tree, n)

    parent, order = bfs_tree(adj, feeder)

    P = np.zeros(n)
    Q = np.zeros(n)

    for i, nd in enumerate(nodes):

        prod = nd["gen_values_kW"][1] * 1e3

        loadP = nd["P_kW"] * 1e3
        loadQ = nd["Q_kVAr"] * 1e3

        P[i] = loadP - prod
        Q[i] = loadQ

    for u in reversed(order):

        if parent[u] != -1:

            P[parent[u]] += P[u]
            Q[parent[u]] += Q[u]

    V = 20e3

    rho = 0.08

    losses = 0

    for u in range(n):

        p = parent[u]

        if p == -1:
            continue

        d = edge_distance(nodes, u, p)

        R = rho * d

        I2 = (P[u]**2 + Q[u]**2) / V**2

        losses += R * I2

    return losses / 1000


# ============================================================
# COUT
# ============================================================

def compute_cost(tree, nodes):

    losses = distflow(tree, nodes)

    total_length = sum(e["w"] for e in tree)

    return losses + 0.05 * total_length

# ============================================================
# BRKGA
# ============================================================

def decode(keys, edges, n):

    modified = []

    for i, e in enumerate(edges):

        modified.append({
            "u": e["u"],
            "v": e["v"],
            "w": e["w"] * (1 + 0.15 * keys[i])
        })

    return kruskal(n, modified)


def run_brkga(nodes, edges,
              n_gen=300,
              pop_size=80,
              elite_size=20):

    ne = len(edges)

    pop = np.random.rand(pop_size, ne)

    def fitness(ind):

        tree = decode(ind, edges, len(nodes))

        return compute_cost(tree, nodes)

    for _ in range(n_gen):

        fit = np.array([fitness(ind) for ind in pop])

        idx = np.argsort(fit)

        elite = pop[idx[:elite_size]]

        new_pop = list(elite)

        while len(new_pop) < pop_size:

            p1 = elite[np.random.randint(elite_size)]
            p2 = pop[np.random.randint(pop_size)]

            mask = np.random.rand(ne) < 0.7

            child = np.where(mask, p1, p2)

            mut = np.random.rand(ne) < 0.02

            child[mut] = np.random.rand(np.sum(mut))

            new_pop.append(child)

        pop = np.array(new_pop)

    fit = np.array([fitness(ind) for ind in pop])

    best = pop[np.argmin(fit)]

    return decode(best, edges, len(nodes))


# ============================================================
# DESSIN
# ============================================================

def plot_network(tree, nodes, title, filename=None):

    fig, ax = plt.subplots(figsize=(10, 10))

    # =========================
    # Lignes électriques
    # =========================
    for e in tree:

        a = nodes[e["u"]]
        b = nodes[e["v"]]

        ax.plot(
            [a["lon"], b["lon"]],
            [a["lat"], b["lat"]],
            color="gray",
            linewidth=1.5,
            alpha=0.7,
            zorder=1
        )


    # =========================
    # Noeuds
    # =========================
    for n in nodes:

        if n["type"] == "turbine":

            ax.scatter(
                n["lon"],
                n["lat"],
                color="royalblue",
                marker="^",
                s=60,
                zorder=3
            )

        elif n["type"] == "commune":

            ax.scatter(
                n["lon"],
                n["lat"],
                color="crimson",
                marker="o",
                s=80,
                zorder=3
            )

            # Nom de la commune
            ax.text(
                n["lon"] + 0.015,
                n["lat"],
                n["name"],
                fontsize=8
            )

        elif n["type"] == "feeder":

            ax.scatter(
                n["lon"],
                n["lat"],
                color="darkgreen",
                marker="s",
                s=180,
                edgecolor="black",
                zorder=4
            )

            ax.text(
                n["lon"] + 0.015,
                n["lat"],
                "Poste source",
                fontsize=9,
                fontweight="bold"
            )


    # =========================
    # Légende
    # =========================
    legend = [
        Line2D(
            [0], [0],
            color="gray",
            lw=2,
            label="Ligne HTA"
        ),

        Line2D(
            [0], [0],
            marker="^",
            color="w",
            markerfacecolor="royalblue",
            markersize=9,
            label="Éolienne"
        ),

        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor="crimson",
            markersize=9,
            label="Commune"
        ),

        Line2D(
            [0], [0],
            marker="s",
            color="w",
            markerfacecolor="darkgreen",
            markeredgecolor="black",
            markersize=11,
            label="Bourges"
        )
    ]

    ax.legend(
        handles=legend,
        loc="upper right"
    )


    # =========================
    # Mise en forme
    # =========================
    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")

    ax.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    ax.set_aspect("equal")

    plt.tight_layout()


    # =========================
    # Sauvegarde éventuelle
    # =========================
    if filename is not None:

        plt.savefig(
            filename,
            bbox_inches="tight",
            dpi=300
        )


# ============================================================
# MAIN
# ============================================================

def main():

    shp_path = "eolien_filtre.shp"

    turbines = load_turbines(shp_path, n=150)

    feeder = build_feeder()

    communes = build_communes()

    nodes = turbines + [feeder] + communes

    edges = build_network(nodes)

    initial_tree = kruskal(len(nodes), edges)

    optimal_tree = run_brkga(
        nodes,
        edges,
        n_gen=100
    )

    loss_initial = distflow(initial_tree, nodes)

    loss_optimal = distflow(optimal_tree, nodes)

    gain_abs = loss_initial - loss_optimal

    gain_rel = 100 * gain_abs / loss_initial

    print("\n=== COMPARAISON ===")

    print(f"Pertes initiales : {loss_initial:.2f} kW")
    print(f"Pertes optimales : {loss_optimal:.2f} kW")

    print(f"Gain absolu : {gain_abs:.2f} kW")
    print(f"Gain relatif : {gain_rel:.2f} %")

    plot_network(
        initial_tree,
        nodes,
        "Réseau réel initial",
        "reseau_initial.pdf"
    )
    plot_network(
        optimal_tree,
        nodes,
        "Réseau réel optimisé BRKGA",
        "reseau_final.pdf"
    )

    plt.show()


if __name__ == "__main__":
    main()