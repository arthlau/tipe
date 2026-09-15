"""
NRP — Optimisation de réseau éolien autour de Bourges
======================================================
Cavalheiro, Vergílio, Lyra — Computers & Operations Research 96 (2018) 272–280

Usage :
    python nrp_bourges.py --shp eolien_filtre.shp

Sorties :
    nrp_bourges.png     visualisation 4 panneaux
    rapport_nrp.txt     résultats détaillés
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial import Delaunay


# ═══════════════════════════════════════════════════════════════════════
# 1.  UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════

def haversine(lat1, lon1, lat2, lon2):
    """Distance en km entre deux points GPS (vectorisé numpy)."""
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


def kruskal(n, edges):
    """
    Algorithme de Kruskal — retourne l'arbre couvrant minimal.
    Utilisé à deux endroits :
      · construction de la topologie initiale (MST géographique)
      · décodage de chromosome BRKGA → arbre radial valide
    """
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a == b:
            return False
        parent[a] = b
        return True

    mst = []
    for e in sorted(edges, key=lambda e: e["dist_km"]):
        if union(e["from"], e["to"]):
            mst.append(e)
        if len(mst) == n - 1:
            break
    return mst


# ═══════════════════════════════════════════════════════════════════════
# 2.  CONSTRUCTION DU RÉSEAU
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# 3.  CALCUL DES PERTES  (DistFlow simplifié)
# ═══════════════════════════════════════════════════════════════════════

def distflow(tree_edges, nodes, feeder_id, gen_kW):
    """
    Calcul des pertes techniques par balayage arrière/avant (DistFlow).

    Hypothèses :
      · Tension nominale 20 kV (réseau HTA)
      · Équations de flux linéarisées (Baran & Wu 1989)
      · P en kW, Q en kVAr, r en Ω, V en kV

    Pertes = Σ_k  r_k · (P_k² + Q_k²) / (V_nom² · 10⁶)  · 10³  [kW]
    """
    n = len(nodes)
    adj = {i: [] for i in range(n)}
    for e in tree_edges:
        adj[e["from"]].append((e["to"],  e["r"]))
        adj[e["to"]].append( (e["from"], e["r"]))

    # BFS depuis le feeder → ordre de parcours + parents
    parent = [-1] * n
    pr     = [0.0] * n
    order  = []
    visited = [False] * n
    queue = [feeder_id]
    visited[feeder_id] = True
    while queue:
        v = queue.pop(0)
        order.append(v)
        for w, r in adj[v]:
            if not visited[w]:
                visited[w] = True
                parent[w]  = v
                pr[w]      = r
                queue.append(w)

    # Injections nettes
    P = np.array([nd.get("P_kW",  0) - gen_kW[i] for i, nd in enumerate(nodes)])
    Q = np.array([nd.get("Q_kVAr", 0)             for i, nd in enumerate(nodes)])

    # Balayage arrière : flux de puissance
    Pb = P.copy()
    Qb = Q.copy()
    for v in reversed(order):
        if v == feeder_id:
            continue
        for w, r in adj[v]:
            if parent[w] == v:
                Pb[v] += Pb[w]
                Qb[v] += Qb[w]

    # Calcul des pertes
    V_kV = 20.0
    loss = 0.0
    for v in order:
        if v == feeder_id:
            continue
        loss += pr[v] * (Pb[v] ** 2 + Qb[v] ** 2) / (V_kV ** 2 * 1e6) * 1e3
    return max(loss, 0.0)


def expected_loss(tree_edges, nodes, feeder_id):
    """Espérance des pertes sur les 3 scénarios de vent."""
    n = len(nodes)
    total = 0.0
    for s, prob in enumerate([0.3, 0.5, 0.2]):
        gen = np.array([nd.get("gen_values_kW", [0, 0, 0])[s] for nd in nodes])
        total += prob * distflow(tree_edges, nodes, feeder_id, gen)
    return total


# ═══════════════════════════════════════════════════════════════════════
# 4.  BRKGA
# ═══════════════════════════════════════════════════════════════════════

def decode_chromosome(keys, edges, n):
    """
    Décode un vecteur de clés [0,1]^m en arbre couvrant via Kruskal.
    Garantit qu'un individu BRKGA est toujours un réseau radial valide.
    """
    return kruskal(n, [dict(e, dist_km=keys[i]) for i, e in enumerate(edges)])


def run_brkga(nodes, edges, N, feeder_id,
              pop_size=120, n_elite=24, n_mutant=12,
              n_gen=300, rho=0.7, restart_patience=50,
              seed=0, verbose=True):
    """
    BRKGA principal.

    Paramètres :
        pop_size          taille de la population
        n_elite           nombre d'individus élite
        n_mutant          nombre de mutants introduits par génération
        n_gen             nombre maximal de générations
        rho               probabilité d'héritage depuis l'élite (croisement)
        restart_patience  générations sans amélioration avant restart
    """
    np.random.seed(seed)
    ne = len(edges)

    def fitness(ind):
        return expected_loss(decode_chromosome(ind, edges, N), nodes, feeder_id)

    # Population initiale
    pop = np.random.uniform(0, 1, (pop_size, ne))
    fit = np.array([fitness(ind) for ind in pop])

    best_loss = fit.min()
    best_tree = decode_chromosome(pop[fit.argmin()], edges, N)
    no_imp    = 0
    history   = []

    for g in range(n_gen):
        order     = np.argsort(fit)
        elite     = pop[order[:n_elite]]
        non_elite = pop[order[n_elite:]]

        new_pop = list(elite)

        # Mutants
        for _ in range(n_mutant):
            new_pop.append(np.random.uniform(0, 1, ne))

        # Croisement biaisé vers l'élite
        while len(new_pop) < pop_size:
            e_  = elite[np.random.randint(n_elite)]
            ne_ = non_elite[np.random.randint(len(non_elite))]
            mask = np.random.uniform(0, 1, ne) < rho
            new_pop.append(np.where(mask, e_, ne_))

        pop = np.array(new_pop)
        fit = np.array([fitness(ind) for ind in pop])

        gen_best = fit.min()
        if gen_best < best_loss:
            best_loss = gen_best
            best_tree = decode_chromosome(pop[fit.argmin()], edges, N)
            no_imp    = 0
        else:
            no_imp += 1

        history.append(best_loss)

        # Restart partiel
        if no_imp >= restart_patience:
            pop[:n_mutant] = np.random.uniform(0, 1, (n_mutant, ne))
            no_imp = 0

        if verbose and g % 50 == 0:
            print(f"  Gen {g:>4d} : best = {best_loss:,.1f} kW")

    return best_tree, best_loss, history


# ═══════════════════════════════════════════════════════════════════════
# 5.  VISUALISATION
# ═══════════════════════════════════════════════════════════════════════

C = {
    "bg":     "#0d0f14", "bg2":    "#13161e", "panel":  "#1a1e29",
    "border": "#252a38", "text":   "#e8eaf0", "muted":  "#6b7280",
    "accent": "#5ee7a0", "blue":   "#60a5fa", "orange": "#f97316",
    "red":    "#f87171", "green":  "#4ade80", "yellow": "#fbbf24",
    "purple": "#a78bfa",
}


def style_ax(ax, title=""):
    ax.set_facecolor(C["bg2"])
    for sp in ax.spines.values():
        sp.set_color(C["border"])
    ax.tick_params(colors=C["muted"], labelsize=7)
    ax.xaxis.label.set_color(C["muted"])
    ax.yaxis.label.set_color(C["muted"])
    if title:
        ax.set_title(title, color=C["text"], fontsize=9,
                     fontweight="bold", pad=6, loc="left")


def draw_network(ax, nodes, edges, tree_edges, to_open=None, to_close=None, mode="current"):
    """Dessine le réseau sur un axe matplotlib."""
    to_open_set  = ({(e["from"], e["to"]) for e in to_open}
                    | {(e["to"], e["from"]) for e in to_open}) if to_open else set()
    to_close_set = ({(e["from"], e["to"]) for e in to_close}
                    | {(e["to"], e["from"]) for e in to_close}) if to_close else set()

    # Tous les arcs NO (fond)
    for e in edges:
        if e["switch"] == "NO":
            ax.plot([nodes[e["from"]]["lon"], nodes[e["to"]]["lon"]],
                    [nodes[e["from"]]["lat"], nodes[e["to"]]["lat"]],
                    color=C["border"], lw=0.4, alpha=0.35, zorder=1)

    if mode == "current":
        for e in tree_edges:
            ax.plot([nodes[e["from"]]["lon"], nodes[e["to"]]["lon"]],
                    [nodes[e["from"]]["lat"], nodes[e["to"]]["lat"]],
                    color=C["blue"], lw=1.2, alpha=0.75, zorder=2)

    elif mode == "optimal":
        for e in tree_edges:
            x = [nodes[e["from"]]["lon"], nodes[e["to"]]["lon"]]
            y = [nodes[e["from"]]["lat"], nodes[e["to"]]["lat"]]
            col = C["green"] if (e["from"], e["to"]) in to_close_set else C["accent"]
            ax.plot(x, y, color=col, lw=1.5 if col == C["green"] else 1.0,
                    alpha=0.85, zorder=3)
        # Lignes à ouvrir (pointillé rouge)
        for e in (to_open or []):
            ax.plot([nodes[e["from"]]["lon"], nodes[e["to"]]["lon"]],
                    [nodes[e["from"]]["lat"], nodes[e["to"]]["lat"]],
                    color=C["red"], lw=1.2, ls="--", alpha=0.8, zorder=2)

    # Nœuds
    for nd in nodes:
        lon, lat = nd["lon"], nd["lat"]
        if nd["type"] == "feeder":
            ax.scatter(lon, lat, c=C["accent"], s=120, zorder=5,
                       marker="*", edgecolors="white", linewidths=0.8)
        elif nd["type"] == "commune":
            ax.scatter(lon, lat, c=C["yellow"], s=60, zorder=4,
                       marker="s", edgecolors=C["bg"], linewidths=0.5)
            ax.annotate(nd["name"].split("-")[0], (lon, lat),
                        xytext=(4, 4), textcoords="offset points",
                        color=C["yellow"], fontsize=5.5, zorder=6)
        else:
            ax.scatter(lon, lat, c=C["blue"], s=18, zorder=3,
                       alpha=0.85, edgecolors=C["bg"], linewidths=0.3)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def make_figure(nodes, edges, current_tree, optimal_tree,
                to_open, to_close, scenarios, results, history, out_path):

    fig = plt.figure(figsize=(22, 14), facecolor=C["bg"])
    gs  = GridSpec(3, 4, figure=fig,
                   left=0.03, right=0.97, top=0.92, bottom=0.05,
                   hspace=0.55, wspace=0.38)

    # ── Carte 1 : réseau actuel ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    style_ax(ax1, "① Réseau actuel (config. initiale — MST géographique)")
    draw_network(ax1, nodes, edges, current_tree, mode="current")
    leg1 = [
        Line2D([0],[0], color=C["blue"],   lw=1.5, label="Ligne NF (active)"),
        Line2D([0],[0], color=C["border"], lw=0.8, ls="--", label="Ligne NO (ouverte)"),
        Line2D([0],[0], marker="*", color="w", markerfacecolor=C["accent"],
               ms=10, lw=0, label="Poste HTB Bourges"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor=C["yellow"],
               ms=7,  lw=0, label="Commune"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C["blue"],
               ms=6,  lw=0, label="Éolienne"),
    ]
    ax1.legend(handles=leg1, loc="upper right", fontsize=6,
               facecolor=C["panel"], edgecolor=C["border"],
               labelcolor=C["text"], framealpha=0.9)

    # ── Carte 2 : réseau optimal ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0:2, 2:4])
    style_ax(ax2, "② Réseau optimal (BRKGA — 300 générations)")
    draw_network(ax2, nodes, edges, optimal_tree,
                 to_open=to_open, to_close=to_close, mode="optimal")
    leg2 = [
        Line2D([0],[0], color=C["accent"], lw=1.5, label="Ligne opt. conservée"),
        Line2D([0],[0], color=C["green"],  lw=1.8, label="Nouvelle ligne (fermer)"),
        Line2D([0],[0], color=C["red"],    lw=1.2, ls="--", label="Ligne à ouvrir"),
    ]
    ax2.legend(handles=leg2, loc="upper right", fontsize=6,
               facecolor=C["panel"], edgecolor=C["border"],
               labelcolor=C["text"], framealpha=0.9)

    # ── Barres de pertes par scénario ────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0:2])
    style_ax(ax3, "③ Pertes espérées par scénario de vent")
    labels    = [s["label"]      for s in scenarios]
    cur_vals  = [s["current_kW"] for s in scenarios]
    opt_vals  = [s["optimal_kW"] for s in scenarios]
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax3.bar(x - w/2, cur_vals, w, color=C["red"],   alpha=0.85, label="Actuel")
    b2 = ax3.bar(x + w/2, opt_vals, w, color=C["green"], alpha=0.85, label="Optimal")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=7, color=C["text"])
    ax3.set_ylabel("Pertes (kW)", fontsize=7)
    ax3.yaxis.grid(True, color=C["border"], lw=0.5, alpha=0.5)
    ax3.set_axisbelow(True)
    for bar in b1:
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 200,
                 f"{bar.get_height()/1000:.1f} MW",
                 ha="center", va="bottom", color=C["red"], fontsize=6.5)
    for bar in b2:
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 200,
                 f"{bar.get_height()/1000:.1f} MW",
                 ha="center", va="bottom", color=C["green"], fontsize=6.5)
    for i, s in enumerate(scenarios):
        ax3.annotate(f"−{s['gain_pct']}%",
                     (i, max(cur_vals[i], opt_vals[i]) + 1500),
                     ha="center", color=C["yellow"], fontsize=7, fontweight="bold")
    ax3.legend(fontsize=7, facecolor=C["panel"], edgecolor=C["border"],
               labelcolor=C["text"])

    # ── KPI panel ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 2])
    style_ax(ax4)
    ax4.axis("off")
    kpis = [
        ("Pertes espérées\nActuelles",
         f"{results['current_exp_kW']:,.0f} kW",  C["red"]),
        ("Pertes espérées\nOptimales",
         f"{results['optimal_exp_kW']:,.0f} kW",  C["green"]),
        ("Réduction",
         f"−{results['gain_pct']} %",             C["accent"]),
        ("Interrupteurs\nmodifiés",
         f"{len(to_open)} ouvrir / {len(to_close)} fermer", C["yellow"]),
        ("Nœuds / Câbles",
         f"{results['N']} / {len(edges)}",        C["blue"]),
    ]
    for j, (lbl, val, col) in enumerate(kpis):
        yy = 0.96 - j * 0.20
        ax4.text(0.05, yy,       lbl, transform=ax4.transAxes,
                 color=C["muted"], fontsize=6.5, va="top")
        ax4.text(0.05, yy - 0.07, val, transform=ax4.transAxes,
                 color=col, fontsize=10, fontweight="bold", va="top",
                 fontfamily="monospace")
        ax4.axhline(y=yy - 0.14, color=C["border"], lw=0.5, xmin=0.03, xmax=0.97)

    # ── Convergence BRKGA ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 3])
    style_ax(ax5, "④ Convergence BRKGA")
    gens = list(range(len(history)))
    ax5.plot(gens, history, color=C["accent"], lw=1.5)
    ax5.fill_between(gens, history,
                     [v * 1.15 for v in history],
                     color=C["accent"], alpha=0.07)
    ax5.axhline(y=results["current_exp_kW"], color=C["red"],
                lw=1, ls="--", alpha=0.7, label="Config. actuelle")
    ax5.axhline(y=results["optimal_exp_kW"], color=C["green"],
                lw=1, ls="--", alpha=0.7, label="Optimal final")
    ax5.set_xlabel("Génération", fontsize=7)
    ax5.set_ylabel("Pertes espérées (kW)", fontsize=7)
    ax5.yaxis.grid(True, color=C["border"], lw=0.5, alpha=0.5)
    ax5.set_axisbelow(True)
    ax5.set_ylim(bottom=0)
    ax5.legend(fontsize=6, facecolor=C["panel"], edgecolor=C["border"],
               labelcolor=C["text"])

    fig.suptitle(
        "NRP — Optimisation réseau éolien · Bourges (50 éoliennes, 6 communes) · BRKGA",
        color=C["text"], fontsize=13, fontweight="bold", y=0.97,
    )
    plt.savefig(out_path, dpi=160, bbox_inches="tight",
                facecolor=C["bg"], edgecolor="none")
    plt.close()
    print(f"  Figure enregistrée : {out_path}")


# ═══════════════════════════════════════════════════════════════════════
# 6.  RAPPORT TEXTE
# ═══════════════════════════════════════════════════════════════════════

def write_report(results, nodes, edges, scenarios,
                 to_open, to_close, out_path):
    lines = []
    def h(s=""): lines.append(s)
    def sep(c="═", n=68): lines.append(c * n)
    def row(l, v, col=30): lines.append(f"  {l:<{col}} {v}")

    now = datetime.now().strftime("%d/%m/%Y %H:%M")

    sep(); h("  RAPPORT D'OPTIMISATION NRP — RÉSEAU ÉOLIEN BOURGES")
    h(f"  Généré le {now}"); sep()

    h(); h("  RÉSEAU MODÉLISÉ"); sep("─")
    row("Nœuds total",             str(results["N"]))
    row("Éoliennes",               "50")
    row("Poste source",            "1  (HTB Bourges)")
    row("Communes",                "6  (Saint-Doulchard, Mehun-sur-Yèvre, Vierzon,")
    row("",                        "    Saint-Florent, Dun-sur-Auron, Châteauneuf-s-Cher)")
    row("Câbles candidats (arcs)", str(len(edges)))
    row("Tensions",                "20 kV (réseau HTA)")
    row("Résistance",              "0.30 Ω/km")
    row("Réactance",               "0.15 Ω/km")
    row("Topologie initiale",      "Arbre couvrant minimal (Kruskal géographique)")
    h()

    h("  ALGORITHME"); sep("─")
    row("Méthode",               "BRKGA (Biased Random-Key Genetic Algorithm)")
    row("Population",            "120")
    row("Élite",                 "20 %  (24 individus)")
    row("Mutants",               "10 %  (12 individus)")
    row("Prob. élite (ρa)",      "0.70")
    row("Générations max",       "300")
    row("Restart patience",      "50 générations sans amélioration")
    row("Décodage chromosome",   "Kruskal → arbre radial garanti")
    row("Fonction objectif",     "E[pertes] = Σ p_i · l_i(P,Q,V)  [DistFlow HTA]")
    h()

    h("  SCÉNARIOS DE GÉNÉRATION ÉOLIENNE (turbine de 2 MW installés)"); sep("─")
    h(f"  {'Scénario':<14} {'Probab.':>8}   {'Génération/turbine':>20}   {'Total 50 turbines':>18}")
    sep("─")
    for s, lbl in enumerate(["Vent faible", "Vent moyen", "Vent fort"]):
        gv = [200, 700, 1400][s]
        pr = [30,   50,   20][s]
        h(f"  {lbl:<14} {pr:>7}%   {gv:>18} kW   {gv*50:>16,} kW")
    h()

    h("  RÉSULTATS — PERTES ESPÉRÉES"); sep("─")
    row("Configuration actuelle",  f"{results['current_exp_kW']:>12,.1f} kW")
    row("Configuration optimale",  f"{results['optimal_exp_kW']:>12,.1f} kW")
    row("Réduction absolue",
        f"{results['current_exp_kW']-results['optimal_exp_kW']:>12,.1f} kW")
    row("Réduction relative",      f"{results['gain_pct']:>12.1f} %")
    h()

    h("  DÉTAIL PAR SCÉNARIO"); sep("─")
    h(f"  {'Scénario':<22} {'Prob':>5}  {'Génér. tot.':>12}  "
      f"{'Actuel (kW)':>12}  {'Optimal (kW)':>12}  {'Gain':>7}")
    sep("─")
    for s in scenarios:
        h(f"  {s['label']:<22} {s['prob']:>4}%  "
          f"{s['gen_total_kW']:>11,.0f}  "
          f"{s['current_kW']:>12,.1f}  "
          f"{s['optimal_kW']:>12,.1f}  "
          f"{s['gain_pct']:>6.1f}%")
    h()

    h("  CHARGES DES 6 COMMUNES"); sep("─")
    h(f"  {'Commune':<24} {'Population':>10}  "
      f"{'Charge P (kW)':>14}  {'Charge Q (kVAr)':>16}")
    sep("─")
    for nd in nodes:
        if nd["type"] == "commune":
            h(f"  {nd['name']:<24} {nd.get('pop',0):>10,}  "
              f"{nd['P_kW']:>14,.0f}  {nd['Q_kVAr']:>16,.0f}")
    h()

    h("  ACTIONS SUR LES INTERRUPTEURS"); sep("─")
    h(f"  Total modifications : {len(to_open)+len(to_close)}  "
      f"({len(to_open)} à ouvrir, {len(to_close)} à fermer)")
    h()
    h("  — INTERRUPTEURS À OUVRIR :")
    for e in to_open:
        na = nodes[e["from"]]["name"]
        nb = nodes[e["to"]]["name"]
        h(f"    ○  N{e['from']:02d} ({na[:20]:<20}) ↔  N{e['to']:02d} ({nb[:20]:<20})"
          f"  {e['dist_km']:.2f} km  r={e['r']:.3f} Ω")
    h()
    h("  — INTERRUPTEURS À FERMER :")
    for e in to_close:
        na = nodes[e["from"]]["name"]
        nb = nodes[e["to"]]["name"]
        h(f"    ●  N{e['from']:02d} ({na[:20]:<20}) ↔  N{e['to']:02d} ({nb[:20]:<20})"
          f"  {e['dist_km']:.2f} km  r={e['r']:.3f} Ω")
    h()

    h("  NOTES MÉTHODOLOGIQUES"); sep("─")
    notes = [
        "Les positions des éoliennes proviennent du shapefile IGN (éolien filtré).",
        "La topologie du réseau est construite par triangulation de Delaunay sur les",
        "  positions GPS réelles — méthode standard quand les tracés réels ne sont",
        "  pas disponibles publiquement (données Enedis/RTE propriétaires).",
        "Charges communes : 0.40 kW/habitant (P), 0.35 kVAr/habitant (Q).",
        "DistFlow simplifié : équations de flux linéarisées, Baran & Wu (1989).",
        "Config. 'actuelle' = MST géographique = câblage naïf sans optimisation.",
        "L'objectif BRKGA minimise E[pertes] sur les 3 scénarios de vent.",
    ]
    for note in notes:
        h(f"  · {note}")
    h()
    sep()
    h("  Référence : Cavalheiro, Vergílio, Lyra — Computers & Operations Research")
    h("              96 (2018) 272-280.  DOI : 10.1016/j.cor.2017.09.021")
    sep()

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"  Rapport enregistré : {out_path}")


# ═══════════════════════════════════════════════════════════════════════
# 7.  PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════

def main(shp_path="eolien_filtre.shp",
         out_png="nrp_bourges.png",
         out_txt="rapport_nrp.txt"):

    print("━" * 60)
    print("  NRP — Optimisation réseau éolien Bourges")
    print("━" * 60)

    # ── Données ──────────────────────────────────────────────────────
    print("\n[1/4] Chargement des données…")
    turbines = load_turbines(shp_path, n=50)
    feeder   = {
        "id": 50, "name": "Poste HTB Bourges", "type": "feeder",
        "lat": 47.0810, "lon": 2.3988, "dist_km": 0.0,
        "P_kW": 0.0, "Q_kVAr": 0.0,
        "gen_values_kW": [0, 0, 0],
        "gen_probs":     [0.3, 0.5, 0.2],
    }
    communes = build_communes()
    nodes, edges, N = build_network(turbines, feeder, communes)
    print(f"  {N} nœuds, {len(edges)} câbles candidats")

    # ── Configuration actuelle ────────────────────────────────────────
    print("\n[2/4] Évaluation de la configuration actuelle…")
    current_tree = [e for e in edges if e["switch"] == "NF"]
    current_loss = expected_loss(current_tree, nodes, feeder["id"])
    print(f"  Pertes espérées (actuel) : {current_loss:,.1f} kW")

    # ── BRKGA ────────────────────────────────────────────────────────
    print("\n[3/4] Optimisation BRKGA…")
    optimal_tree, optimal_loss, history = run_brkga(
        nodes, edges, N, feeder["id"],
        pop_size=120, n_elite=24, n_mutant=12,
        n_gen=300, rho=0.7, restart_patience=50,
        seed=0, verbose=True,
    )
    gain = (current_loss - optimal_loss) / current_loss * 100
    print(f"\n  Pertes espérées (optimal) : {optimal_loss:,.1f} kW")
    print(f"  Réduction : −{gain:.1f} %")

    # ── Changements d'interrupteurs ───────────────────────────────────
    cur_set = ({(e["from"], e["to"]) for e in current_tree}
               | {(e["to"], e["from"]) for e in current_tree})
    opt_set = ({(e["from"], e["to"]) for e in optimal_tree}
               | {(e["to"], e["from"]) for e in optimal_tree})
    to_open  = [e for e in current_tree if (e["from"], e["to"]) not in opt_set]
    to_close = [e for e in optimal_tree if (e["from"], e["to"]) not in cur_set]
    print(f"  Interrupteurs à ouvrir : {len(to_open)}")
    print(f"  Interrupteurs à fermer : {len(to_close)}")

    # ── Détail par scénario ───────────────────────────────────────────
    scenario_labels = ["Vent faible (10%)", "Vent moyen (35%)", "Vent fort (70%)"]
    scenario_probs  = [30, 50, 20]
    scenarios = []
    for s in range(3):
        gen = np.array([nd.get("gen_values_kW", [0,0,0])[s] for nd in nodes])
        cl  = distflow(current_tree, nodes, feeder["id"], gen)
        ol  = distflow(optimal_tree, nodes, feeder["id"], gen)
        scenarios.append({
            "label":        scenario_labels[s],
            "prob":         scenario_probs[s],
            "gen_total_kW": float(gen.sum()),
            "current_kW":   round(cl, 1),
            "optimal_kW":   round(ol, 1),
            "gain_pct":     round((cl - ol) / max(cl, 1) * 100, 1),
        })

    results = {
        "current_exp_kW": round(current_loss, 1),
        "optimal_exp_kW": round(optimal_loss, 1),
        "gain_pct":       round(gain, 1),
        "N": N,
    }

    # ── Sorties ──────────────────────────────────────────────────────
    print("\n[4/4] Génération des sorties…")
    make_figure(nodes, edges, current_tree, optimal_tree,
                to_open, to_close, scenarios, results, history, out_png)
    write_report(results, nodes, edges, scenarios,
                 to_open, to_close, out_txt)

    print("\n━" * 60)
    print(f"  ✓ Terminé.  Pertes : {current_loss:,.0f} → {optimal_loss:,.0f} kW  (−{gain:.1f}%)")
    print("━" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NRP — Optimisation de réseau éolien (Bourges)")
    parser.add_argument("--shp",     default="eolien_filtre.shp",
                        help="Chemin du shapefile éolien")
    parser.add_argument("--out-png", default="nrp_bourges.png",
                        help="Chemin de la figure de sortie")
    parser.add_argument("--out-txt", default="rapport_nrp.txt",
                        help="Chemin du rapport texte")
    args = parser.parse_args()
    main(shp_path=args.shp, out_png=args.out_png, out_txt=args.out_txt)