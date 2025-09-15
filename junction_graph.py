"""
Junction Repair v2 — Core Abstractions + Growth Implementation (no GraphPos)

This version removes the GraphPos adapter and replaces it with small
utility functions: `pos(G,n)`, `set_pos(G,n,p)`, and `edge_len(G,u,v)`.
It includes a working implementation for detecting grown junctions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple, Dict, List, Optional, FrozenSet

import numpy as np
import networkx as nx

from junction_analysis import *
# -----------------------------
# Small primitives
# -----------------------------
NodeId = int
Edge = Tuple[NodeId, NodeId]


class BranchStopReason(Enum):
    ReachedJunction = auto()
    Fork = auto()
    AngleExceeded = auto()
    DeadEnd = auto()
    ClaimedEdge = auto()
    EdgeOwnedByOther = auto()


@dataclass(frozen=True)
class Path:
    """Immutable ordered node path with an optional stop reason."""
    nodes: Tuple[NodeId, ...]
    stop: Optional[BranchStopReason] = None

    def head(self) -> NodeId:
        return self.nodes[0]

    def tail(self) -> NodeId:
        return self.nodes[-1]

    def edges(self) -> Tuple[Edge, ...]:
        if len(self.nodes) < 2:
            return tuple()
        return tuple((min(a, b), max(a, b)) for a, b in zip(self.nodes[:-1], self.nodes[1:]))


@dataclass(frozen=True)
class GrownJunction:
    center: NodeId
    branches: Tuple[Path, ...]
    nodes: FrozenSet[NodeId]
    edges: FrozenSet[Edge]


class JunctionType(Enum):
    Unknown = auto()
    T = auto()
    Y = auto()
    X = auto()


@dataclass(frozen=True)
class JunctionSubtree:
    nodes: FrozenSet[NodeId]
    edges: FrozenSet[Edge]
    centers: Tuple[NodeId, ...] = field(default_factory=tuple)


@dataclass
class Junction:
    subtree: JunctionSubtree
    jtype: JunctionType
    rep_center: NodeId
    metadata: Dict[str, object] = field(default_factory=dict)


# -----------------------------
# Utilities for positions/lengths
# -----------------------------

def pos(G: nx.Graph, n: NodeId) -> np.ndarray:
    return np.asarray(G.nodes[n]["position"], dtype=float)


def set_pos(G: nx.Graph, n: NodeId, p: np.ndarray) -> None:
    G.nodes[n]["position"] = np.asarray(p, dtype=float)


def edge_len(G: nx.Graph, u: NodeId, v: NodeId) -> float:
    return float(np.linalg.norm(pos(G, u) - pos(G, v)))


def _ordered_edge(u: NodeId, v: NodeId) -> Edge:
    return (u, v) if u < v else (v, u)

def _ordered_edge(u: NodeId, v: NodeId) -> Edge:
    return (u, v) if u < v else (v, u)

def _unique_next_neighbor(G: nx.Graph, prev: NodeId, curr: NodeId) -> Optional[NodeId]:
    fwd = [n for n in G.neighbors(curr) if n != prev]
    return fwd[0] if len(fwd) == 1 else None

# -----------------------------
# Growth implementation
# -----------------------------
def grow_from_center_owned(
    G: nx.Graph,
    center: NodeId,
    angle_thresh: float,
    edge_owner: Dict[Edge, NodeId],   # shared across all centers
):
    """
    Grow branches from a degree>=3 node.
    An edge can be owned by at most one center (first-come in sorted center order).
    Stops a branch with EdgeOwnedByOther if it hits an edge already owned by another center.
    """
    assert G.degree[center] >= 3

    branches = []  # list[Path]
    for nb in G.neighbors(center):
        branch = [center, nb]
        prev, curr = center, nb

        while True:
            # (1) stop if we reached another junction (not the seed center)
            if G.degree[curr] >= 3 and curr != center:
                branches.append(Path(tuple(branch), BranchStopReason.ReachedJunction))
                break

            # (2) unique-forward step?
            nxt = _unique_next_neighbor(G, prev, curr)
            if nxt is None:
                # either fork or dead-end (non-unique or zero forward neighbors)
                # we can’t easily tell zero vs >1 without re-check; keep previous semantics: Fork
                branches.append(Path(tuple(branch), BranchStopReason.Fork))
                break

            e = _ordered_edge(curr, nxt)

            # (3) ownership check (global)
            owner = edge_owner.get(e)
            if owner is None or owner == center:
                # append, claim, maybe stop on angle
                angle = float(G[curr][nxt].get('object angle', 0.0))
                branch.append(nxt)
                edge_owner[e] = center  # claim for my center

                if angle > angle_thresh:
                    branches.append(Path(tuple(branch), BranchStopReason.AngleExceeded))
                    break

                prev, curr = curr, nxt
                continue

            # (4) someone else owns it -> stop this branch here
            branches.append(Path(tuple(branch), BranchStopReason.EdgeOwnedByOther))
            break

    # pack grown junction
    nodes = frozenset(n for br in branches for n in br.nodes)
    edges = frozenset(_ordered_edge(u, v) for br in branches for u, v in zip(br.nodes[:-1], br.nodes[1:]))
    return GrownJunction(center=center, branches=tuple(branches), nodes=nodes, edges=edges)


def detect_grown_junctions(G: nx.Graph, angle_thresh: float):
    """
    Deterministic order (sorted node ids). Global ownership prevents overlapping subtrees.
    """
    edge_owner: Dict[Edge, NodeId] = {}
    grown: List[GrownJunction] = []
    for center in sorted(n for n in G.nodes if G.degree[n] >= 3):
        gj = grow_from_center_owned(G, center, angle_thresh, edge_owner)
        grown.append(gj)
    return grown

# -----------------------------
# Clustering & Bridging into Connected Subtrees
# -----------------------------
from heapq import heappush, heappop


def _center_pos(G: nx.Graph, n: NodeId) -> np.ndarray:
    return pos(G, n)


def _center_distance(G: nx.Graph, a: NodeId, b: NodeId) -> float:
    return float(np.linalg.norm(_center_pos(G, a) - _center_pos(G, b)))

def _pair_within_merge_radius(G: nx.Graph, a: NodeId, b: NodeId, merge_radius: float) -> bool:
    pa = np.asarray(G.nodes[a]['position'], float)
    pb = np.asarray(G.nodes[b]['position'], float)
    return float(np.linalg.norm(pa - pb)) <= float(merge_radius)

def _build_proximity_graph(G: nx.Graph, grown: List[GrownJunction], merge_radius: float) -> Dict[int, set[int]]:
    ids = list(range(len(grown)))
    adj: Dict[int, set[int]] = {i: set() for i in ids}
    for i in range(len(ids)):
        ci = grown[i].center
        for j in range(i + 1, len(ids)):
            cj = grown[j].center
            if _pair_within_merge_radius(G, ci, cj, merge_radius):
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _connected_components(adj: Dict[int, set[int]]) -> List[List[int]]:
    unseen = set(adj.keys())
    comps: List[List[int]] = []
    while unseen:
        s = unseen.pop()
        comp = [s]
        stack = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v in unseen:
                    unseen.remove(v)
                    comp.append(v)
                    stack.append(v)
        comps.append(comp)
    return comps


def _subtree_from_grown(g: GrownJunction) -> JunctionSubtree:
    return JunctionSubtree(nodes=frozenset(g.nodes), edges=frozenset(g.edges), centers=(g.center,))


def _reconstruct_path(par: Dict[NodeId, NodeId], end: NodeId) -> List[NodeId]:
    out = [end]
    u = end
    while u in par:
        u = par[u]
        out.append(u)
    out.reverse()
    return out


def _bridge_dijkstra(G: nx.Graph, sources: FrozenSet[NodeId], targets: FrozenSet[NodeId], weight: str = "euclidean") -> Optional[Path]:
    """Multi-source Dijkstra that stops on first settled target. Returns a Path or None."""
    def W(u: NodeId, v: NodeId) -> float:
        return edge_len(G, u, v) if weight == "euclidean" else 1.0

    pq: List[Tuple[float, NodeId]] = []
    dist: Dict[NodeId, float] = {}
    par: Dict[NodeId, NodeId] = {}
    seen: set[NodeId] = set()

    for s in sources:
        dist[s] = 0.0
        heappush(pq, (0.0, s))

    targets_set = set(targets)

    while pq:
        d, u = heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u in targets_set:
            nodes_list = _reconstruct_path(par, u)
            return Path(tuple(nodes_list))
        for v in G.neighbors(u):
            nd = d + W(u, v)
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                par[v] = u
                heappush(pq, (nd, v))
    return None


def _union_subtrees(A: JunctionSubtree, B: JunctionSubtree, bridge: Optional[Path]) -> JunctionSubtree:
    nodes = set(A.nodes) | set(B.nodes)
    edges = set(A.edges) | set(B.edges)
    if bridge is not None and len(bridge.nodes) >= 2:
        nodes.update(bridge.nodes)
        edges.update((_ordered_edge(u, v) for u, v in zip(bridge.nodes[:-1], bridge.nodes[1:])))
    centers = tuple(A.centers + B.centers)
    return JunctionSubtree(nodes=frozenset(nodes), edges=frozenset(edges), centers=centers)


def _prim_like_connect(G: nx.Graph, parts: List[JunctionSubtree], weight: str = "euclidean") -> JunctionSubtree:
    """Connect a list of subtrees into one connected subtree by repeatedly adding the
    shortest bridge from the connected set to any remaining part (Prim's algorithm style)."""
    assert len(parts) >= 1
    connected = parts[0]
    remaining = parts[1:]

    while remaining:
        best_i = None
        best_bridge: Optional[Path] = None
        best_len = float("inf")

        # find the cheapest bridge from connected -> any remaining
        for i, cand in enumerate(remaining):
            br = _bridge_dijkstra(G, connected.nodes, cand.nodes, weight=weight)
            if br is None:
                continue
            # measure length
            L = 0.0
            for u, v in zip(br.nodes[:-1], br.nodes[1:]):
                L += edge_len(G, u, v) if weight == "euclidean" else 1.0
            if L < best_len:
                best_len = L
                best_bridge = br
                best_i = i

        if best_bridge is None or best_i is None:
            # disconnected graph: cannot connect further; return union of what we have
            # (still a valid subtree for the connected component)
            # merge all remaining without bridges to preserve nodes/edges info
            for cand in remaining:
                connected = _union_subtrees(connected, cand, None)
            return connected

        connected = _union_subtrees(connected, remaining[best_i], best_bridge)
        remaining.pop(best_i)

    return connected


def cluster_and_merge(G: nx.Graph, grown: List[GrownJunction], merge_radius: float, weight: str = "euclidean") -> List[JunctionSubtree]:
    if not grown:
        return []

    adj = _build_proximity_graph(G, grown, merge_radius)
    comps = _connected_components(adj)

    subtrees: List[JunctionSubtree] = []
    for comp in comps:
        parts = [JunctionSubtree(nodes=frozenset(grown[i].nodes),
                                 edges=frozenset(grown[i].edges),
                                 centers=(grown[i].center,))
                 for i in comp]
        merged = _prim_like_connect(G, parts, weight=weight)
        subtrees.append(merged)
    return subtrees

# -----------------------------
# Orchestrator: detect + cluster/merge
# -----------------------------

def detect_and_merge_junctions(G: nx.Graph, angle_thresh: float, stroke_width: float, weight: str = "euclidean") -> List[JunctionSubtree]:
    """One-shot convenience: grow all degree>=3 seeds, then cluster and merge them into connected subtrees.

    Returns a list of JunctionSubtree objects.
    """
    grown = detect_grown_junctions(G, angle_thresh)
    merged = cluster_and_merge(G, grown, stroke_width, weight=weight)
    return merged

# -----------------------------
# One-shot orchestration (detect → cluster+merge)
# -----------------------------

def build_junction_subtrees(G: nx.Graph, angle_thresh: float, merge_radius: float, weight: str = "euclidean"):
    grown = detect_grown_junctions(G, angle_thresh)
    subtrees = cluster_and_merge(G, grown, merge_radius, weight=weight)
    return G, subtrees

def process_junctions(
    G: nx.Graph,
    angle_thresh: float = 0.35,
    merge_radius: float = 6.0,
    analysis_colinear_dot: float = 0.95,
    t_rewire_colinear_dot: float = 0.95,
    weight: str = "euclidean",
    mutate: bool = True,
) -> Dict[str, Any]:
    """
    One-call pipeline:
      1) Detect grown junctions from all degree>=3 seeds using 'angle_thresh'.
      2) Cluster by single-link proximity (center distance <= merge_radius), then bridge to connected subtrees.
      3) Analyze each subtree (leaf nodes, two-edge inward tangents, CCW order).
      4) Rewire only 3-leaf T-junctions:
           - find most-colinear leaf pair (|dot| >= t_rewire_colinear_dot),
           - build bar between those leaves,
           - attach the third branch at ray∩bar (or closest point),
           - replace original subtree edges with the new 3-edge configuration via a new center node.
      Returns a report with intermediate artifacts and rewiring actions.

    Parameters
    ----------
    G : nx.Graph
        Graph with node attribute 'position' = [row, col] and edge attribute 'object angle'.
    angle_thresh : float
        Stop criterion for growth (edge appended, then stop if 'object angle' > threshold).
    merge_radius : float
        Single-link clustering radius for merging grown structures by center proximity.
    analysis_colinear_dot : float
        Cosine threshold for classifying 3-leaf subtrees into T vs Y during analysis.
    t_rewire_colinear_dot : float
        Cosine threshold for deciding the colinear pair during T rewiring (noise tolerance).
    weight : str
        'euclidean' (default) for geometric bridging.
    mutate : bool
        If True, rewiring mutates G in place. If False, rewiring is skipped (dry run).

    Returns
    -------
    dict with keys:
        'grown'     : List[GrownJunction]
        'subtrees'  : List[JunctionSubtree]
        'analyses'  : List[SubtreeAnalysis]
        'rewires'   : List[dict] (only if mutate=True; one entry per successful T rewire)
        'skipped'   : List[Tuple[int, str]] (subtree_index, reason)
    """
    # 1) grow
    grown = detect_grown_junctions(G, angle_thresh)

    # 2) cluster & bridge into connected subtrees
    subtrees = cluster_and_merge(G, grown, merge_radius, weight=weight)

    # 3) analyze (with your two-edge tangent improvement integrated in analyze_subtrees)
    analyses = analyze_subtrees(G, subtrees, colinear_dot=analysis_colinear_dot, center_mode="median")

    # 4) rewire 3-leaf T-junctions
    rewires: List[dict] = []
    skipped: List[Tuple[int, str]] = []

    for a in analyses:
        if a.leaf_count != 3:
            skipped.append((a.subtree_index, "not-3-leaf"))
            continue

        # Decide if it's T-like (exact logic is inside rewire_subtree_T via colinear threshold)
        if not mutate:
            # Dry run: just check if it *would* rewire
            leaves = a.leaf_nodes
            tmap = a.leaf_tangents
            # quick best-pair check (same as rewire function)
            i, j, k = leaves
            pairs = [(i, j), (i, k), (j, k)]
            best_s = -1.0
            for u, v in pairs:
                s = abs(float(np.dot(tmap[u], tmap[v])))
                if s > best_s:
                    best_s = s
            if best_s >= float(t_rewire_colinear_dot):
                rewires.append({"type": "T-rewire (dry)", "subtree_index": a.subtree_index, "score_absdot": best_s})
            else:
                skipped.append((a.subtree_index, "3-leaf-but-not-colinear-enough"))
            continue

        # Mutating rewire
        rep = rewire_subtree_T(G, a, colinear_dot_thresh=t_rewire_colinear_dot)
        if rep is not None:
            rep["subtree_index"] = a.subtree_index
            rewires.append(rep)
        else:
            skipped.append((a.subtree_index, "3-leaf-but-not-colinear-enough"))

    return G, {
        "grown": grown,
        "subtrees": subtrees,
        "analyses": analyses,
        "rewires": rewires,
        "skipped": skipped,
    }
