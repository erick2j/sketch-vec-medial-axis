# junction_pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Tuple, Set, Iterable, Optional, Any
import numpy as np
import networkx as nx
from heapq import heappush, heappop

# ================================
# Types & small primitives
# ================================
NodeId = int
Edge   = Tuple[NodeId, NodeId]

def pos(G: nx.Graph, n: NodeId) -> np.ndarray:
    return np.asarray(G.nodes[n]["position"], dtype=float)  # [row, col]

def set_pos(G: nx.Graph, n: NodeId, p: np.ndarray) -> None:
    G.nodes[n]["position"] = np.asarray(p, dtype=float)

def ordered_edge(u: NodeId, v: NodeId) -> Edge:
    return (u, v) if u < v else (v, u)

def edge_len(G: nx.Graph, u: NodeId, v: NodeId) -> float:
    return float(np.linalg.norm(pos(G, u) - pos(G, v)))

def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v

def new_node_id(G: nx.Graph) -> NodeId:
    return (max(G.nodes) + 1) if len(G) else 0

# ================================
# Path & growth results
# ================================
class BranchStopReason(Enum):
    ReachedJunction   = auto()
    Fork              = auto()
    AngleExceeded     = auto()
    DeadEnd           = auto()
    ClaimedEdge       = auto()   # kept for compatibility
    EdgeOwnedByOther  = auto()   # global ownership stop

@dataclass(frozen=True)
class Path:
    nodes: Tuple[NodeId, ...]
    stop: Optional[BranchStopReason] = None
    def edges(self) -> Tuple[Edge, ...]:
        if len(self.nodes) < 2:
            return tuple()
        return tuple(ordered_edge(a, b) for a, b in zip(self.nodes[:-1], self.nodes[1:]))

@dataclass(frozen=True)
class GrownJunction:
    center: NodeId
    branches: Tuple[Path, ...]
    nodes: frozenset[NodeId]
    edges: frozenset[Edge]

@dataclass(frozen=True)
class JunctionSubtree:
    nodes: frozenset[NodeId]
    edges: frozenset[Edge]
    centers: Tuple[NodeId, ...] = ()

# ================================
# Growth (global edge ownership)
# ================================
def _unique_next_neighbor(G: nx.Graph, prev: NodeId, curr: NodeId) -> Optional[NodeId]:
    fwd = [n for n in G.neighbors(curr) if n != prev]
    return fwd[0] if len(fwd) == 1 else None

def grow_from_center_owned(
    G: nx.Graph,
    center: NodeId,
    angle_thresh: float,
    edge_owner: Dict[Edge, NodeId],
) -> GrownJunction:
    """Grow branches from a degree>=3 node. First-come global edge ownership."""
    assert G.degree[center] >= 3
    branches: List[Path] = []

    for nb in G.neighbors(center):
        branch = [center, nb]
        prev, curr = center, nb
        while True:
            # Stop if we reached another junction (not the seed center)
            if G.degree[curr] >= 3 and curr != center:
                branches.append(Path(tuple(branch), BranchStopReason.ReachedJunction))
                break

            nxt = _unique_next_neighbor(G, prev, curr)
            if nxt is None:
                branches.append(Path(tuple(branch), BranchStopReason.Fork))
                break

            e = ordered_edge(curr, nxt)
            owner = edge_owner.get(e)
            if owner is None or owner == center:
                angle = float(G[curr][nxt].get("object angle", 0.0))  # NOTE: space in key
                branch.append(nxt)
                edge_owner[e] = center
                if angle > angle_thresh:
                    branches.append(Path(tuple(branch), BranchStopReason.AngleExceeded))
                    break
                prev, curr = curr, nxt
                continue

            # someone else owns it
            branches.append(Path(tuple(branch), BranchStopReason.EdgeOwnedByOther))
            break

    nodes = frozenset(n for br in branches for n in br.nodes)
    edges = frozenset(e for br in branches for e in br.edges())
    return GrownJunction(center=center, branches=tuple(branches), nodes=nodes, edges=edges)

def detect_grown_junctions(G: nx.Graph, angle_thresh: float) -> List[GrownJunction]:
    """Deterministic detection with global edge ownership."""
    edge_owner: Dict[Edge, NodeId] = {}
    grown: List[GrownJunction] = []
    for center in sorted(n for n in G.nodes if G.degree[n] >= 3):
        gj = grow_from_center_owned(G, center, angle_thresh, edge_owner)
        grown.append(gj)
    return grown

# ================================
# Clustering & bridging (single-link)
# ================================
def _pair_within_merge_radius(G: nx.Graph, a: NodeId, b: NodeId, merge_radius: float) -> bool:
    pa = pos(G, a); pb = pos(G, b)
    return float(np.linalg.norm(pa - pb)) <= float(merge_radius)

def _build_proximity_graph(G: nx.Graph, grown: List[GrownJunction], merge_radius: float) -> Dict[int, set[int]]:
    ids = list(range(len(grown)))
    adj: Dict[int, set[int]] = {i: set() for i in ids}
    for i in range(len(ids)):
        ci = grown[i].center
        for j in range(i + 1, len(ids)):
            cj = grown[j].center
            if _pair_within_merge_radius(G, ci, cj, merge_radius):
                adj[i].add(j); adj[j].add(i)
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
    out = [end]; u = end
    while u in par:
        u = par[u]; out.append(u)
    out.reverse(); return out

def _bridge_dijkstra(G: nx.Graph, sources: frozenset[NodeId], targets: frozenset[NodeId], weight: str="euclidean") -> Optional[Path]:
    """Multi-source Dijkstra; stops on first settled target."""
    def W(u: NodeId, v: NodeId) -> float:
        return edge_len(G, u, v) if weight == "euclidean" else 1.0

    pq: List[Tuple[float, NodeId]] = []
    dist: Dict[NodeId, float] = {}
    par: Dict[NodeId, NodeId] = {}
    seen: set[NodeId] = set()
    for s in sources:
        dist[s] = 0.0; heappush(pq, (0.0, s))
    targets_set = set(targets)

    while pq:
        d, u = heappop(pq)
        if u in seen: continue
        seen.add(u)
        if u in targets_set:
            nodes_list = _reconstruct_path(par, u)
            return Path(tuple(nodes_list))
        for v in G.neighbors(u):
            nd = d + W(u, v)
            if v not in dist or nd < dist[v]:
                dist[v] = nd; par[v] = u; heappush(pq, (nd, v))
    return None

def _union_subtrees(A: JunctionSubtree, B: JunctionSubtree, bridge: Optional[Path]) -> JunctionSubtree:
    nodes = set(A.nodes) | set(B.nodes)
    edges = set(A.edges) | set(B.edges)
    if bridge is not None and len(bridge.nodes) >= 2:
        nodes.update(bridge.nodes)
        edges.update(ordered_edge(u, v) for u, v in zip(bridge.nodes[:-1], bridge.nodes[1:]))
    centers = tuple(A.centers + B.centers)
    return JunctionSubtree(nodes=frozenset(nodes), edges=frozenset(edges), centers=centers)

def _prim_like_connect(G: nx.Graph, parts: List[JunctionSubtree], weight: str="euclidean") -> JunctionSubtree:
    assert len(parts) >= 1
    connected = parts[0]
    remaining = parts[1:]
    while remaining:
        best_i = None; best_bridge: Optional[Path] = None; best_len = float("inf")
        for i, cand in enumerate(remaining):
            br = _bridge_dijkstra(G, connected.nodes, cand.nodes, weight=weight)
            if br is None: continue
            L = 0.0
            for u, v in zip(br.nodes[:-1], br.nodes[1:]):
                L += edge_len(G, u, v) if weight == "euclidean" else 1.0
            if L < best_len:
                best_len = L; best_bridge = br; best_i = i
        if best_bridge is None or best_i is None:
            # cannot connect further; just union remaining without bridges
            for cand in remaining:
                connected = _union_subtrees(connected, cand, None)
            return connected
        connected = _union_subtrees(connected, remaining[best_i], best_bridge)
        remaining.pop(best_i)
    return connected

def cluster_and_merge(G: nx.Graph, grown: List[GrownJunction], merge_radius: float, weight: str="euclidean") -> List[JunctionSubtree]:
    """Single-link cluster (center distance <= merge_radius) → bridge to connected subtrees."""
    if not grown: return []
    adj = _build_proximity_graph(G, grown, merge_radius)
    comps = _connected_components(adj)
    subtrees: List[JunctionSubtree] = []
    for comp in comps:
        parts = [_subtree_from_grown(grown[i]) for i in comp]
        merged = _prim_like_connect(G, parts, weight=weight)
        subtrees.append(merged)
    return subtrees

# ================================
# Subtree analysis (leaves, tangents, CCW)
# ================================
@dataclass(frozen=True)
class SubtreeAnalysis:
    subtree_index: int
    nodes: Set[NodeId]
    edges: Set[Edge]
    leaf_nodes: List[NodeId]
    leaf_tangents: Dict[NodeId, np.ndarray]   # inward unit vectors (row, col)
    leaf_count: int
    jtype: str                                 # 'T-junction'|'Y-junction'|'X-junction'|'2-leaf'|'degenerate'
    ccw_center_rc: np.ndarray
    leaf_order_ccw: List[NodeId]
    leaf_angles: Dict[NodeId, float]

def _build_adj_from_edges(edges: Iterable[Edge]) -> Dict[NodeId, Set[NodeId]]:
    adj: Dict[NodeId, Set[NodeId]] = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj

def _subtree_internal_degrees(edges: Iterable[Edge]) -> Dict[NodeId, int]:
    deg: Dict[NodeId, int] = {}
    for u, v in edges:
        deg[u] = deg.get(u, 0) + 1; deg[v] = deg.get(v, 0) + 1
    return deg

def _subtree_leaf_nodes(nodes: Iterable[NodeId], edges: Iterable[Edge]) -> List[NodeId]:
    nodes = list(nodes); deg = _subtree_internal_degrees(edges)
    return [n for n in nodes if deg.get(n, 0) == 1]

def _pick_forward_neighbor(G: nx.Graph, adj: Dict[NodeId, Set[NodeId]], leaf: NodeId, nb: NodeId) -> Optional[NodeId]:
    candidates = [x for x in adj.get(nb, ()) if x != leaf]
    if not candidates: return None
    if len(candidates) == 1: return candidates[0]
    v1 = unit(pos(G, nb) - pos(G, leaf))
    best, best_dot = None, -1.0
    for c in candidates:
        vc = unit(pos(G, c) - pos(G, nb))
        d = float(np.dot(v1, vc))
        if d > best_dot:
            best_dot = d; best = c
    return best

def _leaf_inward_tangent_two_edge(G: nx.Graph, adj: Dict[NodeId, Set[NodeId]], leaf: NodeId) -> np.ndarray:
    nb = next(iter(adj[leaf]))
    p_leaf = pos(G, leaf); p_nb = pos(G, nb)
    fwd = _pick_forward_neighbor(G, adj, leaf, nb)
    if fwd is not None:
        p_fwd = pos(G, fwd); t = p_fwd - p_leaf
    else:
        t = p_nb - p_leaf
    return unit(t)

def _center_centroid(G: nx.Graph, nodes: Iterable[NodeId]) -> np.ndarray:
    P = np.vstack([pos(G, n) for n in nodes]); return P.mean(axis=0)

def _center_geometric_median(G: nx.Graph, nodes: Iterable[NodeId], iters: int=25, eps: float=1e-6) -> np.ndarray:
    P = np.vstack([pos(G, n) for n in nodes]); x = P.mean(axis=0)
    for _ in range(iters):
        d = np.linalg.norm(P - x, axis=1)
        if np.any(d < eps): return x
        w = 1.0 / (d + eps)
        x_new = (w[:, None] * P).sum(axis=0) / w.sum()
        if np.linalg.norm(x_new - x) < 1e-6:
            x = x_new; break
        x = x_new
    return x

def _ccw_angle_from_center(G: nx.Graph, center_rc: np.ndarray, leaf: NodeId) -> float:
    """True CCW with y-up by using (x=col, y=-row)."""
    leaf_rc = pos(G, leaf)
    cx, cy = center_rc[1], -center_rc[0]
    lx, ly = leaf_rc[1], -leaf_rc[0]
    vx, vy = (lx - cx), (ly - cy)
    ang = np.arctan2(vy, vx)
    if ang < 0: ang += 2.0 * np.pi
    return float(ang)


def _colinear_pair_matching(
    leaves: List[NodeId],
    tangents: Dict[NodeId, np.ndarray],
    colinear_dot: float,
) -> List[Tuple[NodeId, NodeId]]:
    """
    Build disjoint pairs of leaves whose |dot| >= colinear_dot,
    maximizing total alignment score (abs dot). Returns list of (u,v).
    """
    if len(leaves) < 2:
        return []

    # Build a weighted graph over leaves with edges for colinear-enough pairs
    H = nx.Graph()
    H.add_nodes_from(leaves)
    for i in range(len(leaves)):
        u = leaves[i]
        tu = unit(tangents[u])
        for j in range(i + 1, len(leaves)):
            v = leaves[j]
            tv = unit(tangents[v])
            score = abs(float(np.dot(tu, tv)))
            if score >= float(colinear_dot):
                H.add_edge(u, v, weight=score)

    if H.number_of_edges() == 0:
        return []

    # Maximize sum of scores (not min), so use max_weight_matching
    matching = nx.algorithms.matching.max_weight_matching(H, maxcardinality=True, weight="weight")
    # matching is a set of 2-tuples (u, v)
    return [(u, v) for (u, v) in matching]

def analyze_subtree(
    G: nx.Graph,
    subtree: JunctionSubtree | dict | Tuple[Iterable[NodeId], Iterable[Edge]],
    subtree_index: int,
    colinear_dot: float = 0.95,
    center_mode: str = "median",
) -> SubtreeAnalysis:
    # coerce
    if hasattr(subtree, "nodes") and hasattr(subtree, "edges"):
        nodes = set(subtree.nodes); edges = set(subtree.edges)
    elif isinstance(subtree, dict):
        nodes = set(subtree["nodes"]); edges = set(subtree["edges"])
    else:
        nset, eset = subtree; nodes, edges = set(nset), set(eset)

    adj = _build_adj_from_edges(edges)
    leaf_nodes = _subtree_leaf_nodes(nodes, edges)

    # two-edge inward tangents
    leaf_tangents: Dict[NodeId, np.ndarray] = {}
    for leaf in leaf_nodes:
        leaf_tangents[leaf] = _leaf_inward_tangent_two_edge(G, adj, leaf)

    L = len(leaf_nodes)
    # lightweight type
    jtype = "degenerate"
    if L == 2:
        jtype = "2-leaf"
    elif L == 3:
        dirs = [leaf_tangents[n] for n in leaf_nodes]
        pairs = ((0,1),(0,2),(1,2))
        num_colinear = sum(1 for i,j in pairs if abs(float(np.dot(dirs[i], dirs[j]))) > colinear_dot)
        jtype = "T-junction" if num_colinear == 1 else "Y-junction"
    elif L >= 4:
        jtype = "X-junction"

    ccw_center_rc = _center_geometric_median(G, nodes) if center_mode == "median" else _center_centroid(G, nodes)
    leaf_angles: Dict[NodeId, float] = {leaf: _ccw_angle_from_center(G, ccw_center_rc, leaf) for leaf in leaf_nodes}
    leaf_order_ccw: List[NodeId] = sorted(leaf_nodes, key=lambda n: leaf_angles[n])

    return SubtreeAnalysis(
        subtree_index=subtree_index,
        nodes=nodes, edges=edges,
        leaf_nodes=leaf_nodes,
        leaf_tangents=leaf_tangents,
        leaf_count=L,
        jtype=jtype,
        ccw_center_rc=ccw_center_rc,
        leaf_order_ccw=leaf_order_ccw,
        leaf_angles=leaf_angles,
    )

def analyze_subtrees(
    G: nx.Graph,
    subtrees: List[JunctionSubtree],
    colinear_dot: float = 0.92,      # unified default here too
    center_mode: str = "median"
) -> List[SubtreeAnalysis]:
    return [
        analyze_subtree(G, st, i, colinear_dot=colinear_dot, center_mode=center_mode)
        for i, st in enumerate(subtrees)
    ]

# ================================
# Rewiring: T-junction (3-leaf, one colinear pair)
# and No-colinear case (any leaf count)
# ================================
def _segment_ray_intersection(a: np.ndarray, b: np.ndarray, p: np.ndarray, v: np.ndarray, eps: float=1e-9) -> Optional[np.ndarray]:
    """Intersection of segment a->b with ray p + t v (t>=0) in row/col."""
    d = b - a
    M = np.array([[d[0], -v[0]],[d[1], -v[1]]], float)
    rhs = (p - a).astype(float)
    det = float(np.linalg.det(M))
    if abs(det) < eps: return None
    s, t = np.linalg.solve(M, rhs)
    if s < -eps or s > 1+eps or t < -eps: return None
    s = min(max(s, 0.0), 1.0)
    return a + s*d

def _closest_point_on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> np.ndarray:
    ab = b - a; denom = float(np.dot(ab, ab))
    if denom == 0.0: return a.copy()
    t = float(np.dot(p - a, ab)) / denom
    t = max(0.0, min(1.0, t))
    return a + t*ab

def _best_colinear_pair(leaf_ids: List[NodeId], leaf_tangents: Dict[NodeId, np.ndarray]) -> Tuple[Tuple[NodeId, NodeId], NodeId, float]:
    assert len(leaf_ids) == 3
    i, j, k = leaf_ids
    pairs = [(i,j),(i,k),(j,k)]
    best = None; best_score = -1.0
    for a, b in pairs:
        s = abs(float(np.dot(unit(leaf_tangents[a]), unit(leaf_tangents[b]))))
        if s > best_score:
            best_score = s; best = (a, b)
    remaining = ({i, j, k} - set(best)).pop()
    return best, remaining, best_score

def _ray_ray_intersection(p: np.ndarray, u: np.ndarray, q: np.ndarray, v: np.ndarray, eps: float=1e-9) -> Optional[np.ndarray]:
    # Solve p + s u = q + t v, s>=0, t>=0
    M = np.array([[u[0], -v[0]],[u[1], -v[1]]], float)
    rhs = (q - p).astype(float)
    det = float(np.linalg.det(M))
    if abs(det) < eps: return None
    s, t = np.linalg.solve(M, rhs)
    if s < -eps or t < -eps: return None
    s = max(0.0, float(s))
    return p + s*u

def _convex_hull(points: np.ndarray) -> np.ndarray:
    pts = np.unique(points, axis=0)
    if len(pts) <= 2: return pts
    pts_sorted = pts[np.lexsort((pts[:,0], pts[:,1]))]  # by col then row
    def cross(o,a,b): return (a[1]-o[1])*(b[0]-o[0]) - (a[0]-o[0])*(b[1]-o[1])
    lower=[]; 
    for p in pts_sorted:
        while len(lower)>=2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()
        lower.append(p)
    upper=[]
    for p in reversed(pts_sorted):
        while len(upper)>=2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
        upper.append(p)
    return np.vstack((lower[:-1], upper[:-1]))

def _point_in_convex(pt: np.ndarray, hull: np.ndarray, eps: float=1e-9) -> bool:
    if len(hull) == 0: return False
    if len(hull) == 1: return np.linalg.norm(pt - hull[0]) <= eps
    if len(hull) == 2:
        v = hull[1]-hull[0]
        t = np.dot(pt - hull[0], v) / (np.dot(v, v) + eps)
        t = max(0.0, min(1.0, t))
        closest = hull[0] + t*v
        return np.linalg.norm(pt - closest) <= eps
    sign=None; m=len(hull)
    for i in range(m):
        a=hull[i]; b=hull[(i+1)%m]
        edge=b-a; rel=pt-a
        z = edge[1]*rel[0] - edge[0]*rel[1]
        if abs(z) <= eps: continue
        s = 1 if z>0 else -1
        if sign is None: sign=s
        elif s != sign: return False
    return True

def rewire_subtree_colinear_pairs(
    G: nx.Graph,
    analysis: SubtreeAnalysis,
    colinear_dot: float,
) -> Optional[Dict[str, Any]]:
    """
    For any subtree: if there exist colinear leaf pairs (|dot| >= colinear_dot),
    - remove all original subtree edges,
    - add a straight edge between each disjoint colinear pair,
    - ignore any remaining leaves (for now).
    Returns a report or None if no pairs found.
    """
    leaves = list(analysis.leaf_nodes)
    if len(leaves) < 2:
        return None

    pairs = _colinear_pair_matching(leaves, analysis.leaf_tangents, colinear_dot=colinear_dot)
    if not pairs:
        return None

    # Remove original edges strictly inside the subtree
    removed = []
    for (u, v) in list(analysis.edges):
        if G.has_edge(u, v):
            G.remove_edge(u, v)
            removed.append((u, v))

    # Add straight connections between paired leaves
    added = []
    for (u, v) in pairs:
        if not G.has_edge(u, v):
            G.add_edge(u, v)
        added.append((u, v))

    # Determine which leaves were left unmatched (ignored for now)
    matched = set([n for uv in pairs for n in uv])
    unmatched = [n for n in leaves if n not in matched]

    return {
        "type": "ColinearPairs-rewire",
        "subtree_index": analysis.subtree_index,
        "pairs": pairs,
        "unmatched_leaves": unmatched,
        "removed_edges": removed,
        "added_edges": added,
        "leaf_count": analysis.leaf_count,
    }


def rewire_subtree_T(
    G: nx.Graph,
    analysis: SubtreeAnalysis,
    colinear_dot: float = 0.92,   # unified threshold
) -> Optional[Dict[str, Any]]:
    if analysis.leaf_count != 3:
        return None
    leaves = list(analysis.leaf_nodes)
    (l1, l2), l3, score = _best_colinear_pair(leaves, analysis.leaf_tangents)
    if score < float(colinear_dot):   # unified use
        return None
    p1 = pos(G, l1); p2 = pos(G, l2); p3 = pos(G, l3); t3 = unit(analysis.leaf_tangents[l3])
    inter = _segment_ray_intersection(p1, p2, p3, t3)
    attach = _closest_point_on_segment(p1, p2, p3) if inter is None else inter

    removed = []
    for (u, v) in list(analysis.edges):
        if G.has_edge(u, v):
            G.remove_edge(u, v); removed.append((u, v))

    c = new_node_id(G); G.add_node(c); set_pos(G, c, attach)
    G.add_edge(l1, c); G.add_edge(c, l2); G.add_edge(l3, c)
    return {
        "type": "T-rewire",
        "subtree_index": analysis.subtree_index,
        "colinear_pair": (l1, l2),
        "lonely_leaf": l3,
        "score_absdot": score,
        "center_node": c,
        "center_position": attach,
        "removed_edges": removed,
        "added_edges": [(l1, c), (c, l2), (l3, c)],
    }


def rewire_subtree_no_colinear(
    G: nx.Graph,
    analysis: SubtreeAnalysis,
    colinear_dot: float = 0.92,  # unified threshold
) -> Optional[Dict[str, Any]]:
    L = analysis.leaf_count
    if L < 2:
        return None

    leaves = list(analysis.leaf_nodes); tang = analysis.leaf_tangents

    # Require: NO pair is colinear at the same threshold
    for i in range(L):
        ti = unit(tang[leaves[i]])
        for j in range(i+1, L):
            tj = unit(tang[leaves[j]])
            if abs(float(np.dot(ti, tj))) >= float(colinear_dot):
                return None

    # average of ray–ray intersections; snap to closest leaf if outside hull
    pts = []
    for i in range(L):
        p = pos(G, leaves[i]); ui = unit(tang[leaves[i]])
        for j in range(i+1, L):
            q = pos(G, leaves[j]); uj = unit(tang[leaves[j]])
            inter = _ray_ray_intersection(p, ui, q, uj)
            if inter is not None and np.all(np.isfinite(inter)):
                pts.append(inter)
    if not pts:
        pts = [np.vstack([pos(G, n) for n in leaves]).mean(axis=0)]
    center = np.vstack(pts).mean(axis=0)

    leaf_pts = np.vstack([pos(G, n) for n in leaves])
    hull = _convex_hull(leaf_pts)
    if not _point_in_convex(center, hull):
        d = np.linalg.norm(leaf_pts - center[None, :], axis=1)
        center = leaf_pts[int(np.argmin(d))].copy()

    removed = []
    for (u, v) in list(analysis.edges):
        if G.has_edge(u, v):
            G.remove_edge(u, v); removed.append((u, v))

    c = new_node_id(G); G.add_node(c); set_pos(G, c, center)
    added = []
    for n in leaves:
        G.add_edge(n, c); added.append((n, c))
    return {
        "type": "NoColinear-rewire",
        "subtree_index": analysis.subtree_index,
        "center_node": c,
        "center_position": center,
        "removed_edges": removed,
        "added_edges": added,
        "leaf_count": L,
    }


# ================================
# Orchestrator
# ================================
def process_junctions(
    G: nx.Graph,
    angle_thresh: float = 0.35,
    merge_radius: float = 6.0,
    colinear_dot: float = 0.92,   # one knob used everywhere
    weight: str = "euclidean",
) -> Dict[str, Any]:
    """
    End-to-end pipeline (mutates G in-place):
      1) detect grown junctions (global edge ownership, 'object angle' stop),
      2) cluster by single-link center distance <= merge_radius and bridge into connected subtrees,
      3) analyze subtrees (leaves, two-edge inward tangents, CCW angles/order),
      4) rewiring rules:
         - If there is at least one colinear pair (|dot| >= colinear_dot), connect disjoint colinear pairs with straight edges; ignore remaining leaves.
         - Else (no colinear pairs at all, any leaf count >=2): place center at average of ray–ray intersections (snap to closest leaf if outside hull) and connect all leaves to center.
         - Else (leaf_count < 2): skip.
    """
    # 1) grow
    grown = detect_grown_junctions(G, angle_thresh)

    # 2) cluster & bridge
    subtrees = cluster_and_merge(G, grown, merge_radius, weight=weight)

    # 3) analyze
    analyses = analyze_subtrees(G, subtrees, colinear_dot=colinear_dot, center_mode="median")

    rewires: List[dict] = []
    skipped: List[Tuple[int, str]] = []

    for a in analyses:
        # First: try colinear-pairs rewiring (general, any leaf count >= 2)
        rep_pair = rewire_subtree_colinear_pairs(G, a, colinear_dot=colinear_dot)
        if rep_pair is not None:
            rewires.append(rep_pair)
            continue

        # Otherwise: use the "no-colinear" rule (average intersections), if applicable
        rep_no = rewire_subtree_no_colinear(G, a, colinear_dot=colinear_dot)
        if rep_no is not None:
            rewires.append(rep_no)
            continue

        # Nothing to do
        skipped.append((a.subtree_index, "no-rewire-rule"))

    return G, {
        "grown": grown,
        "subtrees": subtrees,
        "analyses": analyses,
        "rewires": rewires,
        "skipped": skipped,
    }

