# junction_pipeline_readable.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from heapq import heappush, heappop
from typing import Dict, List, Tuple, Set, Iterable, Optional, Any

import numpy as np
import networkx as nx


# =============================================================================
# Basic aliases and tiny primitives
# =============================================================================

NodeId = int
Edge   = Tuple[NodeId, NodeId]

def ordered_edge(u: NodeId, v: NodeId) -> Edge:
    return (u, v) if u < v else (v, u)

def pos(G: nx.Graph, n: NodeId) -> np.ndarray:
    """Return node position as float array [row, col]."""
    return np.asarray(G.nodes[n]["position"], dtype=float)

def set_pos(G: nx.Graph, n: NodeId, p: np.ndarray) -> None:
    G.nodes[n]["position"] = np.asarray(p, dtype=float)

def edge_len(G: nx.Graph, u: NodeId, v: NodeId) -> float:
    return float(np.linalg.norm(pos(G, u) - pos(G, v)))

def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v

def new_node_id(G: nx.Graph) -> NodeId:
    return (max(G.nodes) + 1) if len(G) else 0


# =============================================================================
# Growth datatypes
# =============================================================================

class BranchStopReason(Enum):
    """Why a growing branch stopped."""
    ReachedJunction   = auto()
    Fork              = auto()
    AngleExceeded     = auto()
    DeadEnd           = auto()
    ClaimedEdge       = auto()         # kept for compatibility
    EdgeOwnedByOther  = auto()         # global (graph-wide) edge ownership stop

@dataclass(frozen=True)
class Path:
    """A simple node-path wrapper plus a convenience for edges()."""
    nodes: Tuple[NodeId, ...]
    stop: Optional[BranchStopReason] = None

    def edges(self) -> Tuple[Edge, ...]:
        if len(self.nodes) < 2:
            return tuple()
        return tuple(ordered_edge(a, b) for a, b in zip(self.nodes[:-1], self.nodes[1:]))

@dataclass(frozen=True)
class GrownJunction:
    """Result of growing from one junction center (degree >= 3)."""
    center: NodeId
    branches: Tuple[Path, ...]
    nodes: frozenset[NodeId]
    edges: frozenset[Edge]


@dataclass
class _BranchState:
    """Internal mutable state for one growing branch during simultaneous expansion."""
    center: NodeId
    prev: NodeId
    curr: NodeId
    path: List[NodeId]
    active: bool = True
    stop: Optional[BranchStopReason] = None

@dataclass(frozen=True)
class JunctionSubtree:
    """A connected union of grown junctions (with optional bridges)."""
    nodes: frozenset[NodeId]
    edges: frozenset[Edge]
    centers: Tuple[NodeId, ...] = ()


# =============================================================================
# Geometry helpers (compact & self-contained)
# =============================================================================

def ray_segment_intersection_params(
    seg_a: np.ndarray, seg_b: np.ndarray,  # segment endpoints (row, col)
    ray_p: np.ndarray, ray_u: np.ndarray,  # ray: ray_p + t*ray_u, t >= 0
    eps: float = 1e-9
) -> Optional[Tuple[float, float, np.ndarray]]:
    """
    Intersect a ray with a *segment*.
    Returns (t, s, X):
      - t >= 0 is the ray parameter (ray_p + t ray_u),
      - s in [0,1] is the segment parameter (seg_a + s (seg_b - seg_a)),
      - X is the intersection point.
    Returns None if parallel or no valid intersection.
    """
    d = seg_b - seg_a
    M = np.array([[ray_u[0], -d[0]], [ray_u[1], -d[1]]], float)  # ray_p + t u = seg_a + s d
    rhs = (seg_a - ray_p).astype(float)
    det = float(np.linalg.det(M))
    if abs(det) < eps:
        return None
    t, s = np.linalg.solve(M, rhs)
    if t < -eps or s < -eps or s > 1.0 + eps:
        return None
    # clamp mild numeric drift
    t = max(0.0, float(t))
    s = min(max(float(s), 0.0), 1.0)
    X = seg_a + s * d
    return (t, s, X)

def ray_line_intersection_params(
    line_a: np.ndarray, line_b: np.ndarray,
    ray_p: np.ndarray, ray_u: np.ndarray,
    eps: float = 1e-9
) -> Optional[Tuple[float, float, np.ndarray]]:
    """
    Intersect a ray with the *infinite* line through (line_a, line_b).
    Returns (t, s, X) with t >= 0; s is unbounded; X = line_a + s (line_b - line_a).
    """
    d = line_b - line_a
    M = np.array([[ray_u[0], -d[0]], [ray_u[1], -d[1]]], float)
    rhs = (line_a - ray_p).astype(float)
    det = float(np.linalg.det(M))
    if abs(det) < eps:
        return None
    t, s = np.linalg.solve(M, rhs)
    if t < 0.0:
        return None
    X = line_a + s * d
    return (float(t), float(s), X)

def point_to_ray_distance(ray_p: np.ndarray, ray_u: np.ndarray, q: np.ndarray) -> Tuple[float, float]:
    """
    Distance from point q to ray ray_p + t ray_u (t >= 0), and the corresponding t*.
    Returns (t_star, distance).
    """
    w = q - ray_p
    t_star = float(np.dot(w, ray_u))
    if t_star <= 0.0:
        return (0.0, float(np.linalg.norm(q - ray_p)))
    proj = ray_p + t_star * ray_u
    return (t_star, float(np.linalg.norm(q - proj)))

def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew’s monotone chain on unique points ([row, col])."""
    pts = np.unique(points, axis=0)
    if len(pts) <= 2:
        return pts
    pts_sorted = pts[np.lexsort((pts[:, 0], pts[:, 1]))]  # sort by col, then row

    def cross(o, a, b):
        return (a[1] - o[1]) * (b[0] - o[0]) - (a[0] - o[0]) * (b[1] - o[1])

    lower = []
    for p in pts_sorted:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts_sorted):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.vstack((lower[:-1], upper[:-1]))

def point_in_convex(pt: np.ndarray, hull: np.ndarray, eps: float = 1e-9) -> bool:
    """Consistent inside check for convex polygon. Includes border as inside."""
    m = len(hull)
    if m == 0:
        return False
    if m == 1:
        return np.linalg.norm(pt - hull[0]) <= eps
    if m == 2:
        v = hull[1] - hull[0]
        t = np.dot(pt - hull[0], v) / (np.dot(v, v) + eps)
        t = max(0.0, min(1.0, t))
        closest = hull[0] + t * v
        return np.linalg.norm(pt - closest) <= eps

    sign = None
    for i in range(m):
        a = hull[i]
        b = hull[(i + 1) % m]
        edge = b - a
        rel = pt - a
        z = edge[1] * rel[0] - edge[0] * rel[1]
        if abs(z) <= eps:
            continue
        s = 1 if z > 0 else -1
        if sign is None:
            sign = s
        elif s != sign:
            return False
    return True


# =============================================================================
# 1) Grow branches with global edge ownership (prevents overgrowth overlap)
# =============================================================================

def _unique_next_neighbor(G: nx.Graph, prev: NodeId, curr: NodeId) -> Optional[NodeId]:
    fwd = [n for n in G.neighbors(curr) if n != prev]
    return fwd[0] if len(fwd) == 1 else None

def grow_from_center(
    G: nx.Graph,
    center: NodeId,
    angle_thresh: float,
    edge_owner: Dict[Edge, NodeId],
) -> GrownJunction:
    """
    Grow along degree-2 chains from a center (degree >= 3),
    stopping on: junctions, forks, angle threshold, or globally owned edges.
    """
    branches: List[Path] = []

    for nb in G.neighbors(center):
        branch = [center]

        e0 = ordered_edge(center, nb)
        owner0 = edge_owner.get(e0)
        if owner0 is not None and owner0 != center:
            branches.append(Path(tuple(branch), BranchStopReason.EdgeOwnedByOther))
            continue

        # Claim the first edge out of the center before traversing further.
        edge_owner[e0] = center

        branch.append(nb)
        prev, curr = center, nb

        while True:
            # Reached another junction → stop
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
                # Claim and test angle
                angle = float(G[curr][nxt].get("object angle", 0.0))  # NOTE: key includes a space
                branch.append(nxt)
                edge_owner[e] = center
                if angle > angle_thresh:
                    branches.append(Path(tuple(branch), BranchStopReason.AngleExceeded))
                    break
                prev, curr = curr, nxt
                continue

            # Someone else owns it → stop
            branches.append(Path(tuple(branch), BranchStopReason.EdgeOwnedByOther))
            break

    nodes = frozenset(n for br in branches for n in br.nodes)
    edges = frozenset(e for br in branches for e in br.edges())
    return GrownJunction(center=center, branches=tuple(branches), nodes=nodes, edges=edges)

def detect_grown_junctions(G: nx.Graph, angle_thresh: float) -> List[GrownJunction]:
    """Grow all junction centers in lockstep so no single center monopolizes edges."""
    centers = sorted(n for n in G.nodes if G.degree[n] >= 3)
    if not centers:
        return []

    edge_owner: Dict[Edge, NodeId] = {}

    center_data: Dict[NodeId, Dict[str, object]] = {
        c: {
            "branches": [],          # list[Path]
            "nodes": {c},            # set[NodeId]
            "edges": set(),          # set[Edge]
        }
        for c in centers
    }

    states: List[_BranchState] = []
    for center in centers:
        for nb in G.neighbors(center):
            states.append(_BranchState(center=center, prev=center, curr=nb, path=[center]))

    def finalize(state: _BranchState, reason: BranchStopReason) -> None:
        if not state.active:
            return
        state.stop = reason
        state.active = False
        center_data[state.center]["branches"].append(Path(tuple(state.path), reason))

    active = [s for s in states if s.active]
    while active:
        next_active: List[_BranchState] = []
        for state in active:
            if not state.active:
                continue

            prev = state.prev
            curr = state.curr
            edge = ordered_edge(prev, curr)
            owner = edge_owner.get(edge)

            if owner is None:
                edge_owner[edge] = state.center
            elif owner != state.center:
                finalize(state, BranchStopReason.EdgeOwnedByOther)
                continue

            if not state.path or state.path[-1] != curr:
                state.path.append(curr)
                center_data[state.center]["nodes"].add(curr)
                center_data[state.center]["edges"].add(edge)

            angle = float(G[prev][curr].get("object angle", 0.0))
            if angle > angle_thresh:
                finalize(state, BranchStopReason.AngleExceeded)
                continue

            if G.degree[curr] >= 3 and curr != state.center:
                finalize(state, BranchStopReason.ReachedJunction)
                continue

            fwd = [n for n in G.neighbors(curr) if n != prev]
            if not fwd:
                finalize(state, BranchStopReason.DeadEnd)
                continue
            if len(fwd) > 1:
                finalize(state, BranchStopReason.Fork)
                continue

            nxt = fwd[0]
            state.prev = curr
            state.curr = nxt
            next_active.append(state)

        active = next_active

    # Finalize any states that never progressed beyond the center (e.g. isolated centers)
    for state in states:
        if state.active:
            finalize(state, BranchStopReason.DeadEnd)

    grown: List[GrownJunction] = []
    for center in centers:
        data = center_data[center]
        branches = tuple(data["branches"])
        nodes = frozenset(data["nodes"])
        edges = frozenset(data["edges"])
        grown.append(GrownJunction(center=center, branches=branches, nodes=nodes, edges=edges))
    return grown


# =============================================================================
# 2) Cluster nearby centers and bridge into connected subtrees
# =============================================================================

def _within_radius(G: nx.Graph, a: NodeId, b: NodeId, r: float) -> bool:
    return float(np.linalg.norm(pos(G, a) - pos(G, b))) <= float(r)

def _build_center_proximity(grown: List[GrownJunction], G: nx.Graph, merge_radius: float) -> Dict[int, Set[int]]:
    """
    Build an adjacency graph over grown junctions combining two criteria:
      - Euclidean center distance <= merge_radius.
      - Shared graph nodes (overlapping node sets).
    """
    adj: Dict[int, Set[int]] = {i: set() for i in range(len(grown))}
    for i in range(len(grown)):
        gi = grown[i]
        ci = gi.center
        for j in range(i + 1, len(grown)):
            gj = grown[j]
            cj = gj.center

            near = _within_radius(G, ci, cj, merge_radius)
            overlap = bool(gi.nodes & gj.nodes)

            if near or overlap:
                adj[i].add(j)
                adj[j].add(i)
    return adj

def _components(adj: Dict[int, Set[int]]) -> List[List[int]]:
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

def _bridge_multi_source(
    G: nx.Graph,
    A: frozenset[NodeId],
    B: frozenset[NodeId],
    weight: str = "euclidean"
) -> Optional[Path]:
    """
    Dijkstra that starts from all nodes in A, stops on first node in B.
    weight = "euclidean" for geometric length, else unit-weight.
    """
    def W(u: NodeId, v: NodeId) -> float:
        return edge_len(G, u, v) if weight == "euclidean" else 1.0

    pq: List[Tuple[float, NodeId]] = []
    dist: Dict[NodeId, float] = {}
    par: Dict[NodeId, NodeId] = {}
    seen: Set[NodeId] = set()
    targets = set(B)

    for s in A:
        dist[s] = 0.0
        heappush(pq, (0.0, s))

    while pq:
        d, u = heappop(pq)
        if u in seen:
            continue
        seen.add(u)

        if u in targets:
            # reconstruct path A → u
            out = [u]
            while u in par:
                u = par[u]
                out.append(u)
            out.reverse()
            return Path(tuple(out))

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
        edges.update(ordered_edge(u, v) for u, v in zip(bridge.nodes[:-1], bridge.nodes[1:]))
    return JunctionSubtree(
        nodes=frozenset(nodes),
        edges=frozenset(edges),
        centers=tuple(A.centers + B.centers),
    )

def _connect_parts(G: nx.Graph, parts: List[JunctionSubtree], weight: str = "euclidean") -> JunctionSubtree:
    """Prim-like greedy: repeatedly bridge nearest remaining part into the growing tree."""
    assert parts
    tree = parts[0]
    rest = parts[1:]

    while rest:
        best_i = None
        best_bridge = None
        best_cost = float("inf")

        for i, cand in enumerate(rest):
            br = _bridge_multi_source(G, tree.nodes, cand.nodes, weight=weight)
            if br is None:
                continue
            cost = 0.0 if weight != "euclidean" else sum(
                np.linalg.norm(pos(G, a) - pos(G, b)) for a, b in zip(br.nodes[:-1], br.nodes[1:])
            )
            if cost < best_cost:
                best_cost = cost
                best_bridge = br
                best_i = i

        if best_bridge is None:
            # nothing is connectable — just union remaining (structure stays a forest topologically)
            for cand in rest:
                tree = _union_subtrees(tree, cand, None)
            return tree

        tree = _union_subtrees(tree, rest[best_i], best_bridge)
        rest.pop(best_i)

    return tree

def cluster_and_merge(
    G: nx.Graph,
    grown: List[GrownJunction],
    merge_radius: float,
    weight: str = "euclidean",
) -> List[JunctionSubtree]:
    """Single-link cluster on center proximity or shared nodes, then bridge each cluster into one connected subtree."""
    if not grown:
        return []
    prox = _build_center_proximity(grown, G, merge_radius)
    subtrees: List[JunctionSubtree] = []

    for comp in _components(prox):
        parts = [
            JunctionSubtree(nodes=g.nodes, edges=g.edges, centers=(g.center,))
            for g in (grown[i] for i in comp)
        ]
        subtrees.append(_connect_parts(G, parts, weight=weight))
    return subtrees


# =============================================================================
# 3) Subtree analysis (leaves, two-edge tangents, CCW ordering)
# =============================================================================

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

def _internal_degrees(edges: Iterable[Edge]) -> Dict[NodeId, int]:
    deg: Dict[NodeId, int] = {}
    for u, v in edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    return deg

def _subtree_leaf_nodes(nodes: Iterable[NodeId], edges: Iterable[Edge]) -> List[NodeId]:
    deg = _internal_degrees(edges)
    return [n for n in nodes if deg.get(n, 0) == 1]

def _build_adj_from_edges(edges: Iterable[Edge]) -> Dict[NodeId, Set[NodeId]]:
    adj: Dict[NodeId, Set[NodeId]] = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj

def _forward_neighbor(G: nx.Graph, adj: Dict[NodeId, Set[NodeId]], leaf: NodeId, nb: NodeId) -> Optional[NodeId]:
    """
    Pick the neighbor forward from (leaf → nb).
    If multiple, choose the one most aligned with the current direction.
    """
    candidates = [x for x in adj.get(nb, ()) if x != leaf]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    v1 = unit(pos(G, nb) - pos(G, leaf))
    best, best_dot = None, -1.0
    for c in candidates:
        vc = unit(pos(G, c) - pos(G, nb))
        d = float(np.dot(v1, vc))
        if d > best_dot:
            best_dot = d
            best = c
    return best

def _leaf_inward_tangent_two_edges(G: nx.Graph, adj: Dict[NodeId, Set[NodeId]], leaf: NodeId) -> np.ndarray:
    """
    Denoised inward tangent:
      - Prefer the vector leaf → forward-of-neighbor (two-edge estimate),
      - otherwise fall back to leaf → neighbor.
    """
    nb = next(iter(adj[leaf]))
    forward = _forward_neighbor(G, adj, leaf, nb)
    if forward is not None:
        return unit(pos(G, forward) - pos(G, leaf))
    return unit(pos(G, nb) - pos(G, leaf))

def _center_geometric_median(G: nx.Graph, nodes: Iterable[NodeId], iters: int = 25, eps: float = 1e-6) -> np.ndarray:
    P = np.vstack([pos(G, n) for n in nodes])
    x = P.mean(axis=0)
    for _ in range(iters):
        d = np.linalg.norm(P - x, axis=1)
        if np.any(d < eps):  # already at a point
            return x
        w = 1.0 / (d + eps)
        x_new = (w[:, None] * P).sum(axis=0) / w.sum()
        if np.linalg.norm(x_new - x) < 1e-6:
            return x_new
        x = x_new
    return x

def _ccw_angle_y_up(center_rc: np.ndarray, p_rc: np.ndarray) -> float:
    """
    Return CCW angle around center for screen coords [row, col].
    Converts to (x=col, y=-row) so CCW is visual CCW.
    """
    cx, cy = center_rc[1], -center_rc[0]
    px, py = p_rc[1], -p_rc[0]
    vx, vy = (px - cx), (py - cy)
    a = np.arctan2(vy, vx)
    if a < 0:
        a += 2.0 * np.pi
    return float(a)

def analyze_subtree(
    G: nx.Graph,
    subtree: JunctionSubtree | dict | Tuple[Iterable[NodeId], Iterable[Edge]],
    subtree_index: int,
    colinear_dot: float = 0.92,
) -> SubtreeAnalysis:
    """Compute leaves, inward tangents (two-edge), and CCW order."""
    # Coerce structure
    if hasattr(subtree, "nodes") and hasattr(subtree, "edges"):
        nodes = set(subtree.nodes); edges = set(subtree.edges)
    elif isinstance(subtree, dict):
        nodes = set(subtree["nodes"]); edges = set(subtree["edges"])
    else:
        nset, eset = subtree
        nodes, edges = set(nset), set(eset)

    leaf_nodes = _subtree_leaf_nodes(nodes, edges)
    adj = _build_adj_from_edges(edges)

    leaf_tangents: Dict[NodeId, np.ndarray] = {
        leaf: _leaf_inward_tangent_two_edges(G, adj, leaf) for leaf in leaf_nodes
    }

    # Lightweight type string (used only for diagnostics/UI)
    L = len(leaf_nodes)
    if L == 2:
        jtype = "2-leaf"
    elif L == 3:
        dirs = [leaf_tangents[n] for n in leaf_nodes]
        pairs = ((0, 1), (0, 2), (1, 2))
        num_colinear = sum(1 for i, j in pairs if abs(float(np.dot(dirs[i], dirs[j]))) > colinear_dot)
        jtype = "T-junction" if num_colinear == 1 else "Y-junction"
    elif L >= 4:
        jtype = "X-junction"
    else:
        jtype = "degenerate"

    center = _center_geometric_median(G, nodes)
    leaf_angles = {leaf: _ccw_angle_y_up(center, pos(G, leaf)) for leaf in leaf_nodes}
    leaf_order = sorted(leaf_nodes, key=lambda n: leaf_angles[n])

    return SubtreeAnalysis(
        subtree_index=subtree_index,
        nodes=nodes, edges=edges,
        leaf_nodes=leaf_nodes,
        leaf_tangents=leaf_tangents,
        leaf_count=L,
        jtype=jtype,
        ccw_center_rc=center,
        leaf_order_ccw=leaf_order,
        leaf_angles=leaf_angles,
    )

def analyze_subtrees(G: nx.Graph, subtrees: List[JunctionSubtree], colinear_dot: float = 0.92) -> List[SubtreeAnalysis]:
    return [analyze_subtree(G, st, i, colinear_dot=colinear_dot) for i, st in enumerate(subtrees)]


# =============================================================================
# 4) Rewiring framework
#     Rule:
#       a) Connect all disjoint colinear pairs (|dot| >= colinear_dot).
#       b) If none exist, compute center from average ray–ray intersections (snap if outside hull),
#          and connect all leaves to center.
#       c) For remaining leaves (unmatched), project as rays and attach to the closest hit on any paired segment,
#          splitting the segment at intersection. If no hit, snap to endpoint closest to the line–ray
#          intersection point on that segment's infinite line; if no forward line hit, snap to endpoint with
#          smallest distance to the ray.
# =============================================================================

_RAY_INTERSECT_EPS = 1e-6

def colinear_pair_matching(
    leaves: List[NodeId],
    tangents: Dict[NodeId, np.ndarray],
    colinear_dot: float,
) -> List[Tuple[NodeId, NodeId]]:
    """
    Build disjoint pairs of leaves whose |dot| >= colinear_dot, maximizing total |dot|.
    Implemented via max weight matching on a small graph.
    """
    if len(leaves) < 2:
        return []
    H = nx.Graph()
    H.add_nodes_from(leaves)

    # Build candidate edges
    for i, u in enumerate(leaves):
        tu = unit(tangents[u])
        for v in leaves[i + 1:]:
            tv = unit(tangents[v])
            score = abs(float(np.dot(tu, tv)))
            if score >= float(colinear_dot):
                H.add_edge(u, v, weight=score)

    if H.number_of_edges() == 0:
        return []

    matching = nx.algorithms.matching.max_weight_matching(H, maxcardinality=True, weight="weight")
    return [(u, v) for (u, v) in matching]

def split_segment_edge(
    G: nx.Graph,
    u: NodeId, v: NodeId,
    splits: List[Tuple[float, np.ndarray]],
) -> List[NodeId]:
    """
    Split straight edge (u,v) at each (s, point) (0..1 along segment).
    Returns new node ids in ascending s. Replaces (u,v) by a chain.
    """
    if not splits:
        return []
    splits_sorted = sorted(splits, key=lambda x: x[0])

    if G.has_edge(u, v):
        G.remove_edge(u, v)

    created: List[NodeId] = []
    last = u
    for s, pt in splits_sorted:
        m = new_node_id(G)
        G.add_node(m)
        set_pos(G, m, pt)
        G.add_edge(last, m)
        created.append(m)
        last = m

    G.add_edge(last, v)
    return created

def rewire_no_colinear(
    G: nx.Graph,
    analysis: SubtreeAnalysis,
    colinear_dot: float = 0.92
) -> Optional[Dict[str, Any]]:
    """
    No colinear pairs → center is average of all ray–ray intersections.
    If center lies outside convex hull of leaf nodes, snap to closest leaf.
    Connect all leaves to the center node.
    """
    L = analysis.leaf_count
    if L < 2:
        return None

    leaves = list(analysis.leaf_nodes)
    tang = analysis.leaf_tangents

    # Verify: no pair is colinear (at threshold)
    for i in range(L):
        ti = unit(tang[leaves[i]])
        for j in range(i + 1, L):
            tj = unit(tang[leaves[j]])
            if abs(float(np.dot(ti, tj))) >= float(colinear_dot):
                return None

    # Collect ray–ray intersections
    pts = []
    for i in range(L):
        p = pos(G, leaves[i]); ui = unit(tang[leaves[i]])
        for j in range(i + 1, L):
            q = pos(G, leaves[j]); uj = unit(tang[leaves[j]])
            inter = ray_line_intersection_params(q, q + uj, p, ui)  # ray i with line j
            if inter is None:
                continue
            # Require forward on both? We follow prior behavior: forward on i; j is a line.
            t_i, _, X = inter
            if t_i >= 0.0 and np.all(np.isfinite(X)):
                pts.append(X)

    if not pts:  # rare; fallback to mean of leaf positions
        pts = [np.vstack([pos(G, n) for n in leaves]).mean(axis=0)]

    center = np.vstack(pts).mean(axis=0)

    # Snap center to convex hull if necessary
    leaf_pts = np.vstack([pos(G, n) for n in leaves])
    hull = convex_hull(leaf_pts)
    if not point_in_convex(center, hull):
        d = np.linalg.norm(leaf_pts - center[None, :], axis=1)
        center = leaf_pts[int(np.argmin(d))].copy()

    # Remove original subtree edges
    removed = []
    for (u, v) in list(analysis.edges):
        if G.has_edge(u, v):
            G.remove_edge(u, v)
            removed.append((u, v))

    # Add center node and connect all
    c = new_node_id(G)
    G.add_node(c)
    set_pos(G, c, center)

    added = []
    for n in leaves:
        G.add_edge(n, c)
        added.append((n, c))

    return {
        "type": "NoColinear-rewire",
        "subtree_index": analysis.subtree_index,
        "center_node": c,
        "center_position": center,
        "removed_edges": removed,
        "added_edges": added,
        "leaf_count": L,
    }

def rewire_general(
    G: nx.Graph,
    analysis: SubtreeAnalysis,
    colinear_dot: float,
) -> Optional[Dict[str, Any]]:
    """
    General rewiring rule (connect pairs; project unmatched; split or snap):
      1) Remove original subtree edges.
      2) If there are colinear pairs (|dot| >= colinear_dot), connect each pair by an edge.
      3) For unmatched leaves, cast inward tangent as a ray. If it hits any pair segment:
         attach to the nearest valid hit (min t) and split that segment at the hit point.
      4) If no hit:
         - choose the pair segment whose *line* intersects the ray with smallest forward t,
           snap to the endpoint closest to that line intersection point;
         - if no forward line intersects, snap to endpoint with smallest distance to the ray.
      5) If there were *no* colinear pairs at all, use the "no-colinear" rule instead.
    """
    leaves = list(analysis.leaf_nodes)
    if len(leaves) < 2:
        return None

    # Remove original edges in this subtree (we rebuild the junction locally)
    removed = []
    for (u, v) in list(analysis.edges):
        if G.has_edge(u, v):
            G.remove_edge(u, v)
            removed.append((u, v))

    # 1) Pairing step
    pairs = colinear_pair_matching(leaves, analysis.leaf_tangents, colinear_dot=colinear_dot)
    if not pairs:
        # Fallback: no pairs → center-based rewiring
        rep = rewire_no_colinear(G, analysis, colinear_dot=colinear_dot)
        if rep is not None:
            rep["removed_edges_original"] = removed
            rep["paired_mode"] = False
        return rep

    # Add edges for each pair
    added_pair_edges: List[Edge] = []
    for (u, v) in pairs:
        if not G.has_edge(u, v):
            G.add_edge(u, v)
        added_pair_edges.append((u, v))

    matched = {n for uv in pairs for n in uv}
    unmatched = [n for n in leaves if n not in matched]

    # Collect current pair segments in canonical orientation
    pair_segments = {(min(u, v), max(u, v)) for (u, v) in added_pair_edges}

    # Unmatched leaves: intersect rays with pair segments
    splits_by_segment: Dict[Edge, List[Tuple[float, np.ndarray]]] = {}
    leaf_to_attach: Dict[NodeId, Tuple[Edge, float, np.ndarray]] = {}
    snapped_leaf_edges: List[Edge] = []

    for leaf in unmatched:
        p = pos(G, leaf)
        uvec = unit(analysis.leaf_tangents[leaf])

        # Try true segment intersections; keep closest (min t)
        best_t, best_seg, best_s, best_point = float("inf"), None, None, None

        for a, b in pair_segments:
            pa, pb = pos(G, a), pos(G, b)
            res = ray_segment_intersection_params(pa, pb, p, uvec, eps=_RAY_INTERSECT_EPS)
            if res is None:
                continue
            t, s, X = res
            if t < best_t:
                best_t, best_seg, best_s, best_point = t, (a, b), s, X

        if best_seg is not None and best_t < float("inf"):
            splits_by_segment.setdefault(best_seg, []).append((best_s, best_point))
            leaf_to_attach[leaf] = (best_seg, best_s, best_point)
            continue

        # No segment hit → choose line with smallest forward t
        best_t_line = float("inf")
        best_line_seg = None
        best_X_line = None

        for a, b in pair_segments:
            pa, pb = pos(G, a), pos(G, b)
            res_line = ray_line_intersection_params(pa, pb, p, uvec, eps=_RAY_INTERSECT_EPS)
            if res_line is None:
                continue
            t_line, s_line, X_line = res_line
            if t_line < best_t_line:
                best_t_line = t_line
                best_line_seg = (a, b)
                best_X_line = X_line

        if best_line_seg is not None:
            # Snap to the endpoint *closest to the line–ray intersection point*
            a, b = best_line_seg
            pa, pb = pos(G, a), pos(G, b)
            target = a if np.linalg.norm(pa - best_X_line) <= np.linalg.norm(pb - best_X_line) else b
            G.add_edge(leaf, target)
            snapped_leaf_edges.append((leaf, target))
            continue

        # Still nothing: snap to endpoint with smallest distance to the ray
        best_dist, best_endpoint = float("inf"), None
        for a, b in pair_segments:
            pa, pb = pos(G, a), pos(G, b)
            _, dA = point_to_ray_distance(p, uvec, pa)
            _, dB = point_to_ray_distance(p, uvec, pb)
            if dA < best_dist:
                best_dist, best_endpoint = dA, a
            if dB < best_dist:
                best_dist, best_endpoint = dB, b

        if best_endpoint is not None:
            G.add_edge(leaf, best_endpoint)
            snapped_leaf_edges.append((leaf, best_endpoint))
        # else: pathological — leave unattached

    # Apply splits, creating nodes at intersection points
    split_node_at: Dict[Tuple[Edge, float], NodeId] = {}
    for seg, splits in splits_by_segment.items():
        a, b = seg
        created = split_segment_edge(G, a, b, splits)
        for (s, _pt), nid in zip(sorted(splits, key=lambda x: x[0]), created):
            split_node_at[(seg, s)] = nid

    # Connect leaves to the split nodes
    added_leaf_edges: List[Edge] = []
    for leaf, (seg, s, pt) in leaf_to_attach.items():
        nid = split_node_at.get((seg, s))
        if nid is None:  # numeric fallback
            nid = new_node_id(G)
            G.add_node(nid)
            set_pos(G, nid, pt)
        G.add_edge(leaf, nid)
        added_leaf_edges.append((leaf, nid))

    return {
        "type": "General-rewire",
        "subtree_index": analysis.subtree_index,
        "removed_edges_original": removed,
        "paired_mode": True,
        "pairs": list(pairs),
        "unmatched_leaves": unmatched,
        "pair_segments": list(pair_segments),
        "splits_applied": {
            tuple(map(int, seg)): [float(s) for (s, _pt) in sorted(splits_by_segment.get(seg, []), key=lambda x: x[0])]
            for seg in splits_by_segment.keys()
        },
        "added_leaf_edges": added_leaf_edges,
        "snapped_leaf_edges": snapped_leaf_edges,
    }


# =============================================================================
# 5) Orchestrator
# =============================================================================

def process_junctions(
    G: nx.Graph,
    angle_thresh: float = 0.35,
    merge_radius: float = 6.0,
    colinear_dot: float = 0.92,   # single knob used everywhere
    weight: str = "euclidean",
) -> Dict[str, Any]:
    """
    Full pipeline (mutates G in-place):

      1) Grow from all degree>=3 centers with global edge ownership,
         stopping on 'object angle' threshold.
      2) Cluster nearby centers by single-link (<= merge_radius) or
         shared nodes, then bridge the cluster into a connected subtree.
      3) Analyze each subtree (leaves, two-edge inward tangents, CCW order).
      4) Rewire:
         - If ≥1 colinear pair: connect disjoint pairs by straight edges.
           For remaining leaves, project rays; attach to the closest hit on
           any pair segment (splitting as needed). No hit → snap per rules.
         - Else: average ray–ray intersections, snap to hull if needed,
           and connect all leaves to that center.

    Returns a report dictionary with intermediate artifacts and rewiring actions.
    """
    # 1) growth
    grown = detect_grown_junctions(G, angle_thresh)

    # 2) clustering & bridging
    subtrees = cluster_and_merge(G, grown, merge_radius, weight=weight)

    # 3) analysis
    analyses = analyze_subtrees(G, subtrees, colinear_dot=colinear_dot)

    # 4) rewiring
    rewires: List[dict] = []
    skipped: List[Tuple[int, str]] = []

    for a in analyses:
        rep = rewire_general(G, a, colinear_dot=colinear_dot)
        if rep is not None:
            rewires.append(rep)
        else:
            skipped.append((a.subtree_index, "no-rewire-rule"))

    return G, {
        "grown": grown,
        "subtrees": subtrees,
        "analyses": analyses,
        "rewires": rewires,
        "skipped": skipped,
    }
