from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Iterable
import numpy as np
import networkx as nx

NodeId = int
Edge   = Tuple[NodeId, NodeId]


@dataclass(frozen=True)
class SubtreeAnalysis:
    subtree_index: int
    nodes: Set[NodeId]
    edges: Set[Tuple[NodeId, NodeId]]
    leaf_nodes: List[NodeId]
    leaf_tangents: Dict[NodeId, np.ndarray]   # unit vectors pointing INTO the subtree
    leaf_count: int
    jtype: str                                 # 'T-junction' | 'Y-junction' | 'X-junction' | '2-leaf' | 'degenerate'
    ccw_center_rc: np.ndarray                  # chosen center [row, col] used for CCW ordering
    leaf_order_ccw: List[NodeId]               # leaves ordered counterclockwise around ccw_center_rc
    leaf_angles: Dict[NodeId, float]           # angle in [0, 2π) for each leaf, true-CCW (y-up convention)


def _pos(G: nx.Graph, n: int) -> np.ndarray:
    """Return node position as [row, col] float array."""
    return np.asarray(G.nodes[n]['position'], dtype=float)


def _build_adj_from_edges(edges: Iterable[Edge]) -> Dict[NodeId, Set[NodeId]]:
    adj: Dict[NodeId, Set[NodeId]] = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _subtree_internal_degrees(edges: Iterable[Edge]) -> Dict[NodeId, int]:
    deg: Dict[NodeId, int] = {}
    for u, v in edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    return deg


def _subtree_leaf_nodes(nodes: Iterable[NodeId], edges: Iterable[Edge]) -> List[NodeId]:
    nodes = list(nodes)
    deg = _subtree_internal_degrees(edges)
    return [n for n in nodes if deg.get(n, 0) == 1]


def _center_centroid(G: nx.Graph, nodes: Iterable[NodeId]) -> np.ndarray:
    P = np.vstack([_pos(G, n) for n in nodes])
    return P.mean(axis=0)

def _pick_forward_neighbor(
    G: nx.Graph,
    adj: Dict[NodeId, Set[NodeId]],
    leaf: NodeId,
    nb: NodeId
) -> NodeId | None:
    """
    Among nb's neighbors inside the subtree (adj), excluding the leaf,
    pick a forward node. If exactly one exists, take it.
    If multiple, pick the one whose direction from nb is best aligned with (nb - leaf).
    """
    candidates = [x for x in adj.get(nb, ()) if x != leaf]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    v1 = _unit(_pos(G, nb) - _pos(G, leaf))  # direction of the first edge
    best = None
    best_dot = -1.0
    for c in candidates:
        vc = _unit(_pos(G, c) - _pos(G, nb))
        d = float(np.dot(v1, vc))
        if d > best_dot:
            best_dot = d
            best = c
    return best


def _leaf_inward_tangent_two_edge(
    G: nx.Graph,
    adj: Dict[NodeId, Set[NodeId]],
    leaf: NodeId
) -> np.ndarray:
    """
    Prefer a two-edge baseline (forward neighbor if available): tangent ~ (forward - leaf).
    Fallback to one-edge (neighbor - leaf).
    Always points inward (from leaf toward the subtree).
    """
    # unique neighbor within the *subtree* (degree 1 by definition of leaf)
    nb = next(iter(adj[leaf]))
    p_leaf = _pos(G, leaf)
    p_nb   = _pos(G, nb)

    fwd = _pick_forward_neighbor(G, adj, leaf, nb)
    if fwd is not None:
        p_fwd = _pos(G, fwd)
        t = p_fwd - p_leaf   # two-edge baseline
    else:
        t = p_nb - p_leaf    # fallback: one-edge

    t = _unit(t)
    return t



def _center_geometric_median(G: nx.Graph, nodes: Iterable[NodeId], iters: int = 25, eps: float = 1e-6) -> np.ndarray:
    """
    Weiszfeld iterations for a robust center; falls back to centroid if ill-conditioned.
    """
    P = np.vstack([_pos(G, n) for n in nodes])
    x = P.mean(axis=0)
    for _ in range(iters):
        d = np.linalg.norm(P - x, axis=1)
        if np.any(d < eps):
            return x
        w = 1.0 / (d + eps)
        x_new = (w[:, None] * P).sum(axis=0) / w.sum()
        if np.linalg.norm(x_new - x) < 1e-6:
            x = x_new
            break
        x = x_new
    return x


def _ccw_angle_from_center(G: nx.Graph, center_rc: np.ndarray, leaf: NodeId) -> float:
    """
    True geometric CCW angle in [0, 2π) around 'center_rc'.
    Convert [row, col] to (x, y) with y-up by using (x=col, y=-row).
    """
    leaf_rc = _pos(G, leaf)
    cx, cy = center_rc[1], -center_rc[0]
    lx, ly = leaf_rc[1], -leaf_rc[0]
    vx, vy = (lx - cx), (ly - cy)
    ang = np.arctan2(vy, vx)  # [-π, π]
    if ang < 0:
        ang += 2.0 * np.pi
    return float(ang)


def analyze_subtree(
    G: nx.Graph,
    subtree,
    subtree_index: int,
    colinear_dot: float = 0.95,
    center_mode: str = "median"  # 'median' or 'centroid'
) -> SubtreeAnalysis:
    """
    Compute leaf set, inward tangents, a lightweight junction type,
    and the counterclockwise order of leaves (with angles) for one subtree.

    'subtree' may be:
      - an object with .nodes and .edges,
      - a dict {'nodes': ..., 'edges': ...},
      - a (nodes, edges) tuple.
    """
    # coerce
    if hasattr(subtree, 'nodes') and hasattr(subtree, 'edges'):
        nodes = set(subtree.nodes)
        edges = set(subtree.edges)
    elif isinstance(subtree, dict):
        nodes = set(subtree['nodes'])
        edges = set(subtree['edges'])
    else:
        nset, eset = subtree
        nodes, edges = set(nset), set(eset)

    adj = _build_adj_from_edges(edges)

    # leaves are degree-1 *within the subtree*
    leaf_nodes = _subtree_leaf_nodes(nodes, edges)

    # less noisy inward tangent using two edges when available
    leaf_tangents: Dict[NodeId, np.ndarray] = {}
    for leaf in leaf_nodes:
        leaf_tangents[leaf] = _leaf_inward_tangent_two_edge(G, adj, leaf)


    L = len(leaf_nodes)

    # lightweight type label based on leaf count, with T vs Y for 3-leaf using colinearity
    jtype = 'degenerate'
    if L == 2:
        jtype = '2-leaf'
    elif L == 3:
        dirs = [leaf_tangents[n] for n in leaf_nodes]
        pairs = ((0, 1), (0, 2), (1, 2))
        num_colinear = sum(1 for i, j in pairs if abs(float(np.dot(dirs[i], dirs[j]))) > colinear_dot)
        jtype = 'T-junction' if num_colinear == 1 else 'Y-junction'
    elif L >= 4:
        jtype = 'X-junction'  # catch-all for ≥4

    # choose a robust center for angular ordering
    ccw_center_rc = (
        _center_geometric_median(G, nodes)
        if center_mode == "median" else
        _center_centroid(G, nodes)
    )

    # per-leaf CCW angles and CCW order
    leaf_angles: Dict[NodeId, float] = {leaf: _ccw_angle_from_center(G, ccw_center_rc, leaf)
                                        for leaf in leaf_nodes}
    leaf_order_ccw: List[NodeId] = sorted(leaf_nodes, key=lambda n: leaf_angles[n])

    return SubtreeAnalysis(
        subtree_index=subtree_index,
        nodes=nodes,
        edges=edges,
        leaf_nodes=leaf_nodes,
        leaf_tangents=leaf_tangents,
        leaf_count=L,
        jtype=jtype,
        ccw_center_rc=ccw_center_rc,
        leaf_order_ccw=leaf_order_ccw,
        leaf_angles=leaf_angles,
    )


def analyze_subtrees(G: nx.Graph, subtrees, colinear_dot: float = 0.95, center_mode: str = "median") -> List[SubtreeAnalysis]:
    return [
        analyze_subtree(G, st, i, colinear_dot=colinear_dot, center_mode=center_mode)
        for i, st in enumerate(subtrees)
    ]

