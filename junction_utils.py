from __future__ import annotations

from collections import defaultdict, deque
from heapq import heappop, heappush
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np

NodeId = int
Edge = Tuple[NodeId, NodeId]


def ordered_edge(u: NodeId, v: NodeId) -> Edge:
    return (u, v) if u <= v else (v, u)


def edge_length(
    graph: nx.Graph,
    u: NodeId,
    v: NodeId,
    positions: dict[NodeId, np.ndarray],
) -> float:
    if u not in positions:
        positions[u] = np.asarray(graph.nodes[u]["position"], dtype=float)
    if v not in positions:
        positions[v] = np.asarray(graph.nodes[v]["position"], dtype=float)
    return float(np.linalg.norm(positions[u] - positions[v]))


def shortest_path_nodes(
    graph: nx.Graph,
    start: NodeId,
    goal: NodeId,
    positions: dict[NodeId, np.ndarray],
) -> Tuple[NodeId, ...] | None:
    if start == goal:
        return (start,)

    dist = {start: 0.0}
    parent: dict[NodeId, NodeId | None] = {start: None}
    pq: list[tuple[float, NodeId]] = [(0.0, start)]

    while pq:
        d, node = heappop(pq)
        if node == goal:
            path: list[NodeId] = []
            while node is not None:
                path.append(node)
                node = parent[node]
            return tuple(reversed(path))

        if d > dist.get(node, float("inf")):
            continue

        for nb in graph.neighbors(node):
            step = edge_length(graph, node, nb, positions)
            nd = d + step
            if nd < dist.get(nb, float("inf")):
                dist[nb] = nd
                parent[nb] = node
                heappush(pq, (nd, nb))

    return None


def unit_vector(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0.0 else v


def ray_segment_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    eps: float = 1e-8,
) -> Optional[Tuple[float, float, np.ndarray]]:
    """Intersect ray (origin + t*direction, t>=0) with segment [seg_a, seg_b]."""

    direction = np.asarray(direction, dtype=float)
    seg_a = np.asarray(seg_a, dtype=float)
    seg_b = np.asarray(seg_b, dtype=float)
    matrix = np.column_stack((direction, seg_a - seg_b))
    det = float(np.linalg.det(matrix))
    if abs(det) <= eps:
        return None

    rhs = seg_a - origin
    t, s = np.linalg.solve(matrix, rhs)
    if t < -eps or s < -eps or s > 1.0 + eps:
        return None

    t = max(0.0, float(t))
    s = min(max(float(s), 0.0), 1.0)
    point = seg_a + s * (seg_b - seg_a)
    return (t, s, point)


def cluster_centers_within_radius(
    graph: nx.Graph,
    centers: Tuple[NodeId, ...],
    positions: dict[NodeId, np.ndarray],
    radius: float,
) -> Tuple[Tuple[NodeId, ...], ...]:
    radius = max(float(radius), 0.0)
    unused = set(centers)
    clusters: List[Tuple[NodeId, ...]] = []

    def coords_for(nodes: Iterable[NodeId]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for n in nodes:
            if n not in positions:
                positions[n] = np.asarray(graph.nodes[n]["position"], dtype=float)
            out.append(positions[n])
        return out

    def fits(nodes: List[NodeId]) -> bool:
        coords = coords_for(nodes)
        for i, center in enumerate(coords):
            if all(np.linalg.norm(center - coords[j]) <= radius for j in range(len(coords))):
                return True
        return False

    while unused:
        seed = unused.pop()
        cluster = [seed]
        changed = True
        while changed:
            changed = False
            for cand in list(unused):
                tentative = cluster + [cand]
                if fits(tentative):
                    cluster.append(cand)
                    unused.remove(cand)
                    changed = True
        clusters.append(tuple(sorted(cluster)))

    return tuple(clusters)


def grow_junction_forests(
    graph: nx.Graph,
    roots: Iterable[NodeId],
    angle_threshold: float,
) -> Tuple[Dict[NodeId, Optional[NodeId]], Dict[NodeId, List[NodeId]], Dict[NodeId, set[NodeId]]]:
    """Run a multi-root BFS under the angle cutoff and report parent, child, and owned-node maps."""

    parent_of: Dict[NodeId, Optional[NodeId]] = {}
    children_of: Dict[NodeId, List[NodeId]] = defaultdict(list)
    root_nodes: Dict[NodeId, set[NodeId]] = defaultdict(set)

    node_owner: Dict[NodeId, NodeId] = {}
    queue: deque[Tuple[NodeId, NodeId]] = deque()

    for root in roots:
        node_owner[root] = root
        parent_of[root] = None
        root_nodes[root].add(root)
        queue.append((root, root))

    while queue:
        root, node = queue.popleft()
        parent_node = parent_of.get(node)

        for neighbor in graph.neighbors(node):
            if parent_node is not None and neighbor == parent_node:
                continue

            angle = float(graph.edges[node, neighbor].get("object angle", 0.0))
            if angle > angle_threshold:
                continue

            owner_neighbor = node_owner.get(neighbor)
            if owner_neighbor is None:
                node_owner[neighbor] = root
                parent_of[neighbor] = node
                children_of[node].append(neighbor)
                root_nodes[root].add(neighbor)
                queue.append((root, neighbor))
            elif owner_neighbor == root:
                continue

    return parent_of, children_of, root_nodes


def build_branch_paths_for_root(
    graph: nx.Graph,
    root: NodeId,
    parent_of: Dict[NodeId, Optional[NodeId]],
    children_of: Dict[NodeId, List[NodeId]],
    owned_nodes: Iterable[NodeId],
) -> Tuple[Tuple[Tuple[NodeId, ...], Tuple[Edge, ...]], ...]:
    """Return root-to-leaf paths plus single-hop fallbacks for every root incident edge."""

    leaves = [n for n in owned_nodes if n != root and not children_of[n]]

    branches: List[Tuple[Tuple[NodeId, ...], Tuple[Edge, ...]]] = []
    for leaf in leaves:
        path_nodes = [leaf]
        current = leaf

        while current != root:
            parent_node = parent_of.get(current)
            if parent_node is None:
                path_nodes = []
                break
            path_nodes.append(parent_node)
            current = parent_node

        if not path_nodes or path_nodes[-1] != root:
            continue

        path_nodes.reverse()
        path_edges = [
            (path_nodes[i], path_nodes[i + 1])
            for i in range(len(path_nodes) - 1)
        ]

        if path_edges:
            branches.append((tuple(path_nodes), tuple(path_edges)))

    first_step_nodes = {
        branch[0][1]
        for branch in branches
        if len(branch[0]) > 1
    }

    for neighbor in graph.neighbors(root):
        if neighbor in first_step_nodes:
            continue
        branches.append(((root, neighbor), ((root, neighbor),)))

    return tuple(branches)
