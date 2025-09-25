from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np

from junction_utils import (
    build_branch_paths_for_root,
    cluster_centers_within_radius,
    grow_junction_forests,
    ordered_edge,
    ray_segment_intersection,
    shortest_path_nodes,
    unit_vector,
)

NodeId = int
Edge = Tuple[NodeId, NodeId]


@dataclass(frozen=True)
class JunctionBranch:
    nodes: Tuple[NodeId, ...]
    edges: Tuple[Edge, ...]


@dataclass(frozen=True)
class Junction:
    root: NodeId
    branches: Tuple[JunctionBranch, ...]


@dataclass(frozen=True)
class JunctionTree:
    centers: Tuple[NodeId, ...]
    nodes: frozenset[NodeId]
    edges: frozenset[Edge]
    leaves: Tuple[NodeId, ...]

@dataclass(frozen=True)
class JunctionRewire:
    tree_centers: Tuple[NodeId, ...]
    colinear_pairs: Tuple[Edge, ...]
    intersection_pairs: Tuple[Edge, ...]


def _leaf_neighbour_map(edges: Iterable[Edge]) -> dict[NodeId, NodeId]:
    adj: dict[NodeId, set[NodeId]] = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return {
        node: next(iter(neighbours))
        for node, neighbours in adj.items()
        if len(neighbours) == 1
    }


def _rewire_tree(
    graph: nx.Graph,
    tree: JunctionTree,
    positions: dict[NodeId, np.ndarray],
    colinear_dot: float,
) -> Optional[JunctionRewire]:
    if not tree.edges:
        return None

    original_edges = {ordered_edge(u, v) for u, v in tree.edges}
    edges = set(original_edges)
    leaf_to_neighbour = _leaf_neighbour_map(edges)
    if not leaf_to_neighbour:
        return None

    def get_position(node: NodeId) -> Optional[np.ndarray]:
        if node in positions:
            return positions[node]
        pos = graph.nodes.get(node, {}).get("position")
        if pos is None:
            return None
        arr = np.asarray(pos, dtype=float)
        positions[node] = arr
        return arr

    leaf_dirs: dict[NodeId, np.ndarray] = {}
    for leaf, neighbour in leaf_to_neighbour.items():
        p_leaf = get_position(leaf)
        p_neighbour = get_position(neighbour)
        if p_leaf is None or p_neighbour is None:
            continue
        direction = unit_vector(p_leaf - p_neighbour)
        if np.linalg.norm(direction) == 0.0:
            continue
        leaf_dirs[leaf] = direction

    available = set(leaf_dirs.keys())
    colinear_pairs: List[Edge] = []
    unmatched: List[NodeId] = []

    while available:
        leaf = available.pop()
        dir_leaf = leaf_dirs[leaf]
        best_leaf = None
        best_score = colinear_dot
        for other in available:
            score = abs(float(np.dot(dir_leaf, leaf_dirs[other])))
            if score >= best_score:
                best_score = score
                best_leaf = other
        if best_leaf is not None:
            available.remove(best_leaf)
            pair = ordered_edge(leaf, best_leaf)
            if not graph.has_edge(*pair):
                graph.add_edge(*pair)
                graph.edges[pair[0], pair[1]]["junction_rewire"] = "colinear"
            edges.add(pair)
            colinear_pairs.append(pair)
        else:
            unmatched.append(leaf)

    intersection_pairs: List[Edge] = []
    existing_edges_list = list(edges)

    for leaf in unmatched:
        neighbour = leaf_to_neighbour.get(leaf)
        p_leaf = get_position(leaf)
        p_neighbour = get_position(neighbour) if neighbour is not None else None
        if neighbour is None or p_leaf is None or p_neighbour is None:
            continue
        direction = unit_vector(p_leaf - p_neighbour)
        if np.linalg.norm(direction) == 0.0:
            continue

        best_hit: Optional[tuple[float, NodeId, NodeId, np.ndarray]] = None
        for u, v in existing_edges_list:
            if leaf in (u, v):
                continue
            p_u = get_position(u)
            p_v = get_position(v)
            if p_u is None or p_v is None:
                continue
            hit = ray_segment_intersection(p_leaf, direction, p_u, p_v)
            if hit is None:
                continue
            t, _, point = hit
            if t <= 1e-6:
                continue
            if best_hit is None or t < best_hit[0]:
                best_hit = (t, u, v, point)

        if best_hit is None:
            continue

        _, u, v, point = best_hit
        p_u = get_position(u)
        p_v = get_position(v)
        if p_u is None or p_v is None:
            continue
        target = u if np.linalg.norm(point - p_u) <= np.linalg.norm(point - p_v) else v
        pair = ordered_edge(leaf, target)
        if pair in edges:
            continue
        if not graph.has_edge(*pair):
            graph.add_edge(*pair)
            graph.edges[pair[0], pair[1]]["junction_rewire"] = "intersection"
        edges.add(pair)
        existing_edges_list.append(pair)
        intersection_pairs.append(pair)

    new_edges = set(colinear_pairs) | set(intersection_pairs)

    if not new_edges:
        return None

    for edge in original_edges:
        if graph.has_edge(*edge):
            graph.remove_edge(*edge)

    return JunctionRewire(
        tree_centers=tree.centers,
        colinear_pairs=tuple(colinear_pairs),
        intersection_pairs=tuple(intersection_pairs),
    )


def rewire_junction_trees(
    graph: nx.Graph,
    trees: Tuple[JunctionTree, ...],
    colinear_dot: float = 0.92,
) -> Tuple[nx.Graph, Tuple[JunctionRewire, ...]]:
    if not trees:
        return graph.copy(), tuple()

    colinear_dot = max(min(float(colinear_dot), 1.0), -1.0)
    rewired_graph = graph.copy()
    positions: dict[NodeId, np.ndarray] = {}
    reports: List[JunctionRewire] = []

    for tree in trees:
        report = _rewire_tree(rewired_graph, tree, positions, colinear_dot)
        if report is not None:
            reports.append(report)

    return rewired_graph, tuple(reports)


def identify_junctions(
    pruned_graph: nx.Graph,
    angle_threshold: float,
) -> Tuple[Junction, ...]:
    """Return a junction descriptor for every degree>=3 node in the pruned graph."""

    if pruned_graph.number_of_nodes() == 0:
        return tuple()

    # Collect centers (degree >= 3). Sorting keeps the output stable.
    ordered_roots = tuple(sorted(
        n for n in pruned_graph.nodes if pruned_graph.degree[n] >= 3
    ))
    if not ordered_roots:
        return tuple()

    # Grow per-root forests once, capturing parent/children relationships
    # and which nodes ended up owned by each center.
    parent_of, children_of, root_nodes = grow_junction_forests(
        pruned_graph,
        ordered_roots,
        angle_threshold,
    )

    junctions: List[Junction] = []
    for root in ordered_roots:
        owned_nodes = root_nodes.get(root, {root})
        # Recover all root-to-leaf paths for this forest and ensure every
        # incident edge at the root appears in the result exactly once.
        branch_paths = build_branch_paths_for_root(
            pruned_graph,
            root,
            parent_of,
            children_of,
            owned_nodes,
        )

        if not branch_paths:
            continue

        junctions.append(
            Junction(
                root=root,
                branches=tuple(
                    JunctionBranch(nodes=nodes, edges=edges)
                    for nodes, edges in branch_paths
                ),
            )
        )

    return tuple(junctions)

def build_junction_trees(
    graph: nx.Graph,
    junctions: Tuple[Junction, ...],
    radius: float,
) -> Tuple[JunctionTree, ...]:
    if not junctions:
        return tuple()

    radius = max(float(radius), 0.0)
    positions: dict[NodeId, object] = {}

    root_to_junction = {junction.root: junction for junction in junctions}
    centers = tuple(root_to_junction.keys())
    clusters = cluster_centers_within_radius(graph, centers, positions, radius)

    trees: List[JunctionTree] = []
    for cluster in clusters:
        junction_members = [root_to_junction[c] for c in cluster]
        nodes: set[NodeId] = set()
        edges: set[Edge] = set()

        for junction in junction_members:
            for branch in junction.branches:
                nodes.update(branch.nodes)
                branch_nodes = branch.nodes
                edges.update(
                    ordered_edge(branch_nodes[i], branch_nodes[i + 1])
                    for i in range(len(branch_nodes) - 1)
                )

        anchor = cluster[0]
        for other in cluster[1:]:
            path = shortest_path_nodes(graph, anchor, other, positions)
            if path is None or len(path) < 2:
                continue
            nodes.update(path)
            edges.update(
                ordered_edge(u, v) for u, v in zip(path[:-1], path[1:])
            )

        if not edges:
            continue

        degrees: dict[NodeId, int] = {}
        for u, v in edges:
            degrees[u] = degrees.get(u, 0) + 1
            degrees[v] = degrees.get(v, 0) + 1

        leaves = tuple(sorted(n for n, deg in degrees.items() if deg == 1))

        trees.append(
            JunctionTree(
                centers=cluster,
                nodes=frozenset(nodes),
                edges=frozenset(edges),
                leaves=leaves,
            )
        )

    return tuple(trees)
