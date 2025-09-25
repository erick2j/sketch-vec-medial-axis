from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np

from junction_utils import (
    build_branch_paths_for_root,
    cluster_centers_within_radius,
    grow_junction_forests,
    ordered_edge,
    shortest_path_nodes,
    unit_vector,
)

NodeId = int
Edge = Tuple[NodeId, NodeId]

_INTERSECT_EPS = 1e-6


def new_node_id(graph: nx.Graph) -> NodeId:
    return (max(graph.nodes) + 1) if graph.number_of_nodes() else 0


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


@dataclass(frozen=True)
class Ray:
    origin: np.ndarray
    direction: np.ndarray


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


def leaf_tangent_rays(
    graph: nx.Graph,
    tree: JunctionTree,
    positions: Optional[Dict[NodeId, np.ndarray]] = None,
) -> Dict[NodeId, Ray]:
    """Return a ray (origin, unit direction) for every leaf in the junction tree."""

    if positions is None:
        positions = {}

    def get_position(node: NodeId) -> Optional[np.ndarray]:
        if node in positions:
            return positions[node]
        pos = graph.nodes.get(node, {}).get("position")
        if pos is None:
            return None
        arr = np.asarray(pos, dtype=float)
        positions[node] = arr
        return arr

    leaf_to_neighbour = _leaf_neighbour_map(tree.edges)
    rays: Dict[NodeId, Ray] = {}

    for leaf in tree.leaves:
        neighbour = leaf_to_neighbour.get(leaf)
        if neighbour is None:
            continue
        p_leaf = get_position(leaf)
        p_neighbour = get_position(neighbour)
        if p_leaf is None or p_neighbour is None:
            continue
        direction = unit_vector(p_neighbour - p_leaf)
        if np.linalg.norm(direction) == 0.0:
            continue
        rays[leaf] = Ray(origin=p_leaf, direction=direction)

    return rays


def branch_agreement_energy(
    leaf_a: NodeId,
    ray_a: Ray,
    leaf_b: NodeId,
    ray_b: Ray,
    alignment_threshold: float,
) -> float:
    """Return an alignment weight for two branch rays (0.0 means incompatible)."""

    #dot = float(np.clip(np.dot(ray_a.direction, ray_b.direction), -1.0, 1.0))
    ab_dir = unit_vector(ray_b.origin - ray_a.origin)
    a_alignment = np.dot(ray_a.direction, ab_dir)
    b_alignment = np.dot(ray_b.direction, -ab_dir)
    alignment = a_alignment+b_alignment
    if alignment < float(alignment_threshold):
        return 0.0
    return alignment


def colinear_leaf_pairs(
    tree: JunctionTree,
    rays: Dict[NodeId, Ray],
    alignment_threshold: float,
) -> List[Edge]:
    """Return ordered colinear pairs for the given leaf rays via max-weight matching."""

    leaves = [leaf for leaf in tree.leaves if leaf in rays]
    if len(leaves) < 2:
        return []

    H = nx.Graph()
    H.add_nodes_from(leaves)

    for i, u in enumerate(leaves):
        ray_u = rays[u]
        for v in leaves[i + 1:]:
            ray_v = rays[v]
            weight = branch_agreement_energy(u, ray_u, v, ray_v, alignment_threshold)
            if weight <= 0.0:
                continue
            H.add_edge(u, v, weight=float(weight))

    if H.number_of_edges() == 0:
        return []

    matching = nx.algorithms.matching.max_weight_matching(
        H,
        maxcardinality=True,
        weight="weight",
    )

    return [ordered_edge(u, v) for u, v in matching]


def _ray_ray_intersection(
    origin_a: np.ndarray,
    dir_a: np.ndarray,
    origin_b: np.ndarray,
    dir_b: np.ndarray,
    eps: float = 1e-9,
) -> Optional[np.ndarray]:
    """Return intersection point of two rays if it exists (t_a, t_b >= 0)."""

    dir_a = np.asarray(dir_a, dtype=float)
    dir_b = np.asarray(dir_b, dtype=float)
    origin_a = np.asarray(origin_a, dtype=float)
    origin_b = np.asarray(origin_b, dtype=float)

    if np.linalg.norm(dir_a) <= eps or np.linalg.norm(dir_b) <= eps:
        return None

    M = np.column_stack((dir_a, -dir_b))
    det = float(np.linalg.det(M))
    if abs(det) <= eps:
        return None

    rhs = origin_b - origin_a
    t_a, t_b = np.linalg.solve(M, rhs)
    if t_a < -eps or t_b < -eps:
        return None

    t_a = max(0.0, float(t_a))
    return origin_a + t_a * dir_a


def _ray_line_intersection(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    eps: float = 1e-9,
) -> Optional[Tuple[float, float, np.ndarray]]:
    ray_direction = np.asarray(ray_direction, dtype=float)
    seg_a = np.asarray(seg_a, dtype=float)
    seg_b = np.asarray(seg_b, dtype=float)

    if np.linalg.norm(ray_direction) <= eps:
        return None

    seg_dir = seg_b - seg_a
    M = np.column_stack((ray_direction, -seg_dir))
    det = float(np.linalg.det(M))
    if abs(det) <= eps:
        return None

    rhs = seg_a - ray_origin
    t, s = np.linalg.solve(M, rhs)
    if t < -eps:
        return None

    point = seg_a + float(s) * seg_dir
    return float(t), float(s), point


def _convex_hull(points: np.ndarray) -> np.ndarray:
    pts = np.unique(points, axis=0)
    if len(pts) <= 2:
        return pts
    pts_sorted = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[np.ndarray] = []
    for p in pts_sorted:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[np.ndarray] = []
    for p in reversed(pts_sorted):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.vstack((lower[:-1], upper[:-1]))


def _point_in_convex_hull(pt: np.ndarray, hull: np.ndarray, eps: float = 1e-9) -> bool:
    m = len(hull)
    if m == 0:
        return False
    if m == 1:
        return np.linalg.norm(pt - hull[0]) <= eps
    if m == 2:
        a, b = hull
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= eps:
            return np.linalg.norm(pt - a) <= eps
        t = float(np.dot(pt - a, ab) / denom)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return np.linalg.norm(pt - proj) <= eps

    sign = None
    for i in range(m):
        a = hull[i]
        b = hull[(i + 1) % m]
        edge = b - a
        normal = np.array([-edge[1], edge[0]])
        val = float(np.dot(pt - a, normal))
        current_sign = val >= -eps
        if sign is None:
            sign = current_sign
        elif current_sign != sign:
            return False
    return True


def _rewire_no_colinear_case(
    graph: nx.Graph,
    tree: JunctionTree,
    positions: Dict[NodeId, np.ndarray],
    get_position,
    rays: Dict[NodeId, Ray],
) -> Optional[JunctionRewire]:
    """Create a junction center from average ray intersections and connect all leaves."""

    leaf_positions: Dict[NodeId, np.ndarray] = {}
    for leaf in tree.leaves:
        pos = get_position(leaf)
        if pos is not None:
            leaf_positions[leaf] = pos

    if not leaf_positions:
        return None

    intersections: List[np.ndarray] = []
    ray_leaves = [leaf for leaf in tree.leaves if leaf in rays]
    for i, leaf_a in enumerate(ray_leaves):
        ray_a = rays[leaf_a]
        for leaf_b in ray_leaves[i + 1:]:
            ray_b = rays[leaf_b]
            hit = _ray_ray_intersection(ray_a.origin, ray_a.direction, ray_b.origin, ray_b.direction)
            if hit is not None:
                intersections.append(hit)

    if intersections:
        center_point = np.mean(np.vstack(intersections), axis=0)
    else:
        center_point = np.mean(np.vstack(list(leaf_positions.values())), axis=0)

    hull = _convex_hull(np.vstack(list(leaf_positions.values())))
    if not _point_in_convex_hull(center_point, hull):
        # Snap to nearest leaf position
        nearest_leaf = min(leaf_positions.items(), key=lambda item: np.linalg.norm(center_point - item[1]))
        center_point = nearest_leaf[1]

    center_id = new_node_id(graph)
    point_arr = np.asarray(center_point, dtype=float)
    graph.add_node(center_id)
    graph.nodes[center_id]["position"] = point_arr
    positions[center_id] = point_arr

    new_edges: List[Edge] = []
    for leaf in tree.leaves:
        pos = leaf_positions.get(leaf)
        if pos is None:
            continue
        if not graph.has_edge(leaf, center_id):
            graph.add_edge(leaf, center_id)
            graph.edges[leaf, center_id]["junction_rewire"] = "centroid"
        new_edges.append(ordered_edge(leaf, center_id))

    return JunctionRewire(
        tree_centers=tree.centers,
        colinear_pairs=tuple(),
        intersection_pairs=tuple(new_edges),
    )


def _rewire_all_matched_case(
    tree: JunctionTree,
    colinear_pairs: List[Edge],
) -> JunctionRewire:
    """Finalize rewiring when every available branch participated in a pair."""

    return JunctionRewire(
        tree_centers=tree.centers,
        colinear_pairs=tuple(colinear_pairs),
        intersection_pairs=tuple(),
    )


def _rewire_partial_matched_case(
    graph: nx.Graph,
    tree: JunctionTree,
    colinear_pairs: List[Edge],
    unmatched_leaves: List[NodeId],
    positions: Dict[NodeId, np.ndarray],
    rays: Dict[NodeId, Ray],
    get_position,
) -> JunctionRewire:
    """Attach unmatched leaves to the first colinear segment they intersect."""

    if not unmatched_leaves:
        return _rewire_all_matched_case(tree, colinear_pairs)

    segments: Dict[Edge, Tuple[NodeId, NodeId]] = {}
    for u, v in colinear_pairs:
        key = ordered_edge(u, v)
        segments[key] = (u, v)

    intersection_edges: List[Edge] = []

    for leaf in unmatched_leaves:
        ray = rays.get(leaf)
        if ray is None:
            continue

        best_hit: Optional[Tuple[float, Edge, Tuple[NodeId, NodeId], float, np.ndarray]] = None

        for key, (seg_u, seg_v) in list(segments.items()):
            p_u = get_position(seg_u)
            p_v = get_position(seg_v)
            if p_u is None or p_v is None:
                continue
            hit = _ray_line_intersection(ray.origin, ray.direction, p_u, p_v)
            if hit is None:
                continue
            _, s_line, _ = hit
            s_clamped = float(min(max(s_line, 0.0), 1.0))

            # probably a misunderstanding here
            seg_vec = p_v - p_u
            point_on_segment = p_u + s_clamped * seg_vec
            t_candidate = float(np.dot(point_on_segment - ray.origin, ray.direction))
            if t_candidate <= _INTERSECT_EPS:
                continue

            if best_hit is None or t_candidate < best_hit[0]:
                best_hit = (t_candidate, key, (seg_u, seg_v), s_clamped, point_on_segment)

        if best_hit is None:
            continue

        _, key, (seg_u, seg_v), s_hit, point_on_segment = best_hit
        p_u = get_position(seg_u)
        p_v = get_position(seg_v)
        if p_u is None or p_v is None:
            continue

        point_arr = np.asarray(point_on_segment, dtype=float)

        if s_hit <= _INTERSECT_EPS:
            target = seg_u
        elif s_hit >= 1.0 - _INTERSECT_EPS:
            target = seg_v
        else:
            if graph.has_edge(seg_u, seg_v):
                graph.remove_edge(seg_u, seg_v)

            new_id = new_node_id(graph)
            graph.add_node(new_id)
            graph.nodes[new_id]["position"] = point_arr
            positions[new_id] = point_arr

            graph.add_edge(seg_u, new_id)
            graph.edges[seg_u, new_id]["junction_rewire"] = "colinear_split"
            graph.add_edge(new_id, seg_v)
            graph.edges[new_id, seg_v]["junction_rewire"] = "colinear_split"

            segments.pop(key, None)
            segments[ordered_edge(seg_u, new_id)] = (seg_u, new_id)
            segments[ordered_edge(new_id, seg_v)] = (new_id, seg_v)

            target = new_id

        if not graph.has_edge(leaf, target):
            graph.add_edge(leaf, target)
            graph.edges[leaf, target]["junction_rewire"] = "ray_attach"

        intersection_edges.append(ordered_edge(leaf, target))

    return JunctionRewire(
        tree_centers=tree.centers,
        colinear_pairs=tuple(colinear_pairs),
        intersection_pairs=tuple(intersection_edges),
    )


def _rewire_tree(
    graph: nx.Graph,
    tree: JunctionTree,
    positions: dict[NodeId, np.ndarray],
    alignment_threshold: float,
) -> Optional[JunctionRewire]:
    if not tree.edges:
        return None

    original_edges = {ordered_edge(u, v) for u, v in tree.edges}
    for u, v in original_edges:
        if graph.has_edge(u, v):
            graph.remove_edge(u, v)

    def get_position(node: NodeId) -> Optional[np.ndarray]:
        if node in positions:
            return positions[node]
        pos = graph.nodes.get(node, {}).get("position")
        if pos is None:
            return None
        arr = np.asarray(pos, dtype=float)
        positions[node] = arr
        return arr

    rays = leaf_tangent_rays(graph, tree, positions)
    colinear_pairs = colinear_leaf_pairs(tree, rays, alignment_threshold)

    if not colinear_pairs:
        return _rewire_no_colinear_case(graph, tree, positions, get_position, rays)

    for u, v in colinear_pairs:
        if not graph.has_edge(u, v):
            graph.add_edge(u, v)
            graph.edges[u, v]["junction_rewire"] = "colinear"

    matched_nodes = {node for edge in colinear_pairs for node in edge}
    available_leaves = [leaf for leaf in tree.leaves if leaf in rays]
    unmatched_with_rays = [leaf for leaf in available_leaves if leaf not in matched_nodes]

    if available_leaves and not unmatched_with_rays:
        return _rewire_all_matched_case(tree, colinear_pairs)

    return _rewire_partial_matched_case(
        graph,
        tree,
        colinear_pairs,
        unmatched_with_rays,
        positions,
        rays,
        get_position,
    )


def rewire_junction_trees(
    graph: nx.Graph,
    trees: Tuple[JunctionTree, ...],
    alignment_threshold: float = 1.5,
) -> Tuple[nx.Graph, Tuple[JunctionRewire, ...]]:
    if not trees:
        return graph.copy(), tuple()

    rewired_graph = graph.copy()
    positions: dict[NodeId, np.ndarray] = {}
    reports: List[JunctionRewire] = []

    for tree in trees:
        report = _rewire_tree(rewired_graph, tree, positions, float(alignment_threshold))
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
