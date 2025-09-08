from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Literal

import numpy as np
import networkx as nx
from shapely.geometry import LineString

# ---------- Types ----------

NodeId = int
Edge   = Tuple[int, int]
BranchPath = List[NodeId]

JunctionMap  = Dict[int, "Junction"]
JunctionType = Literal["T-junction", "Y-junction", "X-junction", "unknown"]


# ---------- Dataclasses ----------

@dataclass
class BranchRef:
    """A branch grown from a junction center: center->...->endpoint"""
    path_nodes: BranchPath
    endpoint: NodeId

@dataclass
class Junction:
    """Explicit junction object that references graph primitives"""
    jid: int
    center_node: NodeId
    type: JunctionType
    branches: List[BranchRef] = field(default_factory=list)


# ---------- Small helpers ----------

def _pos(graph: nx.Graph, n: NodeId) -> np.ndarray:
    return np.asarray(graph.nodes[int(n)]['position'], dtype=float)

def _set_pos(graph: nx.Graph, n: NodeId, p: np.ndarray) -> None:
    graph.nodes[int(n)]['position'] = np.asarray(p, dtype=float)

def _endpoint_pos(graph: nx.Graph, br: BranchRef) -> np.ndarray:
    return _pos(graph, br.endpoint)

def _ordered_edge(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)

def _is_junction(graph: nx.Graph, n: NodeId) -> bool:
    return graph.degree[int(n)] >= 3

def unique_next_neighbor(graph: nx.Graph, prev: int, curr: int) -> int | None:
    forward = [n for n in graph.neighbors(curr) if n != prev]
    return forward[0] if len(forward) == 1 else None

def _unit(vec: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(vec))
    return vec / n if n > 0 else None

def extended_line(p: np.ndarray, d: np.ndarray, extension: float = 10.0) -> LineString:
    """Build a long segment through p along direction d; robust to degenerate d."""
    p = np.asarray(p, float)
    d = np.asarray(d, float)
    u = _unit(d)
    if u is None:
        return LineString([p, p])
    return LineString([p - u * extension, p + u * extension])

def _line_intersection_point(L1: LineString, L2: LineString) -> Optional[np.ndarray]:
    inter = L1.intersection(L2)
    if inter.is_empty or inter.geom_type != 'Point':
        return None
    return np.asarray(inter.coords[0], dtype=float)

def _remove_branch_from_graph(graph: nx.Graph, br: Branch) -> None:
    """Remove all edges along a branch path from the graph."""
    nodes = br.path_nodes
    for u, v in zip(nodes[:-1], nodes[1:]):
        if graph.has_edge(u, v):
            graph.remove_edge(u, v)
    # optional: remove nodes that become isolated
    for n in nodes[1:-1]:  # don’t delete endpoints or center
        if graph.degree[n] == 0:
            graph.remove_node(n)



# ---------- Seeding ----------

def initialize_junctions(graph: nx.Graph) -> JunctionMap:
    """
    For every degree-3 node, create a Junction with 3 initial branches [node, neighbor].
    Type is set to 'unknown' until classification.
    """
    junctions: JunctionMap = {}
    next_id = 0
    for n in graph.nodes:
        if graph.degree[n] != 3:
            continue
        branches = [BranchRef(path_nodes=[int(n), int(nb)], endpoint=int(nb))
                    for nb in graph.neighbors(n)]
        junctions[next_id] = Junction(
            jid=next_id,
            center_node=int(n),
            branches=branches,
            type='unknown'
        )
        next_id += 1
    return junctions


# ---------- Growth (lockstep) ----------

def grow_junctions(
    graph: nx.Graph,
    junctions: JunctionMap,
    angle_thresh: float
) -> JunctionMap:
    """
    Grow all branches of all junctions in lockstep.
    Each branch stops growing if:
      (1) the next edge has 'object angle' > angle_thresh,
      (2) the next node has already been taken by another branch,
      (3) no unique forward neighbor exists.
    """
    # nodes already on any branch
    claimed_nodes: Set[NodeId] = set()
    for junc in junctions.values():
        for br in junc.branches:
            claimed_nodes.update(br.path_nodes)

    changed = True
    while changed:
        changed = False
        for junc in junctions.values():
            new_branches: List[BranchRef] = []
            for br in junc.branches:
                prev, curr = br.path_nodes[-2], br.path_nodes[-1]

                # (3) no unique forward neighbor → stop
                nxt = unique_next_neighbor(graph, prev, curr)
                if nxt is None:
                    new_branches.append(br)
                    continue

                # (2) node already taken → stop
                if nxt in claimed_nodes:
                    new_branches.append(br)
                    continue

                # (1) high object angle → include node then stop
                angle = graph[curr][nxt].get("object angle", 0.0)
                if angle > angle_thresh:
                    br.path_nodes.append(nxt)
                    br.endpoint = nxt
                    claimed_nodes.add(nxt)
                    new_branches.append(br)
                    continue

                # Otherwise: grow one step
                br.path_nodes.append(nxt)
                br.endpoint = nxt
                claimed_nodes.add(nxt)
                new_branches.append(br)
                changed = True

            junc.branches = new_branches

    return junctions


# ---------- T/Y classification ----------

def _branch_dir_into_center(graph: nx.Graph, br: BranchRef) -> Optional[np.ndarray]:
    """
    Direction estimated by the last edge, pointing from endpoint *toward* the junction center.
    """
    if len(br.path_nodes) < 2:
        return None
    p_prev = _pos(graph, br.path_nodes[-2])
    p_end  = _pos(graph, br.path_nodes[-1])
    v = p_prev - p_end  # into the center
    return _unit(v)

def _branch_end_directions(graph: nx.Graph, branches: List[BranchRef]) -> Optional[List[np.ndarray]]:
    dirs: List[np.ndarray] = []
    for br in branches:
        d = _branch_dir_into_center(graph, br)
        if d is None:
            return None
        dirs.append(d)
    return dirs  # should be 3

def _count_colinear_pairs(dirs: List[np.ndarray], thresh: float) -> int:
    pairs = ((0, 1), (0, 2), (1, 2))
    return sum(1 for i, j in pairs if abs(float(np.dot(dirs[i], dirs[j]))) > thresh)

def classify_ty_junctions(
    graph: nx.Graph,
    junctions: JunctionMap,
    colinear_thresh: float = 0.95
) -> JunctionMap:
    """
    Set j.type to 'T-junction' if exactly 1 colinear pair of branch directions,
    else 'Y-junction'. Only updates junctions currently labeled 'unknown'.
    """
    for j in junctions.values():
        if j.type != 'unknown':
            continue
        dirs = _branch_end_directions(graph, j.branches)
        if dirs is None or len(dirs) != 3:
            # leave as 'unknown' if degenerate
            continue
        colinear_pairs = _count_colinear_pairs(dirs, colinear_thresh)
        j.type = 'T-junction' if colinear_pairs == 1 else 'Y-junction'
    return junctions


# ---------- Merge nearby into X ----------

def build_junction_graph(
    graph: nx.Graph,
    junctions: JunctionMap,
    radius: float
) -> Dict[int, Set[int]]:
    """Undirected graph over junction ids; connect if center nodes within radius."""
    ids = list(junctions.keys())
    centers = {jid: _pos(graph, junctions[jid].center_node) for jid in ids}
    r2 = float(radius) ** 2

    adj: Dict[int, Set[int]] = {jid: set() for jid in ids}
    for i in range(len(ids)):
        ji = ids[i]; ci = centers[ji]
        for j in range(i + 1, len(ids)):
            jj = ids[j]; cj = centers[jj]
            if float(np.dot(ci - cj, ci - cj)) <= r2:
                adj[ji].add(jj)
                adj[jj].add(ji)
    return adj

def connected_junction_components(adj: Dict[int, Set[int]]) -> List[Set[int]]:
    unseen: Set[int] = set(adj.keys())
    comps: List[Set[int]] = []
    while unseen:
        start = unseen.pop()
        comp = {start}
        stack = [start]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v in unseen:
                    unseen.remove(v)
                    comp.add(v)
                    stack.append(v)
        comps.append(comp)
    return comps

def _closest_endpoint_midpoint_for_cluster(
    graph: nx.Graph,
    cluster: List[Junction]
) -> Optional[np.ndarray]:
    """Midpoint of the closest pair among ALL branch endpoints across the cluster."""
    endpoints: List[np.ndarray] = []
    for j in cluster:
        for br in j.branches:
            endpoints.append(_endpoint_pos(graph, br))
    if len(endpoints) < 2:
        return None

    best = None
    best_d2 = float('inf')
    for i in range(len(endpoints)):
        pi = endpoints[i]
        for j in range(i + 1, len(endpoints)):
            pj = endpoints[j]
            d2 = float(np.sum((pi - pj) ** 2))
            if d2 < best_d2:
                best_d2 = d2
                best = (pi + pj) * 0.5
    return best

def _prune_one_branch_per_junction_by_center(
    graph: nx.Graph,
    j: Junction,
    center: np.ndarray
) -> List[BranchRef]:
    """Drop exactly ONE branch from j: the branch whose endpoint is closest to center."""
    if not j.branches:
        return []
    d2 = [float(np.sum((_endpoint_pos(graph, br) - center) ** 2)) for br in j.branches]
    drop_idx = int(np.argmin(d2))
    return [br for k, br in enumerate(j.branches) if k != drop_idx]

def merge_nearby_junctions(
    graph: nx.Graph,
    junctions: JunctionMap,
    radius: float
) -> JunctionMap:
    """
    Group all junctions whose centers lie within distance 'radius'.
    If >1 junctions are merged, mark as 'X-junction' and drop one escaping
    branch per original junction (the one whose endpoint is closest to the
    new X-center). Dropped branches are physically removed from the graph.
    """
    if not junctions:
        return {}

    # Build cluster graph by center-node distance
    adj   = build_junction_graph(graph, junctions, radius)
    comps = connected_junction_components(adj)

    merged: JunctionMap = {}
    new_id = 0

    for comp in comps:
        member_ids = sorted(comp)
        members = [junctions[jid] for jid in member_ids]
        rep = members[0]

        if len(members) == 1:
            # Single junction unchanged
            merged[new_id] = Junction(
                jid=new_id,
                center_node=rep.center_node,
                branches=list(rep.branches),
                type=rep.type
            )
            new_id += 1
            continue

        # (1) New X center = midpoint of closest endpoints across the cluster
        x_center = _closest_endpoint_midpoint_for_cluster(graph, members)

        kept_branches: BranchSet = []
        if x_center is None:
            # Fallback: keep all branches (rare), still mark X
            for j in members:
                kept_branches.extend(j.branches)
        else:
            # (2) For each member, drop the single branch closest to x_center
            #     and physically remove that branch from the graph.
            for j in members:
                kept = _prune_one_branch_per_junction_by_center(graph, j, x_center)
                # remove the dropped ones
                for br in j.branches:
                    if br not in kept:  # dataclass equality by fields
                        _remove_branch_from_graph(graph, br)
                kept_branches.extend(kept)

        # Representative center node is preserved; geometry pass will move it.
        merged[new_id] = Junction(
            jid=new_id,
            center_node=rep.center_node,
            branches=kept_branches,
            type="X-junction"
        )
        new_id += 1

    return merged

# ---------- Geometric centers for T/Y/X & straightening ----------

def _compute_center_x(graph: nx.Graph, junc: Junction) -> np.ndarray:
    """Centroid of all unique nodes in the junction (center + every node along each branch)."""
    nodes: Set[int] = {int(junc.center_node)}
    for br in junc.branches:
        nodes.update(int(n) for n in br.path_nodes)
    pts = np.vstack([_pos(graph, n) for n in nodes])
    return pts.mean(axis=0)

def _compute_center_y(graph: nx.Graph, junc: Junction, dot_thresh: float = 0.95) -> np.ndarray:
    """
    Average of pairwise intersections of rays cast from each endpoint
    toward the center direction; skip nearly-colinear pairs.
    Fallback: average of endpoints.
    """
    if len(junc.branches) != 3:
        return _pos(graph, junc.center_node)

    # rays: from endpoint in the direction of the last edge (pointing toward center)
    rays: List[Tuple[np.ndarray, np.ndarray]] = []
    for br in junc.branches:
        p_end = _endpoint_pos(graph, br)
        d = _branch_dir_into_center(graph, br)
        if d is None:
            return _pos(graph, junc.center_node)
        rays.append((p_end, d))

    lines = [extended_line(p, d) for (p, d) in rays]
    intersections: List[np.ndarray] = []
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(float(np.dot(rays[i][1], rays[j][1]))) > dot_thresh:
                continue
            pt = _line_intersection_point(lines[i], lines[j])
            if pt is not None:
                intersections.append(pt)

    if intersections:
        return np.vstack(intersections).mean(axis=0)

    # fallback: mean of endpoints
    ends = np.vstack([_endpoint_pos(graph, br) for br in junc.branches])
    return ends.mean(axis=0)

def _compute_center_t(graph: nx.Graph, junc: Junction, dot_thresh: float = 0.95) -> np.ndarray:
    """
    Intersect the midline of the two colinear branches' endpoints with the
    ray of the non-colinear branch. Fallback to Y rule if needed.
    """
    if len(junc.branches) != 3:
        return _pos(graph, junc.center_node)

    dirs = _branch_end_directions(graph, junc.branches)
    if dirs is None:
        return _pos(graph, junc.center_node)

    # find a colinear pair
    colinear_pairs = []
    for (i, j) in ((0,1),(0,2),(1,2)):
        if abs(float(np.dot(dirs[i], dirs[j]))) > dot_thresh:
            colinear_pairs.append((i,j))

    if not colinear_pairs:
        return _compute_center_y(graph, junc, dot_thresh)

    i_col, j_col = colinear_pairs[0]
    k_noncol = ({0,1,2} - {i_col, j_col}).pop()

    p_i = _endpoint_pos(graph, junc.branches[i_col])
    p_j = _endpoint_pos(graph, junc.branches[j_col])
    p_k = _endpoint_pos(graph, junc.branches[k_noncol])
    d_k = dirs[k_noncol]

    line_colinear = extended_line((p_i + p_j)/2.0, p_j - p_i)
    line_noncol   = extended_line(p_k, d_k)
    inter = _line_intersection_point(line_colinear, line_noncol)
    return inter if inter is not None else (p_i + p_j)/2.0

def compute_junction_center(graph: nx.Graph, j: Junction) -> np.ndarray:
    if j.type == "X-junction":
        return _compute_center_x(graph, j)
    if j.type == "Y-junction":
        return _compute_center_y(graph, j)
    if j.type == "T-junction":
        return _compute_center_t(graph, j)
    return _pos(graph, j.center_node)

def _straighten_branch_to_center(graph: nx.Graph, br: BranchRef, center: np.ndarray) -> None:
    """
    Move nodes in br.path_nodes (except the endpoint) to lie evenly on the segment
    center → endpoint. Endpoint remains fixed.
    """
    nodes = br.path_nodes
    if not nodes:
        return
    end = _endpoint_pos(graph, br)
    denom = max(1, len(nodes) - 1)
    for idx, nid in enumerate(nodes[:-1]):   # exclude endpoint
        t = float(idx) / float(denom)        # center gets t=0, last internal ~1
        newp = (1.0 - t)*center + t*end
        _set_pos(graph, nid, newp)

def straighten_junction_geometry(graph: nx.Graph, junctions: JunctionMap) -> None:
    """
    Move each junction's center to its computed center (by type),
    then place each branch's internal nodes on straight segments to that center.
    """
    for j in junctions.values():
        center = compute_junction_center(graph, j)
        _set_pos(graph, j.center_node, center)
        for br in j.branches:
            _straighten_branch_to_center(graph, br, center)


# ---------- Orchestrator ----------

def repair_junctions(
    graph: nx.Graph,
    angle_thresh: float,
    merge_radius: float,
    colinear_thresh: float = 0.95
) -> JunctionMap:
    """
    Pipeline:
      1) seed degree-3 junctions
      2) grow branches in lockstep (with object-angle stop & node-claiming)
      3) classify T vs Y (unknowns remain if degenerate)
      4) merge nearby into X and prune one escaping branch per member
      5) geometry pass: compute center for T/Y/X and straighten branches
    """
    g = graph.copy()
    junctions = initialize_junctions(g)
    junctions = grow_junctions(g, junctions, angle_thresh)
    junctions = classify_ty_junctions(g, junctions, colinear_thresh)
    junctions = merge_nearby_junctions(g, junctions, merge_radius)
    straighten_junction_geometry(g, junctions)
    return g, junctions

