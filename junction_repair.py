
from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set
from shapely.geometry import LineString

Branch      = List[int]
BranchSet   = List[Branch]
Junction    = Dict[str, object]      # {'node': int, 'branches': BranchSet, 'type': str}
JunctionMap = Dict[int, Junction]


# =========================
# Small utilities
# =========================

def ordered_edge(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)

def is_junction(graph: nx.Graph, n: int) -> bool:
    return graph.degree[n] >= 3

def _center_pos(graph: nx.Graph, n: int) -> np.ndarray:
    return np.asarray(graph.nodes[n]['position'], dtype=float)

def _squared_distance(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(d @ d)

def direction_from_branch_end(graph: nx.Graph, branch: Branch) -> np.ndarray | None:
    """Unit direction from the last two nodes of a branch; None if degenerate."""
    if len(branch) < 2:
        return None
    p0 = np.asarray(graph.nodes[branch[-2]]['position'])
    p1 = np.asarray(graph.nodes[branch[-1]]['position'])
    v  = p1 - p0
    n  = np.linalg.norm(v)
    return (v / n) if n > 0 else None

def branch_end_directions(graph: nx.Graph, branches: BranchSet) -> List[np.ndarray] | None:
    """Return [dir1, dir2, dir3] if all are valid; otherwise None."""
    dirs: List[np.ndarray] = []
    for br in branches:
        d = direction_from_branch_end(graph, br)
        if d is None:
            return None
        dirs.append(d)
    return dirs  # length 3 by construction

def compute_branch_tangents(graph, branches):
    """
    Takes in a list of branches (list of nodes on a graph) and 
    computes the tangent direction for each branch.
    """
    tangents = []
    # compute a tangent for each branch
    for i, branch in enumerate(branches):
        # if branch has at least two edges, average tangents of last two edges
        if len(branch) >= 3:
            p1 = np.array(graph.nodes[branch[-1]]['position'])
            p2 = np.array(graph.nodes[branch[-3]]['position'])
        # if branch has only one edge, just use that edge as the tangent 
        elif len(branch) >= 2:
            #logger.warning(f"[JUNCTION DETECTION] Branch {i} has fewer than 3 nodes. Using last 2 for tangent.")
            p1 = np.array(graph.nodes[branch[-1]]['position'])
            p2 = np.array(graph.nodes[branch[-2]]['position'])
        # this should probably throw an exception, but skipping for now
        else:
            #logger.warning(f"[JUNCTION DETECTION] Branch {i} is too short to compute a tangent. Skipping.")
            continue

        # append a normalized tangent for each branch
        tangent = p2 - p1
        norm = np.linalg.norm(tangent)
        # check to see if something has gone horribly wrong
        if norm == 0:
            #logger.warning(f"[JUNCTION DETECTION] Branch {i} has zero-length direction vector. Skipping.")
            continue

        # normalize the tangent
        tangent /= norm
        # append tangent AND origin to tangents list 
        tangents.append((p1, tangent))
    return tangents


def count_colinear_pairs(dirs: List[np.ndarray], thresh: float) -> int:
    """Number of (i,j) pairs whose |dot| exceeds the threshold."""
    assert len(dirs) == 3
    pairs = ((0, 1), (0, 2), (1, 2))
    return sum(1 for i, j in pairs if abs(np.dot(dirs[i], dirs[j])) > thresh)


def compute_colinear_map(tangents, dot_threshold=0.90):
    """
    Takes in a list of tangents and returns a hash map that tells us what other 
    tangents are roughly colinear. 
    """
    # initially no tangents are colinear 
    colinear_map = {i: set() for i in range(len(tangents))}
    for i in range(len(tangents)):
        # compare tangent i to all subsequent tangents
        _, t1 = tangents[i]
        for j in range(i + 1, len(tangents)):
            _, t2 = tangents[j]
            if abs(np.dot(t1, t2)) > dot_threshold:
                colinear_map[i].add(j)
                colinear_map[j].add(i)
    return colinear_map


def _unique_next_neighbor(graph: nx.Graph, prev: int, curr: int) -> int | None:
    """Return the unique neighbor forward of (prev -> curr), or None if it doesn't exist."""
    forward = [n for n in graph.neighbors(curr) if n != prev]
    return forward[0] if len(forward) == 1 else None

def _endpoint(graph: nx.Graph, branch: Branch) -> np.ndarray:
    """Position of the last node in a branch."""
    return np.asarray(graph.nodes[branch[-1]]['position'], dtype=float) 

def _unit(v: np.ndarray) -> np.ndarray | None:
    """Return unit vector or None if degenerate."""
    n = float(np.linalg.norm(v))
    return (v / n) if n > 0 else None

def _pos(graph: nx.Graph, n: int) -> np.ndarray:
    """Return node position as float np.array([y, x]) (or [row, col])."""
    return np.asarray(graph.nodes[n]['position'], dtype=float)


def _set_pos(graph: nx.Graph, n: int, p: np.ndarray) -> None:
    """Set node position (accepts ndarray)."""
    graph.nodes[n]['position'] = np.asarray(p, dtype=float)


def _unique_nodes_in_junction(rec: JunctionRec) -> Set[int]:
    """All unique node IDs in center + all branches."""
    s: Set[int] = {rec['node']}
    for br in rec['branches']:
        s.update(br)
    return s


def _centroid_of_nodes(graph: nx.Graph, nodes: Set[int]) -> np.ndarray:
    """Centroid of given node set."""
    if not nodes:
        raise ValueError("Empty node set for centroid.")
    pts = np.vstack([_pos(graph, n) for n in nodes])
    return pts.mean(axis=0)


def _avg_endpoints(graph: nx.Graph, branches: BranchSet) -> np.ndarray:
    """Average of the three branch endpoints (fallback center)."""
    pts = np.vstack([_endpoint(graph, br) for br in branches])
    return pts.mean(axis=0)

def _branch_dirs(graph: nx.Graph, branches: BranchSet) -> list[np.ndarray] | None:
    """Unit directions from each branch’s last two nodes; None if any degenerate."""
    dirs = []
    for br in branches:
        d = direction_from_branch_end(graph, br)
        if d is None:
            return None
        dirs.append(d)
    return dirs  # length 3

def _line_intersection_point(L1: LineString, L2: LineString) -> np.ndarray | None:
    inter = L1.intersection(L2)
    if inter.is_empty or inter.geom_type != 'Point':
        return None
    return np.asarray(inter.coords[0], dtype=float)

def extended_line(p: np.ndarray, d: np.ndarray, extension: float = 10.0) -> LineString:
    """
    Robust line builder: if direction is degenerate, return a point segment.
    """
    p = np.asarray(p, float)
    d = np.asarray(d, float)
    u = _unit(d)
    if u is None:
        return LineString([p, p])
    return LineString([p - u * extension, p + u * extension])

# =========================
# 1) Seed degree-3 nodes
# =========================

def initialize_junctions(graph: nx.Graph) -> Dict[int, BranchSet]:
    """
    For every degree-3 node, create 3 initial branches [node, neighbor].
    """
    initial: Dict[int, BranchSet] = {}
    for node in graph:
        if graph.degree[node] != 3:
            continue
        nbrs = list(graph.neighbors(node))
        initial[node] = [[node, nbr] for nbr in nbrs]
    return initial


# =========================
# 2) Lockstep growth with object-angle stop
# =========================

def grow_branches_lockstep(
    graph: nx.Graph,
    seeds: Dict[int, BranchSet],
    angle_thresh: float
) -> Dict[int, BranchSet]:
    """
    Grow branches in locksetp
    """
    # to begin with, we will attemp to grow every branch
    growing = {n: [b[:] for b in brs] for n, brs in seeds.items()}
    # keep track of which __edges__ have already been taken
    claimed_interior: Set[Tuple[int, int]] = set()
    # as long is this is true, we will continiue growing all remaining 'growing' branches
    changed = True

    while changed:
        changed = False
        for node, branches in list(growing.items()):
            # create a new empty set of branches for the current node
            new_branches: BranchSet = []
            for br in branches:
                # look at the outermost edge
                prev, curr = br[-2], br[-1]

                # stop growing current branch if last node is a junction 
                if is_junction(graph, curr):
                    new_branches.append(br)
                    continue

                # stop growing current branch if there is no unique next neighbor 
                nxt = _unique_next_neighbor(graph, prev, curr)
                if nxt is None:
                    new_branches.append(br)
                    continue

                e = ordered_edge(curr, nxt)
                # set this to true if always want to claim the edge
                interior_edge = (not is_junction(graph, nxt)) and (not is_junction(graph,curr))

                # if we hit the angle threshold, stop growing AND add edge to branch 
                angle = graph[curr][nxt].get('object angle', 0.0)
                if angle > angle_thresh:
                    br.append(nxt)
                    if interior_edge:
                        claimed_interior.add(e)
                    new_branches.append(br)
                    continue

                # stop growing this branch if we have already hit this edge before 
                if interior_edge and (e in claimed_interior):
                    new_branches.append(br)
                    continue

                # if we have gotten here, we should grow the branch and mark the edge as claimed 
                br.append(nxt)
                if interior_edge:
                    claimed_interior.add(e)
                new_branches.append(br)
                changed = True

            growing[node] = new_branches

    return growing


# =========================
# 3) T/Y classification
# =========================

def _build_proximity_graph(
    graph: nx.Graph,
    junctions: JunctionMap,
    radius: float
) -> Dict[int, Set[int]]:
    """
    Build an adjacency map over junction IDs where an edge exists
    if their centers are within 'radius'.
    """
    ids = list(junctions.keys())
    centers = {jid: _center_pos(graph, junctions[jid]['node']) for jid in ids}
    r2 = float(radius) ** 2

    adj: Dict[int, Set[int]] = {jid: set() for jid in ids}
    # O(n^2) pairwise check; for many junctions you could swap to KDTree
    for i in range(len(ids)):
        ji = ids[i]
        ci = centers[ji]
        for j in range(i + 1, len(ids)):
            jj = ids[j]
            cj = centers[jj]
            if _squared_distance(ci, cj) <= r2:
                adj[ji].add(jj)
                adj[jj].add(ji)
    return adj

def _connected_components(adj: Dict[int, Set[int]]) -> List[Set[int]]:
    """Connected components over an adjacency map."""
    unseen = set(adj.keys())
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

def classify_grown_junctions(
    graph: nx.Graph,
    grown: Dict[int, BranchSet],
    colinear_thresh: float = 0.95
) -> JunctionMap:
    """
    Classify each degree-3 node as T or Y via branch-end colinearity:
      - compute unit directions from the last two nodes of each branch
      - count colinear pairs via |dot| > colinear_thresh
      - exactly one colinear pair => T; otherwise Y
    """
    junctions: JunctionMap = {}
    next_id = 0

    for node, branches in grown.items():
        dirs = branch_end_directions(graph, branches)
        if dirs is None:
            continue  # skip degenerate branches

        num_colinear = count_colinear_pairs(dirs, colinear_thresh)
        jtype = 'T-junction' if num_colinear == 1 else 'Y-junction'

        junctions[next_id] = {
            'node': node,
            'branches': branches,
            'type': jtype
        }
        next_id += 1

    return junctions

#######################
## JUNCTION MERGING
#######################


def merge_nearby_junctions(
    graph: nx.Graph,
    junctions: JunctionMap,
    radius: float
) -> JunctionMap:
    """
    Group all junctions whose centers lie within distance 'radius'.
    If >1 junctions are merged, mark as 'X-junction'; otherwise keep type.
    """
    if not junctions:
        return {}

    adj   = _build_proximity_graph(graph, junctions, radius)
    comps = _connected_components(adj)

    merged: JunctionMap = {}
    new_id = 0

    for comp in comps:
        members = sorted(comp)
        rep_jid = members[0]
        rep     = junctions[rep_jid]

        # Combine branches from all members
        combined_branches: BranchSet = []
        for jid in members:
            combined_branches.extend(junctions[jid]['branches'])

        # Type depends on size of component
        if len(members) == 1:
            jtype = rep.get('type', 'unknown')
        else:
            jtype = 'X-junction'

        tangents = compute_branch_tangents(graph, combined_branches)
        colinear_map = compute_colinear_map(tangents)

        merged[new_id] = {
            'node':     rep['node'],       # keep representative's node
            'branches': combined_branches,
            'type':     jtype,
            'colinear_map' : colinear_map,
            'tangents': tangents
        }
        new_id += 1

    return merged


###############################

def compute_center_x(graph: nx.Graph, rec: Junction) -> np.ndarray:
    """X: centroid of all unique nodes in the junction."""
    nodes = {rec['node']}
    for br in rec['branches']:
        nodes.update(br)
    pts = np.vstack([_center_pos(graph, n) for n in nodes])
    return pts.mean(axis=0)


def compute_center_y(graph: nx.Graph, rec: Junction, colinear_dot: float = 0.95) -> np.ndarray:
    """
    Y: average of intersections between non‑colinear tangent rays.
    Falls back to average endpoints, then current center.
    """
    branches = rec['branches']
    # Prefer given tangents; otherwise compute from branches.
    if rec.get('tangents') and len(rec['tangents']) == 3:
        rays = rec['tangents']  # list of (point, dir)
    else:
        dirs = _branch_dirs(graph, branches)
        if dirs is None:
            return _avg_endpoints(graph, branches)
        rays = [( _endpoint(graph, br), dirs[i] ) for i, br in enumerate(branches)]

    # Collect intersections of non‑colinear pairs
    lines = [extended_line(p, d) for (p, d) in rays]
    intersections = []
    for i in range(3):
        for j in range(i + 1, 3):
            # Skip nearly colinear pairs
            if abs(np.dot(rays[i][1], rays[j][1])) > colinear_dot:
                continue
            pt = _line_intersection_point(lines[i], lines[j])
            if pt is not None:
                intersections.append(pt)

    if intersections:
        return np.vstack(intersections).mean(axis=0)

    # Fallbacks
    return _avg_endpoints(graph, branches)


def compute_center_t(graph: nx.Graph, rec: Junction, colinear_dot: float = 0.95) -> np.ndarray:
    """
    T: intersect (midpoint line of colinear endpoints) with the non‑colinear branch ray.
    Falls back to average endpoints on failure.
    """
    branches = rec['branches']

    # Prefer given tangents; otherwise compute from branches.
    if rec.get('tangents') and len(rec['tangents']) == 3:
        rays = rec['tangents']
    else:
        dirs = _branch_dirs(graph, branches)
        if dirs is None:
            return _avg_endpoints(graph, branches)
        rays = [( _endpoint(graph, br), dirs[i] ) for i, br in enumerate(branches)]

    # Find a single colinear pair
    col_pair = None
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(np.dot(rays[i][1], rays[j][1])) > colinear_dot:
                col_pair = (i, j)
                break
        if col_pair:
            break

    if col_pair is None:
        # If nothing looks colinear, treat as Y fallback
        return compute_center_y(graph, rec, colinear_dot)

    i_col, j_col = col_pair
    k = ({0, 1, 2} - {i_col, j_col}).pop()

    p_i = _endpoint(graph, branches[i_col])
    p_j = _endpoint(graph, branches[j_col])
    mid = 0.5 * (p_i + p_j)
    dir_col = p_j - p_i
    L_col = extended_line(mid, dir_col)

    p_k, d_k = rays[k]
    L_non = extended_line(p_k, d_k)

    pt = _line_intersection_point(L_col, L_non)
    return pt if pt is not None else _avg_endpoints(graph, branches)


def compute_junction_center(graph: nx.Graph, rec: Junction) -> np.ndarray:
    """Dispatch by type; always returns a valid center point."""
    jtype = rec.get('type', '')
    """
    if jtype == 'X-junction':
        return compute_center_x(graph, rec)
    elif jtype == 'Y-junction':
        return compute_center_y(graph, rec)
    elif jtype == 'T-junction':
        return compute_center_t(graph, rec)
    # unknown -> keep current center
    """
    return _center_pos(graph, rec['node'])


def _straighten_branch_toward_center(graph: nx.Graph, branch: Branch, center: np.ndarray) -> None:
    """
    Place all internal nodes of `branch` evenly on the straight segment
    from its endpoint to `center`. Endpoint remains fixed.
    """
    if len(branch) <= 1:
        return
    end_pos = _endpoint(graph, branch)
    m = len(branch) - 1  # number of internal nodes + center index
    # Interpolate internal nodes from end_pos -> center (excluding the endpoint)
    for i, nid in enumerate(branch[:-1]):  # exclude endpoint
        t = (i + 1) / (m)  # m steps to approach the endpoint; smooth distribution
        new_pos = (1.0 - t) * end_pos + t * center
        _set_pos(graph, nid, new_pos)

def straighten_junction_geometry(graph: nx.Graph, junctions: JunctionMap) -> JunctionMap:
    """
    For each junction:
      1) Compute a robust center based on its type (T/Y/X).
      2) Move the center node there.
      3) Straighten each branch along a straight segment (endpoint -> center).
    Returns the junction map for convenient chaining.
    """
    for rec in junctions.values():
        center = compute_junction_center(graph, rec)  # never None
        for br in rec['branches']:
            _straighten_branch_toward_center(graph, br, center)
        _set_pos(graph, rec['node'], center)
    return junctions
        

# =========================
# Orchestrator
# =========================

def classify_junctions(
    graph: nx.Graph,
    angle_thresh: float,
    colinear_thresh: float = 0.95,
    merge_radius: float = 5.0,
) -> JunctionMap:
    """
    One call:
      1) Initialize degree-3 junction seeds
      2) Lockstep grow branches w/ object-angle stop (edges already have 'object angle')
      3) Classify T vs Y
      4) Identify X pairs (shared node + center distance)
      5) Geometrically merge X pairs by straightening all branches to the midpoint
         (no node/edge creation/deletion; positions only)

    Returns the final JunctionMap with junction 'type' fields updated.
    """
    seeds   = initialize_junctions(graph)
    grown   = grow_branches_lockstep(graph, seeds, angle_thresh)
    junctions = classify_grown_junctions(graph, grown, colinear_thresh)
    junctions = merge_nearby_junctions(graph, junctions, merge_radius)
    straighten_junction_geometry(graph, junctions)
    return junctions

