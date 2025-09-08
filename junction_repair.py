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
    """
    Returns the canonical ordering of an edge from node id's pair (u,v).
    """
    return (u, v) if u < v else (v, u)

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
            p1 = np.array(graph.nodes[branch[-1]]['position'])
            p2 = np.array(graph.nodes[branch[-2]]['position'])
        # this should probably throw an exception, but skipping for now
        else:
            continue

        # append a normalized tangent for each branch
        tangent = p2 - p1
        norm = np.linalg.norm(tangent)
        # check to see if something has gone horribly wrong
        if norm == 0:
            continue

        # normalize the tangent
        tangent /= norm
        # append tangent AND origin to tangents list 
        tangents.append((p1, tangent))
    return tangents


def count_colinear_pairs(dirs: List[np.ndarray], thresh: float) -> int:
    """
    Number of (i,j) pairs whose |dot| exceeds the threshold.
    """
    assert len(dirs) == 3
    pairs = ((0, 1), (0, 2), (1, 2))
    return sum(1 for i, j in pairs if abs(np.dot(dirs[i], dirs[j])) > thresh)


def compute_colinear_map(tangents, dot_threshold=0.95):
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
    This mean that a single edge could be part of two branches...
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


def is_junction(graph: nx.Graph, node_id: int) -> bool:
    """
    Returns True if the node_id is a junction (degree >=3). False otherwise.
    """
    return graph.degree[node_id] >= 3


def unique_next_neighbor(graph: nx.Graph, prev: int, curr: int) -> int | None:
    """
    Return the unique neighbor forward of (prev -> curr), or None if it doesn't exist.
    """
    forward = [n for n in graph.neighbors(curr) if n != prev]
    return forward[0] if len(forward) == 1 else None

def grow_branches_lockstep(
    graph: nx.Graph,
    seeds: Dict[int, BranchSet],
    angle_thresh: float
) -> Dict[int, BranchSet]:
    """
    Grow branches in lockstep so that one junction doesn't eat up the whole
    branch between two close junctions.
    
    Branches will grow until:
    - (1) hit a degree >=3 node
    - (2) hit a dead end/ fork in the road 
    - (3) hit an edge with high object angle (exceeds threshold)
        - in this case, will still add the edge to get a good tangent estimate
    - (4) meets an edge that has already been claimed by another branch
    """
    # to begin with, we will attemp to grow every branch
    growing = {n: [b[:] for b in brs] for n, brs in seeds.items()}
    # keep track of which __edges__ have already been taken
    claimed_edges: Set[Tuple[int, int]] = set()
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

                # condition (1): stop growing current branch if last node is a junction 
                if is_junction(graph, curr):
                    new_branches.append(br)
                    continue

                # stop growing current branch if there is no unique next neighbor 
                nxt = unique_next_neighbor(graph, prev, curr)
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
                        claimed_edges.add(e)
                    new_branches.append(br)
                    continue

                # stop growing this branch if we have already hit this edge before 
                if interior_edge and (e in claimed_edges):
                    new_branches.append(br)
                    continue

                # if we have gotten here, we should grow the branch and mark the edge as claimed 
                br.append(nxt)
                if interior_edge:
                    claimed_edges.add(e)
                new_branches.append(br)
                changed = True

            growing[node] = new_branches

    return growing


# =====================================
#  T- and Y- junction classification 
# =====================================

def direction_from_branch_end(graph: nx.Graph, branch: Branch) -> np.ndarray | None:
    """
    Computes the branch direction for a single Branch as estimated by the 
    last edge in the branch. The last edge in the branch is a ''good enough''
    estimate of the stroke tangent.

    Direction __is normalized__. Direction is pointing INTO the branch/junction.

    Note: Returns None if there are no edges in the branch (i think this impossible unless
    i haven't changed the code back)
    """
    if len(branch) < 2:
        return None
    p0 = np.asarray(graph.nodes[branch[-2]]['position'])
    p1 = np.asarray(graph.nodes[branch[-1]]['position'])
    v  = p0 - p1
    n  = np.linalg.norm(v)
    return (v / n) if n > 0 else None

def branch_end_directions(graph: nx.Graph, branches: BranchSet) -> List[np.ndarray] | None:
    """
    Computes all of the directions for each branch in a BranchSet. 

    Note: Returns None if even one of the directions is None (again, i don't
    think this can happen)
    """
    directions: List[np.ndarray] = []
    for branch in branches:
        dir = direction_from_branch_end(graph, branch)
        if dir is None:
            return None
        directions.append(dir)
    return directions 


def classify_grown_junctions(
    graph: nx.Graph,
    grown: Dict[int, BranchSet],
    colinear_thresh: float = 0.95
) -> JunctionMap:
    """
    Classify each degree-3 node as T or Y:
        A junction is a T-junction if exactly 1 pair of branch directions
        are parallel.
        All other junctions are Y-junctions.
    """
    junctions: JunctionMap = {}
    next_id = 0

    for node, branches in grown.items():
        dirs = branch_end_directions(graph, branches)
        # prettry sure this is impossible
        if dirs is None:
            continue 

        num_colinear = count_colinear_pairs(dirs, colinear_thresh)
        jtype = 'T-junction' if num_colinear == 1 else 'Y-junction'

        junctions[next_id] = {
            'node': node,
            'branches': branches,
            'type': jtype
        }
        next_id += 1

    return junctions

# =====================================
#  Junction merging to form X-junctions (or higher) 
# =====================================

def build_junction_graph(
    graph: nx.Graph,
    junctions: JunctionMap,
    radius: float
) -> Dict[int, Set[int]]:
    """
    Make a graph for all the junctions where an edge between i ~ j if
    junction i's central node is within a specified radius of junction j.
    """
    ids = list(junctions.keys())
    centers = {jid: np.asarray(graph.nodes[junctions[jid]['node']]['position'], dtype=float) for jid in ids}
    r2 = float(radius) ** 2

    adj: Dict[int, Set[int]] = {jid: set() for jid in ids}
    # brute force for now 
    for i in range(len(ids)):
        ji = ids[i]
        ci = centers[ji]
        for j in range(i + 1, len(ids)):
            jj = ids[j]
            cj = centers[jj]
            if np.dot(ci-cj, ci-cj) <= r2:
                adj[ji].add(jj)
                adj[jj].add(ji)
    return adj

def connected_junction_components(adj: Dict[int, Set[int]]) -> List[Set[int]]:
    """
    Connected components of the junction graph are unique junctions.
    This function finds them.
    """
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

    adj   = build_junction_graph(graph, junctions, radius)
    comps = connected_junction_components(adj)

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
            'node':     rep['node'],       # keep representative node
            'branches': combined_branches,
            'type':     jtype,
            'colinear_map' : colinear_map,
            'tangents': tangents
        }
        new_id += 1

    return merged


###############################

def compute_center_x(graph: nx.Graph, junc: Junction) -> np.ndarray:
    """
    For X-junctions, a first guess is to place the junction at the 
    centroid of all nodes in the junction.

    TODO:
        come up with something better.
    """
    nodes = {junc['node']}
    for br in junc['branches']:
        nodes.update(br)
    pts = np.vstack([np.asarray(graph.nodes[n]['position'], dtype=float) for n in nodes])
    return pts.mean(axis=0)


def compute_center_y(graph: nx.Graph, junc: Junction, colinear_dot: float = 0.95) -> np.ndarray:
    """
    For Y-junctions, we place the junction node at the average of all intersection
    points of branch rays. Skip pairs of branch rays that are close to colinear,
    they will move the junction too much.
    """
    branches = junc['branches']
    rays = junc['tangents'] 

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

    # fallback
    print("Y-junction fail")
    return _avg_endpoints(graph, branches)


def compute_center_t(graph: nx.Graph, junc: Junction, colinear_dot: float = 0.95) -> np.ndarray:
    """
    T: intersect (midpoint line of colinear endpoints) with the non‑colinear branch ray.
    Falls back to average endpoints on failure.
    """
    branches = junc['branches']
    rays = junc['tangents']
    colinear_map = junc['colinear_map']

    colinear_pairs = [(i, j) for i in range(3) for j in colinear_map[i] if j > i]
    if not colinear_pairs:
        return compute_center_y(graph, junc, colinear_dot)

    i_col, j_col = colinear_pairs[0]
    k_noncol = ({0, 1, 2} - {i_col, j_col}).pop()

    p_i = np.array(graph.nodes[branches[i_col][-1]]['position'])
    p_j = np.array(graph.nodes[branches[j_col][-1]]['position'])
    p_k, d_k = rays[k_noncol]

    line_colinear = extended_line((p_i + p_j) / 2.0, p_j - p_i)
    line_noncol = extended_line(p_k, d_k)
    inter = line_colinear.intersection(line_noncol)

    # hopefully doesn't happen
    if inter.is_empty or inter.geom_type != 'Point':
        return (p_i + p_j) / 2.0

    intersection_point = np.array(inter.coords[0])
    return intersection_point


def compute_junction_center(graph: nx.Graph, junc: Junction) -> np.ndarray:
    """Dispatch by type; always returns a valid center point."""
    jtype = junc.get('type', '')
    if jtype == 'X-junction':
        return compute_center_x(graph, junc)
    elif jtype == 'Y-junction':
        return compute_center_y(graph, junc)
    elif jtype == 'T-junction':
        return compute_center_t(graph, junc)
    # unknown -> keep current center
    return np.asarray(graph.nodes[junc['node']]['position'], dtype=float)


def straighten_branch_toward_center(graph: nx.Graph, branch: Branch, center: np.ndarray) -> None:
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
    for junc in junctions.values():
        # skip x-junctions
        if junc.get('type', '') == 'X-junction':
            continue
        center = compute_junction_center(graph, junc) 
        for br in junc['branches']:
            straighten_branch_toward_center(graph, br, center)
        _set_pos(graph, junc['node'], center)
    return junctions


# =========================
# X-junction refinement
# =========================

def _euclid_len(graph: nx.Graph, u: int, v: int) -> float:
    pu, pv = _pos(graph, u), _pos(graph, v)
    return float(np.linalg.norm(pu - pv))

def _pairwise_min_dist_nodes(graph: nx.Graph, nodes: list[int]) -> tuple[int, int, float]:
    """Return (i_node, j_node, distance) with minimal Euclidean distance."""
    best = (None, None, float('inf'))
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            d = _euclid_len(graph, nodes[i], nodes[j])
            if d < best[2]:
                best = (nodes[i], nodes[j], d)
    return best  # type: ignore

def _branch_edge_set(br: Branch) -> set[tuple[int, int]]:
    return {ordered_edge(br[k], br[k + 1]) for k in range(len(br) - 1)}

def _leaf_nodes_of_junction(junc: Junction) -> list[int]:
    """Leaf = branch endpoint (last node in the branch list)."""
    return [br[-1] for br in junc['branches']]

def _geom_shortest_path(graph: nx.Graph, s: int, t: int) -> list[int] | None:
    """Geometric (Euclidean) shortest path node list; None if no path."""
    try:
        return nx.shortest_path(graph, s, t, weight=lambda u, v, d: _euclid_len(graph, u, v))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

def _remove_path_edges(graph: nx.Graph, path_nodes: list[int]) -> set[tuple[int, int]]:
    """Remove all edges along the given node path; return the removed edge set (ordered)."""
    removed: set[tuple[int, int]] = set()
    for a, b in zip(path_nodes[:-1], path_nodes[1:]):
        e = ordered_edge(a, b)
        if graph.has_edge(*e):
            graph.remove_edge(*e)
            removed.add(e)
    return removed

def _recompute_tangent_bundle(graph: nx.Graph, branches: BranchSet):
    tangents = compute_branch_tangents(graph, branches)
    colinear_map = compute_colinear_map(tangents)
    return tangents, colinear_map

def refine_x_junctions(
    graph: nx.Graph,
    junctions: JunctionMap,
) -> JunctionMap:
    """
    For each X-junction:
      1) Find the two closest leaf nodes.
      2) Remove edges along the geometric shortest path between them.
      3) Drop any branches that touched those removed edges.
      4) Set the junction center to the midpoint of the two closest leaves.
      5) Ensure every remaining branch is connected to the center node and
         straighten all remaining branches as straight segments to the center.
    """
    for jid, junc in list(junctions.items()):
        if junc.get('type', '') != 'X-junction':
            continue

        # --- Step 1: closest leaf pair
        leaves = _leaf_nodes_of_junction(junc)
        if len(leaves) < 4:
            # Not really an X (or degenerate); skip safely.
            continue

        li, lj, _ = _pairwise_min_dist_nodes(graph, leaves)

        # --- Step 2: geometric shortest path and removal
        path = _geom_shortest_path(graph, li, lj)
        if path is None or len(path) < 2:
            # No path (already disconnected); still proceed to recenter/straighten.
            removed_edges: set[tuple[int, int]] = set()
        else:
            removed_edges = _remove_path_edges(graph, path)

        # --- Step 3: drop branches that touched removed edges
        kept_branches: BranchSet = []
        for br in junc['branches']:
            br_edges = _branch_edge_set(br)
            if br_edges.isdisjoint(removed_edges):
                kept_branches.append(br)
        junc['branches'] = kept_branches

        # If everything got removed, keep a minimal stub and continue
        if not junc['branches']:
            # Recreate two stubs to the two closest leaves so we still have a junction.
            junc['branches'] = [[junc['node'], li], [junc['node'], lj]]

        # --- Step 4: set center to midpoint of closest leaves
        p_li = _pos(graph, li)
        p_lj = _pos(graph, lj)
        center = 0.5 * (p_li + p_lj)

        center_node = junc['node']  # reuse representative node id
        _set_pos(graph, center_node, center)

        # --- Step 5: connect all remaining branches to the center + straighten
        fixed_branches: BranchSet = []
        for br in junc['branches']:
            # Ensure the branch is rooted at the center node (hard connection).
            if br[0] != center_node:
                # Add an edge from center -> first node if missing, and
                # make center the first element of the branch path.
                if not graph.has_edge(center_node, br[0]):
                    graph.add_edge(center_node, br[0])
                br = [center_node] + br  # prepend center
            # Straighten along the segment (endpoint -> center)
            straighten_branch_toward_center(graph, br, center)
            fixed_branches.append(br)

        # Update junction bundle + recompute tangents/colinearity (optional but nice)
        junc['branches'] = fixed_branches
        tangents, colinear_map = _recompute_tangent_bundle(graph, fixed_branches)
        junc['tangents'] = tangents
        junc['colinear_map'] = colinear_map
        junc['type'] = 'X-junction'  # still an X (>=4 leaves typically)

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
    junctions = refine_x_junctions(graph, junctions)
    #straighten_junction_geometry(graph, junctions)
    return junctions

