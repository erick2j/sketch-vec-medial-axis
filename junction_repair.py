# junctions_xy_straighten.py

from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set

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


def count_colinear_pairs(dirs: List[np.ndarray], thresh: float) -> int:
    """Number of (i,j) pairs whose |dot| exceeds the threshold."""
    assert len(dirs) == 3
    pairs = ((0, 1), (0, 2), (1, 2))
    return sum(1 for i, j in pairs if abs(np.dot(dirs[i], dirs[j])) > thresh)




def _branch_arc_params(graph: nx.Graph, branch: Branch) -> np.ndarray:
    """
    Return cumulative arc-length parameters t in [0,1] along a branch.
    t[0] = 0 at the center node, t[-1] = 1 at the endpoint.
    """
    if len(branch) == 1:
        return np.array([0.0], dtype=float)
    pts = np.array([graph.nodes[i]['position'] for i in branch], dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    L = cum[-1]
    if L <= 1e-12:
        # degenerate—spread uniformly by index
        return np.linspace(0.0, 1.0, len(branch))
    return (cum / L).astype(float)

def _straighten_branch_to_center(graph: nx.Graph, branch: Branch, center_pos: np.ndarray):
    """
    Move all nodes on the branch onto the straight line from center -> endpoint,
    preserving each node's relative arc-length parameter.
    Assumes branch[0] is the junction center and branch[-1] is the endpoint.
    """
    t = _branch_arc_params(graph, branch)
    end_pos = _center_pos(graph, branch[-1])
    # linear interpolation: p(t) = (1 - t) * center + t * endpoint
    new_pts = (1.0 - t)[:, None] * center_pos[None, :] + t[:, None] * end_pos[None, :]
    for nid, p in zip(branch, new_pts):
        graph.nodes[nid]['position'] = p

def _unique_next_neighbor(graph: nx.Graph, prev: int, curr: int) -> int | None:
    """Return the unique neighbor forward of (prev -> curr), or None if it doesn't exist."""
    forward = [n for n in graph.neighbors(curr) if n != prev]
    return forward[0] if len(forward) == 1 else None

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

# =========================
# 4) X detection (shared-node + center-distance rule)
# =========================

def _nodes_in_record(rec: Junction) -> Set[int]:
    s: Set[int] = set()
    for br in rec['branches']:
        s.update(br)
    s.add(int(rec['node']))
    return s

def find_x_pairs_shared_node_and_distance(
    graph: nx.Graph,
    junctions: JunctionMap,
    max_center_distance: float
) -> List[Tuple[int, int]]:
    """
    Two T/Y junctions form an X pair iff:
      1) Their branch-node sets (including centers) share EXACTLY ONE node.
      2) The Euclidean distance between their centers <= max_center_distance.
    Returns list of (jid_a, jid_b) pairs (each pair once, i<j).
    """
    items = list(junctions.items())
    node_sets = {jid: _nodes_in_record(rec) for jid, rec in items}
    centers = {jid: _center_pos(graph, rec['node']) for jid, rec in items}
    max_d2 = float(max_center_distance) ** 2

    pairs: List[Tuple[int, int]] = []
    for i in range(len(items)):
        jid_i, rec_i = items[i]
        ci = centers[jid_i]
        for j in range(i + 1, len(items)):
            jid_j, rec_j = items[j]
            inter = node_sets[jid_i] & node_sets[jid_j]
            if len(inter) != 1:
                continue
            cj = centers[jid_j]
            if np.sum((ci - cj) ** 2) <= max_d2:
                pairs.append((jid_i, jid_j))
    return pairs


# =========================
# 5) Straighten/merge X pairs (geometry only; no topology edits)
# =========================

def _ensure_branch_center_first(branch: Branch, center_id: int) -> Branch:
    """
    Ensure the branch is oriented [center, ..., endpoint].
    If reversed [endpoint, ..., center], flip in-place and return it.
    If neither end is center, leave as-is (we'll still straighten).
    """
    if not branch:
        return branch
    if branch[0] == center_id:
        return branch
    if branch[-1] == center_id:
        branch.reverse()
        return branch
    return branch

def straighten_x_pairs_to_midpoint(
    graph: nx.Graph,
    junctions: JunctionMap,
    x_pairs: List[Tuple[int, int]],
    mark_as_x: bool = True,
) -> JunctionMap:
    """
    For each (jid_a, jid_b) X pair:
      • Move BOTH junction center nodes to their midpoint.
      • For every branch (from both junctions), move nodes so each branch becomes
        a straight spoke from the shared midpoint to its endpoint, preserving each
        node's arc-length fraction along the branch.
      • Optionally set both records' 'type' to 'X-junction'.
    """
    processed: Set[Tuple[int, int]] = set()

    for jid_a, jid_b in x_pairs:
        if jid_a not in junctions or jid_b not in junctions:
            continue
        if (jid_a, jid_b) in processed or (jid_b, jid_a) in processed:
            continue

        rec_a = junctions[jid_a]
        rec_b = junctions[jid_b]

        n_a, n_b = int(rec_a['node']), int(rec_b['node'])
        p_mid = 0.5 * (_center_pos(graph, n_a) + _center_pos(graph, n_b))

        # move BOTH existing center nodes to the same midpoint
        graph.nodes[n_a]['position'] = p_mid
        graph.nodes[n_b]['position'] = p_mid

        # straighten all branches for both records
        for rec in (rec_a, rec_b):
            c = int(rec['node'])
            for br in rec['branches']:
                _ensure_branch_center_first(br, c)
                _straighten_branch_to_center(graph, br, p_mid)

        if mark_as_x:
            rec_a['type'] = 'X-junction'
            rec_b['type'] = 'X-junction'

        processed.add((jid_a, jid_b))

    return junctions


# =========================
# Orchestrator
# =========================

def classify_junctions(
    graph: nx.Graph,
    angle_thresh: float,
    colinear_thresh: float = 0.95,
    x_center_distance: float = 5.0,
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
    return junctions

