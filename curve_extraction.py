import numpy as np
import logging
import time
import triangle as tr
import networkx as nx
from scipy.spatial import KDTree 
from shapely.geometry import Point, Polygon, LineString
from shapely.geometry import Point, LinearRing, MultiPolygon
from shapely.prepared import prep

logger = logging.getLogger(__name__)

######################################
#####  CONTOUR FIXING UTILITIES  ##### 
######################################

def resample_contours(contours : list[np.ndarray], spacing: float  = 0.5, tol: float = 1e-5) -> list[np.ndarray]:
    """
    Resamples each contour to have equal spacing after removing near-duplicate points.
    Also prints how many vertices were removed in total.
    """
    start = time.time()
    resampled_contours = []

    for contour in contours:
        # check proper dimension
        if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 3:
            raise ValueError("Each contour must be a 2D NumPy array with shape (N,2) with N >= 4")
            continue

        # compute the difference in x and y for all consecutive points 
        dxdy = np.diff(contour, axis=0)

        # compute total arc length along contouor
        distances = np.hypot(dxdy[:, 0], dxdy[:, 1])
        arc_length = np.concatenate(([0], np.cumsum(distances)))
        total_length = arc_length[-1]

        # generate evenly spaced samples along total length of contour with provided spacing
        num_points = max(int(np.floor(total_length / spacing)) + 1, 2)
        even_spaced = np.linspace(0, total_length, num_points)

        # linearly interpolate to resample points along the contour
        even_rows = np.interp(even_spaced, arc_length, contour[:, 0])
        even_cols = np.interp(even_spaced, arc_length, contour[:, 1])
        resampled = np.stack((even_rows, even_cols), axis=-1)

        # add to final list of contours
        resampled_contours.append(resampled)

    end = time.time()
    elapsed_ms = (end-start) * 1000

    logger.info(f"[CONTOUR CLEANING] Resampled {len(resampled_contours)} contours in {elapsed_ms:.2f} ms")

    return resampled_contours

######################################
#####  MEDIAL AXIS STUFF         ##### 
######################################

def unique_contour_points(contours: list[np.ndarray], tol: float=1e-5):
    """
    Extracts all of the unique points in a list of contours.

    NOTES: Assumes that vertices are all in CW/CCW order so only possible duplicates
    are at the beginning and end.
    """

    start = time.time()
    num_original_points = 0
    num_unique_points   = 0

    unique_points = []

    for contour in contours:
        num_original_points += len(contour)
        if len(contour) < 2:
            unique_points.append(contour)
            continue

        if np.linalg.norm(contour[0] - contour[-1]) <= tol:
            contour = contour[:-1]

        unique_points.append(contour)
        num_unique_points += len(contour)

    end = time.time()
    logger.info(f"[CONTOUR CLEANING] Removed {num_original_points - num_unique_points} points from the original contour set with {num_original_points} [{((num_original_points - num_unique_points) / num_original_points)*100:.2f}%].")

    if unique_points:
        return np.vstack(unique_points)
    else:
        return np.empty((0,2), dtype=float)


def bounding_box_voronoi_pruning(positions: np.ndarray, edges: list[tuple[int, int]], boundary_points: np.ndarray):
    """
    Prunes the points and edges of a voronoi diagram by using the bounding box of the 
    input boundary points.
    """
    start = time.perf_counter()
    # compute corners of bounding box
    min_coords = boundary_points.min(axis=0)
    max_coords = boundary_points.max(axis=0)
    # keep only positions inside the bounding box
    inside_points = np.all((positions >= min_coords) & (positions <= max_coords), axis=1)

    # reindex all vertices to start from 0 to N-1 
    keep_indices = np.where(inside_points)[0]
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(keep_indices)}

    # create positions dictionary based on reindexing
    pos_dict = {
            new_id : tuple(positions[old_id])
            for old_id, new_id in old_to_new.items()
            }

    # create new edges list based on reindexing
    edge_list = [
            tuple(sorted((old_to_new[i], old_to_new[j])))
            for i, j in edges
            if i in old_to_new and j in old_to_new
            ]


    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"[PRUNING] Peformed bounding box pruning in {elapsed_ms} ms")

    return pos_dict, edge_list 

def fast_voronoi_diagram(points):
    """
    Constructs the Voronoi diagram from a set of boundary points.

    Parameters
    ----------
    points : ndarray
        Array of shape (N, 2) containing (row, col) coordinates of boundary points.
    """       
    # compute voronoi diagram using the Triangle library
    positions, edges, _, _ = tr.voronoi(points[:, [0,1]])
    # prune voronoi diagram using the bounding box of input seed points
    pos_dict, pruned_edges = bounding_box_voronoi_pruning(positions[:,[0,1]], edges, points)
    # construct voronoi diagram
    G = nx.Graph()
    G.add_edges_from(pruned_edges)
    nx.set_node_attributes(G, pos_dict, 'position')
    return G


def build_bounding_polygon(contours):
    exteriors, interiors = [], []
    for c in contours:
        if len(c) < 4: continue
        ring = LinearRing(c)
        if ring.is_ccw:
            exteriors.append(Polygon(ring))
        else:
            interiors.append(Polygon(ring))
    polygons = []
    for exterior in exteriors:
        holes = [hole.exterior for hole in interiors if exterior.contains(hole)]
        polygons.append(Polygon(exterior.exterior, holes) if holes else exterior)
    return MultiPolygon(polygons)


def fast_medial_axis(contours, field: np.ndarray, isovalue: float):
    all_points = np.vstack(contours)
    positions, edges, _, _ = tr.voronoi(all_points[:, [0,1]])

    pos_array = np.array(positions)
    row_idx = np.round(pos_array[:, 0]).astype(int)
    col_idx = np.round(pos_array[:, 1]).astype(int)

    h, w = field.shape
    valid_mask = np.full(len(pos_array), False, dtype=bool)

    inside_bounds = (
        (0 <= row_idx) & (row_idx < h) &
        (0 <= col_idx) & (col_idx < w)
    )

    if np.any(inside_bounds):
        idx_in_bounds = np.where(inside_bounds)[0]
        valid_mask[idx_in_bounds] = field[row_idx[idx_in_bounds], col_idx[idx_in_bounds]] <= isovalue

    valid_indices = np.where(valid_mask)[0]
    pos_dict = {i: tuple(pos_array[i]) for i in valid_indices}

    edge_array = np.array(edges)
    edge_mask = np.isin(edge_array[:, 0], valid_indices) & np.isin(edge_array[:, 1], valid_indices)
    valid_edges = edge_array[edge_mask]

    G = nx.Graph()
    G.add_edges_from((int(i), int(j)) for i, j in valid_edges)
    nx.set_node_attributes(G, pos_dict, 'position')

    return G

def compute_object_angles(graph: nx.Graph, boundary_points: np.ndarray, k: int = 2) -> None:
    """
    Computes the object angle for each edge in the medial axis graph and stores
    it as an edge attribute 'object angle'.

    Parameters
    ----------
    graph : networkx.Graph
        The medial axis graph with node positions stored under the 'position' attribute.
    boundary_points : np.ndarray
        Array of shape (N, 2) containing (row, col) coordinates of the boundary points.
    k : int
        Number of nearest neighbors to use (typically 2).
    """
    positions = {n: np.array(p, dtype=float) for n, p in nx.get_node_attributes(graph, 'position').items()}
    if not positions or graph.number_of_edges() == 0:
        return

    node_ids = sorted(positions)
    pos_array = np.array([positions[n] for n in node_ids])
    node_index = {n: i for i, n in enumerate(node_ids)}

    # Build KDTree from boundary points
    kdtree = KDTree(boundary_points)

    # Collect all edges and compute their midpoints
    edge_list = [(u, v) for u, v in graph.edges]
    midpoints = np.array([
        (positions[u] + positions[v]) / 2.0 for u, v in edge_list
    ])

    # Find two closest boundary points to each midpoint
    dists, idxs = kdtree.query(midpoints, k=k)
    a = boundary_points[idxs[:, 0]]
    b = boundary_points[idxs[:, 1]]

    # Compute vectors to nearest boundary points
    v1 = a - midpoints
    v2 = b - midpoints

    # Compute dot product and norms
    dot = np.sum(v1 * v2, axis=1)
    norm_prod = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    cos_theta = np.clip(dot / norm_prod, -1.0, 1.0)

    # Compute object angle
    angles = np.arccos(cos_theta) / 2.0

    # Assign object angles as edge attributes
    attr_dict = {
        tuple(sorted((u, v))): float(angle) for (u, v), angle in zip(edge_list, angles)
    }
    nx.set_edge_attributes(graph, attr_dict, name='object angle')


def prune_by_object_angle(graph: nx.Graph, angle_threshold: float) -> nx.Graph:
    """
    Prunes leaf paths in the medial axis graph based on the precomputed
    'object angle' edge attribute. Stops pruning when the angle exceeds threshold.

    Parameters
    ----------
    graph : nx.Graph
        Medial axis graph with 'object angle' on each edge.
    angle_threshold : float
        Angle threshold in radians.

    Returns
    -------
    nx.Graph
        A pruned copy of the input graph.
    """
    G = graph.copy()
    removed = set()
    leaves = [n for n in G.nodes if G.degree[n] == 1]

    while leaves:
        current = leaves.pop(0)
        if current in removed or current not in G:
            continue

        neighbors = list(G.neighbors(current))
        if not neighbors:
            continue

        neighbor = neighbors[0]
        edge_key = tuple(sorted((current, neighbor)))

        angle = G.edges[current, neighbor].get('object angle', 0)

        if angle < angle_threshold:
            G.remove_node(current)
            removed.add(current)

            if G.degree(neighbor) == 1 and neighbor not in removed:
                leaves.append(neighbor)

    return G



######################################
###  JUNCTION LABELLING UTILITIES  ### 
######################################

def ordered_edge(u, v):
    return (min(u, v), max(u, v))


def compute_branch_tangents(branches, graph):
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
            logger.warning(f"[JUNCTION DETECTION] Branch {i} has fewer than 3 nodes. Using last 2 for tangent.")
            p1 = np.array(graph.nodes[branch[-1]]['position'])
            p2 = np.array(graph.nodes[branch[-2]]['position'])
        # this should probably throw an exception, but skipping for now
        else:
            logger.warning(f"[JUNCTION DETECTION] Branch {i} is too short to compute a tangent. Skipping.")
            continue

        # append a normalized tangent for each branch
        tangent = p2 - p1
        norm = np.linalg.norm(tangent)
        # check to see if something has gone horribly wrong
        if norm == 0:
            logger.warning(f"[JUNCTION DETECTION] Branch {i} has zero-length direction vector. Skipping.")
            continue

        # normalize the tangent
        tangent /= norm
        # append tangent AND origin to tangents list 
        tangents.append((p1, tangent))
    return tangents

def compute_colinear_map(tangents, dot_threshold=np.sqrt(3)/ 2.0):
    """
    Takes in a list of tangents and returns a hash map that tells us what other 
    tangents are roughly colinear. 

    dot product 0.9848 means the angle is within 10 degrees
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


def classify_junction(colinear_map):
    """
    Determines whether a junction is a Y or a T junction by 
    looking at the number of colinear pairs there are.
    A T-junction will have exactly one colinear pair
    """
    # count unique colinear pairs (each appears twice in the map)
    unique_colinear_pairs = sum(len(partners) for partners in colinear_map.values()) // 2
    return 'T-junction' if unique_colinear_pairs == 1 else 'Y-junction'

def classify_junction_by_angle(junction_node, branches, graph):
    """
    Classify junction as Y- or T-junction based on angles between endpoint vectors.

    Parameters
    ----------
    junction_node : int
        Node ID of the junction point.
    branches : List[List[int]]
        List of 3 branches (each a list of node IDs from junction to endpoint).
    graph : networkx.Graph
        Graph with node positions.

    Returns
    -------
    str
        'Y-junction' or 'T-junction'
    """
    if len(branches) != 3:
        return None

    J = np.array(graph.nodes[junction_node]['position'])

    vectors = []
    for branch in branches:
        if len(branch) < 2:
            return None  # malformed branch
        end_node = branch[-1]
        P = np.array(graph.nodes[end_node]['position'])
        v = P - J
        if np.linalg.norm(v) == 0:
            return None  # skip degenerate case
        vectors.append(v / np.linalg.norm(v))

    # Compute dot products between all pairs
    dot_products = [np.dot(vectors[i], vectors[j]) for i in range(3) for j in range(i+1, 3)]

    if any(dot > 0 for dot in dot_products):
        return 'Y-junction'
    else:
        return 'T-junction'


def extended_line(p, d, extension=10.0):
    d = d / np.linalg.norm(d)
    return LineString([p - d * extension, p + d * extension])


def identify_junctions(graph, object_angle_threshold):
    nx.set_edge_attributes(graph, False, 'collapsible')
    nx.set_edge_attributes(graph, 'None', 'junction_type')

    junctions = {}
    junction_id = 0
    used_edges = set()

    # this needs to go here because we need to keep track of
    # used edges
    def trace_branch(u, v):
        """
        Helper function to mark branches until we hit a edge with
        high enough object angle or a junction node
        """
        branch = [u, v]
        prev, current = u, v
        while graph.degree(current) < 3:
            nbrs = [n for n in graph.neighbors(current) if n != prev]
            if len(nbrs) != 1:
                break
            nxt = nbrs[0]
            edge = ordered_edge(current, nxt)
            if edge in used_edges:
                break
            angle = graph[current][nxt].get('object angle', 0)
            if angle > object_angle_threshold:
                break
            branch.append(nxt)
            prev, current = current, nxt
        return branch

    # MARKING CODE 

    for node in graph.nodes:
        # im only going to look at T and Y junction for now
        if graph.degree(node) !=  3:
            continue

        # trace a branch for each outgoing edge
        branches = [trace_branch(node, nbr) for nbr in graph.neighbors(node)]
        # compute tangnets for each branch
        tangents = compute_branch_tangents(branches, graph)
        # identify which tangents are colinear
        colinear_map = compute_colinear_map(tangents)
        # classify junctions based on how many branches are colinear
        jtype = classify_junction(colinear_map)

        # mark the edges for visualization 
        for branch in branches:
            for u, v in zip(branch, branch[1:]):
                graph[u][v]['collapsible'] = True
                graph[u][v]['junction_type'] = jtype
                used_edges.add(ordered_edge(u, v))

        # keep track of the traits for each junction
        junctions[junction_id] = {
            'node': node,
            'branches': branches,
            'type': jtype,
            'colinear_map': colinear_map,
            'tangents': tangents
        }
        junction_id += 1

    return junctions


def collapse_t_junction(graph, junc_data):
    """
    Procedure for collapsing T-junction
    """
    node = junc_data['node']
    branches = junc_data['branches']
    colinear_map = junc_data['colinear_map']
    tangents = junc_data['tangents']

    colinear_pairs = [(i, j) for i in range(3) for j in colinear_map[i] if j > i]
    if not colinear_pairs:
        return

    i_col, j_col = colinear_pairs[0]
    k_noncol = ({0, 1, 2} - {i_col, j_col}).pop()

    p_i = np.array(graph.nodes[branches[i_col][-1]]['position'])
    p_j = np.array(graph.nodes[branches[j_col][-1]]['position'])
    p_k, d_k = tangents[k_noncol]

    line_colinear = extended_line((p_i + p_j) / 2.0, p_j - p_i)
    line_noncol = extended_line(p_k, d_k)
    inter = line_colinear.intersection(line_noncol)

    if inter.is_empty or inter.geom_type != 'Point':
        return

    intersection_point = np.array(inter.coords[0])
    move_branch_nodes_toward_point(branches, intersection_point, graph, node)


def collapse_y_junction(graph, junc_data):
    """
    Procedure for collapsing Y-junctions
    """
    node = junc_data['node']
    branches = junc_data['branches']
    tangents = junc_data['tangents']
    colinear_map = junc_data['colinear_map']

    lines = [extended_line(p, d) for p, d in tangents]
    intersections = []

    for i in range(3):
        for j in range(i + 1, 3):
            if j in colinear_map[i]:
                continue
            inter = lines[i].intersection(lines[j])
            if not inter.is_empty and inter.geom_type == 'Point':
                intersections.append(np.array(inter.coords[0]))

    if not intersections:
        return

    intersection_point = np.mean(intersections, axis=0)
    move_branch_nodes_toward_point(branches, intersection_point, graph, node)


def collapse_to_tangent_intersection(graph, junctions):
    for junc_id, data in junctions.items():
        if data['type'] == 'T-junction':
            collapse_t_junction(data, graph)
        elif data['type'] == 'Y-junction':
            collapse_y_junction(data, graph)


def repair_junctions(graph, object_angle_thresh):
    junctions = identify_junctions(graph, object_angle_thresh)
    return junctions
    #collapse_to_tangent_intersection(graph, junctions)


