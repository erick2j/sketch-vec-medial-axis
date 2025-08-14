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

