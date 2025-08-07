import numpy as np
import logging
import time
import triangle as tr
import networkx as nx
from scipy.spatial import Voronoi
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
    G = nx.Graph()
    # compute voronoi diagram using the Triangle library
    positions, edges, _, _ = tr.voronoi(points[:, [0,1]])
    # prune voronoi diagram using the bounding box of input seed points
    pos_dict, pruned_edges = bounding_box_voronoi_pruning(positions[:,[0,1]], edges, points)
    # construct voronoi diagram
    G.add_edges_from(pruned_edges)
    nx.set_node_attributes(G, pos_dict, 'position')
    return G


def fast_medial_axis(contours):
    # Step 1: Flatten all unique boundary points
    all_points = np.vstack(contours)
    
    # Step 2: Compute Voronoi diagram using `triangle`
    positions, edges, _, _ = tr.voronoi(all_points[:, [0,1]])

    # Step 3: Clean/prune Voronoi edges based on polygon containment
    pos_dict = {i: pt for i, pt in enumerate(positions)}
    G = nx.Graph()
    G.add_edges_from(edges)
    nx.set_node_attributes(G, pos_dict, 'position')

    # Step 4: Build MultiPolygon from CCW/CW contours
    poly = build_bounding_polygon(contours)
    prepared_poly = prep(poly)

    # Step 5: Keep only nodes inside polygon
    inside = [n for n, p in pos_dict.items() if prepared_poly.contains(Point(p))]
    medial_axis = G.subgraph(inside).copy()
    return medial_axis

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

