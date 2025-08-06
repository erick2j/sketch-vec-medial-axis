import numpy as np
import logging
import time
import networkx as nx
from scipy.spatial import Voronoi
from shapely.geometry import Point, Polygon, LineString

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
