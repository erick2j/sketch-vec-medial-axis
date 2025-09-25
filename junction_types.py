from __future__ import annotations
from typing import Callable, Iterable, Tuple

import numpy as np
import networkx as nx

from distance_to_measure import distance_to_measure_roi_sparse_cpu_numba
from curve_extraction import (
    compute_object_angles,
    fast_medial_axis,
    prune_by_object_angle,
    resample_contours,
    unique_contour_points,
)
from junction_procedure import (
    Junction,
    JunctionRewire,
    JunctionTree,
    build_junction_trees,
    identify_junctions,
    rewire_junction_trees,
)
from image_processing import find_contours

NodeId = int
Edge   = Tuple[NodeId, NodeId]



class StrokeGraph:
    """
    A wrapper for a networkx graph meant for representing a vectorized sketch.
    """

    def __init__(self, measure: np.ndarray, stroke_width: float = 2.0, iso_scale: float = 0.2,
                 pruning_object_angle: float = 5 * np.pi / 6.0, junction_object_angle: float = 2 * np.pi / 6.0):
        self.image_measure = measure

        self.stroke_width = float(stroke_width)
        self.iso_scale = float(iso_scale)
        self.object_angle_pruning_threshold = float(pruning_object_angle)
        self.object_angle_junction_threshold = float(junction_object_angle)

        self.distance_field: np.ndarray | None = None
        self.mass_threshold: float | None = None
        self.distance_threshold: float | None = None

        self.boundary_contours: list[np.ndarray] = []
        self.boundary_points: np.ndarray | None = None

        self.medial_axis: nx.Graph | None = None
        self.pruned_graph: nx.Graph | None = None
        self.stroke_graph: nx.Graph | None = None

        self.junctions: Tuple[Junction, ...] = tuple()
        self.junction_trees: Tuple[JunctionTree, ...] = tuple()
        self.junction_rewires: Tuple[JunctionRewire, ...] = tuple()

        self._run_pipeline_from("distance")



    ##############################################
    #          Stroke Graph Properties           #
    ##############################################

    @property
    def measure(self) -> np.ndarray:
        return self.image_measure

    @property
    def height(self) -> int:
        return self.image_measure.shape[0]

    @property
    def width(self) -> int:
        return self.image_measure.shape[1]

    @property
    def distance_function(self) -> np.ndarray:
        if self.distance_field is None:
            raise ValueError("Distance field has not been computed yet.")
        return self.distance_field

    @property
    def isovalue(self) -> float:
        if self.distance_threshold is None:
            raise ValueError("Iso-value has not been computed yet.")
        return self.distance_threshold

    ##############################################
    #        Update Vectorization Methods        #
    ##############################################

    def update_stroke_width(self, val: float) -> None:
        if np.isclose(val, self.stroke_width):
            return
        self.stroke_width = float(val)
        self._run_pipeline_from("distance")

    def update_iso_scale(self, val: float) -> None:
        if np.isclose(val, self.iso_scale):
            return
        self.iso_scale = float(val)
        self._run_pipeline_from("isovalue")

    def update_distance_threshold(self, val: float) -> None:
        if self.distance_field is None:
            raise ValueError("Distance field must exist before updating the threshold directly.")

        new_threshold = float(val)
        if self.distance_threshold is not None and np.isclose(new_threshold, self.distance_threshold):
            return

        dmin = float(np.min(self.distance_field))
        dmax = float(np.max(self.distance_field))
        if dmax > dmin:
            self.iso_scale = (new_threshold - dmin) / (dmax - dmin)

        self.distance_threshold = new_threshold
        self._run_pipeline_from("boundary")

    def update_pruning_threshold(self, val: float) -> None:
        if np.isclose(val, self.object_angle_pruning_threshold):
            return
        self.object_angle_pruning_threshold = float(val)
        self._run_pipeline_from("pruned_graph")

    def update_junction_threshold(self, val: float) -> None:
        if np.isclose(val, self.object_angle_junction_threshold):
            return
        self.object_angle_junction_threshold = float(val)
        self._run_pipeline_from("stroke_graph")



    ##############################################
    #            "Private" Methods               #
    ##############################################

    def _run_pipeline_from(self, stage: str) -> None:
        stages = tuple(self._pipeline())
        valid_names = {name for name, _ in stages}
        if stage not in valid_names:
            raise ValueError(f"Unknown pipeline stage '{stage}'. Expected one of {sorted(valid_names)}")

        should_run = False
        for name, fn in stages:
            if not should_run and name == stage:
                should_run = True
            if should_run:
                fn()

    def _pipeline(self) -> Iterable[tuple[str, Callable[[], None]]]:
        return (
            ("distance", self._compute_distance_field),
            ("isovalue", self._compute_isovalue),
            ("boundary", self._compute_boundary_contours),
            ("medial_axis", self._compute_medial_axis),
            ("object_angles", self._compute_medial_object_angles),
            ("pruned_graph", self._compute_pruned_graph),
            ("stroke_graph", self._compute_stroke_graph),
        )

    def _compute_distance_field(self) -> None:
        self.distance_field, self.mass_threshold = distance_to_measure_roi_sparse_cpu_numba(
            self.image_measure,
            0.5 * self.stroke_width,
        )

    def _compute_isovalue(self) -> None:
        if self.distance_field is None:
            raise ValueError("Distance field must be computed before the iso-value.")
        dmin = float(np.min(self.distance_field))
        dmax = float(np.max(self.distance_field))
        self.distance_threshold = dmin + self.iso_scale * (dmax - dmin)

    def _compute_boundary_contours(self) -> None:
        if self.distance_threshold is None:
            raise ValueError("Iso-value must be computed before boundary extraction.")

        contours = find_contours(self.distance_function, self.isovalue, fully_connected="high")
        valid_contours = [c for c in contours if c.ndim == 2 and len(c) >= 3]

        if not valid_contours:
            self.boundary_contours = []
            self.boundary_points = np.empty((0, 2))
            return

        self.boundary_contours = resample_contours(valid_contours, 0.5, 1e-5)

        self.boundary_points = unique_contour_points(self.boundary_contours)

    def _compute_medial_axis(self) -> None:
        if not self.boundary_contours:
            self.medial_axis = nx.Graph()
            return

        self.medial_axis = fast_medial_axis(self.boundary_contours, self.distance_function, self.isovalue)

    def _compute_medial_object_angles(self) -> None:
        if self.medial_axis is None or self.medial_axis.number_of_edges() == 0:
            return

        if self.boundary_points is None or len(self.boundary_points) == 0:
            self.boundary_points = unique_contour_points(self.boundary_contours)

        compute_object_angles(self.medial_axis, self.boundary_points)

    def _compute_pruned_graph(self) -> None:
        if self.medial_axis is None:
            self.pruned_graph = nx.Graph()
            return

        self.pruned_graph = prune_by_object_angle(
            self.medial_axis,
            self.object_angle_pruning_threshold,
        )

    def _compute_stroke_graph(self) -> None:
        if self.pruned_graph is None or self.pruned_graph.number_of_nodes() == 0:
            self.stroke_graph = nx.Graph()
            self.junctions = tuple()
            self.junction_trees = tuple()
            self.junction_rewires = tuple()
            return

        graph = self.pruned_graph.copy()
        self.junctions = identify_junctions(
            graph,
            self.object_angle_junction_threshold,
        )
        self.junction_trees = build_junction_trees(
            graph,
            self.junctions,
            self.stroke_width,
        )
        self.stroke_graph, rewires = rewire_junction_trees(
            graph,
            self.junction_trees,
            colinear_dot=0.92,
        )
        self.junction_rewires = rewires
