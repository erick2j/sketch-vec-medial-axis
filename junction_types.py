from __future__ import annotations
from typing import Tuple

import numpy as np
import networkx as nx

NodeId = int
Edge   = Tuple[NodeId, NodeId]



class StrokeGraph:
    """
    A wrapper for a networkx graph meant for representing a vectorized sketch.
    """
    graph: nx.Graph
    shape: Tuple[int, int] # (height, width)

    def __init__(self, image):
        pass

    @property
    def image(self):
        pass

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def height(self) -> int:
        return self.shape[1]








