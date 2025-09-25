from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx


NodeId = int
Edge = Tuple[NodeId, NodeId]


def _ordered_edge(u: NodeId, v: NodeId) -> Edge:
    return (u, v) if u <= v else (v, u)


@dataclass(frozen=True)
class JunctionBranch:
    nodes: Tuple[NodeId, ...]
    edges: Tuple[Edge, ...]


@dataclass(frozen=True)
class Junction:
    root: NodeId
    branches: Tuple[JunctionBranch, ...]


def identify_junctions(
    pruned_graph: nx.Graph,
    angle_threshold: float,
) -> Tuple[Junction, ...]:
    """
    Group pruned-graph edges into junction trees grown from high-degree centers.

    Each junction is rooted at a node whose degree is at least three and has
    branches that extend while the traversed edges stay below the supplied
    angle threshold. The input graph is treated as read-only; only junction
    descriptions are returned.
    """

    graph = pruned_graph
    if graph.number_of_nodes() == 0:
        return tuple()

    # sort high-degree nodes for deterministic results
    ordered_roots = tuple(sorted(
        n for n in graph.nodes if graph.degree[n] >= 3
    ))
    if not ordered_roots:
        return tuple()

    queue: deque[Tuple[NodeId, NodeId]] = deque()
    edge_owner: Dict[Edge, NodeId] = {}
    node_owner: Dict[NodeId, NodeId] = {}
    parent_of: Dict[NodeId, Optional[NodeId]] = {}

    for root in ordered_roots:
        node_owner[root] = root
        parent_of[root] = None
        queue.append((root, root))

    while queue:
        root, node = queue.popleft()
        parent_node = parent_of.get(node)

        for neighbor in graph.neighbors(node):
            if parent_node is not None and neighbor == parent_node:
                continue

            edge = _ordered_edge(node, neighbor)
            if edge in edge_owner:
                continue

            angle = float(graph.edges[node, neighbor].get("object angle", 0.0))
            if angle > angle_threshold:
                continue

            owner_neighbor = node_owner.get(neighbor)
            if owner_neighbor is None:
                node_owner[neighbor] = root
                parent_of[neighbor] = node
                edge_owner[edge] = root
                queue.append((root, neighbor))
            elif owner_neighbor == root:
                # Adding this edge would introduce a cycle in the root's tree.
                continue
            else:
                continue

    nodes_with_children = {parent for parent in parent_of.values() if parent is not None}

    junctions: List[Junction] = []
    for root in ordered_roots:
        owned_nodes = {node for node, owner in node_owner.items() if owner == root}
        leaves = [n for n in owned_nodes if n != root and n not in nodes_with_children]

        branches: List[JunctionBranch] = []
        for leaf in leaves:
            path_nodes = [leaf]
            current = leaf

            while current != root:
                parent_node = parent_of.get(current)
                if parent_node is None:
                    path_nodes = []
                    break
                path_nodes.append(parent_node)
                current = parent_node

            if not path_nodes or path_nodes[-1] != root:
                continue

            path_nodes.reverse()
            path_edges = [
                (path_nodes[i], path_nodes[i + 1])
                for i in range(len(path_nodes) - 1)
            ]

            if path_edges:
                branches.append(
                    JunctionBranch(
                        nodes=tuple(path_nodes),
                        edges=tuple(path_edges),
                    )
                )

        if not branches:
            continue

        junctions.append(
            Junction(
                root=root,
                branches=tuple(branches),
            )
        )

    return tuple(junctions)
