# This Python file uses the following encoding: utf-8
import numpy as np
import random
import string
from PySide6 import QtWidgets
import matplotlib as plt
from matplotlib.figure import Figure
from matplotlib.colors import Normalize, Colormap
from matplotlib.patches import Polygon, PathPatch
from matplotlib.path import Path
from matplotlib.collections import LineCollection 
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg 
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

class MPLWidget(QtWidgets.QWidget):
    """
    A Qt Widget that manages a Matplotlib Figure. 
    This is the main visualizer for the application.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        # initialize matplotlib figure
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_axis_off()
        self.axes.set_frame_on(True)
        #self.figure.set_layout_engine('constrained')

        # store the current image being displayed
        self.current_image = None

        # initialize a bunch of artists 
        self.levelset_contour_plots = []

        # maintain a dictionary (attribute_name : artist)
        self.artists = {}
        # maintain a dictionary (attribute_name : colormap)
        self.colormaps = {}

        # set layout of the widget to embed the canvas
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)

        # add matplotlib toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        # set layout and draw the canvas
        self.setLayout(layout)
        self.canvas.draw()

    def clear_all(self):
        '''
        Clears all figures and stored data from the widget.
        '''
        if self.current_image is not None:
            self.current_image.remove()
            self.current_image = None

        self.axes.clear()
        self.axes.axis('image')
        self.axes.axis('off')
        # clear stored data
        self.levelset_contour_plots = []
        self.artists  = {}
        self.colormaps = {}

        # redraw canvas after clearing everything.
        self.canvas.draw()


    def show_image(self, img, cmap='pink', normalize=False):
        '''
        Display an image on the axes.
        '''

        if img is None:
            self.show_blank_image()
            return

        # clear previous image
        if self.current_image is not None:
            self.current_image.remove()

        if normalize:
            norm = Normalize(vmin=np.min(img), vmax=np.max(img), clip=False)
        else:
            norm = None

        # show the input image
        self.current_image = self.axes.imshow(img, cmap=cmap, norm=norm,
                                              interpolation='None', origin='upper')
        # display options
        self.axes.axis('image')
        self.axes.axis('off')
        self.axes.set_aspect('equal')

        # redraw canvas
        self.canvas.draw()

    def show_blank_image(self, shape=None):
        """
        Displays a blank image on the widget.
        """

        if shape is None and self.current_image is not None:
            shape = self.current_image.get_array().shape
        else:
            shape = (512, 512)

        if len(shape) == 2:
            shape = (*shape, 3)

        blank_img = np.ones(shape, dtype=np.uint8) * 255

        if self.current_image is not None:
            self.current_image.remove()

        self.current_image = self.axes.imshow(blank_img, cmap='gray', interpolation='None', origin='upper')
        self.axes.set_xlim(0, blank_img.shape[1])
        self.axes.set_ylim(blank_img.shape[0], 0)
        self.axes.axis('image')
        self.axes.axis('off')
        
        #self.axes.set_aspect('equal')
        self.canvas.draw()


    def plot_contours(self, contours):
        """
        Vectorized + fast rendering of all contours.
        """
        # Clear previous plots
        for contour_plot in self.levelset_contour_plots:
            contour_plot.remove()
        self.levelset_contour_plots = []

        path_vertices = []
        path_codes = []

        all_scatters = []

        # convert contours to paths
        for contour in contours:
            contour = contour[:, [1, 0]] 

            if not np.array_equal(contour[0], contour[-1]):
                contour = np.vstack([contour, contour[0]])

            n_points = len(contour)
            path_vertices.extend(contour)
            path_codes.extend([Path.MOVETO] + [Path.LINETO] * (n_points - 2) + [Path.CLOSEPOLY])

        # plot curves
        compound_path = Path(np.array(path_vertices), path_codes)
        patch = PathPatch(compound_path, facecolor='#00FFFF', edgecolor='black', lw=1.0, alpha=0.1, zorder=1)
        self.axes.add_patch(patch)
        self.levelset_contour_plots.append(patch)

        # plot points
        all_points = np.vstack([c[:, [1, 0]] for c in contours])  
        scatter = self.axes.scatter(all_points[:, 0], all_points[:, 1], color='#00003f', s=1, zorder=2)
        self.levelset_contour_plots.append(scatter)

        self.canvas.draw()

   
    def hide_levelset_contours(self):
        '''
        Remove plotted voronoi centers
        '''
        for contour_plot in self.levelset_contour_plots:
            contour_plot.remove()
        self.levelset_contour_plots = []
        self.canvas.draw()


    def plot_voronoi_diagram(self, graph, color='blue', linewidth=1.0):
        '''
        Plots the Voronoi diagram from a VectorSketch 
        '''
        if graph is None or graph.number_of_edges() == 0:
            return

        # make sure to clear any previously drawn voronoi diagram
        self.hide_voronoi_diagram()

        # Extract node positions into array (sorted by node index for consistency)
        node_ids = sorted(graph.nodes)
        id_to_index = {node: i for i, node in enumerate(node_ids)}
        positions = np.array([graph.nodes[n]['position'] for n in node_ids])  # shape (N, 2)

        # Build edge list in index form and convert (row, col) → (x, y)
        edge_array = np.array([
            [id_to_index[u], id_to_index[v]] for u, v in graph.edges
        ])
        segments = positions[edge_array]  # shape (E, 2, 2)
        segments = segments[:, :, ::-1]   # convert to (x, y) for display

        # Plot with LineCollection
        lines = LineCollection(segments, colors=color, linewidths=linewidth, zorder=3)
        self.axes.add_collection(lines)
        self.artists['voronoi_diagram'] = lines
        self.canvas.draw()


    def plot_medial_axis(self, graph, color='red', linewidth=1.0, vertex_color='black', vertex_size=1):
        """
        Plots medial axis edges and vertices.
        """
        if graph is None or graph.number_of_edges() == 0:
            return

        self.hide_medial_axis()

        # Extract node positions into array (sorted by node index for consistency)
        node_ids = sorted(graph.nodes)
        id_to_index = {node: i for i, node in enumerate(node_ids)}
        positions = np.array([graph.nodes[n]['position'] for n in node_ids])  # shape (N, 2)

        # Build edge list in index form and convert (row, col) → (x, y)
        edge_array = np.array([
            [id_to_index[u], id_to_index[v]] for u, v in graph.edges
        ])
        segments = positions[edge_array]  # shape (E, 2, 2)
        segments = segments[:, :, ::-1]   # convert to (x, y) for display

        # Plot edges
        lines = LineCollection(segments, colors=color, linewidths=linewidth, zorder=3)
        self.axes.add_collection(lines)

        # Plot vertices
        xs, ys = positions[:, 1], positions[:, 0]  # flip to (x, y)
        vertex_plot = self.axes.scatter(xs, ys, c=vertex_color, s=vertex_size, zorder=4)

        # Store references
        self.artists['medial_axis_edges'] = lines
        self.artists['medial_axis_vertices'] = vertex_plot 
        self.canvas.draw()



    def plot_medial_axis_object_angles(self, graph, colormap='jet', linewidth=1.0, vmin=None, vmax=None):
        """
        Plots the medial axis with color mapped to the 'object angle' attribute on each edge.
        """

        if graph is None or graph.number_of_edges() == 0:
            return

        self.hide_medial_axis_object_angles()

        # Get node positions and object angles
        node_ids = sorted(graph.nodes)
        id_to_index = {node: i for i, node in enumerate(node_ids)}
        positions = np.array([graph.nodes[n]['position'] for n in node_ids])

        edge_list = list(graph.edges)
        edge_array = np.array([[id_to_index[u], id_to_index[v]] for u, v in edge_list])
        segments = positions[edge_array][:, :, ::-1]  # (E, 2, 2) in (x, y)

        # Extract object angles
        angles = np.array([graph.edges[u, v].get('object angle', 0.0) for u, v in edge_list])

        # Normalize colors
        norm = Normalize(vmin=vmin if vmin is not None else angles.min(),
                         vmax=vmax if vmax is not None else angles.max())
        cmap = plt.colormaps.get_cmap(colormap)
        colors = cmap(norm(angles))

        # Create colored LineCollection
        lines = LineCollection(segments, colors=colors, linewidths=linewidth, zorder=3)
        self.axes.add_collection(lines)
        self.artists['medial_axis_object_angles'] = lines
        self.canvas.draw()


    def plot_medial_axis_junctions(self, graph, junctions):
        """
        Plot junctions on the Matplotlib canvas.
        Assumes `junctions` is a dict of Junction dataclasses.
        """
        # Clear previous junction plots
        self.hide_medial_axis_junctions()
        self.artists['junctions'] = []

        color_map = {
            'T-junction': 'red',
            'Y-junction': 'blue',
            'X-junction': 'green',
            'unknown':    'cyan',
        }

        for junc in junctions.values():
            center = junc.center_node
            jtype  = junc.type
            color  = color_map.get(jtype, 'cyan')

            # Plot each branch
            for br in junc.branches:
                path = br.path_nodes
                if len(path) < 2:
                    continue
                pts = [np.array(graph.nodes[n]['position'])[::-1] for n in path]
                segments = [np.vstack([pts[i], pts[i+1]]) for i in range(len(pts)-1)]
                lc = LineCollection(segments, colors=color, linewidths=2, zorder=1)
                self.axes.add_collection(lc)
                self.artists['junctions'].append(lc)

            # Plot center point
            cpos = np.array(graph.nodes[center]['position'])[::-1]
            sc = self.axes.scatter([cpos[0]], [cpos[1]], s=10, c=color, edgecolors='k', zorder=1)
            self.artists['junctions'].append(sc)

        self.canvas.draw()


    def plot_junction_subtrees(
        self,
        graph,
        subtrees,
        linewidth=2.0,
        cmap_name='tab20',
        alpha=0.9,
        show_nodes=False,
        node_size=8,
        node_edgecolor='black',
    ):
        """
        Draw each subtree in a distinct color.

        Parameters
        ----------
        graph : nx.Graph
            The full graph containing positions at each node: G.nodes[n]['position'] = (row, col).
        subtrees : iterable
            Each item may be:
              - a dataclass/object with attributes .edges (iterable of (u,v)) and .nodes (iterable of n),
              - a dict with keys 'edges' and 'nodes',
              - a tuple (nodes, edges).
        linewidth : float
            Line width for subtree edges.
        cmap_name : str
            Matplotlib categorical colormap name (e.g., 'tab20', 'tab10', 'Set3').
        alpha : float
            Edge alpha for visibility.
        show_nodes : bool
            If True, also scatter the nodes of each subtree in the same color.
        node_size : float
            Marker size for nodes if show_nodes=True.
        node_edgecolor : str
            Edge color for node markers (for contrast).
        """
        if graph is None:
            return
        if not subtrees:
            return

        # Remove any previously drawn subtrees first
        self.hide_junction_subtrees()

        # Ensure storage
        self.artists['junction_subtrees'] = []

        # Get deterministic ordering of node positions (not strictly required here)
        # but we’ll fetch directly per edge/node to avoid extra mapping.

        # Colormap for distinct colors
        cmap = plt.colormaps.get_cmap(cmap_name)
        num_colors = getattr(cmap, 'N', 20)

        def _coerce_subtree(st):
            """Return (nodes_set, edges_set) from various subtree shapes."""
            if hasattr(st, 'nodes') and hasattr(st, 'edges'):
                return set(st.nodes), set(st.edges)
            if isinstance(st, dict) and 'nodes' in st and 'edges' in st:
                return set(st['nodes']), set(st['edges'])
            if isinstance(st, tuple) and len(st) == 2:
                n, e = st
                return set(n), set(e)
            raise TypeError("Subtree must have 'nodes' and 'edges' (object, dict, or (nodes, edges)).")

        for i, st in enumerate(subtrees):
            nodes_set, edges_set = _coerce_subtree(st)
            color = cmap(i % num_colors)

            # Build edge segments (convert stored (row, col) -> (x, y) = (col, row))
            segs = []
            for (u, v) in edges_set:
                if not graph.has_node(u) or not graph.has_node(v):
                    continue
                pu = np.asarray(graph.nodes[u]['position'], dtype=float)
                pv = np.asarray(graph.nodes[v]['position'], dtype=float)
                segs.append(np.vstack([pu[::-1], pv[::-1]]))  # flip to (x,y)

            if segs:
                lc = LineCollection(
                    segs,
                    colors=[color],
                    linewidths=linewidth,
                    alpha=alpha,
                    zorder=5
                )
                self.axes.add_collection(lc)
                self.artists['junction_subtrees'].append(lc)

            if show_nodes and nodes_set:
                pts = []
                for n in nodes_set:
                    if not graph.has_node(n):
                        continue
                    p = np.asarray(graph.nodes[n]['position'], dtype=float)[::-1]
                    pts.append(p)
                if pts:
                    pts = np.asarray(pts)
                    sc = self.axes.scatter(
                        pts[:, 0], pts[:, 1],
                        s=node_size,
                        c=[color],
                        edgecolors=node_edgecolor,
                        linewidths=0.3,
                        zorder=6
                    )
                    self.artists['junction_subtrees'].append(sc)

        # Keep axes tidy & redraw
        self.axes.axis('image')
        self.axes.axis('off')
        self.canvas.draw()


    def plot_junction_subtrees_by_leaf_count(
        self,
        graph,
        analyses,                # List[SubtreeAnalysis]
        cmap_name='viridis',     # continuous colormap
        linewidth=2.5,
        alpha=0.95,
        vmin=None,
        vmax=None,
        show_nodes=False,
        node_size=10,
        node_edgecolor='black',
    ):
        """
        Color each subtree by its number of leaves. 'analyses' is the output of analyze_subtrees(...).
        """
        if graph is None or not analyses:
            return

        # clear previous
        self.hide_junction_subtrees()
        self.artists['junction_subtrees'] = []

        # color scale based on leaf_count
        counts = np.array([a.leaf_count for a in analyses], dtype=float)
        if vmin is None: vmin = float(counts.min())
        if vmax is None: vmax = float(counts.max())
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.colormaps.get_cmap(cmap_name)

        for a in analyses:
            color = cmap(norm(float(a.leaf_count)))

            # edge segments
            segs = []
            for (u, v) in a.edges:
                if not graph.has_node(u) or not graph.has_node(v):
                    continue
                pu = np.asarray(graph.nodes[u]['position'], float)
                pv = np.asarray(graph.nodes[v]['position'], float)
                segs.append(np.vstack([pu[::-1], pv[::-1]]))
            if segs:
                lc = LineCollection(segs, colors=[color], linewidths=linewidth, alpha=alpha, zorder=5)
                self.axes.add_collection(lc)
                self.artists['junction_subtrees'].append(lc)

            if show_nodes and a.nodes:
                pts = np.array([np.asarray(graph.nodes[n]['position'], float)[::-1] for n in a.nodes if graph.has_node(n)])
                if len(pts):
                    sc = self.axes.scatter(pts[:,0], pts[:,1], s=node_size, c=[color],
                                           edgecolors=node_edgecolor, linewidths=0.3, zorder=6)
                    self.artists['junction_subtrees'].append(sc)

        self.axes.axis('image'); self.axes.axis('off')
        self.canvas.draw()


    def plot_subtree_leaf_tangents(
        self,
        graph,
        analyses,                 # List[SubtreeAnalysis]
        length=10.0,
        linewidth=2.0,
        color='red',
        alpha=0.8,               
        linestyle='dotted',       
    ):
        """
        Draw short tangent segments for each leaf node, pointing inward.
        """
        if graph is None or not analyses:
            return

        self.hide_subtree_leaf_tangents()
        self.artists['subtree_leaf_tangents'] = []

        segs = []
        for a in analyses:
            for leaf, u_rc in a.leaf_tangents.items():
                if not graph.has_node(leaf):
                    continue
                p_rc = np.asarray(graph.nodes[leaf]['position'], float)  # [row, col]
                start_xy = p_rc[::-1]  # (x, y)
                end_xy = np.array([p_rc[1] + length * u_rc[1],
                                   p_rc[0] + length * u_rc[0]], dtype=float)
                segs.append(np.vstack([start_xy, end_xy]))

        if segs:
            lc = LineCollection(
                segs,
                colors=[color],
                linewidths=linewidth,
                alpha=alpha,
                linestyle=linestyle,
                zorder=7
            )
            self.axes.add_collection(lc)
            self.artists['subtree_leaf_tangents'].append(lc)

        self.canvas.draw()

    def plot_leaf_order_labels(self, graph, analyses, color='cyan', fontsize=8, dy=0.2):
        """
        Draw small numbers near each leaf showing its CCW index (0..L-1).
        'dy' nudges the label upward in display pixels (negative is up).
        """
        if graph is None or not analyses:
            return

        # clear old labels
        if 'leaf_order_labels' in self.artists:
            for t in self.artists['leaf_order_labels']:
                try: t.remove()
                except Exception: pass
            del self.artists['leaf_order_labels']

        self.artists['leaf_order_labels'] = []
        for a in analyses:
            for k, leaf in enumerate(a.leaf_order_ccw):
                p = np.asarray(graph.nodes[leaf]['position'], float)
                x, y = p[1], p[0]
                txt = self.axes.text(x, y + dy, str(k), color=color, fontsize=fontsize,
                                     ha='center', va='center', zorder=10)
                self.artists['leaf_order_labels'].append(txt)
        self.canvas.draw()

        
    def hide_leaf_order_labels(self):
        if 'leaf_order_labels' in self.artists:
            for t in self.artists['leaf_order_labels']:
                try: t.remove()
                except Exception: pass
            del self.artists['leaf_order_labels']
            self.canvas.draw()

    def hide_subtree_leaf_tangents(self):
        if 'subtree_leaf_tangents' in self.artists:
            for artist in self.artists['subtree_leaf_tangents']:
                try:
                    artist.remove()
                except (AttributeError, ValueError):
                    pass
            del self.artists['subtree_leaf_tangents']
            self.canvas.draw()


    def hide_junction_subtrees(self):
        """
        Remove all previously drawn subtree artists.
        """
        if 'junction_subtrees' in self.artists:
            for artist in self.artists['junction_subtrees']:
                try:
                    artist.remove()
                except (AttributeError, ValueError):
                    pass
            del self.artists['junction_subtrees']
            self.canvas.draw()

    def hide_medial_axis(self):
        if 'medial_axis_edges' in self.artists:
            self.artists['medial_axis_edges'].remove()
            self.artists['medial_axis_vertices'].remove()
            del self.artists['medial_axis_edges']
            del self.artists['medial_axis_vertices']
            self.canvas.draw()

    def hide_medial_axis_object_angles(self):
        if 'medial_axis_object_angles' in self.artists:
            self.artists['medial_axis_object_angles'].remove()
            del self.artists['medial_axis_object_angles']
            self.canvas.draw()

    def hide_voronoi_diagram(self):
        if 'voronoi_diagram' in self.artists:
            self.artists['voronoi_diagram'].remove()
            del self.artists['voronoi_diagram']
            self.canvas.draw()

    def hide_medial_axis_junctions(self):
        """
        Clear all previously plotted junctions (T-junctions and Y-junctions).
        """
        if 'junctions' in self.artists:
            for artist in self.artists['junctions']:
                try:
                    artist.remove()
                except (AttributeError, ValueError):
                    pass
            del self.artists['junctions']
            self.canvas.draw()






