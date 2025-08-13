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
        Plot T-junctions and Y-junctions on the Matplotlib canvas.
        
        junctions : dict
            The dictionary of junctions returned by the mark_collapsible_edges function.
        graph : networkx.Graph
            The input graph with node positions and edges.
        """
        # Clear previous junction plots
        self.hide_medial_axis_junctions()

        # Ensure the 'junctions' key exists
        self.artists['junctions'] = []

        # Loop over the junctions
        for junc_id, data in junctions.items():
            node = data['node']
            junction_type = data['type']
            branches = data['branches']

            # Get node position
            position = np.array(graph.nodes[node]['position'])
            position = position[::-1]

            # Define the edge colors based on junction type
            if junction_type == 'T-junction':
                edge_color = 'red'
            elif junction_type == 'Y-junction':
                edge_color = 'blue'
            elif junction_type == 'X-junction':
                edge_color = 'green'
            else:
                edge_color = 'gray'  # Default for non-T, non-Y junctions

            # For each branch, plot the edges
            for branch in branches:
                # Create segments from branch
                positions = [np.array(graph.nodes[branch[i]]['position']) for i in range(len(branch))]
                positions = [pos[::-1] for pos in positions]
                segments = [np.vstack([positions[i], positions[i + 1]]) for i in range(len(positions) - 1)]
                
                # Create line collection for these segments
                line_collection = LineCollection(segments, colors=edge_color, linewidths=2, zorder=2)
                self.axes.add_collection(line_collection)
                self.artists['junctions'].append(line_collection)

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






