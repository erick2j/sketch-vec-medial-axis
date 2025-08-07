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
        '''
        Plots a set of points on the figure
        '''

        # Clear previous plots
        for contour_plot in self.levelset_contour_plots:
            contour_plot.remove()
        self.levelset_contour_plots = []

        path_vertices = []
        path_codes = []

        for contour in contours:
            contour = contour[:, [1, 0]]  # Convert (row, col) → (x, y)

            # Ensure the contour is closed
            if not np.array_equal(contour[0], contour[-1]):
                contour = np.vstack([contour, contour[0]])

            # Add to path
            n_points = len(contour)
            path_vertices.extend(contour)
            path_codes.extend([Path.MOVETO] + [Path.LINETO] * (n_points - 2) + [Path.CLOSEPOLY])

            # Plot scatter points
            scatter = self.axes.scatter(contour[:, 0], contour[:, 1], color='#00003f', s=1, zorder=2)
            self.levelset_contour_plots.append(scatter)

        # Plot the filled region
        compound_path = Path(path_vertices, path_codes)
        patch = PathPatch(compound_path, facecolor='#00FFFF', edgecolor='black', lw=3.0, alpha=0.1, zorder=1)
        self.axes.add_patch(patch)
        self.levelset_contour_plots.append(patch)

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
        self.artists['voronoi diagram'] = lines
        self.canvas.draw()


    def plot_medial_axis(self, graph, color='red', linewidth=1.0):
        """
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

        # Plot with LineCollection
        lines = LineCollection(segments, colors=color, linewidths=linewidth, zorder=3)
        self.axes.add_collection(lines)
        self.artists['medial_axis'] = lines
        self.canvas.draw()


    def hide_medial_axis(self):
        if 'medial_axis' in self.artists:
            self.artists['medial_axis'].remove()
            del self.artists['medial_axis']
            self.canvas.draw()

    def hide_voronoi_diagram(self):
        if 'voronoi_diagram' in self.artists:
            self.artists['voronoi_diagram'].remove()
            del self.artists['voronoi_diagram']
            self.canvas.draw()






