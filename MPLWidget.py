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
        self.figure.set_layout_engine('constrained')

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
        '''
        if self.current_image is not None:
            self.current_image.remove()

        if img is None:
            self.show_blank_image()
            return

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
        #self.figure.tight_layout()

        # redraw canvas
        self.canvas.draw()

    def show_blank_image(self):
        """
        Displays a blank image on the widget.
        """
        #if self.current_image is not None:
        img_shape = self.current_image.get_array().shape
        if len(img_shape) == 2:
           img_shape = (*img_shape, 3)
        blank_img = np.ones(img_shape, dtype=np.uint8) * 255  
        #self.current_image.remove()
        self.current_image = self.axes.imshow(blank_img, cmap='gray')
        self.axes.axis('image')
        self.axes.axis('off')
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





