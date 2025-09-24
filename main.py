import os
import sys
import logging

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog

from ui_viewer import Ui_MainWindow
from image_processing import normalize_to_measure, process_image
from vector_utils import export_contours_to_svg
from junction_types import StrokeGraph

logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
        )
logger = logging.getLogger(__name__)

########################### IMPORTANT ###########################
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic viewer.ui -o ui_viewer.py, or
#     pyside2-uic viewer.ui -o ui_viewer.py
#################################################################

MIN_STROKE_WIDTH = 1
MAX_STROKE_WIDTH = 20
MIN_ISO_SCALE = 0.0
MAX_ISO_SCALE = 1.0
DEFAULT_ISO_SCALE = 0.2
DEFAULT_OBJECT_ANGLE = 7.0 * np.pi / 16.0
DEFAULT_JUNCTION_OBJECT_ANGLE = 6.0 * np.pi / 16.0
MIN_OBJECT_ANGLE = 0
MAX_OBJECT_ANGLE = np.pi /2.0

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self._initialize_slider_defaults()
        self.connect()
        
        self.base_image = None
        self.image_measure = None
        self.stroke_graph_model: StrokeGraph | None = None

        self.distance_function = None
        self.boundary_contours = None
        self.medial_axis = None
        self.pruned_medial_axis = None
        self.stroke_graph = None
        self.subtrees = tuple()
        self.analyses = tuple()

        self.iso_scale = float(self.get_iso_scale())
        self.ui.isovalue_display.display(self.iso_scale)
        self.stroke_width = float(self.get_stroke_width())
        self.ui.stroke_width_display.display(self.stroke_width)
        self.isovalue = 0.0
        self.object_angle = float(self.get_object_angle())
        self.ui.object_angle_display.display(self.object_angle)
        self.junction_object_angle = float(self.get_junction_object_angle())
        self.ui.junction_object_angle_display.display(self.junction_object_angle)

        self.voronoi_diagram = None

        

    def connect(self):
        '''
        Connects all UI elements to their respective functions.
        '''
        # image loading buttons 
        self.ui.load_image_pushbutton.clicked.connect(self.load_image_from_file)
        self.ui.original_image_radiobutton.toggled.connect(self.toggle_original_image)
        self.ui.distance_radiobutton.toggled.connect(self.toggle_distance_function)
        self.ui.clean_canvas_radiobutton.toggled.connect(self.toggle_clean_canvas)

        # sliders
        self.ui.stroke_width_slider.sliderMoved.connect(self.update_on_move_stroke_width)
        self.ui.stroke_width_slider.sliderReleased.connect(self.update_on_release_stroke_width)

        self.ui.isovalue_slider.sliderMoved.connect(self.update_on_move_iso_value)
        self.ui.isovalue_slider.sliderReleased.connect(self.update_on_release_iso_value)

        self.ui.object_angle_slider.sliderMoved.connect(self.update_on_move_object_angle)
        self.ui.object_angle_slider.sliderReleased.connect(self.update_on_release_object_angle)

        self.ui.junction_object_angle_slider.sliderMoved.connect(self.update_on_move_junction_object_angle)
        self.ui.junction_object_angle_slider.sliderReleased.connect(self.update_on_release_junction_object_angle)


        # checkboxes
        self.ui.level_set_contour_checkbox.stateChanged.connect(self.toggle_boundary)
        self.ui.voronoi_diagram_checkbox.stateChanged.connect(self.toggle_voronoi_diagram)
        self.ui.stroke_graph_checkbox.stateChanged.connect(self.toggle_medial_axis)
        self.ui.object_angle_checkbox.stateChanged.connect(self.toggle_medial_axis_object_angles)
        self.ui.junctions_checkbox.stateChanged.connect(self.toggle_medial_axis_junctions)


        self.ui.upsampling_combobox.currentIndexChanged.connect(self.compute_image_measure)
        self.ui.gaussian_blur_checkbox.toggled.connect(self.compute_image_measure)


        self.ui.export_svg_pushbutton.clicked.connect(self.export_contour_svg)
        

    def uncheck_all_checkboxes(self):
        '''
        Unchecks all checkboxes 
        '''
        pass


    def uncheck_all_buttons(self):
        '''
        Unchecks all radiobuttons and checkboxes
        '''
        pass

    def clear_all_state(self):
        '''
        Clears all data and figures currently stored.
        '''
        pass

    def _initialize_slider_defaults(self):
        self._set_slider_default(self.ui.isovalue_slider, DEFAULT_ISO_SCALE)
        self._set_slider_default(self.ui.object_angle_slider, DEFAULT_OBJECT_ANGLE, MIN_OBJECT_ANGLE, MAX_OBJECT_ANGLE)
        self._set_slider_default(self.ui.junction_object_angle_slider, DEFAULT_JUNCTION_OBJECT_ANGLE, MIN_OBJECT_ANGLE, MAX_OBJECT_ANGLE)

    def _set_slider_default(self, slider, value, min_val=0.0, max_val=1.0):
        if slider is None:
            return

        slider_max = slider.maximum() or 1
        if max_val > min_val:
            ratio = (value - min_val) / (max_val - min_val)
        else:
            ratio = 0.0

        default_pos = int(round(np.clip(ratio, 0.0, 1.0) * slider_max))
        was_blocked = slider.blockSignals(True)
        slider.setValue(default_pos)
        slider.blockSignals(was_blocked)


    def _set_slider_from_value(self, slider, value, min_val, max_val):
        if slider is None:
            return

        slider_max = slider.maximum() or 1
        if max_val > min_val:
            ratio = (value - min_val) / (max_val - min_val)
        else:
            ratio = 0.0

        pos = int(round(np.clip(ratio, 0.0, 1.0) * slider_max))
        was_blocked = slider.blockSignals(True)
        slider.setValue(pos)
        slider.blockSignals(was_blocked)

    def _sync_from_model(self):
        model = self.stroke_graph_model
        if model is None:
            self.distance_function = None
            self.boundary_contours = None
            self.medial_axis = None
            self.pruned_medial_axis = None
            self.stroke_graph = None
            self.subtrees = tuple()
            self.analyses = tuple()
            return

        self.distance_function = model.distance_function
        self.boundary_contours = model.boundary_contours
        self.medial_axis = model.medial_axis
        self.pruned_medial_axis = model.pruned_graph
        self.stroke_graph = model.stroke_graph
        self.subtrees = model.junction_subtrees
        self.analyses = model.junction_analyses

    def _sync_slider_positions(self):
        self.ui.stroke_width_display.display(self.stroke_width)
        self._set_slider_from_value(self.ui.stroke_width_slider, self.stroke_width, MIN_STROKE_WIDTH, MAX_STROKE_WIDTH)

        self.ui.isovalue_display.display(self.iso_scale)
        self._set_slider_from_value(self.ui.isovalue_slider, self.iso_scale, MIN_ISO_SCALE, MAX_ISO_SCALE)

        self.ui.object_angle_display.display(self.object_angle)
        self._set_slider_from_value(self.ui.object_angle_slider, self.object_angle, MIN_OBJECT_ANGLE, MAX_OBJECT_ANGLE)

        self.ui.junction_object_angle_display.display(self.junction_object_angle)
        self._set_slider_from_value(self.ui.junction_object_angle_slider, self.junction_object_angle, MIN_OBJECT_ANGLE, MAX_OBJECT_ANGLE)

    def _refresh_views(self):
        if self.ui.original_image_radiobutton.isChecked():
            self.toggle_original_image()
        elif self.ui.distance_radiobutton.isChecked():
            self.toggle_distance_function()
        elif self.ui.clean_canvas_radiobutton.isChecked():
            self.toggle_clean_canvas()

        self.toggle_boundary()
        self.toggle_medial_axis()
        self.toggle_medial_axis_object_angles()
        self.toggle_medial_axis_junctions()

    def _post_model_update(self):
        if self.stroke_graph_model is None:
            self._sync_from_model()
            self._refresh_views()
            return

        self.stroke_width = float(self.stroke_graph_model.stroke_width)
        self.iso_scale = float(self.stroke_graph_model.iso_scale)
        self.object_angle = float(self.stroke_graph_model.object_angle_pruning_threshold)
        self.junction_object_angle = float(self.stroke_graph_model.object_angle_junction_threshold)
        self.isovalue = float(self.stroke_graph_model.isovalue)

        self._sync_from_model()
        self._sync_slider_positions()
        self._refresh_views()

    def _rebuild_stroke_graph(self):
        if self.image_measure is None:
            return

        self.stroke_width = float(self.get_stroke_width())
        self.iso_scale = float(self.get_iso_scale())
        self.object_angle = float(self.get_object_angle())
        self.junction_object_angle = float(self.get_junction_object_angle())

        self.stroke_graph_model = StrokeGraph(
            self.image_measure,
            stroke_width=self.stroke_width,
            iso_scale=self.iso_scale,
            pruning_object_angle=self.object_angle,
            junction_object_angle=self.junction_object_angle,
        )
        self._post_model_update()


    def load_image_from_file(self):
        '''
        Opens a QFileDialog to select an image and display it on the axes and store it
        for future use.
        '''
        # open file dialog to choose an image
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp *.tiff)")
        if not file_path:
            return
        # read image and display it
        self.base_image = 255 - process_image(file_path, padding=0)
        self.compute_image_measure()
        self.ui.original_image_radiobutton.setChecked(True)

      
    def extract_one_stable_manifold(self):
        '''
        Computes the 1-stable manifold of the current distance function.
        '''
        pass

    def export_distance_function(self):
        '''
        Writes the distance function to a .obj for visualization
        '''
        pass



    def update_on_move_stroke_width(self):
        '''
        Updates ON MOVE the stroke width value + slider + display
        '''
        self.stroke_width = self.get_stroke_width()
        self.ui.stroke_width_display.display(self.stroke_width)
        
    def update_on_move_iso_value(self):
        '''
        Updates ON MOVE the isovalue slider + display
        '''
        self.iso_scale = self.get_iso_scale()
        self.ui.isovalue_display.display(self.iso_scale)

    def update_on_move_object_angle(self):
        '''
        Updates ON MOVE the isovalue slider + display
        '''
        self.object_angle = self.get_object_angle()
        self.ui.object_angle_display.display(self.object_angle)

    def update_on_move_junction_object_angle(self):
        '''
        Updates ON MOVE the isovalue slider + display
        '''
        self.junction_object_angle = self.get_junction_object_angle()
        self.ui.junction_object_angle_display.display(self.junction_object_angle)

    def update_on_release_stroke_width(self):
        '''
        Performs relevant computation ON RELEASE of isovalue slider
        '''
        if self.stroke_graph_model is None:
            return

        self.stroke_width = float(self.get_stroke_width())
        self.ui.stroke_width_display.display(self.stroke_width)
        self.stroke_graph_model.update_stroke_width(self.stroke_width)
        self._post_model_update()

    def update_on_release_iso_value(self):
        '''
        Performs relevant computation ON RELEASE of isovalue slider
        '''
        if self.stroke_graph_model is None:
            return

        self.iso_scale = float(self.get_iso_scale())
        self.ui.isovalue_display.display(self.iso_scale)
        self.stroke_graph_model.update_iso_scale(self.iso_scale)
        self._post_model_update()

    def update_on_release_object_angle(self):
        '''
        Performs relevant computation ON RELEASE of isovalue slider
        '''
        if self.stroke_graph_model is None:
            return

        self.object_angle = float(self.get_object_angle())
        self.ui.object_angle_display.display(self.object_angle)
        self.stroke_graph_model.update_pruning_threshold(self.object_angle)
        self._post_model_update()

    def update_on_release_junction_object_angle(self):
        '''
        Performs relevant computation ON RELEASE of isovalue slider
        '''
        if self.stroke_graph_model is None:
            return

        self.junction_object_angle = float(self.get_junction_object_angle())
        self.ui.junction_object_angle_display.display(self.junction_object_angle)
        self.stroke_graph_model.update_junction_threshold(self.junction_object_angle)
        self._post_model_update()
        
    def compute_image_measure(self):
        """
        Recompute self.image_measure from self.base_image based on:
          - Gaussian blur checkbox + radius
          - Upsampling combobox
        Then update the display.
        """
        if self.base_image is None:
            return

        img = self.base_image

        # optional gaussian blur
        if self.ui.gaussian_blur_checkbox.isChecked():
            radius = float(self.ui.gaussian_blur_spinbox.value())
            if radius > 0:
                ksize = 2 * int(np.ceil(radius)) + 1
                img = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=radius)

        # optional upsampling
        match self.ui.upsampling_combobox.currentText():
            case "2x Naive":
                img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            case "4x Naive":
                img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            case "2x Bilinear":
                img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            case "4x Bilinear":
                img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
            case _:  # "None" or anything else
                pass

        # 3) Normalize and show
        self.image_measure = normalize_to_measure(img)
        self._rebuild_stroke_graph()

    def export_contour_svg(self):
        if self.boundary_contours:
            export_contours_to_svg(self.boundary_contours, "output.svg")

    ### Toggle Functions
    def toggle_original_image(self):
        if self.ui.original_image_radiobutton.isChecked() and self.image_measure is not None:
            self.ui.mpl_widget.show_image(self.image_measure, cmap='gray')

    def toggle_clean_canvas(self):
        if self.ui.clean_canvas_radiobutton.isChecked():
            self.ui.mpl_widget.show_blank_image()

    def toggle_distance_function(self):
        if self.ui.distance_radiobutton.isChecked() and self.distance_function is not None:
            self.ui.mpl_widget.show_image(self.distance_function, normalize=False)

    def toggle_boundary(self):
        if self.ui.level_set_contour_checkbox.isChecked() and self.boundary_contours:
            self.ui.mpl_widget.plot_contours(self.boundary_contours)
        else:
            self.ui.mpl_widget.hide_levelset_contours()

    def toggle_voronoi_diagram(self):
        if self.ui.voronoi_diagram_checkbox.isChecked() and self.voronoi_diagram is not None:
            self.ui.mpl_widget.plot_voronoi_diagram(self.voronoi_diagram)
        else:
            self.ui.mpl_widget.hide_voronoi_diagram()
     

    def toggle_medial_axis(self):
        if self.ui.stroke_graph_checkbox.isChecked() and self.stroke_graph is not None:
            self.ui.mpl_widget.plot_medial_axis(self.stroke_graph)
        else:
            self.ui.mpl_widget.hide_medial_axis()

    def toggle_medial_axis_object_angles(self):
        if self.ui.object_angle_checkbox.isChecked() and self.stroke_graph is not None:
            self.ui.mpl_widget.plot_medial_axis_object_angles(self.stroke_graph)
        else:
            self.ui.mpl_widget.hide_medial_axis_object_angles()

    def toggle_medial_axis_junctions(self):
        if (
            self.ui.junctions_checkbox.isChecked()
            and self.stroke_graph is not None
            and self.subtrees
            and self.analyses
        ):
            self.ui.mpl_widget.plot_junction_subtrees(self.stroke_graph, self.subtrees)
            self.ui.mpl_widget.plot_subtree_leaf_tangents(
                self.stroke_graph,
                self.analyses,
                length=14.0,
                color='black',
                linewidth=1.2,
            )
            self.ui.mpl_widget.plot_leaf_order_labels(self.stroke_graph, self.analyses)
        else:
            self.ui.mpl_widget.hide_junction_subtrees()
            self.ui.mpl_widget.hide_subtree_leaf_tangents()
            self.ui.mpl_widget.hide_leaf_order_labels()


    ### Getters for sliders
    def get_stroke_width(self):
        step = (MAX_STROKE_WIDTH - MIN_STROKE_WIDTH) / self.ui.stroke_width_slider.maximum()
        return float(MIN_STROKE_WIDTH + self.ui.stroke_width_slider.value() * step)
        
    def get_iso_scale(self):
        slider_max = self.ui.isovalue_slider.maximum() or 1
        step = (MAX_ISO_SCALE - MIN_ISO_SCALE) / slider_max
        return float(MIN_ISO_SCALE + self.ui.isovalue_slider.value() * step)
        
    def get_object_angle(self):
        step = (MAX_OBJECT_ANGLE - MIN_OBJECT_ANGLE) / self.ui.object_angle_slider.maximum() 
        return float(MIN_OBJECT_ANGLE + self.ui.object_angle_slider.value() * step)

    def get_junction_object_angle(self):
        step = (MAX_OBJECT_ANGLE - MIN_OBJECT_ANGLE) / self.ui.junction_object_angle_slider.maximum() 
        return float(MIN_OBJECT_ANGLE + self.ui.junction_object_angle_slider.value() * step)

        
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
