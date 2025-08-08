import os 
import sys
import logging
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
from ui_viewer import Ui_MainWindow
from image_processing import *
from distance_to_measure import *
from curve_extraction import *
from vector_utils import *

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

MAX_STROKE_WIDTH = 20 
MIN_ISO_VALUE = 0 
MAX_ISO_VALUE = 100 

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.connect()
        
        self.base_image = None
        self.image_measure = None
        self.distance_function = None
        self.isovalue = 0.0
        self.stroke_width = 1 
        self.boundary_contours = None

        

    def connect(self):
        '''
        Connects all UI elements to their respective functions.
        '''
        # image IO
        self.ui.load_image_pushbutton.clicked.connect(self.load_image_from_file)
        self.ui.original_image_radiobutton.toggled.connect(self.toggle_original_image)
        self.ui.distance_radiobutton.toggled.connect(self.toggle_distance_function)
        self.ui.clean_canvas_radiobutton.toggled.connect(self.toggle_clean_canvas)



        self.ui.stroke_width_slider.sliderMoved.connect(self.update_on_move_stroke_width)
        self.ui.stroke_width_slider.sliderReleased.connect(self.update_on_release_stroke_width)


        self.ui.isovalue_slider.sliderMoved.connect(self.update_on_move_iso_value)
        self.ui.isovalue_slider.sliderReleased.connect(self.update_on_release_iso_value)

        self.ui.level_set_contour_checkbox.stateChanged.connect(self.toggle_boundary)
        self.ui.stroke_graph_checkbox.stateChanged.connect(self.toggle_medial_axis)
        self.ui.voronoi_diagram_checkbox.stateChanged.connect(self.toggle_voronoi_diagram)


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
        self.isovalue = self.get_distance_iso_value()
        self.ui.isovalue_display.display(self.isovalue)

    def update_on_move_persistence_threshold(self):
        '''
        Updates ON MOVE the persistence threshold slider + display
        '''
        pass
        
    def update_on_release_stroke_width(self):
        '''
        Performs relevant computation ON RELEASE of isovalue slider
        '''
        global MIN_ISO_VALUE, MAX_ISO_VALUE
        if self.image_measure is not None:
            self.stroke_width = self.get_stroke_width()
            self.ui.stroke_width_display.display(self.stroke_width)
            self.distance_function, _ = distance_to_measure_roi_sparse_cpu_numba(self.image_measure, 0.5*self.stroke_width)
            MAX_ISO_VALUE = np.max(self.distance_function)
            MIN_ISO_VALUE = np.min(self.distance_function)
            self.toggle_distance_function()

    def update_on_release_iso_value(self):
        '''
        Performs relevant computation ON RELEASE of isovalue slider
        '''
        if self.distance_function is not None:
            self.boundary_contours = find_contours(self.distance_function, self.isovalue, fully_connected='high')
            self.boundary_contours = resample_contours(self.boundary_contours, 0.5, 1e-5)
            self.toggle_boundary()
            self.voronoi_diagram = fast_voronoi_diagram(unique_contour_points(self.boundary_contours))
            self.medial_axis = fast_medial_axis(self.boundary_contours)
        
    def update_on_release_persistence_threshold(self):
        '''
        Performs relevant computation ON RELEASE of isovalue slider
        '''
        pass


    def compute_image_measure(self):
        """
        Recompute self.image_measure from self.base_image based on:
          - Gaussian blur checkbox + radius
          - Upsampling combobox
        Then update the display.
        """
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
        self.toggle_original_image()

    def export_contour_svg(self):
        if self.boundary_contours is not None:
            self.boundary_contours = find_contours(self.distance_function, self.isovalue, fully_connected='high')
            self.boundary_contours = resample_contours(self.boundary_contours, 0.5, 1e-5)
            export_contours_to_svg(self.boundary_contours, "output.svg")

    ### Toggle Functions
    def toggle_original_image(self):
        if self.ui.original_image_radiobutton.isChecked():
            self.ui.mpl_widget.show_image(self.image_measure, cmap='gray')

    def toggle_clean_canvas(self):
        if self.ui.clean_canvas_radiobutton.isChecked():
            self.ui.mpl_widget.show_blank_image()

    def toggle_distance_function(self):
        if self.ui.distance_radiobutton.isChecked():
            self.ui.mpl_widget.show_image(self.distance_function, normalize=False)

    def toggle_boundary(self):
        if self.ui.level_set_contour_checkbox.isChecked() and self.boundary_contours is not None:
            self.ui.mpl_widget.plot_contours(self.boundary_contours)
        else:
            self.ui.mpl_widget.hide_levelset_contours()

    def toggle_voronoi_diagram(self):
        if self.ui.voronoi_diagram_checkbox.isChecked() and self.voronoi_diagram is not None:
            self.ui.mpl_widget.plot_voronoi_diagram(self.voronoi_diagram)
        else:
            self.ui.mpl_widget.hide_voronoi_diagram()
     

    def toggle_medial_axis(self):
        if self.ui.stroke_graph_checkbox.isChecked() and self.medial_axis is not None:
            self.ui.mpl_widget.plot_medial_axis(self.medial_axis)
        else:
            self.ui.mpl_widget.hide_medial_axis()
        

    def toggle_image_measure_option(self):
        pass

    def toggle_boundary_measure_option(self):
        pass

    def toggle_complement_measure_option(self):
        pass


    ### Getters for sliders
    def get_stroke_width(self):
        step = MAX_STROKE_WIDTH / self.ui.stroke_width_slider.maximum()
        return self.ui.stroke_width_slider.value() * step
        
    def get_distance_iso_value(self):
        step = (MAX_ISO_VALUE - MIN_ISO_VALUE) / self.ui.isovalue_slider.maximum() 
        return float(MIN_ISO_VALUE + self.ui.isovalue_slider.value() * step)
        
    def get_persistence_threshold(self):
        pass
        
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
