# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'viewer.ui'
##
## Created by: Qt User Interface Compiler version 6.9.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLCDNumber,
    QLabel, QMainWindow, QMenuBar, QPushButton,
    QRadioButton, QSizePolicy, QSlider, QStatusBar,
    QTabWidget, QVBoxLayout, QWidget)

from MPLWidget import MPLWidget

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1178, 776)
        MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        palette = QPalette()
        brush = QBrush(QColor(0, 144, 255, 255))
        brush.setStyle(Qt.BrushStyle.SolidPattern)
        palette.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.Highlight, brush)
        palette.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Highlight, brush)
        brush1 = QBrush(QColor(145, 145, 145, 255))
        brush1.setStyle(Qt.BrushStyle.SolidPattern)
        palette.setBrush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, brush1)
        self.centralwidget.setPalette(palette)
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.ControlsBox = QGroupBox(self.centralwidget)
        self.ControlsBox.setObjectName(u"ControlsBox")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(2)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.ControlsBox.sizePolicy().hasHeightForWidth())
        self.ControlsBox.setSizePolicy(sizePolicy1)
        self.verticalLayout_2 = QVBoxLayout(self.ControlsBox)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.sliders_groupbox = QGroupBox(self.ControlsBox)
        self.sliders_groupbox.setObjectName(u"sliders_groupbox")
        self.gridLayout_2 = QGridLayout(self.sliders_groupbox)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.stroke_width_slider = QSlider(self.sliders_groupbox)
        self.stroke_width_slider.setObjectName(u"stroke_width_slider")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(4)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.stroke_width_slider.sizePolicy().hasHeightForWidth())
        self.stroke_width_slider.setSizePolicy(sizePolicy2)
        self.stroke_width_slider.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.stroke_width_slider.setMinimum(1)
        self.stroke_width_slider.setMaximum(1000)
        self.stroke_width_slider.setSingleStep(1)
        self.stroke_width_slider.setPageStep(1)
        self.stroke_width_slider.setValue(100)
        self.stroke_width_slider.setSliderPosition(100)
        self.stroke_width_slider.setOrientation(Qt.Horizontal)
        self.stroke_width_slider.setTickPosition(QSlider.NoTicks)
        self.stroke_width_slider.setTickInterval(1)

        self.gridLayout_2.addWidget(self.stroke_width_slider, 1, 0, 1, 1)

        self.isovalue_label = QLabel(self.sliders_groupbox)
        self.isovalue_label.setObjectName(u"isovalue_label")
        self.isovalue_label.setAlignment(Qt.AlignCenter)

        self.gridLayout_2.addWidget(self.isovalue_label, 2, 0, 1, 1)

        self.isovalue_display = QLCDNumber(self.sliders_groupbox)
        self.isovalue_display.setObjectName(u"isovalue_display")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy3.setHorizontalStretch(1)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.isovalue_display.sizePolicy().hasHeightForWidth())
        self.isovalue_display.setSizePolicy(sizePolicy3)
        self.isovalue_display.setFrameShape(QFrame.StyledPanel)
        self.isovalue_display.setFrameShadow(QFrame.Plain)
        self.isovalue_display.setSmallDecimalPoint(True)
        self.isovalue_display.setDigitCount(5)
        self.isovalue_display.setSegmentStyle(QLCDNumber.Flat)
        self.isovalue_display.setProperty(u"value", 2.000000000000000)
        self.isovalue_display.setProperty(u"intValue", 2)

        self.gridLayout_2.addWidget(self.isovalue_display, 3, 1, 1, 1)

        self.isovalue_slider = QSlider(self.sliders_groupbox)
        self.isovalue_slider.setObjectName(u"isovalue_slider")
        sizePolicy2.setHeightForWidth(self.isovalue_slider.sizePolicy().hasHeightForWidth())
        self.isovalue_slider.setSizePolicy(sizePolicy2)
        self.isovalue_slider.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.isovalue_slider.setMaximum(1000)
        self.isovalue_slider.setValue(100)
        self.isovalue_slider.setSliderPosition(100)
        self.isovalue_slider.setOrientation(Qt.Horizontal)
        self.isovalue_slider.setTickPosition(QSlider.NoTicks)
        self.isovalue_slider.setTickInterval(50)

        self.gridLayout_2.addWidget(self.isovalue_slider, 3, 0, 1, 1)

        self.stroke_width_label = QLabel(self.sliders_groupbox)
        self.stroke_width_label.setObjectName(u"stroke_width_label")
        self.stroke_width_label.setAlignment(Qt.AlignCenter)

        self.gridLayout_2.addWidget(self.stroke_width_label, 0, 0, 1, 1)

        self.object_angle_slider = QSlider(self.sliders_groupbox)
        self.object_angle_slider.setObjectName(u"object_angle_slider")
        self.object_angle_slider.setMaximum(1000)
        self.object_angle_slider.setValue(0)
        self.object_angle_slider.setSliderPosition(0)
        self.object_angle_slider.setOrientation(Qt.Horizontal)

        self.gridLayout_2.addWidget(self.object_angle_slider, 5, 0, 1, 1)

        self.stroke_width_display = QLCDNumber(self.sliders_groupbox)
        self.stroke_width_display.setObjectName(u"stroke_width_display")
        sizePolicy3.setHeightForWidth(self.stroke_width_display.sizePolicy().hasHeightForWidth())
        self.stroke_width_display.setSizePolicy(sizePolicy3)
        self.stroke_width_display.setFrameShape(QFrame.StyledPanel)
        self.stroke_width_display.setFrameShadow(QFrame.Plain)
        self.stroke_width_display.setSmallDecimalPoint(True)
        self.stroke_width_display.setDigitCount(6)
        self.stroke_width_display.setSegmentStyle(QLCDNumber.Flat)
        self.stroke_width_display.setProperty(u"value", 1.000000000000000)
        self.stroke_width_display.setProperty(u"intValue", 1)

        self.gridLayout_2.addWidget(self.stroke_width_display, 1, 1, 1, 1)

        self.label = QLabel(self.sliders_groupbox)
        self.label.setObjectName(u"label")
        self.label.setAlignment(Qt.AlignCenter)

        self.gridLayout_2.addWidget(self.label, 4, 0, 1, 1)

        self.object_angle_display = QLCDNumber(self.sliders_groupbox)
        self.object_angle_display.setObjectName(u"object_angle_display")
        self.object_angle_display.setFrameShape(QFrame.StyledPanel)
        self.object_angle_display.setFrameShadow(QFrame.Plain)
        self.object_angle_display.setSegmentStyle(QLCDNumber.Flat)
        self.object_angle_display.setProperty(u"value", 1.000000000000000)

        self.gridLayout_2.addWidget(self.object_angle_display, 5, 1, 1, 1)


        self.verticalLayout_2.addWidget(self.sliders_groupbox)

        self.ImageOptionsBox = QGroupBox(self.ControlsBox)
        self.ImageOptionsBox.setObjectName(u"ImageOptionsBox")
        self.verticalLayout_4 = QVBoxLayout(self.ImageOptionsBox)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.ImageTab = QTabWidget(self.ImageOptionsBox)
        self.ImageTab.setObjectName(u"ImageTab")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(1)
        sizePolicy4.setHeightForWidth(self.ImageTab.sizePolicy().hasHeightForWidth())
        self.ImageTab.setSizePolicy(sizePolicy4)
        self.ImageTab.setTabPosition(QTabWidget.North)
        self.ImageTab.setTabShape(QTabWidget.Rounded)
        self.ImageTab.setElideMode(Qt.ElideNone)
        self.ImageTab.setUsesScrollButtons(False)
        self.ImageTab.setDocumentMode(False)
        self.ImageTab.setTabsClosable(False)
        self.ImageTab.setMovable(False)
        self.ImageTab.setTabBarAutoHide(False)
        self.ImageVisualizationTab = QWidget()
        self.ImageVisualizationTab.setObjectName(u"ImageVisualizationTab")
        self.ImageVisualizationTab.setMouseTracking(False)
        self.verticalLayout_7 = QVBoxLayout(self.ImageVisualizationTab)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.original_image_radiobutton = QRadioButton(self.ImageVisualizationTab)
        self.original_image_radiobutton.setObjectName(u"original_image_radiobutton")
        self.original_image_radiobutton.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.verticalLayout_7.addWidget(self.original_image_radiobutton)

        self.distance_radiobutton = QRadioButton(self.ImageVisualizationTab)
        self.distance_radiobutton.setObjectName(u"distance_radiobutton")
        self.distance_radiobutton.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.verticalLayout_7.addWidget(self.distance_radiobutton)

        self.clean_canvas_radiobutton = QRadioButton(self.ImageVisualizationTab)
        self.clean_canvas_radiobutton.setObjectName(u"clean_canvas_radiobutton")

        self.verticalLayout_7.addWidget(self.clean_canvas_radiobutton)

        self.ImageTab.addTab(self.ImageVisualizationTab, "")
        self.upsampling_tab = QWidget()
        self.upsampling_tab.setObjectName(u"upsampling_tab")
        self.verticalLayout_8 = QVBoxLayout(self.upsampling_tab)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.upsampling_combobox = QComboBox(self.upsampling_tab)
        self.upsampling_combobox.addItem("")
        self.upsampling_combobox.addItem("")
        self.upsampling_combobox.addItem("")
        self.upsampling_combobox.setObjectName(u"upsampling_combobox")
        self.upsampling_combobox.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.verticalLayout_8.addWidget(self.upsampling_combobox)

        self.ImageTab.addTab(self.upsampling_tab, "")

        self.verticalLayout_4.addWidget(self.ImageTab)

        self.OverlayTab = QTabWidget(self.ImageOptionsBox)
        self.OverlayTab.setObjectName(u"OverlayTab")
        sizePolicy4.setHeightForWidth(self.OverlayTab.sizePolicy().hasHeightForWidth())
        self.OverlayTab.setSizePolicy(sizePolicy4)
        self.OverlaysTab = QWidget()
        self.OverlaysTab.setObjectName(u"OverlaysTab")
        self.ImageOverlays = QGroupBox(self.OverlaysTab)
        self.ImageOverlays.setObjectName(u"ImageOverlays")
        self.ImageOverlays.setGeometry(QRect(0, -10, 282, 174))
        sizePolicy4.setHeightForWidth(self.ImageOverlays.sizePolicy().hasHeightForWidth())
        self.ImageOverlays.setSizePolicy(sizePolicy4)
        self.verticalLayout_5 = QVBoxLayout(self.ImageOverlays)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.voronoi_diagram_checkbox = QCheckBox(self.ImageOverlays)
        self.voronoi_diagram_checkbox.setObjectName(u"voronoi_diagram_checkbox")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.MinimumExpanding)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.voronoi_diagram_checkbox.sizePolicy().hasHeightForWidth())
        self.voronoi_diagram_checkbox.setSizePolicy(sizePolicy5)
        self.voronoi_diagram_checkbox.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.verticalLayout_3.addWidget(self.voronoi_diagram_checkbox)

        self.level_set_contour_checkbox = QCheckBox(self.ImageOverlays)
        self.level_set_contour_checkbox.setObjectName(u"level_set_contour_checkbox")
        sizePolicy5.setHeightForWidth(self.level_set_contour_checkbox.sizePolicy().hasHeightForWidth())
        self.level_set_contour_checkbox.setSizePolicy(sizePolicy5)
        self.level_set_contour_checkbox.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.verticalLayout_3.addWidget(self.level_set_contour_checkbox)

        self.stroke_graph_checkbox = QCheckBox(self.ImageOverlays)
        self.stroke_graph_checkbox.setObjectName(u"stroke_graph_checkbox")
        sizePolicy5.setHeightForWidth(self.stroke_graph_checkbox.sizePolicy().hasHeightForWidth())
        self.stroke_graph_checkbox.setSizePolicy(sizePolicy5)
        self.stroke_graph_checkbox.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.verticalLayout_3.addWidget(self.stroke_graph_checkbox)

        self.object_angle_checkbox = QCheckBox(self.ImageOverlays)
        self.object_angle_checkbox.setObjectName(u"object_angle_checkbox")
        sizePolicy5.setHeightForWidth(self.object_angle_checkbox.sizePolicy().hasHeightForWidth())
        self.object_angle_checkbox.setSizePolicy(sizePolicy5)

        self.verticalLayout_3.addWidget(self.object_angle_checkbox)

        self.circumradius_checkbox = QCheckBox(self.ImageOverlays)
        self.circumradius_checkbox.setObjectName(u"circumradius_checkbox")
        sizePolicy5.setHeightForWidth(self.circumradius_checkbox.sizePolicy().hasHeightForWidth())
        self.circumradius_checkbox.setSizePolicy(sizePolicy5)

        self.verticalLayout_3.addWidget(self.circumradius_checkbox)


        self.verticalLayout_5.addLayout(self.verticalLayout_3)

        self.OverlayTab.addTab(self.OverlaysTab, "")
        self.ExperimentalOverlayTab = QWidget()
        self.ExperimentalOverlayTab.setObjectName(u"ExperimentalOverlayTab")
        self.OverlayTab.addTab(self.ExperimentalOverlayTab, "")

        self.verticalLayout_4.addWidget(self.OverlayTab)


        self.verticalLayout_2.addWidget(self.ImageOptionsBox)

        self.ActionButtons = QGroupBox(self.ControlsBox)
        self.ActionButtons.setObjectName(u"ActionButtons")
        self.verticalLayout_6 = QVBoxLayout(self.ActionButtons)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.repair_junctions_pushbutton = QPushButton(self.ActionButtons)
        self.repair_junctions_pushbutton.setObjectName(u"repair_junctions_pushbutton")

        self.verticalLayout_6.addWidget(self.repair_junctions_pushbutton)

        self.export_svg_pushbutton = QPushButton(self.ActionButtons)
        self.export_svg_pushbutton.setObjectName(u"export_svg_pushbutton")
        self.export_svg_pushbutton.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.verticalLayout_6.addWidget(self.export_svg_pushbutton)


        self.verticalLayout_2.addWidget(self.ActionButtons)


        self.gridLayout.addWidget(self.ControlsBox, 0, 1, 1, 1)

        self.VisualizerBox = QGroupBox(self.centralwidget)
        self.VisualizerBox.setObjectName(u"VisualizerBox")
        sizePolicy6 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy6.setHorizontalStretch(5)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.VisualizerBox.sizePolicy().hasHeightForWidth())
        self.VisualizerBox.setSizePolicy(sizePolicy6)
        self.verticalLayout = QVBoxLayout(self.VisualizerBox)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.mpl_widget = MPLWidget(self.VisualizerBox)
        self.mpl_widget.setObjectName(u"mpl_widget")
        sizePolicy7 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy7.setHorizontalStretch(0)
        sizePolicy7.setVerticalStretch(15)
        sizePolicy7.setHeightForWidth(self.mpl_widget.sizePolicy().hasHeightForWidth())
        self.mpl_widget.setSizePolicy(sizePolicy7)
        self.mpl_widget.setMinimumSize(QSize(400, 400))

        self.verticalLayout.addWidget(self.mpl_widget)

        self.gb_visualizer = QGroupBox(self.VisualizerBox)
        self.gb_visualizer.setObjectName(u"gb_visualizer")
        sizePolicy8 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy8.setHorizontalStretch(0)
        sizePolicy8.setVerticalStretch(1)
        sizePolicy8.setHeightForWidth(self.gb_visualizer.sizePolicy().hasHeightForWidth())
        self.gb_visualizer.setSizePolicy(sizePolicy8)
        self.gb_visualizer.setToolTipDuration(0)
        self.gb_visualizer.setAutoFillBackground(False)
        self.gb_visualizer.setFlat(False)
        self.gb_visualizer.setCheckable(False)
        self.gb_visualizer.setChecked(False)
        self.horizontalLayout = QHBoxLayout(self.gb_visualizer)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.load_image_pushbutton = QPushButton(self.gb_visualizer)
        self.load_image_pushbutton.setObjectName(u"load_image_pushbutton")
        sizePolicy9 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy9.setHorizontalStretch(3)
        sizePolicy9.setVerticalStretch(0)
        sizePolicy9.setHeightForWidth(self.load_image_pushbutton.sizePolicy().hasHeightForWidth())
        self.load_image_pushbutton.setSizePolicy(sizePolicy9)
        self.load_image_pushbutton.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.load_image_pushbutton.setFocusPolicy(Qt.NoFocus)
        icon = QIcon()
        if QIcon.hasThemeIcon(QIcon.ThemeIcon.DocumentOpen):
            icon = QIcon.fromTheme(QIcon.ThemeIcon.DocumentOpen)
        else:
            icon.addFile(u".", QSize(), QIcon.Mode.Normal, QIcon.State.Off)

        self.load_image_pushbutton.setIcon(icon)
        self.load_image_pushbutton.setFlat(False)

        self.horizontalLayout.addWidget(self.load_image_pushbutton)

        self.clear_image_pushbutton = QPushButton(self.gb_visualizer)
        self.clear_image_pushbutton.setObjectName(u"clear_image_pushbutton")
        sizePolicy10 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy10.setHorizontalStretch(1)
        sizePolicy10.setVerticalStretch(0)
        sizePolicy10.setHeightForWidth(self.clear_image_pushbutton.sizePolicy().hasHeightForWidth())
        self.clear_image_pushbutton.setSizePolicy(sizePolicy10)
        self.clear_image_pushbutton.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        icon1 = QIcon()
        if QIcon.hasThemeIcon(QIcon.ThemeIcon.ApplicationExit):
            icon1 = QIcon.fromTheme(QIcon.ThemeIcon.ApplicationExit)
        else:
            icon1.addFile(u".", QSize(), QIcon.Mode.Normal, QIcon.State.Off)

        self.clear_image_pushbutton.setIcon(icon1)

        self.horizontalLayout.addWidget(self.clear_image_pushbutton)


        self.verticalLayout.addWidget(self.gb_visualizer)


        self.gridLayout.addWidget(self.VisualizerBox, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1178, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.ImageTab.setCurrentIndex(0)
        self.OverlayTab.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Measure Medial Axis", None))
        self.ControlsBox.setTitle("")
        self.sliders_groupbox.setTitle("")
        self.isovalue_label.setText(QCoreApplication.translate("MainWindow", u"Distance Multiplier", None))
        self.stroke_width_label.setText(QCoreApplication.translate("MainWindow", u"Stroke Width", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Object Angle Threshold", None))
        self.ImageOptionsBox.setTitle("")
        self.original_image_radiobutton.setText(QCoreApplication.translate("MainWindow", u"Original Image", None))
        self.distance_radiobutton.setText(QCoreApplication.translate("MainWindow", u"Distance Function", None))
        self.clean_canvas_radiobutton.setText(QCoreApplication.translate("MainWindow", u"Clean Canvas", None))
        self.ImageTab.setTabText(self.ImageTab.indexOf(self.ImageVisualizationTab), QCoreApplication.translate("MainWindow", u"Image", None))
        self.upsampling_combobox.setItemText(0, QCoreApplication.translate("MainWindow", u"None", None))
        self.upsampling_combobox.setItemText(1, QCoreApplication.translate("MainWindow", u"2x Naive", None))
        self.upsampling_combobox.setItemText(2, QCoreApplication.translate("MainWindow", u"4x Naive", None))

        self.upsampling_combobox.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Upsampling Technique", None))
        self.ImageTab.setTabText(self.ImageTab.indexOf(self.upsampling_tab), QCoreApplication.translate("MainWindow", u"Upsampling", None))
        self.ImageOverlays.setTitle("")
        self.voronoi_diagram_checkbox.setText(QCoreApplication.translate("MainWindow", u"Voronoi Diagram", None))
        self.level_set_contour_checkbox.setText(QCoreApplication.translate("MainWindow", u"Boundary Contour", None))
        self.stroke_graph_checkbox.setText(QCoreApplication.translate("MainWindow", u"Stroke Graph", None))
        self.object_angle_checkbox.setText(QCoreApplication.translate("MainWindow", u"Object Angle Measure", None))
        self.circumradius_checkbox.setText(QCoreApplication.translate("MainWindow", u"Circumradius Measure", None))
        self.OverlayTab.setTabText(self.OverlayTab.indexOf(self.OverlaysTab), QCoreApplication.translate("MainWindow", u"Overlays", None))
        self.OverlayTab.setTabText(self.OverlayTab.indexOf(self.ExperimentalOverlayTab), QCoreApplication.translate("MainWindow", u"Experimental", None))
        self.ActionButtons.setTitle("")
        self.repair_junctions_pushbutton.setText(QCoreApplication.translate("MainWindow", u"Repair Junctions", None))
        self.export_svg_pushbutton.setText(QCoreApplication.translate("MainWindow", u"Export SVG", None))
        self.VisualizerBox.setTitle("")
#if QT_CONFIG(tooltip)
        self.gb_visualizer.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.gb_visualizer.setTitle("")
        self.load_image_pushbutton.setText(QCoreApplication.translate("MainWindow", u"Load Image", None))
        self.clear_image_pushbutton.setText(QCoreApplication.translate("MainWindow", u"Clear Image", None))
    # retranslateUi

