# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import numpy.ma as ma  # For masked arrays (hiding zeros in colormaps)
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from astropy.io import fits
from skimage import measure
from matplotlib.colors import LogNorm
import tools.KMOS_readout as tools
import scipy.stats as stats

filenames = []
for entry in os.scandir('Gaussian_fits'):
    if entry.is_file():
        path = entry.path.replace('\\', '\\\\')
        filenames.append(path)
if 'Gaussian_fits\\\\desktop.ini' in filenames:
    filenames.remove('Gaussian_fits\\\\desktop.ini')

def set_app_style():
    QtWidgets.QApplication.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 70))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(35, 35, 45))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 70))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 70))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    QtWidgets.QApplication.setPalette(palette)

# Remove redundant QApplication creation here; ensure it's only created in main or run_my_app

set_app_style()
pg.setConfigOption('background', (53, 53, 70))
pg.setConfigOption('foreground', 'w')
pg.setConfigOption('antialias', True)
pg.setConfigOption('leftButtonPan', False)
pg.setConfigOption('imageAxisOrder', 'row-major')
halpha_wavelength = 0.6563

# Add this new class to your file (before the MyApp class)
class HistogramWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Residual Histogram")
        self.setGeometry(100, 100, 600, 400)
        
        # Apply the same dark theme as main window and ensure pyqtgraph view has a solid background
        self.setStyleSheet("""
            QWidget {
            background-color: rgb(53, 53, 70);
            color: white;
            }
            /* pyqtgraph uses QGraphicsView for the plotting canvas; force its background too */
            QGraphicsView, QGraphicsScene, QGraphicsProxyWidget {
            background-color: rgb(53, 53, 70);
            }
            QLabel, QPushButton {
            color: white;
            }
        """)
        # Also set a Qt palette and enable auto-fill to make sure the widget background is painted
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 70))
        pal.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
        self.setPalette(pal)
        self.setAutoFillBackground(True)

        # Allow moving the window by dragging anywhere on it
        # (keeps the native title bar movable as well)
        self._drag_active = False
        self._drag_offset = None
        
        # Create layout
        layout = QtWidgets.QVBoxLayout(self)
        
        # Create plot widget for histogram
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Frequency')
        self.plot_widget.setLabel('bottom', 'Residual Value (σ)')
        self.plot_widget.setTitle('Distribution of Residuals')
        layout.addWidget(self.plot_widget)
        
        # Add statistics label
        self.stats_label = QtWidgets.QLabel()
        self.stats_label.setAlignment(QtCore.Qt.AlignCenter)
        self.stats_label.setStyleSheet("font-size: 12px; padding: 5px;")
        layout.addWidget(self.stats_label)
        
        # Close button
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        close_button.setStyleSheet("QPushButton { background-color: #353546; padding: 5px; }")
        layout.addWidget(close_button)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            # start dragging
            self._drag_active = True
            # store offset between mouse global pos and top-left of window
            self._drag_offset = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if getattr(self, "_drag_active", False) and self._drag_offset is not None:
            new_pos = event.globalPos() - self._drag_offset
            self.move(new_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_active = False
            self._drag_offset = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)
    
    def update_histogram(self, residuals):
        """Update the histogram with new residual data"""
        self.plot_widget.clear()
        
        # Filter finite values
        finite_residuals = residuals[np.isfinite(residuals)]
        
        if len(finite_residuals) < 2:
            self.stats_label.setText("Insufficient data for histogram")
            return
        
        # Create histogram
        hist, bin_edges = np.histogram(finite_residuals, bins=30, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = min(1/0.8,bin_edges[1] - bin_edges[0])
        
        # Plot histogram as bars
        bar_graph = pg.BarGraphItem(
            x=bin_centers, 
            height=hist, 
            width=bin_width*0.8, 
            brush='cyan', 
            pen='white'
        )
        self.plot_widget.addItem(bar_graph)
        
        # Add theoretical normal distribution for comparison
        x_theory = np.linspace(-4, 4, 200)
        y_theory = stats.norm.pdf(x_theory, 0, 1)
        self.plot_widget.plot(x_theory, y_theory, pen=pg.mkPen('red', width=2), name='Normal(0,1)')
        
        # Add reference lines
        self.plot_widget.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('white', style=QtCore.Qt.DashLine)))
        self.plot_widget.addItem(pg.InfiniteLine(pos=3, angle=90, pen=pg.mkPen('red', style=QtCore.Qt.DashLine)))
        self.plot_widget.addItem(pg.InfiniteLine(pos=-3, angle=90, pen=pg.mkPen('red', style=QtCore.Qt.DashLine)))
        
        # Calculate and display statistics
        mean_res = np.mean(finite_residuals)
        std_res = np.std(finite_residuals)
        n_points = len(finite_residuals)
        n_outliers = np.sum(np.abs(finite_residuals) > 3)
        outlier_percent = (n_outliers / n_points) * 100 if n_points > 0 else 0
        
        # Kolmogorov-Smirnov test for normality
        ks_stat, ks_p = stats.kstest(finite_residuals, 'norm', args=(0, 1))
        
        stats_text = (
            f"<b>Statistics:</b> N={n_points}, Mean={mean_res:.3f}, Std={std_res:.3f}<br>"
            f"<b>Outliers (|σ|>3):</b> {n_outliers} ({outlier_percent:.1f}%)<br>"
            f"<b>Normality Test:</b> KS={ks_stat:.3f}, p={ks_p:.3f}"
        )
        self.stats_label.setText(stats_text)
        
        # Add legend
        legend = self.plot_widget.addLegend()
        legend.addItem(bar_graph, "Residuals")

class MyApp(QtWidgets.QMainWindow):
    def __init__(self, main_cube=None, fit_data=None, z_redshift=None):
        super().__init__()
        self.setWindowTitle("Galaxy Viewer")
        self.resize(1230, 702)

        self.main_cube = main_cube
        self.main_cube_data = main_cube[1].data
        self.main_cube_error = main_cube[2].data
        self.z_redshift = z_redshift
        self.psf = tools.psf(main_cube)
        self.fit_data = fit_data

        self.x = 0
        self.y = 0
        self.MouseClicked = False

        self.contour_items = []

        self.init_ui()
        self.define_spectral_axis()
        self.create_cube_from_fits_values(fit_data, file_fit_results=file_fit_results)
        self.update_image_display()

    def init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        split = QtWidgets.QSplitter()
        split.setChildrenCollapsible(False)  # Prevent panels from collapsing
        layout.addWidget(split)

        left_panel = QtWidgets.QVBoxLayout()
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left_panel)
        split.addWidget(left_widget)

        self.comboBox_target = QtWidgets.QComboBox()
        self.comboBox_target.addItems(["Fluss","W80", "Sigma", "Offset", "[NII] Amplitude",
                                       "Chi² Single", "Chi² Double", "bin_num","S/N", "S/N Double", "S/N NII", "2nd Component Probability"])
        self.comboBox_target.currentIndexChanged.connect(self.update_image_display)
        left_panel.addWidget(self.comboBox_target)
        
        # Colormap selector (works for all visualizations)
        cmap_layout = QtWidgets.QHBoxLayout()
        cmap_label = QtWidgets.QLabel("Colormap:")
        cmap_label.setStyleSheet("color: white; font-size: 12px;")
        self.comboBox_colormap = QtWidgets.QComboBox()
        self.comboBox_colormap.addItems(["inferno", "plasma", "magma", "viridis", "cividis", 
                                          "hot", "YlOrRd", "turbo", "Spectral", "coolwarm", "RdBu", "Oranges", "Blues", "Greens"])
        self.comboBox_colormap.setCurrentText("inferno")
        self.comboBox_colormap.currentIndexChanged.connect(self.update_image_display)
        self.comboBox_colormap.setStyleSheet("background-color: #353546; color: white; padding: 2px;")
        cmap_layout.addWidget(cmap_label)
        cmap_layout.addWidget(self.comboBox_colormap)
        left_panel.addLayout(cmap_layout)
        
        # Mini-Gaussian overlay toggle
        self.checkbox_mini_gaussians = QtWidgets.QCheckBox("Show Mini-Gaussians")
        self.checkbox_mini_gaussians.setChecked(True)  # Default on
        self.checkbox_mini_gaussians.setStyleSheet("color: white; font-size: 12px;")
        self.checkbox_mini_gaussians.stateChanged.connect(self.update_image_display)
        left_panel.addWidget(self.checkbox_mini_gaussians)
        
        # Velocity toggle for Offset view (Δλ → km/s)
        self.checkbox_velocity = QtWidgets.QCheckBox("Offset in km/s anzeigen")
        self.checkbox_velocity.setChecked(False)  # Default: wavelength
        self.checkbox_velocity.setStyleSheet("color: white; font-size: 12px;")
        self.checkbox_velocity.stateChanged.connect(self.update_image_display)
        left_panel.addWidget(self.checkbox_velocity)

        self.label_chisquare = QtWidgets.QLabel("Chi²: Single = -, Double = -")
        self.label_chisquare.setFixedHeight(25)  # Fixed height to prevent layout shifts
        self.label_chisquare.setMinimumWidth(400)  # Minimum width for content
        left_panel.addWidget(self.label_chisquare)

        # === CREATE PLOT WITH SEPARATE COLORBAR ===
        plot_and_colorbar_layout = QtWidgets.QHBoxLayout()
        
        # Main plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.plot_widget.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        # Disable auto-range to prevent view changes during interaction
        self.plot_widget.getViewBox().setAutoVisible(x=False, y=False)
        self.plot_widget.getViewBox().enableAutoRange(enable=False)
        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_moved)
        self.plot_widget.scene().sigMouseClicked.connect(self.toggle_mouse_tracking)
        # Connect view range changes to update colorbar dynamically
        self.plot_widget.getViewBox().sigRangeChanged.connect(self.on_view_range_changed)
        plot_and_colorbar_layout.addWidget(self.plot_widget, stretch=1)
        
        # Colorbar container (vertical: colorbar + controls)
        colorbar_container = QtWidgets.QVBoxLayout()
        colorbar_container.setSpacing(1)
        colorbar_container.setContentsMargins(0, 0, 0, 0)
        
        # Separate colorbar widget
        self.colorbar_widget = pg.GraphicsLayoutWidget()
        self.colorbar_widget.setFixedWidth(70)
        self.colorbar_widget.setMinimumHeight(200)
        self.colorbar_widget.setBackground((53, 53, 70))
        
        # Create colorbar
        try:
            self.colorbar = pg.ColorBarItem(
                interactive=False, 
                values=(0, 1), 
                colorMap=pg.colormap.get('inferno', source='matplotlib'),
                orientation='vertical'
            )
            self.colorbar_widget.addItem(self.colorbar)
        except (AttributeError, TypeError):
            self.colorbar = None
        
        colorbar_container.addWidget(self.colorbar_widget)
        
        # Compact vmin/vmax controls
        range_widget = QtWidgets.QWidget()
        range_widget.setFixedWidth(70)
        range_layout = QtWidgets.QGridLayout(range_widget)
        range_layout.setSpacing(1)
        range_layout.setContentsMargins(2, 2, 2, 2)
        
        # vmax control (top value)
        self.vmax_spin = QtWidgets.QDoubleSpinBox()
        self.vmax_spin.setRange(-1e10, 1e10)
        self.vmax_spin.setDecimals(4)
        self.vmax_spin.setValue(1.0)
        self.vmax_spin.setFixedWidth(65)
        self.vmax_spin.setStyleSheet("""
            QDoubleSpinBox {
                background: #404050; color: #ff9944; 
                border: 1px solid #555; font-size: 9px;
                padding: 1px;
            }
        """)
        self.vmax_spin.valueChanged.connect(self._on_spin_changed)
        
        # vmin control (bottom value)  
        self.vmin_spin = QtWidgets.QDoubleSpinBox()
        self.vmin_spin.setRange(-1e10, 1e10)
        self.vmin_spin.setDecimals(4)
        self.vmin_spin.setValue(0.0)
        self.vmin_spin.setFixedWidth(65)
        self.vmin_spin.setStyleSheet("""
            QDoubleSpinBox {
                background: #404050; color: #44aaff;
                border: 1px solid #555; font-size: 9px;
                padding: 1px;
            }
        """)
        self.vmin_spin.valueChanged.connect(self._on_spin_changed)
        
        # Auto-scale button (small)
        self.auto_btn = QtWidgets.QPushButton("⟲")
        self.auto_btn.setFixedSize(20, 20)
        self.auto_btn.setToolTip("Auto-scale")
        self.auto_btn.setStyleSheet("background: #404050; color: white; border: 1px solid #555; font-size: 11px;")
        self.auto_btn.clicked.connect(self._reset_colorbar_range)
        
        range_layout.addWidget(self.vmax_spin, 0, 0)
        range_layout.addWidget(self.auto_btn, 0, 1)
        range_layout.addWidget(self.vmin_spin, 1, 0)
        
        colorbar_container.addWidget(range_widget)
        
        # Unit label
        self.colorbar_unit_label = QtWidgets.QLabel("")
        self.colorbar_unit_label.setStyleSheet("color: #888; font-size: 9px;")
        self.colorbar_unit_label.setAlignment(QtCore.Qt.AlignCenter)
        self.colorbar_unit_label.setFixedWidth(70)
        colorbar_container.addWidget(self.colorbar_unit_label)
        
        plot_and_colorbar_layout.addLayout(colorbar_container)
        
        # Add the plot+colorbar layout to left panel
        left_panel.addLayout(plot_and_colorbar_layout)
        
        # Store current image data for colorbar updates
        self._current_display_data = None
        self._current_display_selection = None
        self._current_vmin_actual = 0
        self._current_vmax_actual = 1
        self._current_image_item = None

        right_panel = QtWidgets.QVBoxLayout()
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right_panel)
        split.addWidget(right_widget)

        # Spectrum and Residual Plots
        self.plot_widget_2 = pg.PlotWidget()
        self.plot_widget_2.setMouseEnabled(x=True, y=True)
        self.plot_widget_2.getViewBox().setMouseMode(pg.ViewBox.RectMode)

        self.plot_widget_residual = pg.PlotWidget()
        self.plot_widget_residual.setMouseEnabled(x=True, y=True)

        # Link the x-axes so they always share the same range/zoom
        self.plot_widget_residual.getViewBox().setXLink(self.plot_widget_2.getViewBox())
        self.plot_widget_residual.setMaximumHeight(250)
        self.plot_widget_residual.setLabel("bottom", "Wavelength")
        self.plot_widget_residual.setLabel("left", "Residual")

        # Add a small window showing the index of the galaxy
        self.idx_label = QtWidgets.QLabel(f"Galaxy Index: {idx}")
        self.idx_label.setAlignment(QtCore.Qt.AlignCenter)
        self.idx_label.setStyleSheet("font-size: 16px; color: white; background-color: #353546; padding: 4px; border-radius: 6px;")
        right_panel.addWidget(self.idx_label)

        # Container for both plots
        spectrum_container = QtWidgets.QWidget()
        spectrum_layout = QtWidgets.QVBoxLayout()
        spectrum_layout.setContentsMargins(0, 0, 0, 0)
        spectrum_container.setLayout(spectrum_layout)

        spectrum_layout.addWidget(self.plot_widget_2)
        spectrum_layout.addWidget(self.plot_widget_residual)

        # Add to right panel
        right_panel.addWidget(spectrum_container)

        self.zoom_button = QtWidgets.QPushButton("Zoom Halpha")
        self.zoom_button.clicked.connect(self.zoom_on_halpha)
        right_panel.addWidget(self.zoom_button)

        self.zerlegung_button = QtWidgets.QPushButton("Zerlegungsmodus")
        self.zerlegung_button.clicked.connect(self.plot_zerlegung_fit)
        right_panel.addWidget(self.zerlegung_button)

        self.change_file_button = QtWidgets.QPushButton("Datei wechseln")
        self.change_file_button.clicked.connect(self.change_file)
        right_panel.addWidget(self.change_file_button)

    def plot_zerlegung_fit(self):
        x, y = self.x, self.y
        x_data = self.wavelength
        continuum = self.continuum_slope[x, y] * x_data + self.continuum_intercept[x, y]
        y_double_1 = (
            self.cube_amps_double_one[x, y] * np.exp(-0.5 * ((x_data - self.cube_offsets_double_one[x, y]) / self.cube_sigmas_double_one[x, y]) ** 2) + continuum  # Assuming continuum is defined in the main_cube_data
        )
        y_double_2 = (self.cube_amps_double_two[x, y] * np.exp(-0.5 * ((x_data - self.cube_offsets_double_two[x, y]) / self.cube_sigmas_double_two[x, y]) ** 2) + continuum)

        double_1_curve = self.plot_widget_2.plot(x_data, y_double_1, pen=pg.mkPen('purple', width=3))
        double_2_curve = self.plot_widget_2.plot(x_data, y_double_2, pen=pg.mkPen('cyan', width=3))

        legend = pg.LegendItem(offset=(70, 30))
        legend.setParentItem(self.plot_widget_2.graphicsItem())
        if self.cube_sigmas_double_two[x, y] < self.cube_sigmas_double_one[x, y]:
            legend.addItem(double_1_curve, "Broad Component")
            legend.addItem(double_2_curve, "Narrow Component")
        else:
            legend.addItem(double_1_curve, "Narrow Component")
            legend.addItem(double_2_curve, "Broad Component")
        legend.setBrush(pg.mkBrush(53, 53, 70, 220))
        legend.setPen(pg.mkPen('w'))
        legend.setLabelTextColor('w')
        legend.setFont(QtGui.QFont("Arial", 10))

    def define_spectral_axis(self):
        header = self.main_cube[1].header
        self.crval3 = header['CRVAL3']
        self.cdelt3 = header['CDELT3']
        self.naxis3 = header['NAXIS3']
        self.wavelength = (self.crval3 + self.cdelt3 * np.arange(self.naxis3)) / (1 + self.z_redshift)

    def toggle_mouse_tracking(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.MouseClicked = not self.MouseClicked
            if self.MouseClicked:
                self.plot_widget.scene().sigMouseMoved.disconnect(self.on_mouse_moved)
            else:
                self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_moved)

    def _on_spin_changed(self):
        """Update image levels when vmin/vmax spin boxes change"""
        if self._current_image_item is None:
            return
        
        try:
            vmin = self.vmin_spin.value()
            vmax = self.vmax_spin.value()
            
            if vmin >= vmax:
                return
            
            self._current_image_item.setLevels([vmin, vmax])
            if self.colorbar:
                self.colorbar.setLevels([vmin, vmax])
        except Exception:
            pass
    
    def _reset_colorbar_range(self):
        """Reset to original auto-scaled range"""
        self.vmin_spin.blockSignals(True)
        self.vmax_spin.blockSignals(True)
        self.vmin_spin.setValue(self._current_vmin_actual)
        self.vmax_spin.setValue(self._current_vmax_actual)
        self.vmin_spin.blockSignals(False)
        self.vmax_spin.blockSignals(False)
        
        if self._current_image_item is not None:
            self._current_image_item.setLevels([self._current_vmin_actual, self._current_vmax_actual])
        if self.colorbar:
            self.colorbar.setLevels([self._current_vmin_actual, self._current_vmax_actual])
    
    def _update_colorbar_range(self, vmin, vmax, img_item):
        """Store range and update spin boxes"""
        self._current_vmin_actual = vmin
        self._current_vmax_actual = vmax
        self._current_image_item = img_item
        
        # Set appropriate step size based on range
        step = (vmax - vmin) / 100 if vmax != vmin else 0.01
        
        self.vmin_spin.blockSignals(True)
        self.vmax_spin.blockSignals(True)
        self.vmin_spin.setSingleStep(step)
        self.vmax_spin.setSingleStep(step)
        self.vmin_spin.setValue(vmin)
        self.vmax_spin.setValue(vmax)
        self.vmin_spin.blockSignals(False)
        self.vmax_spin.blockSignals(False)

    def on_view_range_changed(self, view_box, ranges):
        """Placeholder - colorbar now controlled by sliders"""
        pass

    def on_mouse_moved(self, pos):
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        x, y= int(mouse_point.x()), int(mouse_point.y())
        if 0 <= x < self.main_cube_data.shape[1] and 0 <= y < self.main_cube_data.shape[2]:
            self.x, self.y = x, y
            self.plot_1D_spectra()

    def plot_1D_spectra(self):
        chi1 = self.cube_chi_squared_one[self.x, self.y]
        chi2 = self.cube_chi_squared_two[self.x, self.y]
        use_nii_single = self.nii_single_used[self.x, self.y]
        use_nii_double = self.nii_double_used[self.x, self.y]

        nii_used = []
        if use_nii_single:
            nii_used.append("1")
        if use_nii_double:
            nii_used.append("2")
        nii_text = "Yes (" + ", ".join(nii_used) + ")" if nii_used else "No"

        self.label_chisquare.setText(
            f"<span style='font-size:16px; color:#FFA500;'><b>χ² ₁</b> = <b>{chi1:.2f}</b></span>, "
            f"<span style='font-size:16px; color:#00FF00;'><b>χ² ₂</b> = <b>{chi2:.2f}</b></span>, "
            f"<span style='font-size:16px; color:#D3D3D3;'>S/N₁ = <b>{self.cube_son_one[self.x, self.y]:.2f}</b></span>, "
            f"<span style='font-size:16px; color:#FF69B4;'>S/N₂ = <b>{self.cube_son_two[self.x, self.y]:.2f}</b></span>, "
            f"<span style='font-size:16px; color:#B0B0B0;'>NII used: <b>{nii_text}</b></span>"
        )

        x, y = self.x, self.y
        bin_num = self.cube_bin_num[x, y]
        mask = self.cube_bin_num == bin_num

        x, y = self.x, self.y
        current_data = self.main_cube_data[:, mask] * self.psf[mask]
        current_error = self.main_cube_error[:, mask] * self.psf[mask]


        # Stack
        summed_data = np.sum(current_data, axis=1)
        summed_error = np.sqrt(np.sum((current_error) ** 2, axis=1))

        self.current_data = summed_data
        self.current_error = summed_error

        # === Main spectrum plot ===
        self.plot_widget_2.clear()
        self.plot_widget_2.plot(self.wavelength, self.current_data, pen='w')

        lower = self.current_data - self.current_error
        upper = self.current_data + self.current_error
        fill = pg.FillBetweenItem(
            pg.PlotCurveItem(self.wavelength, lower, pen=None),
            pg.PlotCurveItem(self.wavelength, upper, pen=None),
            brush=pg.mkBrush(150, 150, 150, 80)
        )

        nii_wavelength_1 = 0.6548
        halpha_wavelength = 0.6563
        nii_wavelength_2 = 0.6584

        self.plot_widget_2.addItem(pg.InfiniteLine(pos=halpha_wavelength, angle=90, pen=pg.mkPen(color=(180, 180, 180), width=2, style=QtCore.Qt.DashLine)))
        self.plot_widget_2.addItem(pg.InfiniteLine(pos=nii_wavelength_1, angle=90, pen=pg.mkPen(color=(100,150,255), width=2, style=QtCore.Qt.DotLine)))
        self.plot_widget_2.addItem(pg.InfiniteLine(pos=nii_wavelength_2, angle=90, pen=pg.mkPen(color=(100,150,255), width=2, style=QtCore.Qt.DotLine)))
        self.plot_widget_2.addItem(fill)

        # === Residual plot ===
        self.plot_widget_residual.clear()

        # === Fit results & residuals ===
        self.plot_fit_results()

    def toggle_second_residual(self):
                # Redraw the residual plot to show/hide the second residual
                self.plot_widget_residual.clear()
                self.plot_fit_results()

    def plot_fit_results(self):
        x, y = self.x, self.y
        single_nii_used = self.nii_single_used[x, y]
        double_nii_used = self.nii_double_used[x, y]

        x_data = np.linspace(self.wavelength[0], self.wavelength[-1], len(self.wavelength)*10)
        continuum = self.continuum_slope[x, y] * x_data + self.continuum_intercept[x, y]

        # Model fits
        y_single = (
            np.abs(self.cube_amps_single[x, y]) * np.exp(-0.5 * ((x_data - self.cube_offsets_single[x, y]) / self.cube_sigmas_single[x, y]) ** 2)
            + continuum
        )
        y_double = (
            self.cube_amps_double_one[x, y] * np.exp(-0.5 * ((x_data - self.cube_offsets_double_one[x, y]) / self.cube_sigmas_double_one[x, y]) ** 2)
            + self.cube_amps_double_two[x, y] * np.exp(-0.5 * ((x_data - self.cube_offsets_double_two[x, y]) / self.cube_sigmas_double_two[x, y]) ** 2)
            + continuum
        )
        y_nii = self.cube_amps_nii1[x, y] * np.exp(-0.5 * ((x_data - self.cube_offsets_nii1[x, y]) / self.cube_sigmas_nii1[x, y]) ** 2) + self.cube_amps_nii2[x, y] * np.exp(-0.5 * ((x_data - self.cube_offsets_nii2[x, y]) / self.cube_sigmas_nii2[x, y]) ** 2) + continuum

        # Plot models
        if single_nii_used:
            self.plot_widget_2.plot(x_data, y_single + y_nii - continuum, pen=pg.mkPen('r', width=2))
        else:
            self.plot_widget_2.plot(x_data, y_single, pen=pg.mkPen('r', width=2))
        if double_nii_used:
            self.plot_widget_2.plot(x_data, y_double + y_nii - continuum , pen=pg.mkPen('g', width=2))
        else:
            self.plot_widget_2.plot(x_data, y_double, pen=pg.mkPen('g', width=2))
        if single_nii_used or double_nii_used:
            self.plot_widget_2.plot(x_data, y_nii, pen=pg.mkPen('b', width=2))

        mask = self.current_error > 0

        # === Residuals (data - best fit model) ===
        mask = self.current_error > 0
        chi1 = self.cube_chi_squared_one[x, y]
        chi2 = self.cube_chi_squared_two[x, y]
        if chi1 < chi2:
            residual = np.full_like(self.current_data, np.nan)
            residual[mask] = (self.current_data[mask] - y_single[::10][mask]) / self.current_error[mask]
            model_label = "Single Gaussian"
            second_residual = np.full_like(self.current_data, np.nan)
            second_residual[mask] = (self.current_data[mask] - y_double[::10][mask]) / self.current_error[mask]
            second_label = "Double Gaussian"
        else:
            residual = np.full_like(self.current_data, np.nan)
            residual[mask] = (self.current_data[mask] - y_double[::10][mask]) / self.current_error[mask]
            model_label = "Double Gaussian"
            second_residual = np.full_like(self.current_data, np.nan)
            second_residual[mask] = (self.current_data[mask] - y_single[::10][mask]) / self.current_error[mask]
            second_label = "Single Gaussian"

        # Store residuals for histogram
        self.current_residual = residual

        self.plot_widget_residual.setTitle(f"Best Fit Model: {model_label}")
        self.plot_widget_residual.plot(self.wavelength, residual, pen=pg.mkPen('y'))
        self.plot_widget_residual.plot(self.wavelength, np.full(self.wavelength.shape, 3), pen=pg.mkPen('r', style=QtCore.Qt.DashLine, width=1))
        self.plot_widget_residual.plot(self.wavelength, np.full(self.wavelength.shape, -3), pen=pg.mkPen('r', style=QtCore.Qt.DashLine, width=1))
        self.plot_widget_residual.plot(self.wavelength, np.full(self.wavelength.shape, 0), pen=pg.mkPen('w', style=QtCore.Qt.DashLine, width=1))
        self.plot_widget_residual.addItem(pg.InfiniteLine(pos=halpha_wavelength, angle=90, pen=pg.mkPen(color=(180, 180, 180), width=2, style=QtCore.Qt.DashLine)))

        # Add histogram button (only once)
        if not hasattr(self, 'histogram_button'):
            self.histogram_button = QtWidgets.QPushButton("Show Histogram")
            self.histogram_button.clicked.connect(self.show_histogram)
            self.histogram_button.setStyleSheet("background-color: #353546; color: white; padding: 5px;")
            # Add to right panel
            parent_widget = self.plot_widget_residual.parentWidget()
            if parent_widget and hasattr(parent_widget, 'layout') and parent_widget.layout() is not None:
                parent_widget.layout().addWidget(self.histogram_button)

        # Add button to show second best fit residual
        if not hasattr(self, 'show_second_residual_button'):
            self.show_second_residual_button = QtWidgets.QPushButton("Show 2nd Best Fit Residual")
            self.show_second_residual_button.setCheckable(True)
            self.show_second_residual_button.setStyleSheet("background-color: #353546; color: white;")
            self.show_second_residual_button.clicked.connect(self.toggle_second_residual)
            # Add to right panel if not already added
            parent_widget = self.plot_widget_residual.parentWidget()
            if parent_widget and hasattr(parent_widget, 'layout') and parent_widget.layout() is not None:
                parent_widget.layout().addWidget(self.show_second_residual_button)

        # Show/hide second residual based on button state
        if hasattr(self, 'show_second_residual_button') and self.show_second_residual_button.isChecked():
            self.plot_widget_residual.plot(self.wavelength, second_residual, pen=pg.mkPen('c', width=2))
            self.plot_widget_residual.setTitle(
                f"Best Fit Model: <span style='color:yellow;'>{model_label}</span> | "
                f"2nd: <span style='color:cyan;'>{second_label}</span>"
            )

        # Keep residual plot labels consistent
        self.plot_widget_residual.setLabel("left", "Residuals / \n error")
        self.plot_widget_residual.setLabel("bottom", "Wavelength")

        # === Legend ===
        legend = pg.LegendItem(offset=(70, 30))
        legend.setParentItem(self.plot_widget_2.graphicsItem())
        legend.addItem(pg.PlotCurveItem(x_data, y_single, pen=pg.mkPen('r')), "Single Gaussian")
        legend.addItem(pg.PlotCurveItem(x_data, y_double, pen=pg.mkPen('g')), "Double Gaussian")
        legend.addItem(pg.PlotCurveItem(x_data, y_nii, pen=pg.mkPen('b')), "[NII] Line")
        legend.setBrush(pg.mkBrush(53, 53, 70, 220))
        legend.setPen(pg.mkPen('w'))
        legend.setLabelTextColor('w')
        legend.setFont(QtGui.QFont("Arial", 10))

    # Add this method to your MyApp class
    def show_histogram(self):
        """Show histogram popup window"""
        if not hasattr(self, 'histogram_window'):
            self.histogram_window = HistogramWindow(self)
        
        # Update histogram with current residuals
        if hasattr(self, 'current_residual'):
            self.histogram_window.update_histogram(self.current_residual)
        
        self.histogram_window.show()
        self.histogram_window.raise_()  # Bring to front

    def zoom_on_halpha(self):
        halpha_wavelength = 0.6563
        x = self.x
        y = self.y
        current_data = self.main_cube_data[:, x, y] * self.psf[x, y]
        current_model1 = self.cube_amps_single[x, y] * np.exp(-0.5 * ((self.wavelength - self.cube_offsets_single[x, y]) / self.cube_sigmas_single[x, y]) ** 2) + self.continuum_slope[x, y] * self.wavelength + self.continuum_intercept[x, y]
        current_model2 = self.cube_amps_double_one[x, y] * np.exp(-0.5 * ((self.wavelength - self.cube_offsets_double_one[x, y]) / self.cube_sigmas_double_one[x, y]) ** 2) + self.continuum_slope[x, y] * self.wavelength + self.continuum_intercept[x, y]
        delta = 0.01 * halpha_wavelength
        x_min = halpha_wavelength - delta
        x_max = halpha_wavelength + delta
        self.plot_widget_2.setXRange(x_min, x_max)
        self.plot_widget_residual.setXRange(x_min, x_max)
        local_data = current_data[(self.wavelength >= x_min) & (self.wavelength <= x_max)]
        local_residual = local_data - current_model1[(self.wavelength >= x_min) & (self.wavelength <= x_max)]/ self.current_error[(self.wavelength >= x_min) & (self.wavelength <= x_max)]
        
        y_min = min(0, np.nanmin(local_data))
        # Avoid RuntimeWarning by checking for all-NaN arrays
        def safe_nanmax(arr, default=1.0):
            return np.nanmax(arr) if np.any(np.isfinite(arr)) else default

        y_max = max(
            1.2 * safe_nanmax(local_data, default=1.0),
            safe_nanmax(current_model1, default=1.0),
            safe_nanmax(current_model2, default=1.0)
        )
        if local_residual.size > 0 and np.any(np.isfinite(local_residual)):
            y_residual = min(3.2, np.nanmax(np.abs(local_residual)))
        else:
            y_residual = 3.2
        self.plot_widget_residual.setYRange(-y_residual, y_residual)
        print(f"Setting Y range to: {y_residual}")
        self.plot_widget_2.setYRange(y_min, y_max)

    def update_image_display(self):

        # Compute mask using minimum of non-NaN values along the chi-squared arrays
        chi1 = self.cube_chi_squared_one
        chi2 = self.cube_chi_squared_two
        son1 = self.cube_son_one
        son2 = self.cube_son_two
        #ston = self.cube_ston
        min_chi = np.fmin(chi1, chi2)  # elementwise minimum, ignoring NaNs
        max_son = np.fmax(son1, son2)  # elementwise maximum, ignoring NaNs
        #mask_strict = max_son > 3  # Boolean mask for S/N > 3
        #mask_strict = min_chi < 2  # Boolean mask for chi-squared < 3
        #mask = mask_strict  # Use all data for now, can be adjusted later
        mask_strict = np.ones_like(self.cube_amps_single, dtype=bool)
        mask = np.ones_like(self.cube_amps_single, dtype=bool)  # Use all data for now, can be adjusted later
        self.cube_amps_single = np.where(mask, self.cube_amps_single, np.nan)
        self.cube_sigmas_single = np.where(mask, self.cube_sigmas_single, np.nan)
        self.cube_offsets_single = np.where(mask_strict, self.cube_offsets_single, np.nan)
        self.cube_amps_double_one = np.where(mask, self.cube_amps_double_one, np.nan)
        self.cube_sigmas_double_one = np.where(mask, self.cube_sigmas_double_one, np.nan)
        self.cube_offsets_double_one = np.where(mask_strict, self.cube_offsets_double_one, np.nan)
        self.cube_amps_double_two = np.where(mask, self.cube_amps_double_two, np.nan)
        self.cube_sigmas_double_two = np.where(mask, self.cube_sigmas_double_two, np.nan)
        self.cube_offsets_double_two = np.where(mask_strict, self.cube_offsets_double_two, np.nan)
        self.cube_amps_nii = np.where(mask, self.cube_amps_nii1, np.nan)
        self.cube_sigmas_nii = np.where(mask, self.cube_sigmas_nii1, np.nan)
        self.cube_offsets_nii = np.where(mask, self.cube_offsets_nii1, np.nan)

        # Find the bounding box where not all values are np.nan
        # Only set range if this is the first display or if file changed
        if not hasattr(self, '_initial_range_set') or not self._initial_range_set:
            valid_mask = ~np.isnan(self.cube_amps_single)
            y_indices, x_indices = np.where(valid_mask)
            if y_indices.size > 0 and x_indices.size > 0:
                row_min, row_max = y_indices.min(), y_indices.max()
                col_min, col_max = x_indices.min(), x_indices.max()
                # Because the image is transposed (img.setImage(norm_data.T)), 
                # x corresponds to columns, y to rows, but the display origin is lower left.
                # So set ranges accordingly:
                self.plot_widget.setXRange(row_min, row_max)
                self.plot_widget.setYRange(col_min, col_max)
                self._initial_range_set = True


        selection = self.comboBox_target.currentText()
        self.plot_widget.clear()
        
        # Store current selection for colorbar zoom updates
        self._current_display_selection = selection

        img = pg.ImageItem()
        
        # === SET COLORBAR UNIT LABEL BASED ON SELECTION ===
        unit_labels = {
            "Fluss": "Flux [erg/s/cm²]",
            "W80": "W80 [km/s]",
            "Sigma": "σ [μm]",
            "Offset": "Δλ [μm]",  # Will be overwritten if velocity mode
            "[NII] Amplitude": "Flux [erg/s/cm²]",
            "Chi² Single": "χ² (reduced)",
            "Chi² Double": "χ² (reduced)",
            "bin_num": "Bin Number",
            "S/N": "Signal/Noise",
            "S/N Double": "Signal/Noise",
            "S/N NII": "Signal/Noise",
            "2nd Component Probability": "Probability"
        }
        # For Offset, check velocity checkbox (will be updated in Offset section)
        if selection != "Offset" and hasattr(self, 'colorbar_unit_label'):
            self.colorbar_unit_label.setText(unit_labels.get(selection, ""))
        
        # === HELPER: Get selected colormap and create LUT ===
        selected_cmap_name = self.comboBox_colormap.currentText()
        selected_cmap = pg.colormap.get(selected_cmap_name, source='matplotlib')
        
        # Set up colorbar for each section
        if selection == "Fluss":
            data = self.cube_amps_single
            
            # Mask zeros/invalid for better color distribution
            data_clean = np.where((data <= 0) | ~np.isfinite(data), np.nan, data)
            
            # Store for colorbar zoom - use sqrt-scaled data
            norm_data = np.sqrt(np.clip(data_clean - np.nanmin(data_clean), 0, None))
            if np.nanmax(norm_data) > 0:
                norm_data = norm_data / np.nanmax(norm_data)
            self._current_display_data = norm_data
            
            lut = selected_cmap.getLookupTable(0.0, 1.0, 256)
            # Distinct background for zeros (RGB only, no alpha)
            lut[0] = lut[0].copy()
            lut[0][:3] = [20, 20, 30]
            img.setLookupTable(lut)
            
            img.setImage(norm_data.T)
            img.setLevels([0, 1])
            if self.colorbar:
                self.colorbar.setColorMap(selected_cmap)
                self.colorbar.setLevels([0, 1])
            self.plot_widget.addItem(img)
            self._update_colorbar_range(0, 1, img)
        elif selection == "W80":
            data = np.where(mask, self.w80, np.nan)
            
            # === KEY IMPROVEMENT: Mask zeros to exclude them from colormap ===
            # Replace zeros with NaN so they don't dominate the color scale
            data_clean = np.where((data == 0) | ~np.isfinite(data), np.nan, data)
            
            # Calculate percentiles only on valid non-zero data
            valid_data = data_clean[np.isfinite(data_clean)]
            if len(valid_data) > 0:
                p1 = np.nanpercentile(valid_data, 1)
                p99 = np.nanpercentile(valid_data, 99)
            else:
                p1, p99 = 1e-3, 1.0
            
            # Use log scale for better dynamic range
            data_log = np.log10(np.clip(data_clean, a_min=max(p1, 1e-3), a_max=None))
            vmin_log = np.log10(max(p1, 1e-3))
            vmax_log = np.log10(max(p99, 1e-3))
            
            # Store log-scaled data for colorbar zoom
            self._current_display_data = data_log

            # === USE SELECTED COLORMAP from dropdown ===
            colormap = pg.ColorMap(
                pos=np.linspace(0.0, 1.0, 512),
                color=selected_cmap.getLookupTable(0.0, 1.0, 512)
            )
            lut = colormap.getLookupTable(0.0, 1.0, 512)
            
            # Set NaN/masked values to a distinct color (dark gray)
            # Handle both RGB and RGBA LUTs
            lut[0] = lut[0].copy()
            lut[0][:3] = [40, 40, 50]
            
            img.setLookupTable(lut)
            img.setImage(data_log.T, autoLevels=False)
            img.setLevels([vmin_log, vmax_log])
            if self.colorbar:
                self.colorbar.setColorMap(colormap)
                self.colorbar.setLevels([vmin_log, vmax_log])
            self.plot_widget.addItem(img)
            self._update_colorbar_range(vmin_log, vmax_log, img)

            # ensure previous overlays removed
            if not hasattr(self, 'contour_items'):
                self.contour_items = []
            for it in self.contour_items:
                try:
                    self.plot_widget.removeItem(it)
                except Exception:
                    pass
            self.contour_items = []

            # overlay small Gaussian line-shapes only if checkbox is checked
            if self.checkbox_mini_gaussians.isChecked():
                try:
                    # small helper gaussians
                    def gaussian(x, A, B, C):
                        return np.abs(A) * np.exp(-0.5 * ((x - C) / B) ** 2)

                    def double_gaussian(x, A1, B1, C1, A2, B2, C2):
                        return (np.abs(A1) * np.exp(-0.5 * ((x - C1) / B1) ** 2) +
                                np.abs(A2) * np.exp(-0.5 * ((x - C2) / B2) ** 2))
                    
                    mask_map = np.isfinite(data) & (data > 0)
                    valid_indices = np.argwhere(mask_map)
                    if valid_indices.size > 0:
                        # limit number of drawn mini-spectra for performance
                        max_draw = 2000
                        step = max(1, len(valid_indices) // max_draw)
                        chi1 = getattr(self, 'cube_chi_squared_one', None)
                        chi2 = getattr(self, 'cube_chi_squared_two', None)

                        for k in range(0, len(valid_indices), step):
                            i, j = valid_indices[k]
                            try:
                                use_single = True
                                if chi1 is not None and chi2 is not None and np.isfinite(chi1[i, j]) and np.isfinite(chi2[i, j]):
                                    use_single = abs(1 - chi1[i, j]) < abs(1 - chi2[i, j])

                                if use_single:
                                    A = self.cube_amps_single[i, j] if hasattr(self, 'cube_amps_single') else np.nan
                                    B = self.cube_sigmas_single[i, j] if hasattr(self, 'cube_sigmas_single') else np.nan
                                    C = self.cube_offsets_single[i, j] if hasattr(self, 'cube_offsets_single') else np.nan
                                    if not np.isfinite([A, B, C]).all():
                                        continue
                                    x_wave = np.linspace(C - B * 3, C + B * 3, 40)
                                    y = gaussian(x_wave, A, B, C)
                                else:
                                    A1 = self.cube_amps_double_one[i, j]
                                    B1 = self.cube_sigmas_double_one[i, j]
                                    C1 = self.cube_offsets_double_one[i, j]
                                    A2 = self.cube_amps_double_two[i, j]
                                    B2 = self.cube_sigmas_double_two[i, j]
                                    C2 = self.cube_offsets_double_two[i, j]
                                    if not np.isfinite([A1, B1, C1, A2, B2, C2]).all():
                                        continue
                                    dominant = 1 if B1 > B2 else 2
                                    width = (B1 if dominant == 1 else B2) * 3
                                    center = C1 if dominant == 1 else C2
                                    x_wave = np.linspace(center - width, center + width, 40)
                                    y = double_gaussian(x_wave, A1, B1, C1, A2, B2, C2)

                                # normalize and scale vertically so shapes fit in one pixel height
                                if np.nanmax(y) == 0 or np.all(np.isnan(y)):
                                    continue
                                y_norm = y / np.nanmax(y) * 0.6  # amplitude scaled to ~0.4 pixel

                                # matplotlib-style coordinates (columns, rows)
                                x_plot_mat = np.linspace(-0.4, 0.4, len(x_wave)) + j      # columns
                                y_plot_mat = y_norm + i - 0.4                            # rows

                                # Use native column (x) and row (y) coordinates for the mini-spectra so
                                # the small Gaussian shapes are oriented along the image's horizontal axis.
                                # Previously these were swapped (x_pg = y_plot_mat, y_pg = x_plot_mat),
                                # which rotated the curves by 90°; keep the natural ordering here.
                                x_pg = x_plot_mat
                                y_pg = y_plot_mat


                                xp_rel = np.linspace(-0.4, 0.4, len(x_wave))           # relative horizontal shape
                                yp_rel = y_norm                                  # relative vertical shape

                                # Swap centers: use row index (i) as x (horizontal) and column (j) as y (vertical)
                                x_pg_swapped = xp_rel + i + 0.5
                                y_pg_swapped = yp_rel + j + 0.1

                                curve_swapped = pg.PlotCurveItem(x_pg_swapped, y_pg_swapped,
                                                                    pen=pg.mkPen('k', width=0.75), antialias=True)
                                self.plot_widget.addItem(curve_swapped)
                                self.contour_items.append(curve_swapped)
                            except Exception:
                                pass
                            continue


                    # lock aspect ratio to keep pixels square (so overlays align)
                    try:
                        self.plot_widget.getViewBox().setAspectLocked(True, ratio=1)
                    except Exception:
                        pass

                except Exception:
                    # keep GUI usable on any error
                    pass
        elif selection == "Sigma":
            data = self.cube_sigmas_single
            lut = selected_cmap.getLookupTable(0.0, 1.0, 256)
            lut[0] = lut[0].copy()
            lut[0][:3] = [20, 20, 30]
            img.setLookupTable(lut)
            norm_data = np.sqrt(np.clip(data - np.nanmin(data), 0, None))
            if np.nanmax(norm_data) > 0:
                norm_data = norm_data / np.nanmax(norm_data)
            img.setImage(norm_data.T)
            if self.colorbar:
                self.colorbar.setColorMap(selected_cmap)
            self.colorbar.setLevels([0, 1])
            self.plot_widget.addItem(img)
            self._current_display_data = norm_data  # Store for colorbar zoom
        elif selection == "Offset":
            # Speed of light in km/s
            c_km_s = 299792.458
            
            # Calculate wavelength offset from Hα
            data_wavelength = self.cube_offsets_single - halpha_wavelength
            
            # Check if velocity mode is enabled
            show_velocity = self.checkbox_velocity.isChecked()
            
            if show_velocity:
                # Convert Δλ to velocity: v = c * Δλ / λ₀
                data = (data_wavelength / halpha_wavelength) * c_km_s
                unit_label = "v [km/s]"
            else:
                data = data_wavelength
                unit_label = "Δλ [μm]"
            
            # Update unit label
            if hasattr(self, 'colorbar_unit_label'):
                self.colorbar_unit_label.setText(unit_label)
            
            # Store data for colorbar zoom
            self._current_display_data = data
            
            # Calculate symmetric color range around mean
            mean_val = np.nanmean(data[np.isfinite(data)])
            p5 = np.nanpercentile(data[mask_strict], 5)
            p95 = np.nanpercentile(data[mask_strict], 95)
            delta = max(abs(mean_val - p5), abs(p95 - mean_val))
            vmin = mean_val - delta
            vmax = mean_val + delta
            
            # For velocity/offset, use selected colormap but invert for intuitive red/blue shift
            colormap = pg.ColorMap(
                pos=np.linspace(0.0, 1.0, 512),
                color=selected_cmap.getLookupTable(1.0, 0.0, 512)  # inverted
            )
            lut = colormap.getLookupTable(0.0, 1.0, 512)
            img.setLookupTable(lut)
            img.setImage(data.T, autoLevels=False)
            img.setLevels([vmin, vmax])
            if self.colorbar:
                self.colorbar.setColorMap(colormap)
                self.colorbar.setLevels([vmin, vmax])
            self.plot_widget.addItem(img)
            self._update_colorbar_range(vmin, vmax, img)
        elif selection == "[NII] Amplitude":
            data = self.cube_amps_nii
            lut = selected_cmap.getLookupTable(0.0, 1.0, 256)
            lut[0] = lut[0].copy()
            lut[0][:3] = [20, 20, 30]
            img.setLookupTable(lut)
            norm_data = np.sqrt(np.clip(data - np.nanmin(data), 0, None))
            if np.nanmax(norm_data) > 0:
                norm_data = norm_data / np.nanmax(norm_data)
            img.setImage(norm_data.T)
            img.setLevels([0, 1])
            if self.colorbar:
                self.colorbar.setColorMap(selected_cmap)
                self.colorbar.setLevels([0, 1])
            self.plot_widget.addItem(img)
            self._current_display_data = norm_data
            self._update_colorbar_range(0, 1, img)
        elif selection == "Chi² Single":
            data = self.cube_chi_squared_one
            lut = selected_cmap.getLookupTable(0.0, 1.0, 256)
            lut[0] = lut[0].copy()
            lut[0][:3] = [20, 20, 30]
            img.setLookupTable(lut)
            norm_data = np.sqrt(np.clip(data - np.nanmin(data), 0, None))
            if np.nanmax(norm_data) > 0:
                norm_data = norm_data / np.nanmax(norm_data)
            img.setImage(norm_data.T)
            img.setLevels([0, 1])
            if self.colorbar:
                self.colorbar.setColorMap(selected_cmap)
                self.colorbar.setLevels([0, 1])
            self.plot_widget.addItem(img)
            self._current_display_data = norm_data
            self._update_colorbar_range(0, 1, img)
        elif selection == "Chi² Double":
            data = self.cube_chi_squared_two
            lut = selected_cmap.getLookupTable(0.0, 1.0, 256)
            lut[0] = lut[0].copy()
            lut[0][:3] = [20, 20, 30]
            img.setLookupTable(lut)
            norm_data = np.sqrt(np.clip(data - np.nanmin(data), 0, None))
            if np.nanmax(norm_data) > 0:
                norm_data = norm_data / np.nanmax(norm_data)
            img.setImage(norm_data.T)
            img.setLevels([0, 1])
            if self.colorbar:
                self.colorbar.setColorMap(selected_cmap)
                self.colorbar.setLevels([0, 1])
            self.plot_widget.addItem(img)
            self._current_display_data = norm_data
            self._update_colorbar_range(0, 1, img)
        elif selection == "2nd Component Probability":
            probability_1_s = stats.chi2.sf((chi1)*5, 5) 
            #print(reduced_chi2_1, reduced_chi2_2)
            probability_2_s = stats.chi2.sf(chi2*7, 7) 
            #print(probability_1_s.shape, probability_2_s.shape)
            percentage_outflow = (probability_2_s - probability_1_s)/2 + 0.5 

            data = np.where(mask, percentage_outflow, np.nan)

            lut = selected_cmap.getLookupTable(0.0, 1.0, 256)
            lut[0] = lut[0].copy()
            lut[0][:3] = [20, 20, 30]
            img.setLookupTable(lut)
            norm_data = data
            img.setImage(norm_data.T)
            img.setLevels([0, 1])
            if self.colorbar:
                self.colorbar.setColorMap(selected_cmap)
                self.colorbar.setLevels([0, 1])
            self.plot_widget.addItem(img)
            self._current_display_data = norm_data
            self._update_colorbar_range(0, 1, img)
            levels = [1/3, 1/2, 2/3]
            
            for lev in levels:
                contours = measure.find_contours(data, level=lev)
                for contour in contours:
                    x = contour[:, 1]
                    y = contour[:, 0]
                    # Swap x and y for display due to transpose
                    curve = pg.PlotCurveItem(y+0.5, x+0.5, pen=pg.mkPen('w', width=1, style=QtCore.Qt.DashLine))
                    if lev == 1/2:
                        curve.setPen(pg.mkPen('r', width=3))
                    self.plot_widget.addItem(curve)
                    self.contour_items.append(curve)
            
        elif selection == "bin_num":
            data = self.cube_bin_num
            # Use 'flag' colormap for bin numbers (good for discrete values), but allow user override
            lut = selected_cmap.getLookupTable(0.0, 1.0, 256)
            img.setLookupTable(lut)
            norm_data = np.sqrt(np.clip(data - np.nanmin(data), 0, None))
            if np.nanmax(norm_data) > 0:
                norm_data = norm_data / np.nanmax(norm_data)
            img.setImage(norm_data.T)
            img.setLevels([0, 1])
            if self.colorbar:
                self.colorbar.setColorMap(selected_cmap)
                self.colorbar.setLevels([0, 1])
            self.plot_widget.addItem(img)
            self._update_colorbar_range(0, 1, img)
        elif selection == "S/N":
            data = np.where(mask, self.cube_son_one, np.nan)
            
            # Mask zeros and invalid values for better color distribution
            data_clean = np.where((data <= 0) | ~np.isfinite(data), np.nan, data)
            
            # Calculate proper normalization from valid data only
            valid_data = data_clean[np.isfinite(data_clean)]
            if len(valid_data) > 0:
                vmin = 0
                vmax = np.nanpercentile(valid_data, 98)  # Use 98th percentile to avoid outlier stretching
            else:
                vmin, vmax = 0, 1
            
            # Normalize data for display
            if vmax > 0:
                norm_data = np.clip(data_clean / vmax, 0, 1)
            else:
                norm_data = data_clean
            
            # Store for colorbar zoom (use original data, not normalized)
            self._current_display_data = data_clean

            # Set up colormap and LUT using selected colormap
            lut = selected_cmap.getLookupTable(0.0, 1.0, 256)
            # Make the lowest value (zeros/NaN) a distinct dark color
            lut[0] = lut[0].copy()
            lut[0][:3] = [20, 20, 30]
            img.setLookupTable(lut)

            # Contour calculation
            contour_data = np.nan_to_num(data, nan=0.0)
            
            if not hasattr(self, 'contour_items'):
                self.contour_items = []
            for item in self.contour_items:
                self.plot_widget.removeItem(item)
            self.contour_items = []

            img.setImage(norm_data.T)
            img.setLevels([0, 1])
            if self.colorbar:
                self.colorbar.setColorMap(selected_cmap)
                self.colorbar.setLevels([0, vmax])
            self.plot_widget.addItem(img)
            self._update_colorbar_range(0, vmax, img)
            levels = [1., 3., 5., 10.]

            for lev in levels:
                contours = measure.find_contours(contour_data, level=lev)
                for contour in contours:
                    x = contour[:, 1]
                    y = contour[:, 0]
                    # Swap x and y for display due to transpose
                    curve = pg.PlotCurveItem(y+0.5, x+0.5, pen=pg.mkPen('w', width=1))
                    if lev == 5:
                        curve.setPen(pg.mkPen('r', width=3))
                    if lev == 3:
                        curve.setPen(pg.mkPen('w', width=3))
                    self.plot_widget.addItem(curve)
                    self.contour_items.append(curve)
            
        elif selection == "S/N Double":
            data = np.where(mask, self.cube_son_two, np.nan)
            lut = selected_cmap.getLookupTable(0.0, 1.0, 256)
            lut[0] = lut[0].copy()
            lut[0][:3] = [20, 20, 30]
            img.setLookupTable(lut)
            norm_data = np.sqrt(np.clip(data - np.nanmin(data), 0, None))
            if np.nanmax(norm_data) > 0:
                norm_data = norm_data / np.nanmax(norm_data)
            img.setImage(norm_data.T)
            img.setLevels([0, 1])
            if self.colorbar:
                self.colorbar.setColorMap(selected_cmap)
                self.colorbar.setLevels([0, 1])
            self.plot_widget.addItem(img)
            self._current_display_data = norm_data
            self._update_colorbar_range(0, 1, img)
        elif selection == "S/N NII":
            data = np.where(mask, self.cube_son_one, np.nan)
            lut = selected_cmap.getLookupTable(0.0, 1.0, 256)
            lut[0] = lut[0].copy()
            lut[0][:3] = [20, 20, 30]
            img.setLookupTable(lut)
            norm_data = np.sqrt(np.clip(data - np.nanmin(data), 0, None))
            if np.nanmax(norm_data) > 0:
                norm_data = norm_data / np.nanmax(norm_data)
            img.setImage(norm_data.T)
            img.setLevels([0, 1])
            if self.colorbar:
                self.colorbar.setColorMap(selected_cmap)
                self.colorbar.setLevels([0, 1])
            self.plot_widget.addItem(img)
            self._current_display_data = norm_data
            self._update_colorbar_range(0, 1, img)
        else:
            data = np.zeros_like(self.cube_amps_single)
            lut = selected_cmap.getLookupTable(0.0, 1.0, 256)
            img.setLookupTable(lut)
            img.setImage(data.T)
            img.setLevels([0, 1])
            if self.colorbar:
                self.colorbar.setColorMap(selected_cmap)
                self.colorbar.setLevels([0, 1])
            self.plot_widget.addItem(img)
            self._current_display_data = data
            self._update_colorbar_range(0, 1, img)
        
        # === DRAW MINI-GAUSSIANS FOR ALL VISUALIZATIONS (except W80 which handles it separately) ===
        if selection != "W80" and self.checkbox_mini_gaussians.isChecked():
            self._draw_mini_gaussians()

    def _draw_mini_gaussians(self):
        """Helper method to draw mini-Gaussian overlays on any visualization"""
        # ensure previous overlays removed
        if not hasattr(self, 'contour_items'):
            self.contour_items = []
        # Don't clear contour_items here as some visualizations (S/N, 2nd Component) add their own contours
        
        try:
            # small helper gaussians
            def gaussian(x, A, B, C):
                return np.abs(A) * np.exp(-0.5 * ((x - C) / B) ** 2)

            def double_gaussian(x, A1, B1, C1, A2, B2, C2):
                return (np.abs(A1) * np.exp(-0.5 * ((x - C1) / B1) ** 2) +
                        np.abs(A2) * np.exp(-0.5 * ((x - C2) / B2) ** 2))
            
            # Use flux data to determine where to draw
            data = self.cube_amps_single
            mask_map = np.isfinite(data) & (data > 0)
            valid_indices = np.argwhere(mask_map)
            
            if valid_indices.size > 0:
                # limit number of drawn mini-spectra for performance
                max_draw = 2000
                step = max(1, len(valid_indices) // max_draw)
                chi1 = getattr(self, 'cube_chi_squared_one', None)
                chi2 = getattr(self, 'cube_chi_squared_two', None)

                for k in range(0, len(valid_indices), step):
                    i, j = valid_indices[k]
                    try:
                        use_single = True
                        if chi1 is not None and chi2 is not None and np.isfinite(chi1[i, j]) and np.isfinite(chi2[i, j]):
                            use_single = abs(1 - chi1[i, j]) < abs(1 - chi2[i, j])

                        if use_single:
                            A = self.cube_amps_single[i, j] if hasattr(self, 'cube_amps_single') else np.nan
                            B = self.cube_sigmas_single[i, j] if hasattr(self, 'cube_sigmas_single') else np.nan
                            C = self.cube_offsets_single[i, j] if hasattr(self, 'cube_offsets_single') else np.nan
                            if not np.isfinite([A, B, C]).all():
                                continue
                            x_wave = np.linspace(C - B * 3, C + B * 3, 40)
                            y = gaussian(x_wave, A, B, C)
                        else:
                            A1 = self.cube_amps_double_one[i, j]
                            B1 = self.cube_sigmas_double_one[i, j]
                            C1 = self.cube_offsets_double_one[i, j]
                            A2 = self.cube_amps_double_two[i, j]
                            B2 = self.cube_sigmas_double_two[i, j]
                            C2 = self.cube_offsets_double_two[i, j]
                            if not np.isfinite([A1, B1, C1, A2, B2, C2]).all():
                                continue
                            dominant = 1 if B1 > B2 else 2
                            width = (B1 if dominant == 1 else B2) * 3
                            center = C1 if dominant == 1 else C2
                            x_wave = np.linspace(center - width, center + width, 40)
                            y = double_gaussian(x_wave, A1, B1, C1, A2, B2, C2)

                        # normalize and scale vertically so shapes fit in one pixel height
                        if np.nanmax(y) == 0 or np.all(np.isnan(y)):
                            continue
                        y_norm = y / np.nanmax(y) * 0.6

                        xp_rel = np.linspace(-0.4, 0.4, len(x_wave))
                        yp_rel = y_norm

                        # Swap centers: use row index (i) as x (horizontal) and column (j) as y (vertical)
                        x_pg_swapped = xp_rel + i + 0.5
                        y_pg_swapped = yp_rel + j + 0.1

                        curve_swapped = pg.PlotCurveItem(x_pg_swapped, y_pg_swapped,
                                                            pen=pg.mkPen('k', width=0.75), antialias=True)
                        self.plot_widget.addItem(curve_swapped)
                        self.contour_items.append(curve_swapped)
                    except Exception:
                        pass

            # lock aspect ratio to keep pixels square
            try:
                self.plot_widget.getViewBox().setAspectLocked(True, ratio=1)
            except Exception:
                pass

        except Exception:
            # keep GUI usable on any error
            pass



    def create_cube_from_fits_values(self, data, file_fit_results=None):
        self.cube_chi_squared_one = data[12].data
        self.cube_chi_squared_two = data[13].data
        self.cube_son_one = data[14].data
        self.cube_son_two = data[15].data
        self.w80 = tools.master_map_w80(file_fit_results)
        self.cube_amps_single = data[1].data
        self.cube_sigmas_single = data[4].data
        self.cube_offsets_single = data[7].data
        self.cube_amps_double_one = data[2].data
        self.cube_sigmas_double_one = data[5].data
        self.cube_offsets_double_one = data[8].data
        self.cube_amps_double_two = data[3].data
        self.cube_sigmas_double_two = data[6].data
        self.cube_offsets_double_two = data[9].data
        self.continuum_slope = data[10].data
        self.continuum_intercept = data[11].data
        
        self.cube_amps_nii1 = data[16].data
        self.cube_sigmas_nii1 = data[17].data
        self.cube_offsets_nii1 = data[18].data
        self.cube_amps_nii2 = data[19].data
        self.cube_sigmas_nii2 = data[20].data
        self.cube_offsets_nii2 = data[21].data
        self.nii_single_used = data[26].data
        self.nii_double_used = data[27].data
        try:
            self.cube_bin_num = data[24].data
        except IndexError:
            shape = np.shape(data[1].data)
            self.cube_bin_num = np.arange(shape[0] * shape[1]).reshape(shape[0], shape[1])

    def change_file(self):
        idx, ok = QtWidgets.QInputDialog.getInt(self, "Neue Datei wählen", f"Index (0 - {len(filenames)-1}):", 0, 0, len(filenames)-1, 1)
        if not ok:
            return
        file_fit_results = filenames[idx]
        print(f"Selected file index: {idx}, file: {file_fit_results}")
        print(f"Loading file: {file_fit_results}")
        file_main_cube = file_fit_results.replace('Gaussian_fits\\', 'KMOS3D_ALL\\').replace('_voronoi_binned.fits', '.fits')
        main_cube = fits.open(file_main_cube)
        z_redshift = tools.redshift(main_cube)
        fit_data = fits.open(file_fit_results)

        self.main_cube = main_cube
        self.main_cube_data = main_cube[1].data
        self.main_cube_error = main_cube[2].data
        self.z_redshift = z_redshift
        self.psf = tools.psf(main_cube)

        self.define_spectral_axis()
        self.create_cube_from_fits_values(fit_data, file_fit_results=file_fit_results)
        
        # Reset range flag so new file gets proper zoom
        self._initial_range_set = False
        
        self.update_image_display()
        self.idx_label.setText(f"Galaxy Index: {idx}")
        self.plot_widget_2.clear()

def run_my_app(main_cube, fit_data, z_redshift):
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    window = MyApp(main_cube, fit_data, z_redshift)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    idx, ok = QtWidgets.QInputDialog.getInt(None, "Select FITS file", f"Enter index (0 - {len(filenames)-1}):", 0, 0, len(filenames)-1, 1)
    if not ok:
        sys.exit(0)
    file_fit_results = filenames[idx]
    file_main_cube = file_fit_results.replace('Gaussian_fits\\', 'KMOS3D_ALL\\').replace('_voronoi_binned.fits', '.fits')
    main_cube = fits.open(file_main_cube)
    z_redshift = tools.redshift(main_cube)
    fit_data = fits.open(file_fit_results)
    run_my_app(main_cube, fit_data, z_redshift)