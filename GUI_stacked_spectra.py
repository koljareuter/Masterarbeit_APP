import os
import sys
from pathlib import Path

# Force software rendering to fix MESA/GLX errors
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

# Platform-specific Qt settings
if sys.platform == 'win32':
    # On Windows, use the native platform
    pass  # Qt will use the default Windows platform
elif sys.platform == 'darwin':
    # macOS
    pass
else:
    # Linux - sometimes needs offscreen or xcb
    if 'QT_QPA_PLATFORM' not in os.environ:
        # Only set if not already configured by user
        pass  # Let Qt auto-detect

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt backend for standalone GUI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Ellipse
from astropy.io import fits
import pandas as pd
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QPushButton, QLabel, 
                             QComboBox, QCheckBox, QFileDialog, QGroupBox,
                             QMessageBox, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor
import warnings
warnings.filterwarnings('ignore')

####################### HELP FUNCTIONS ###########################

def normalize_path(filepath):
    """Normalize a filepath to work on both Windows and Linux.
    
    Converts all path separators to the OS-native format and resolves the path.
    """
    if filepath is None:
        return None
    # Convert to Path object which handles cross-platform normalization
    return str(Path(filepath).resolve())


def extract_filename_from_path(filepath):
    """Extract just the filename from a full path, cross-platform compatible.
    
    Works with both Windows (backslash) and Unix (forward slash) paths,
    regardless of the current operating system.
    """
    if filepath is None:
        return None
    
    # Use pathlib for cross-platform handling
    # First, try to handle the path as-is
    p = Path(filepath)
    filename = p.name
    
    # If the filename still contains path separators (e.g., Windows path on Linux),
    # manually extract the last component
    if '\\' in filename:
        filename = filename.split('\\')[-1]
    if '/' in filename:
        filename = filename.split('/')[-1]
    
    return filename


# Determine the redshift of the galaxy
def redshift(hdulist1, FILE='k3d_fnlsp_table_v3.fits', testdata=False):
    '''Determine the redshift of the galaxy from the FITS file header and a reference FITS table.
    Parameters: hdulist1: astropy.io.fits.HDUList'''
    if testdata:
        return 0, False
    
    # Try to find the reference file
    reference_file = Path(FILE)
    if not reference_file.exists():
        # Try looking in the same directory as the script
        script_dir = Path(__file__).parent
        reference_file = script_dir / FILE
        if not reference_file.exists():
            # Try looking in the current working directory
            reference_file = Path.cwd() / FILE
            if not reference_file.exists():
                raise FileNotFoundError(
                    f"Reference file '{FILE}' not found. "
                    f"Searched in: current directory, script directory."
                )
    
    hdulist = fits.open(str(reference_file))
    
    filename = hdulist1.filename()
    data = hdulist[1].data
    
    # Convert the FITS data table to a structured array and then to a pandas DataFrame
    structured_array = np.array(data).byteswap().newbyteorder()
    df = pd.DataFrame(structured_array)
    
    # Extract just the filename using cross-platform method
    ID = extract_filename_from_path(filename)
    
    # Clean up the ID - remove common path prefixes that might be embedded
    # Use a pattern that matches both forward and back slashes
    patterns_to_remove = [
        r'.*[/\\]KMOS3D_ALL[/\\]',
        r'.*[/\\]KMOS3D[/\\]',
        r'.*[/\\]Gaussian_fits[/\\]?',
        r'^KMOS3D_ALL[/\\]',
        r'^KMOS3D[/\\]',
        r'^Gaussian_fits[/\\]?',
        r'_voronoi_binned\.fits$',
    ]
    
    for pattern in patterns_to_remove:
        ID = re.sub(pattern, '', ID, flags=re.IGNORECASE)
    
    # Remove trailing '.fits' if present, then add it back consistently
    ID = re.sub(r'\.fits$', '', ID, flags=re.IGNORECASE)
    ID = ID.strip() + '.fits'
    
    # Clean both the ID and the dataframe column
    # Convert bytes to strings first
    df['FILE'] = df['FILE'].str.decode('utf-8')

    # Now you can do string operations
    ID = ID.strip()
    matching_rows = df[df['FILE'].str.strip() == ID]
    
    if len(matching_rows) == 0:
        # Try matching without the .fits extension as well
        ID_no_ext = re.sub(r'\.fits$', '', ID, flags=re.IGNORECASE)
        matching_rows = df[df['FILE'].str.strip().str.replace('.fits', '', case=False, regex=False) == ID_no_ext]
        
    if len(matching_rows) == 0:
        raise ValueError(f'Error: Filename not found in the database. '
                        f'Looking for: "{ID}", '
                        f'Sample from DB: "{df["FILE"].iloc[0] if len(df) > 0 else "empty"}"')

    df_filtered = matching_rows
    z = df_filtered['Z'].iloc[0]
    flag = False
    if z < 0:
        flag = True
        z = df_filtered['Z_TARGETED'].iloc[0]

    return z, flag


def psf(hdulist):
    ref_pixel_flux_x = hdulist[1].header['CRPIX1']
    ref_pixel_flux_y = hdulist[1].header['CRPIX2']
    ref_pixel_psf_x = hdulist[4].header['CRPIX1']
    ref_pixel_psf_y = hdulist[4].header['CRPIX2']
    delta_x = ref_pixel_flux_x - ref_pixel_psf_x
    delta_y = ref_pixel_flux_y - ref_pixel_psf_y

    psf_fwhm = hdulist[4].header['FWHM']
    psf_raw = hdulist[4].data
    psf = np.zeros_like(hdulist[1].data[1800])
    for i in range(hdulist[1].data[1800].shape[0]):
        for j in range(hdulist[1].data[1800].shape[1]):
            if delta_x % 1 == 0 and delta_y % 1 == 0:
                psf[i][j] = psf_raw[i - int(delta_y)][j - int(delta_x)]
            elif delta_x % 1 == 0.5 and delta_y % 1 == 0.5:
                factors = []
                if i - int(delta_y + 0.5) < 20 and j - int(delta_x + 0.5) < 20:
                    factors.append(psf_raw[i - int(delta_y + 0.5)][j - int(delta_x + 0.5)])
                if i - int(delta_y + 0.5) < 20 and j - int(delta_x - 0.5) < 20:
                    factors.append(psf_raw[i - int(delta_y + 0.5)][j - int(delta_x - 0.5)])
                if i - int(delta_y - 0.5) < 20 and j - int(delta_x + 0.5) < 20:
                    factors.append(psf_raw[i - int(delta_y - 0.5)][j - int(delta_x + 0.5)])
                if i - int(delta_y - 0.5) < 20 and j - int(delta_x - 0.5) < 20:
                    factors.append(psf_raw[i - int(delta_y - 0.5)][j - int(delta_x - 0.5)])
                if factors:
                    psf[i][j] = np.nanmean(factors)
                else:
                    psf[i][j] = np.nan
            else:
                print('Error: Delta X and Delta Y are not integer or half-integer values, but a mixture of both.')
                break
    return psf


# Suppress matplotlib warnings
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# ==================== Color Theme ====================
class ModernTheme:
    # Dark modern theme
    BG_DARK = '#0F172A'
    BG_MEDIUM = '#1E293B'
    BG_LIGHT = '#334155'
    ACCENT_CYAN = '#06B6D4'
    ACCENT_EMERALD = '#10B981'
    ACCENT_BLUE = '#3B82F6'
    ACCENT_PURPLE = '#8B5CF6'
    ACCENT_PINK = '#EC4899'
    TEXT_PRIMARY = '#F1F5F9'
    TEXT_SECONDARY = '#94A3B8'
    LINE_RED = '#EF4444'
    LINE_ORANGE = '#F97316'
    LINE_LIME = '#84CC16'

# ==================== Helper Functions ====================
def calculate_distance_grid(shape, center_x, center_y):
    """Calculate Euclidean distance from center for each pixel."""
    y, x = np.indices(shape)
    return np.sqrt((x - center_x)**2 + (y - center_y)**2)


def extract_psf_parameters(hdu_list):
    """Extract elliptical PSF parameters from FITS header [4]."""
    try:
        # Access header from extension 4 (as requested)
        header = hdu_list[4].header
        
        # Extract keys (using the HIERARCH keys you provided)
        fwhm_maj = header.get('HIERARCH ESO K3D PSF GAUSS FWHM_MAJ', 3.5)
        fwhm_min = header.get('HIERARCH ESO K3D PSF GAUSS FWHM_MIN', 3.5)
        pa = header.get('HIERARCH ESO K3D PSF GAUSS PA', 0.0)
        
        # KMOS Headers often give FWHM in arcseconds. 
        # We need to convert to pixels. 
        # Standard KMOS pixel scale is 0.2 arcsec/pixel (verify if needed)
        pixel_scale = 0.2 
        
        fwhm_maj_px = fwhm_maj / pixel_scale
        fwhm_min_px = fwhm_min / pixel_scale
        
        return fwhm_maj_px, fwhm_min_px, pa
        
    except (IndexError, KeyError, ValueError):
        # Fallback if header[4] doesn't exist or keys are missing
        print("Warning: Could not extract PSF params. Using default.")
        return 3.7, 3.7, 0.0

def extract_wavelength_calibration(header, n_spectral):
    """Extract wavelength solution from FITS header."""
    try:
        crval = header['CRVAL3']
        cdelt = header['CDELT3']
        crpix = header['CRPIX3']
        indices = np.arange(n_spectral)
        wavelengths = crval + (indices - (crpix - 1)) * cdelt
        return wavelengths
    except KeyError:
        return np.arange(n_spectral)

def create_aperture_mask(shape, center_x, center_y, radius_px, exp_map, exp_threshold=0.5):
    """Create spatial mask combining aperture and exposure criteria."""
    dist_grid = calculate_distance_grid(shape, center_x, center_y)
    aperture_mask = dist_grid <= radius_px
    exp_normalized = exp_map / np.nanmax(exp_map)
    valid_exp = exp_normalized > exp_threshold
    return aperture_mask & valid_exp

def weighted_stack_spectrum(flux_cube, noise_cube, mask_2d):
    """Perform inverse-variance weighted stacking of spectrum."""
    mask_3d = np.broadcast_to(mask_2d, flux_cube.shape)
    variance = noise_cube**2
    weights = np.divide(1.0, variance, out=np.zeros_like(variance), where=variance > 0)
    weights[~mask_3d] = 0
    weights[np.isnan(weights)] = 0
    
    numerator = np.nansum(flux_cube * weights, axis=(1, 2))
    sum_weights = np.nansum(weights, axis=(1, 2))
    
    stacked_flux = np.divide(numerator, sum_weights, 
                             out=np.zeros_like(numerator), 
                             where=sum_weights > 0)
    stacked_noise = np.sqrt(np.divide(1.0, sum_weights, 
                                       out=np.zeros_like(sum_weights), 
                                       where=sum_weights > 0))
    stacked_snr = np.divide(stacked_flux, stacked_noise, 
                            out=np.zeros_like(stacked_flux), 
                            where=stacked_noise > 0)
    
    return stacked_flux, stacked_noise, stacked_snr



# ==================== Main GUI Application ====================
class KMOSViewerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('KMOS Spectrum Viewer - Stacking Edition')
        self.setGeometry(100, 100, 1600, 1000)
        
        # Data storage
        self.flux_cube = None
        self.noise_cube = None
        self.exp_map = None
        self.header = None
        self.wavelengths = None
        self.white_light = None
        self.center_x = None
        self.center_y = None
        self.obj_name = None
        self.redshift = 2.2  # Random float, will be updated upon loading
        self.flag_redshift = False
        
        # Interactive state
        self.dragging_aperture = False
        self.aperture_circle = None
        self.psf_circle = None
        self.ax_img = None
        self.ax_zoom = None
        self.ax_full = None
        
        self.plot_initialized = False
        self.im_obj = None       # The image object
        self.aperture_obj = None # The circle
        self.zoom_line = None    # The zoom plot line
        self.zoom_fill = None    # The zoom fill
        self.full_line = None    # The full spectrum line
        self.full_fill = None    # The full fill


        # Apply modern dark theme
        self.apply_dark_theme()
        
        # Create GUI
        self.init_ui()
        
    def apply_dark_theme(self):
        """Apply modern dark theme to the application."""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(ModernTheme.BG_DARK))
        palette.setColor(QPalette.WindowText, QColor(ModernTheme.TEXT_PRIMARY))
        palette.setColor(QPalette.Base, QColor(ModernTheme.BG_MEDIUM))
        palette.setColor(QPalette.AlternateBase, QColor(ModernTheme.BG_LIGHT))
        palette.setColor(QPalette.ToolTipBase, QColor(ModernTheme.TEXT_PRIMARY))
        palette.setColor(QPalette.ToolTipText, QColor(ModernTheme.TEXT_PRIMARY))
        palette.setColor(QPalette.Text, QColor(ModernTheme.TEXT_PRIMARY))
        palette.setColor(QPalette.Button, QColor(ModernTheme.BG_LIGHT))
        palette.setColor(QPalette.ButtonText, QColor(ModernTheme.TEXT_PRIMARY))
        palette.setColor(QPalette.Highlight, QColor(ModernTheme.ACCENT_CYAN))
        palette.setColor(QPalette.HighlightedText, QColor(ModernTheme.BG_DARK))
        self.setPalette(palette)
        
        # Set stylesheet for modern look
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {ModernTheme.BG_DARK};
            }}
            QLabel {{
                color: {ModernTheme.TEXT_PRIMARY};
                font-size: 13px;
            }}
            QPushButton {{
                background-color: {ModernTheme.ACCENT_CYAN};
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: #0891B2;
            }}
            QPushButton:pressed {{
                background-color: #0E7490;
            }}
            QSlider::groove:horizontal {{
                border: 1px solid {ModernTheme.BG_LIGHT};
                height: 8px;
                background: {ModernTheme.BG_MEDIUM};
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {ModernTheme.ACCENT_CYAN};
                border: 2px solid {ModernTheme.ACCENT_CYAN};
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }}
            QSlider::handle:horizontal:hover {{
                background: #0891B2;
            }}
            QComboBox {{
                background-color: {ModernTheme.BG_LIGHT};
                color: {ModernTheme.TEXT_PRIMARY};
                border: 2px solid {ModernTheme.BG_LIGHT};
                padding: 8px;
                border-radius: 6px;
                font-size: 13px;
            }}
            QComboBox:hover {{
                border: 2px solid {ModernTheme.ACCENT_CYAN};
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox QAbstractItemView {{
                background-color: {ModernTheme.BG_LIGHT};
                color: {ModernTheme.TEXT_PRIMARY};
                selection-background-color: {ModernTheme.ACCENT_CYAN};
            }}
            QCheckBox {{
                color: {ModernTheme.TEXT_PRIMARY};
                font-size: 13px;
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 20px;
                height: 20px;
                border: 2px solid {ModernTheme.BG_LIGHT};
                border-radius: 4px;
                background-color: {ModernTheme.BG_MEDIUM};
            }}
            QCheckBox::indicator:checked {{
                background-color: {ModernTheme.ACCENT_CYAN};
                border: 2px solid {ModernTheme.ACCENT_CYAN};
            }}
            QGroupBox {{
                color: {ModernTheme.TEXT_PRIMARY};
                border: 2px solid {ModernTheme.BG_LIGHT};
                border-radius: 8px;
                margin-top: 12px;
                font-weight: bold;
                font-size: 14px;
                padding-top: 15px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
            }}
        """)
    
    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # ========== Left Panel: Controls ==========
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(15)
        control_panel.setMaximumWidth(320)
        control_panel.setMinimumWidth(320)
        
        # Title
        title_label = QLabel('🔭 KMOS Viewer')
        title_font = QFont('Arial', 40, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet(f'color: {ModernTheme.ACCENT_CYAN}; padding: 10px;')
        control_layout.addWidget(title_label)
        
        # File selection
        file_group = QGroupBox('Data Source')
        file_layout = QVBoxLayout()
        self.load_btn = QPushButton('📁 Load FITS File')
        self.load_btn.clicked.connect(self.load_fits_file)
        file_layout.addWidget(self.load_btn)
        self.file_label = QLabel('No file loaded')
        self.file_label.setStyleSheet(f'color: {ModernTheme.TEXT_SECONDARY}; font-size: 11px;')
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        file_group.setLayout(file_layout)
        control_layout.addWidget(file_group)
        
        # Aperture control
        aperture_group = QGroupBox('Extraction Aperture')
        aperture_layout = QVBoxLayout()
        
        self.aperture_label = QLabel('Radius: 3.5 pixels')
        self.aperture_label.setStyleSheet(f'color: {ModernTheme.TEXT_PRIMARY}; font-weight: bold;')
        aperture_layout.addWidget(self.aperture_label)
        
        self.aperture_slider = QSlider(Qt.Horizontal)
        self.aperture_slider.setMinimum(10)
        self.aperture_slider.setMaximum(100)
        self.aperture_slider.setValue(35)
        self.aperture_slider.setTickPosition(QSlider.TicksBelow)
        self.aperture_slider.setTickInterval(10)
        self.aperture_slider.valueChanged.connect(self.update_aperture_label)
        self.aperture_slider.valueChanged.connect(self.update_plot)
        aperture_layout.addWidget(self.aperture_slider)
        
        # Add instruction label
        instruction_label = QLabel('💡 Click image to move aperture')
        instruction_label.setStyleSheet(f'color: {ModernTheme.ACCENT_EMERALD}; font-size: 11px; font-style: italic;')
        aperture_layout.addWidget(instruction_label)
        
        aperture_group.setLayout(aperture_layout)
        control_layout.addWidget(aperture_group)
        
        # Display mode
        display_group = QGroupBox('Display Mode')
        display_layout = QVBoxLayout()
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Signal-to-Noise', 'Flux'])
        self.mode_combo.currentTextChanged.connect(self.update_plot)
        display_layout.addWidget(self.mode_combo)
        
        display_group.setLayout(display_layout)
        control_layout.addWidget(display_group)
        
        # Options
        options_group = QGroupBox('Visualization Options')
        options_layout = QVBoxLayout()
        
        self.psf_checkbox = QCheckBox('Show PSF Reference Circle')
        self.psf_checkbox.setChecked(True)
        self.psf_checkbox.stateChanged.connect(self.update_plot)
        options_layout.addWidget(self.psf_checkbox)
        
        self.grid_checkbox = QCheckBox('Show Grid Lines')
        self.grid_checkbox.setChecked(True)
        self.grid_checkbox.stateChanged.connect(self.update_plot)
        options_layout.addWidget(self.grid_checkbox)
        
        options_group.setLayout(options_layout)
        control_layout.addWidget(options_group)
        
        # Info display
        info_group = QGroupBox('Source Information')
        info_layout = QVBoxLayout()
        
        self.info_object = QLabel('Object: —')
        self.info_redshift = QLabel('Redshift: —')
        self.info_pixels = QLabel('Pixels: —')
        self.info_center = QLabel('Center: —')
        
        for label in [self.info_object, self.info_redshift, self.info_pixels, self.info_center]:
            label.setStyleSheet(f'color: {ModernTheme.TEXT_SECONDARY}; font-size: 12px;')
            info_layout.addWidget(label)
        
        info_group.setLayout(info_layout)
        control_layout.addWidget(info_group)
        
        control_layout.addStretch()
        
        # Credits
        credits = QLabel('Master Thesis \n \nKMOS Viewer v1.0\n© 2025\n by Kolja Reuter \n \nFeatures:\n• Zoomable plots\n• Draggable aperture\n• Click to reposition\n')
        credits.setStyleSheet(f'color: {ModernTheme.TEXT_SECONDARY}; font-size: 10px;')
        credits.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(credits)
        
        # ========== Right Panel: Plots ==========
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create matplotlib figure with dark background
        self.figure = Figure(figsize=(14, 10), facecolor=ModernTheme.BG_MEDIUM)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet(f'background-color: {ModernTheme.BG_MEDIUM}; border-radius: 8px;')
        
        # Add navigation toolbar for zoom/pan
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet(f"""
            QToolBar {{
                background-color: {ModernTheme.BG_LIGHT};
                border: none;
                spacing: 5px;
                padding: 5px;
            }}
            QToolButton {{
                background-color: {ModernTheme.BG_MEDIUM};
                color: {ModernTheme.TEXT_PRIMARY};
                border: none;
                padding: 5px;
                border-radius: 4px;
            }}
            QToolButton:hover {{
                background-color: {ModernTheme.ACCENT_CYAN};
            }}
        """)
        
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Add panels to main layout
        main_layout.addWidget(control_panel)
        main_layout.addWidget(plot_panel, stretch=1)
        
    def on_click(self, event):
        """Handle mouse click events."""
        if self.flux_cube is None or event.inaxes != self.ax_img:
            return
        
        # Check if click is near aperture circle
        if self.aperture_circle is not None:
            contains, _ = self.aperture_circle.contains(event)
            if contains:
                self.dragging_aperture = True
                return
        
        # Otherwise, move aperture to clicked location
        if event.xdata is not None and event.ydata is not None:
            self.center_x = event.xdata
            self.center_y = event.ydata
            self.update_center_info()
            self.update_plot()
    
    def on_release(self, event):
        """Handle mouse release events."""
        self.dragging_aperture = False
    
    def on_motion(self, event):
        """Handle mouse motion events."""
        if self.dragging_aperture and event.inaxes == self.ax_img:
            if event.xdata is not None and event.ydata is not None:
                self.center_x = event.xdata
                self.center_y = event.ydata
                self.update_center_info()
                self.update_plot()
    
    def on_scroll(self, event):
        """Zoom in/out on mouse scroll."""
        if event.inaxes is None:
            return

        # Get the current axis
        ax = event.inaxes
        
        # Determine zoom factor (0.8 = zoom in, 1.2 = zoom out)
        base_scale = 1.2
        scale_factor = 1/base_scale if event.button == 'up' else base_scale

        # Get current limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        
        # Set the zoom center to the mouse cursor location
        xdata = event.xdata
        ydata = event.ydata
        
        # Calculate new limits
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        
        ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        
        self.canvas.draw_idle()  # Efficient redraw



    def update_center_info(self):
        """Update center position display."""
        self.info_center.setText(f'Center: ({self.center_x:.1f}, {self.center_y:.1f})')
    
    def update_aperture_label(self):
        """Update aperture label when slider changes."""
        radius = self.aperture_slider.value() / 10.0
        self.aperture_label.setText(f'Radius: {radius:.1f} pixels')
    
    def load_fits_file(self):
        """Load FITS file through file dialog."""
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Select KMOS FITS File', '', 'FITS Files (*.fits *.fit);;All Files (*)'
        )
        
        if not filename:
            return
        
        try:
            # Normalize the filepath for cross-platform compatibility
            filename = normalize_path(filename)
            
            with fits.open(filename) as hdu:
                self.flux_cube = hdu[1].data
                self.noise_cube = hdu[2].data
                self.exp_map = hdu[3].data
                self.header = hdu[1].header
                # Use cross-platform filename extraction
                self.obj_name = self.header.get('OBJECT', extract_filename_from_path(filename))
                self.psf_maj, self.psf_min, self.psf_pa = extract_psf_parameters(hdu)
                
                # Wavelength calibration
                self.wavelengths = extract_wavelength_calibration(
                    self.header, self.flux_cube.shape[0]
                )
                
                # White light image
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.white_light = np.nanmedian(self.flux_cube, axis=0)
                
                # Center position
                self.center_x = self.header.get('CRPIX1', self.flux_cube.shape[2] / 2) - 1
                self.center_y = self.header.get('CRPIX2', self.flux_cube.shape[1] / 2) - 1
                self.redshift, self.flag_redshift = redshift(hdu)
            
            # Update UI - use cross-platform filename extraction
            display_name = extract_filename_from_path(filename)
            self.file_label.setText(f'✓ {display_name}')
            self.file_label.setStyleSheet(f'color: {ModernTheme.ACCENT_EMERALD}; font-size: 11px;')
            self.info_object.setText(f'Object: {self.obj_name}')
            self.info_redshift.setText(f'Redshift: z = {self.redshift:.4f}')
            self.update_center_info()
            
            self.info_redshift.setText(f'Redshift: z = {self.redshift:.4f}')
            self.update_center_info()
            
            self.plot_initialized = False  # Force a full redraw for new files

            # Generate plot
            self.update_plot()
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load file:\n{str(e)}')
    
    def update_plot(self):
        """Optimized plot update with fixed collection clearing."""
        if self.flux_cube is None:
            return
        
        radius_px = self.aperture_slider.value() / 10.0
        view_mode = self.mode_combo.currentText()
        show_psf = self.psf_checkbox.isChecked()
        show_grid = self.grid_checkbox.isChecked()
        
        # --- 1. Calculations (Run every time) ---
        aperture_mask = create_aperture_mask(
            self.white_light.shape, self.center_x, self.center_y, 
            radius_px, self.exp_map
        )
        
        if np.sum(aperture_mask) == 0:
            return

        stacked_flux, stacked_noise, stacked_snr = weighted_stack_spectrum(
            self.flux_cube, self.noise_cube, aperture_mask
        )
        
        n_pixels = np.sum(aperture_mask)
        self.info_pixels.setText(f'Pixels: {n_pixels}')
        
        # Select Data
        if view_mode == 'Signal-to-Noise':
            y_data = np.abs(stacked_snr)
            y_label = 'S/N Ratio'
            main_color = ModernTheme.ACCENT_EMERALD
        else:
            y_data = stacked_flux
            y_label = 'Flux'
            main_color = ModernTheme.ACCENT_BLUE

        # --- 2. Initialization (Runs ONLY once per file load) ---
        if not self.plot_initialized:
            self.figure.clear()
            gs = self.figure.add_gridspec(2, 2, height_ratios=[1, 0.5], 
                                        hspace=0.35, wspace=0.30,
                                        left=0.08, right=0.96, top=0.94, bottom=0.08)
            
            # === Image Panel ===
            self.ax_img = self.figure.add_subplot(gs[0, 0])
            self.ax_img.set_facecolor(ModernTheme.BG_DARK)
            vmin, vmax = np.nanpercentile(self.white_light, [1, 99.5])
            self.im_obj = self.ax_img.imshow(self.white_light, origin='lower', cmap='magma', 
                                      interpolation='gaussian', vmin=vmin, vmax=vmax)
            
            # Colorbar styling
            cbar = self.figure.colorbar(self.im_obj, ax=self.ax_img, fraction=0.046, pad=0.04)
            cbar.ax.set_facecolor(ModernTheme.BG_LIGHT)
            cbar.set_label('Median Flux', fontsize=10, color=ModernTheme.TEXT_PRIMARY)
            cbar.ax.tick_params(axis='y', colors=ModernTheme.TEXT_PRIMARY, labelsize=9) 
            cbar.outline.set_edgecolor(ModernTheme.BG_LIGHT)

            # Aperture Circle
            self.aperture_circle = Circle((self.center_x, self.center_y), radius_px, 
                                         edgecolor=ModernTheme.ACCENT_CYAN, facecolor='none', 
                                         linewidth=1.5, picker=5)
            self.ax_img.add_patch(self.aperture_circle)
            self.ax_img.set_title(f'{self.obj_name} [Median gaussian-smoothed Map]', color=ModernTheme.TEXT_PRIMARY, fontweight='bold')
            self.ax_img.set_xlabel('X [pixels]', color=ModernTheme.TEXT_PRIMARY)
            self.ax_img.set_ylabel('Y [pixels]', color=ModernTheme.TEXT_PRIMARY)

            # PSF 
            self.psf_patch = Ellipse((self.center_x, self.center_y), 
                                     width=self.psf_min,  # Minor axis
                                     height=self.psf_maj, # Major axis
                                     angle=self.psf_pa,   # Rotation angle
                                     edgecolor=ModernTheme.LINE_LIME, 
                                     linestyle='--', 
                                     facecolor='none',
                                     visible=show_psf,
                                     label='PSF Fit')
            self.ax_img.add_patch(self.psf_patch)

            # === Zoom Panel ===
            self.ax_zoom = self.figure.add_subplot(gs[0, 1])
            self.ax_zoom.set_facecolor(ModernTheme.BG_DARK)
            if self.flag_redshift:
                self.ax_zoom.set_title(f'Hα & [NII] (z = {self.redshift:.4f}) (lit.)', 
                                   color=ModernTheme.TEXT_PRIMARY, fontweight='bold')
            else:
                self.ax_zoom.set_title(f'Hα & [NII] (z = {self.redshift:.4f})', 
                                   color=ModernTheme.TEXT_PRIMARY, fontweight='bold')
            self.ax_zoom.set_xlabel('Wavelength [μm]', color=ModernTheme.TEXT_PRIMARY)
            
            # Initialize empty line for Zoom
            self.zoom_line, = self.ax_zoom.plot([], [], color=main_color, linewidth=1.5, zorder=3)
            
            # Static H-Alpha Lines
            if self.redshift > 0:
                ha_obs = 0.656280 * (1 + self.redshift)
                nii_6583_obs = 0.658345 * (1 + self.redshift)
                nii_6548_obs = 0.654805 * (1 + self.redshift)
                
                self.ax_zoom.axvline(ha_obs, color=ModernTheme.LINE_RED, linestyle='--', 
                                   linewidth=1.5, alpha=0.9, zorder=1)
                self.ax_zoom.text(ha_obs, 0.95, r'  Hα', transform=self.ax_zoom.get_xaxis_transform(),
                                color=ModernTheme.LINE_RED, fontweight='bold', fontsize=11, va='top')
                
                self.ax_zoom.axvline(nii_6583_obs, color=ModernTheme.LINE_ORANGE, linestyle=':', 
                                   linewidth=1.5, alpha=0.8, zorder=1)
                self.ax_zoom.text(nii_6583_obs, 0.85, r'  [NII]', transform=self.ax_zoom.get_xaxis_transform(),
                                color=ModernTheme.LINE_ORANGE, fontsize=10, va='top')

                self.ax_zoom.axvline(nii_6548_obs, color=ModernTheme.LINE_ORANGE, linestyle=':', 
                                   linewidth=1.5, alpha=0.6, zorder=1)
    
                self.ax_zoom.text(nii_6548_obs, 0.85, r'[NII]', transform=self.ax_zoom.get_xaxis_transform(),
                                color=ModernTheme.LINE_ORANGE, fontsize=10, va='top', ha='right')


            # === Full Spectrum Panel ===
            self.ax_full = self.figure.add_subplot(gs[1, :])
            self.ax_full.set_facecolor(ModernTheme.BG_DARK)
            self.ax_full.set_title('Integrated Spectrum', color=ModernTheme.TEXT_PRIMARY, fontweight='bold')
            self.ax_full.set_xlabel('Observed Wavelength [μm]', color=ModernTheme.TEXT_PRIMARY)
            self.full_line, = self.ax_full.plot(self.wavelengths, y_data, color=main_color, linewidth=1.5)
            self.ax_full.set_xlim(self.wavelengths[0], self.wavelengths[-1])

            # Apply styling to all axes
            for ax in [self.ax_img, self.ax_zoom, self.ax_full]:
                ax.tick_params(colors=ModernTheme.TEXT_SECONDARY)
                for spine in ax.spines.values():
                    spine.set_edgecolor(ModernTheme.BG_LIGHT)
                    spine.set_linewidth(1.5)
                if show_grid:  # activate if you dont want it added to image
                    ax.grid(show_grid, alpha=0.3, linestyle=':')

            self.plot_initialized = True

        # --- 3. Fast Update (Runs on every slider move) ---
        
        # Update Image Overlays
        self.aperture_circle.center = (self.center_x, self.center_y)
        self.aperture_circle.set_radius(radius_px)
        
        # Update PSF Ellipse Position
        self.psf_patch.center = (self.center_x, self.center_y)
        self.psf_patch.set_visible(show_psf)
        # Update Full Spectrum
        self.full_line.set_ydata(y_data)
        self.full_line.set_color(main_color)
        self.ax_full.set_ylim(np.nanmin(y_data), np.nanmax(y_data)*1.1)
        self.ax_full.set_ylabel(y_label, color=ModernTheme.TEXT_PRIMARY)

        # === FIX: Safe Collection Removal ===
        # We convert to a list first to safely iterate while modifying
        for c in list(self.ax_full.collections):
            c.remove()
        self.ax_full.fill_between(self.wavelengths, 0, y_data, color=main_color, alpha=0.15)

        # Update Zoom Spectrum
        if self.redshift > 0:
            ha_obs = 0.656280 * (1 + self.redshift)
            window = 0.015
            mask = (self.wavelengths > ha_obs - window) & (self.wavelengths < ha_obs + window)
            
            if np.any(mask):
                wl_zoom = self.wavelengths[mask]
                y_zoom = y_data[mask]
                
                self.zoom_line.set_data(wl_zoom, y_zoom)
                self.zoom_line.set_color(main_color)
                self.ax_zoom.set_xlim(wl_zoom[0], wl_zoom[-1])
                
                y_min, y_max = np.nanmin(y_zoom), np.nanmax(y_zoom)
                if not np.isnan(y_max):
                     self.ax_zoom.set_ylim(y_min * 0.95, y_max * 1.1)

                # === FIX: Safe Collection Removal for Zoom ===
                for c in list(self.ax_zoom.collections):
                    c.remove()
                self.ax_zoom.fill_between(wl_zoom, 0, y_zoom, color=main_color, alpha=0.2, zorder=2)

        self.canvas.draw_idle()

# ==================== Main Entry Point ====================
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern Qt style
    
    # Set application font
    font = QFont('Segoe UI', 10)
    app.setFont(font)
    
    window = KMOSViewerGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()