"""
Merged W80 Histogram GUI with AGN Classification
- Data-reading + SIMBAD classification & handling from the NOT improved version
- GUI, threading, plotting, exporting from the improved version
- AGN classification integration
"""

from fileinput import filename
from operator import is_
import os
import sys
import re
import pickle
from collections import defaultdict
from pathlib import Path
from turtle import mode

from matplotlib.pylab import True_
from tools import Fitting_Voronoi as fittingvoronoi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.simbad import Simbad
from scipy.ndimage import gaussian_filter1d

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QPushButton, QLabel, QSplitter,
    QMessageBox, QFileDialog, QProgressBar, QStatusBar, QGroupBox,
    QSpinBox, QCheckBox, QTabWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import QScrollArea

def master_map_w80(filename, single_map_and_double_map = False, plot = False):
    hdulist = fits.open(filename)
    single_popt = {
            'A': np.array(hdulist['A'].data),
            'B': np.array(hdulist['B'].data),
            'C': np.array(hdulist['C'].data),
      }
    double_popt = {
            'A1': np.array(hdulist['A1'].data),
            'B1': np.array(hdulist['B1'].data),
            'C1': np.array(hdulist['C1'].data),
            'A2': np.array(hdulist['A2'].data),
            'B2': np.array(hdulist['B2'].data),
            'C2': np.array(hdulist['C2'].data),
      }
    
    chi = {
            'chi_single': hdulist['chi1'].data,
            'chi_double': hdulist['chi2'].data
      }
    
    son = {
            'son_single': hdulist['StoN'].data,
            'son_double': hdulist['son2'].data
      }
    if plot:
        plt.imshow(np.abs(1 - chi['chi_single']), cmap='plasma', origin='lower',
                   vmin=np.nanmin(np.abs(1 - chi['chi_single'])),
                   vmax=np.nanpercentile(np.abs(1 - chi['chi_single']), 97))

        plt.colorbar(label='Reduced Chi-Squared (Single)')
        plt.title(r'$|\chi^2_{red} - 1|$ (Single)')
        plt.show()


        plt.imshow(np.abs(1 - chi['chi_double']), cmap='plasma', origin='lower',
                   vmin=np.nanmin(np.abs(1 - chi['chi_double'])),
                   vmax=np.nanpercentile(np.abs(1 - chi['chi_double']), 97))
        plt.colorbar(label='Reduced Chi-Squared (Double)')
        plt.title(r'$|\chi^2_{red} - 1|$ (Double)')
        plt.show()

        plt.imshow(son['son_single'], cmap='plasma', origin='lower')
        plt.colorbar(label='S/N (Single)')
        contour_levels = [5, 10]
        CS = plt.contour(son['son_single'], levels=contour_levels, colors=['red', 'white'], linewidths=1.5)
        plt.clabel(CS, fmt={5: '5', 10: '10'}, colors=['red', 'white'], fontsize=12)
        
        plt.title('S/N (Single)')
        plt.show()

        plt.imshow(son['son_double'], cmap='plasma', origin='lower')
        plt.colorbar(label='S/N (Double)')
        CS2 = plt.contour(son['son_double'], levels=contour_levels, colors=['red', 'white'], linewidths=1.5)
        plt.clabel(CS2, fmt={5: '5', 10: '10'}, colors=['red', 'white'], fontsize=12)

        
        plt.title('S/N (Double)')
        plt.show()

    hdulist.close()
    filename1 = filename.replace( 'Gaussian_fits', 'KMOS3D_ALL')
    filename1 = filename1.replace( '_voronoi_binned.fits', '.fits')
    hdulist1 = fits.open(filename1)

    wavelength, flux, flux_err = fittingvoronoi.process_data_cube(hdulist1)
    hdulist1.close()
    width_single_map = np.zeros_like(single_popt['A'])
    width_double_map = np.zeros_like(double_popt['A1'])
    
    for i in range(single_popt['A'].shape[0]):
        for j in range(single_popt['A'].shape[1]):
            
            single_function = fittingvoronoi.gaussian(wavelength, single_popt['A'][i][j], single_popt['B'][i][j], single_popt['C'][i][j])
            double_function = fittingvoronoi.double_gaussian(wavelength, double_popt['A1'][i][j], double_popt['B1'][i][j], double_popt['C1'][i][j], double_popt['A2'][i][j], double_popt['B2'][i][j], double_popt['C2'][i][j])
            # Calculate the cumulative sum of the double_function
            cumulative_sum_double = np.cumsum(double_function)
            total_sum_double = cumulative_sum_double[-1]
            # Calculate the cumulative sum of the single_function
            cumulative_sum_single = np.cumsum(single_function)
            total_sum_single = cumulative_sum_single[-1]

            # Determine the 10% and 90% thresholds
            lower_threshold_single = 0.1 * total_sum_single
            upper_threshold_single = 0.9 * total_sum_single
            lower_threshold_double = 0.1 * total_sum_double
            upper_threshold_double = 0.9 * total_sum_double

            # Find the indices corresponding to the thresholds
            lower_index_single = np.searchsorted(cumulative_sum_single, lower_threshold_single)
            upper_index_single = np.searchsorted(cumulative_sum_single, upper_threshold_single)

            lower_index_double = np.searchsorted(cumulative_sum_double, lower_threshold_double)
            upper_index_double = np.searchsorted(cumulative_sum_double, upper_threshold_double)

            # Calculate the width of the single_function excluding the first and last 10% of the area
            width_single = wavelength[upper_index_single] - wavelength[lower_index_single]
            # Calculate the width of the double_function excluding the first and last 10% of the area
            width_double = wavelength[upper_index_double] - wavelength[lower_index_double]      
            if width_single == 0:
                width_single = np.nan
            if width_double == 0:
                width_double = np.nan
            width_single_map[i][j] = width_single 
            width_double_map[i][j] = width_double 

    k, m = 7, 15
    #print(width_double_map[k][m], width_single_map[k][m])
    Halpha = 0.65628    
    c = 299792.458  # Speed of light in km/s 
    width_single_map, width_double_map = c * width_single_map /Halpha, c * width_double_map / Halpha
    #print(width_double_map[k][m], width_single_map[k][m])
    
    if single_map_and_double_map:
        if plot:
            plt.imshow(np.log10(width_single_map), cmap='plasma', origin='lower')
            plt.colorbar(label='log10(Width of Single Gaussian) (km/s)')
            plt.title('log10(Width of Single Gaussian)')
            plt.show()

            plt.imshow(np.log10(width_double_map), cmap='plasma', origin='lower')
            plt.colorbar(label='log10(Width of Double Gaussian) (km/s)')
            plt.title('log10(Width of Double Gaussian)')
            plt.show()
        #return np.log10(width_single_map), np.log10(width_double_map)

    width80_master_map = np.zeros_like(width_single_map)
    mask_bad_data = son['son_single'] < 5
    

    mask = np.abs(chi['chi_single'] -1 ) < np.abs( chi['chi_double'] -1)
    width80_master_map = np.where(mask, width_single_map, width_double_map)
    width80_master_map = np.where(mask_bad_data, np.nan, width80_master_map)

    if plot:

        plt.imshow(np.log10(width80_master_map), cmap='plasma', origin='lower')
        plt.colorbar(label='log10(Width of Gaussian) (km/s) ')
        plt.title(r'$\log_{10}(\mathrm{Width\ of\ Gaussian})$ with $\chi^2_{red}$ cut')
        plt.show()

    return width80_master_map


# ---------------------------
# Legacy (NOT improved) data-reading + SIMBAD handling
# Adapted from program_statistics.py to work with a filename (not index)
# ---------------------------

def legacy_redshift(hdulist1, FILE='k3d_fnlsp_table_v3.fits', testdata=False):
    """Legacy redshift() from NOT improved file (slightly adapted)."""
    if testdata:
        return 0.0
    with fits.open(FILE) as hdulist:
        filename = hdulist1.filename()
        data = hdulist[1].data
        # Convert to DataFrame safely
        structured_array = np.array(data).byteswap().newbyteorder()
        df = pd.DataFrame(structured_array)
        # Ensure FILE is string
        if df['FILE'].dtype != object:
            try:
                df['FILE'] = df['FILE'].str.decode('utf-8')
            except Exception:
                df['FILE'] = df['FILE'].astype(str)
        # Build ID from filename (KMOS3D_ALL/... .fits)
        ID = re.sub(r'(KMOS3D_ALL|KMOS3D)[\\/]+', '', filename)
        ID = re.sub(r'\.fits$', '', ID)
        ID = str(ID).strip() + '.fits'
        matching = df[df['FILE'].str.strip() == ID]
        if len(matching) == 0:
            raise ValueError('Filename not found in database', ID)
        z = float(matching['Z'].iloc[0])
        if z < 0:
            z = float(matching['Z_TARGETED'].iloc[0])
        return z


def legacy_get_ra_dec(hdulist1, FILE='k3d_fnlsp_table_v3.fits', testdata=False):
    """Legacy get_ra_dec() from NOT improved file (slightly adapted)."""
    if testdata:
        return (0.0, 0.0)
    with fits.open(FILE) as hdulist:
        filename = hdulist1.filename()
        data = hdulist[1].data
        structured_array = np.array(data).byteswap().newbyteorder()
        df = pd.DataFrame(structured_array)
        if df['FILE'].dtype != object:
            try:
                df['FILE'] = df['FILE'].str.decode('utf-8')
            except Exception:
                df['FILE'] = df['FILE'].astype(str)
        ID = re.sub(r'(KMOS3D_ALL|KMOS3D)[\\/]+', '', filename)
        ID = re.sub(r'\.fits$', '', ID)
        ID = str(ID).strip() + '.fits'
        matching = df[df['FILE'].str.strip() == ID]
        if len(matching) == 0:
            raise ValueError('Filename not found in database', ID)
        ra = float(matching['RA'].iloc[0])
        dec = float(matching['DEC'].iloc[0])
        return (ra, dec)


def load_agn_catalog(filename='AGN_SAMPLE.fits'):
    """Load AGN catalog and return list of AGN IDs."""
    try:
        if not Path(filename).exists():
            return []
        with fits.open(filename) as hdulist:
            list_agn_ids = hdulist[1].data['ID_TARGETED']
            # Replace underscores with spaces in each element
            list_agn_ids = [element.replace('_', ' ') for element in list_agn_ids]
            return list_agn_ids
    except Exception as e:
        print(f"Error loading AGN catalog: {e}")
        return []


def remove_last_part(name):
    """Remove the last part of a galaxy name for matching."""
    return ' '.join(name.split(' ')[:1])


def is_agn_galaxy(galaxy_id, agn_list):
    """Check if a galaxy is in the AGN list."""
    if not galaxy_id or not agn_list:
        return False
    name_no_last = galaxy_id.replace('_', ' ')
    if name_no_last.startswith("COS"):
        # Turn "COSMOS 4 25458" into "COS4 25458"
        name_no_last = re.sub(r'COSMOS\s*(\d+)', r'COS\1', name_no_last)
    #print(agn_list, name_no_last)
    return name_no_last in agn_list


def legacy_extract_information_from_filename(filename, agn_list=None):
    """
    Legacy extract_information() logic from NOT improved file,
    adapted to take a filename string instead of an index.
    Returns a dict with galaxy_id, redshift, filter, ra, dec, simbad_class, is_agn.
    """
    # Extract galaxy_id
    match = re.search(r'([A-Z0-9]+_\d+)', filename)
    galaxy_id = match.group(1) if match else None
    if galaxy_id is not None and galaxy_id.startswith("COS"):
        galaxy_id = re.sub(r'COS(\d+)_', r'COSMOS \1 ', galaxy_id)

    # Extract filter between _ID_ and _voronoi_binned.fits
    filter_match = re.search(r'_([A-Za-z]+)_voronoi_binned\.fits', filename)
    filt = filter_match.group(1) if filter_match else None

    # Open corresponding KMOS3D_ALL file
    filename_1 = filename.replace('Gaussian_fits', 'KMOS3D_ALL')
    filename_1 = filename_1.replace('_voronoi_binned.fits', '.fits')
    with fits.open(filename_1) as hdulist_kmos:
        try:
            redshift = round(float(legacy_redshift(hdulist_kmos)), 3)
        except Exception:
            redshift = 0.0
        try:
            ra, dec = legacy_get_ra_dec(hdulist_kmos, testdata=False)
        except Exception:
            ra, dec = (0.0, 0.0)

    # Legacy SIMBAD classification: progressive-radius query, minimal cleanup
    with fits.open(filename, mode='readonly') as hdul:
        header = hdul[0].header
        simbad_class = header.get('SIMBAD_CL', 'Unknown')
        w80_mean = header.get('W80_MEAN', np.nan)
        w80_std = header.get('W80_STD', np.nan)

    # Check if galaxy is AGN
    is_agn = is_agn_galaxy(galaxy_id, agn_list) if agn_list else False

    return {
        'galaxy_id': galaxy_id,
        'redshift': redshift,
        'filter': filt,
        'ra': ra,
        'dec': dec,
        'simbad_class': str(simbad_class),
        'w80_mean': w80_mean,
        'w80_std': w80_std,
        'is_agn': is_agn
    }


# ---------------------------
# Improved processing & GUI with AGN integration
# ---------------------------

class DataProcessor:
    """Handles data processing operations"""
    def __init__(self):
        self.filenames = self._scan_files()
        self.agn_list = load_agn_catalog()

    def _scan_files(self):
        """Scan for FITS files in Gaussian_fits directory"""
        fits_dir = Path('Gaussian_fits')
        if not fits_dir.exists():
            return []
        return [str(f) for f in fits_dir.glob('*_voronoi_binned.fits')]

    @staticmethod
    def _gaussian(x, a, b, c):
        return a * np.exp(-0.5 * ((x - b) / c) ** 2)

    @staticmethod
    def _double_gaussian(x, a1, b1, c1, a2, b2, c2):
        return (a1 * np.exp(-0.5 * ((x - b1) / c1) ** 2) +
                a2 * np.exp(-0.5 * ((x - b2) / c2) ** 2))

    @staticmethod
    def _calculate_w80_from_profile(wavelength, profile):
        if np.sum(profile) == 0:
            return np.nan
        cumsum = np.cumsum(profile)
        total = cumsum[-1]
        if total == 0:
            return np.nan
        idx_10 = np.searchsorted(cumsum, 0.1 * total)
        idx_90 = np.searchsorted(cumsum, 0.9 * total)
        if idx_90 >= len(wavelength) or idx_10 >= len(wavelength):
            return np.nan
        return wavelength[idx_90] - wavelength[idx_10]


class DataLoader(QThread):
    """Background thread for loading data using legacy info + improved W80 + AGN classification."""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished_loading = pyqtSignal(dict)

    def __init__(self, filenames, use_cache=True):
        super().__init__()
        self.filenames = filenames
        self.use_cache = use_cache
        self.processor = DataProcessor()

    def run(self):
        cache_file = 'pkl_files\sorted_w80_cache.pkl'
        if self.use_cache and Path(cache_file).exists():
            self.status.emit("Loading cached data...")
            try:
                with open(cache_file, 'rb') as f:
                    w80_data = pickle.load(f)
                self.finished_loading.emit(w80_data)
                return
            except Exception as e:
                print("Cache load failed, recalculating...", e)

        # Initialize data structures for both SIMBAD and AGN classification
        w80_by_class = defaultdict(list)
        w80_by_agn = defaultdict(list)
        w80_means_by_class = defaultdict(list)
        w80_stds_by_class = defaultdict(list)
        w80_means_by_agn = defaultdict(list)
        w80_stds_by_agn = defaultdict(list)
        
        total_files = max(1, len(self.filenames))

        for i, filename in enumerate(self.filenames, 1):
            self.status.emit(f"Processing {Path(filename).name}")
            self.progress.emit(int((i / total_files) * 100))

            try:
                # Use legacy (NOT improved) info extraction & SIMBAD handling + AGN
                info = legacy_extract_information_from_filename(filename, self.processor.agn_list)
                simbad_class = info.get('simbad_class', 'Unknown') or 'Unknown'
                is_agn = info.get('is_agn', False)
                w80_mean = info.get('w80_mean', np.nan)
                w80_std = info.get('w80_std', np.nan)
                
                # Calculate W80 map (improved method)
                w80_map = master_map_w80(filename)
                if w80_map is not None:
                    valid = w80_map[~np.isnan(w80_map)]
                    if valid.size > 0:
                        # Store by SIMBAD class
                        w80_by_class[simbad_class].extend(valid.astype(float).tolist())
                        
                        # Store by AGN classification
                        agn_key = "AGN" if is_agn else "Non-AGN"
                        w80_by_agn[agn_key].extend(valid.astype(float).tolist())

                        # Store per-galaxy means/stds
                        try:
                            mean_val = float(w80_mean)
                            if np.isfinite(mean_val):
                                w80_means_by_class[simbad_class].append(mean_val)
                                w80_means_by_agn[agn_key].append(mean_val)
                        except Exception:
                            pass

                        try:
                            std_val = float(w80_std)
                            if np.isfinite(std_val):
                                w80_stds_by_class[simbad_class].append(std_val)
                                w80_stds_by_agn[agn_key].append(std_val)
                        except Exception:
                            pass

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        # Combine all data
        result = dict(w80_by_class)
        result['_w80_means_'] = {k: v[:] for k, v in w80_means_by_class.items()}
        result['_w80_stds_'] = {k: v[:] for k, v in w80_stds_by_class.items()}
        result['_w80_by_agn_'] = dict(w80_by_agn)
        result['_w80_means_by_agn_'] = {k: v[:] for k, v in w80_means_by_agn.items()}
        result['_w80_stds_by_agn_'] = {k: v[:] for k, v in w80_stds_by_agn.items()}
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception:
            pass

        self.finished_loading.emit(result)


class ModernHistogramWidget(QMainWindow):
    """Modern, optimized histogram widget with improved UI and AGN integration"""
    def __init__(self, w80_data=None):
        super().__init__()
        self.w80_by_class = w80_data or {}
        self.classes = sorted([k for k in self.w80_by_class.keys() 
                              if not k.startswith('_')]) if self.w80_by_class else []
        self.trend_mode = False
        self.spaxel_mode = True
        self.agn_mode = False  # New: toggle between SIMBAD and AGN classification
        self.bins = None

        self.init_ui()
        self.setup_style()

        if not self.w80_by_class:
            self.load_data()
        else:
            self.setup_bins()
            self.update_plot()

    def init_ui(self):
        self.setWindowTitle('W80 Distribution Analyzer - KMOS3D Data with AGN Classification')
        self.setGeometry(100, 100, 1600, 900)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)

        plot_panel = self.create_plot_panel()
        splitter.addWidget(plot_panel)

        splitter.setSizes([400, 1200])

    def create_control_panel(self):
        panel = QWidget()
        panel.setMaximumWidth(450)
        layout = QVBoxLayout(panel)

        # Classification mode selector
        class_mode_group = QGroupBox("Classification Mode")
        class_mode_layout = QVBoxLayout(class_mode_group)
        
        self.btn_simbad_mode = QPushButton("SIMBAD Classes")
        self.btn_simbad_mode.setCheckable(True)
        self.btn_simbad_mode.setChecked(True)
        self.btn_simbad_mode.clicked.connect(lambda: self.set_classification_mode(False, True))
        
        self.btn_agn_mode = QPushButton("Million Quasar Catalog")
        self.btn_agn_mode.setCheckable(True)
        self.btn_agn_mode.clicked.connect(lambda: self.set_classification_mode(True, False))

        # Decoupled button like the AGN button (acts independently)
        self.btn_imp_mode = QPushButton("AGN or Non-AGN, that's the question")
        self.btn_imp_mode.setCheckable(True)
        self.btn_imp_mode.clicked.connect(lambda: self.set_classification_mode(True, True))
        
        class_mode_layout.addWidget(self.btn_imp_mode)
        class_mode_layout.addWidget(self.btn_simbad_mode)
        class_mode_layout.addWidget(self.btn_agn_mode)
        layout.addWidget(class_mode_group)

        # Class selection list
        title = QLabel("Class Selection")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)

        self.class_list = QListWidget()
        self.class_list.setSelectionMode(QListWidget.MultiSelection)
        self.class_list.itemSelectionChanged.connect(self.on_selection_changed)
        layout.addWidget(self.class_list)

        btn_group = QGroupBox("Display Controls")
        btn_layout = QVBoxLayout(btn_group)

        select_layout = QHBoxLayout()
        self.btn_all = QPushButton("Select All")
        self.btn_none = QPushButton("Clear All")
        self.btn_all.clicked.connect(self.select_all)
        self.btn_none.clicked.connect(self.clear_all)
        select_layout.addWidget(self.btn_all)
        select_layout.addWidget(self.btn_none)
        btn_layout.addLayout(select_layout)

        self.btn_toggle_view = QPushButton("Show Trends")
        self.btn_toggle_view.setCheckable(True)
        self.btn_toggle_view.clicked.connect(self.toggle_view_mode)
        btn_layout.addWidget(self.btn_toggle_view)

        self.btn_toggle_data = QPushButton("Show Galaxy Means")
        self.btn_toggle_data.setCheckable(True)
        self.btn_toggle_data.clicked.connect(self.toggle_data_mode)
        btn_layout.addWidget(self.btn_toggle_data)

        self.btn_export = QPushButton("Export Plot")
        self.btn_export.clicked.connect(self.export_plot)
        btn_layout.addWidget(self.btn_export)

        bin_layout = QHBoxLayout()
        bin_layout.addWidget(QLabel("Bins:"))
        self.bin_spinner = QSpinBox()
        self.bin_spinner.setRange(10, 100)
        self.bin_spinner.setValue(40)
        self.bin_spinner.valueChanged.connect(self.update_bins)
        bin_layout.addWidget(self.bin_spinner)
        btn_layout.addLayout(bin_layout)

        layout.addWidget(btn_group)

        stats_group = QGroupBox("Statistics")
        self.stats_label = QLabel("No data loaded")
        self.stats_label.setWordWrap(True)
        self.stats_label.setAlignment(Qt.AlignTop)

        stats_inner = QWidget()
        stats_inner_layout = QVBoxLayout(stats_inner)
        stats_inner_layout.addWidget(self.stats_label)
        stats_inner_layout.addStretch(1)

        stats_scroll = QScrollArea()
        stats_scroll.setWidgetResizable(True)
        stats_scroll.setWidget(stats_inner)
        stats_scroll.setMinimumHeight(180)

        stats_layout = QVBoxLayout(stats_group)
        stats_layout.addWidget(stats_scroll)
        layout.addWidget(stats_group)

        return panel

    def set_classification_mode(self, agn_mode, miliqua_mode=False):
        """Switch between SIMBAD and AGN classification modes."""
        self.agn_mode = agn_mode
        self.miliqua_mode = miliqua_mode
        self.btn_simbad_mode.setChecked(not agn_mode)
        self.btn_imp_mode.setChecked(agn_mode and miliqua_mode)
        self.btn_agn_mode.setChecked(agn_mode and (not miliqua_mode))

        # Update the class list
        self.update_class_list()
        self.setup_bins()
        self.update_plot()
        self.update_statistics()

    def update_class_list(self):
        """Update the class list based on current classification mode."""
        self.class_list.clear()
        
        # Determine mode combinations:
        # - AGN-only mode: agn_mode=True, miliqua_mode=False  -> show Million Quasar / AGN groups
        # - SIMBAD mode:    agn_mode=False                     -> show SIMBAD classes
        # - Imp/combined:   agn_mode=True, miliqua_mode=True   -> group Active / Inactive
        miliqua = getattr(self, 'miliqua_mode', False)

        if self.agn_mode and not miliqua:
            # Show AGN classification (Million Quasar / AGN groups)
            agn_data = self.w80_by_class.get('_w80_by_agn_', {})
            means_data = self.w80_by_class.get('_w80_means_by_agn_', {})
            for cls in sorted(agn_data.keys()):
                count = len(means_data.get(cls, []))
                if count > 0:
                    item = QListWidgetItem(f"{cls} (N={count})")
                    item.setData(Qt.UserRole, cls)
                    item.setSelected(True)
                    self.class_list.addItem(item)

        elif not self.agn_mode:
            # Show SIMBAD classification
            means_data = self.w80_by_class.get('_w80_means_', {})
            for cls in self.classes:
                count = len(means_data.get(cls, []))
                if count > 0:
                    item = QListWidgetItem(f"{cls} (N={count})")
                    item.setData(Qt.UserRole, cls)
                    item.setSelected(True)
                    self.class_list.addItem(item)

        else:
            # Combined "imp" mode: create Active / Inactive groups from SIMBAD + AGN catalog
            self.btn_imp_mode.setChecked(True)
            active_simbad = {'AGN', 'AG?', 'Q', 'Sy1', 'Sy2', 'QSO', 'Q?'}

            means_by_agn = self.w80_by_class.get('_w80_means_by_agn_', {})
            means_by_simbad = self.w80_by_class.get('_w80_means_', {})

            active_keys = []
            inactive_keys = []

            # map AGN catalog groups
            if means_by_agn.get('AGN'):
                active_keys.append('AGN')
            if means_by_agn.get('Non-AGN'):
                inactive_keys.append('Non-AGN')

            # map SIMBAD classes into active/inactive lists
            for cls, lst in means_by_simbad.items():
                if not lst:
                    print(f"Skipping empty class {cls}")
                    continue
                if cls in active_simbad:
                    active_keys.append(cls)
                else:
                    inactive_keys.append(cls)

            def get_count(key):
                if key in self.w80_by_class.get('_w80_means_by_agn_', {}):
                    return len(self.w80_by_class['_w80_means_by_agn_'][key])
                elif key in self.w80_by_class.get('_w80_means_', {}):
                    return len(self.w80_by_class['_w80_means_'][key])
                else:
                    return len(self.w80_by_class.get(key, []))

            n_active = sum(get_count(k) for k in active_keys)
            n_inactive = sum(get_count(k) for k in inactive_keys)

            item_active = QListWidgetItem(f"Active (N={n_active})")
            # store the actual class keys (list) in UserRole
            item_active.setData(Qt.UserRole, active_keys)
            item_active.setSelected(True)
            self.class_list.addItem(item_active)

            item_inactive = QListWidgetItem(f"Inactive (N={n_inactive})")
            item_inactive.setData(Qt.UserRole, inactive_keys)
            item_inactive.setSelected(False)
            self.class_list.addItem(item_inactive)
        self.on_selection_changed()
                    

    def create_plot_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.figure = Figure(figsize=(12, 8), facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, panel)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        return panel

    def setup_style(self):
        style = """
        QMainWindow { background-color: #f5f5f5; }
        QPushButton {
            background-color: #4CAF50; color: white; border: none;
            padding: 8px 16px; border-radius: 4px; font-weight: bold;
        }
        QPushButton:hover { background-color: #45a049; }
        QPushButton:pressed { background-color: #3d8b40; }
        QPushButton:checked { background-color: #2196F3; }
        QListWidget { border: 1px solid #ddd; border-radius: 4px; background-color: white; }
        QListWidget::item { padding: 4px; border-bottom: 1px solid #eee; }
        QListWidget::item:selected { background-color: #e3f2fd; }
        QGroupBox {
            font-weight: bold; border: 2px solid #ccc; border-radius: 5px;
            margin-top: 1ex; padding-top: 10px;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        """
        self.setStyleSheet(style)

    def load_data(self):
        processor = DataProcessor()
        if not processor.filenames:
            QMessageBox.warning(self, "No Data", "No FITS files found in Gaussian_fits directory")
            return

        self.progress_bar.setVisible(True)
        self.loader = DataLoader(processor.filenames)
        self.loader.progress.connect(self.progress_bar.setValue)
        self.loader.status.connect(self.status_bar.showMessage)
        self.loader.finished_loading.connect(self.on_data_loaded)
        self.loader.start()

    def on_data_loaded(self, w80_data):
        self.w80_by_class = w80_data
        self.classes = sorted([k for k in w80_data.keys() 
                              if not k.startswith('_')])

        self.update_class_list()
        self.setup_bins()
        self.update_plot()
        self.update_statistics()
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Data loaded successfully", 2000)

    def setup_bins(self):
        if not self.w80_by_class:
            return
        selected = self.get_selected_classes()
        if not selected:
            return
        
        all_vals = []
        if self.agn_mode:
            agn_data = self.w80_by_class.get('_w80_by_agn_', {})
            for cls in selected:
                # support grouped selections where cls can be a list/tuple/ndarray of keys
                if isinstance(cls, (list, tuple, np.ndarray)):
                    for sub in cls:
                        key = str(sub)
                        if key in agn_data:
                            vals = np.array(agn_data[key])
                            all_vals.extend(vals[np.isfinite(vals) & (vals > 0)])
                else:
                    key = str(cls)
                    if key in agn_data:
                        vals = np.array(agn_data[key])
                        all_vals.extend(vals[np.isfinite(vals) & (vals > 0)])
        else:
            for cls in selected:
                if isinstance(cls, (list, tuple, np.ndarray)):
                    for sub in cls:
                        key = str(sub)
                        if key in self.w80_by_class:
                            vals = np.array(self.w80_by_class[key])
                            all_vals.extend(vals[np.isfinite(vals) & (vals > 0)])
                else:
                    key = str(cls)
                    if key in self.w80_by_class:
                        vals = np.array(self.w80_by_class[key])
                        all_vals.extend(vals[np.isfinite(vals) & (vals > 0)])
        
        if not all_vals:
            return
        min_val = min(all_vals) * 0.9
        max_val = max(all_vals) * 1.1
        n_bins = self.bin_spinner.value()
        self.bins = np.logspace(np.log10(min_val), np.log10(max_val), n_bins)

    def get_selected_classes(self):
        selected = []
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            if item.isSelected():
                selected.append(item.data(Qt.UserRole))
        return selected

    def update_plot(self, is_mean=True, miliqua_mode=False):
        if not self.w80_by_class:
            return
        selected = self.get_selected_classes()
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if not selected:
            ax.text(0.5, 0.5, "No classes selected\nSelect classes from the left panel",
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            self.canvas.draw()
            return

        plot_data = []
        labels = []
        colors = plt.cm.Set3(np.linspace(0, 1, len(selected)))
        simbad_agn_names = {'AGN', 'AG?', 'Q', 'Sy1', 'Sy2', 'QSO', 'Q?'}
        for cls in selected:
            # Support single class keys and grouped keys (lists) created by the "Active/Inactive" button
            if isinstance(cls, (list, tuple, np.ndarray)):
                combined = []
                for sub in cls:
                    key = str(sub)
                    vals_sub = []

                    if is_mean:
                        if key in self.w80_by_class.get('_w80_means_by_agn_', {}):
                            vals_sub = self.w80_by_class['_w80_means_by_agn_'][key]
                        elif key in self.w80_by_class.get('_w80_means_', {}):
                            vals_sub = self.w80_by_class['_w80_means_'][key]
                        else:
                            vals_sub = self.w80_by_class.get(key, [])
                    else:
                        if key in self.w80_by_class.get('_w80_by_agn_', {}):
                            arr = np.array(self.w80_by_class['_w80_by_agn_'][key])
                        else:
                            arr = np.array(self.w80_by_class.get(key, []))
                        vals_sub = arr[np.isfinite(arr) & (arr > 0)].tolist()

                    if isinstance(vals_sub, np.ndarray):
                        combined.extend(vals_sub.tolist())
                    else:
                        combined.extend(vals_sub)

                vals = np.array(combined)

            else:
                key = str(cls)
                # Single-key path: try AGN-specific containers first, then SIMBAD containers, then fall back
                if is_mean:
                    if key in self.w80_by_class.get('_w80_means_by_agn_', {}):
                        vals = self.w80_by_class.get('_w80_means_by_agn_', {}).get(key, [])
                    elif key in self.w80_by_class.get('_w80_means_', {}):
                        vals = self.w80_by_class.get('_w80_means_', {}).get(key, [])
                    else:
                        vals = self.w80_by_class.get(key, [])
                else:
                    if key in self.w80_by_class.get('_w80_by_agn_', {}):
                        vals = np.array(self.w80_by_class.get('_w80_by_agn_', {}).get(key, []))
                        vals = vals[np.isfinite(vals) & (vals > 0)]
                    else:
                        vals = np.array(self.w80_by_class.get(key, []))
                        vals = vals[np.isfinite(vals) & (vals > 0)]
            
            if len(vals) > 0:
                plot_data.append(vals)
                if isinstance(cls, (list, tuple, np.ndarray)):
                    parts = [str(x) for x in cls]
                    label = "\n".join(parts) + f"\n(N={len(vals)})"
                else:
                    label = f"{cls}\n(N={len(vals)})"
                labels.append(label)
                

        if not plot_data:
            ax.text(0.5, 0.5, "No valid data in selected classes",
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            self.canvas.draw()
            return

        if self.trend_mode:
            self.create_trend_plot(ax, plot_data, labels, colors)
        else:
            self.create_histogram_plot(ax, plot_data, labels, colors, is_mean=is_mean)

        self.figure.tight_layout()
        self.canvas.draw()

    def create_histogram_plot(self, ax, data, labels, colors, is_mean=True):
        ax.hist(data, bins=self.bins, label=labels, color=colors, alpha=0.8,
            stacked=True, edgecolor='black', linewidth=0.5)
        ax.set_xscale('log')
        ax.set_xlabel('W80 [km/s]', fontsize=12, fontweight='bold')
        if is_mean:
            ax.set_ylabel('Number of Galaxies', fontsize=12, fontweight='bold')
        else:
            ax.set_ylabel('Number of Spaxels', fontsize=12, fontweight='bold')
        
        title_suffix = "AGN Classification" if self.agn_mode else "SIMBAD Classification"
        ax.set_title(f'W80 Distribution by {title_suffix}',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

    def create_trend_plot(self, ax, data, labels, colors):
        bin_centers = np.sqrt(self.bins[:-1] * self.bins[1:])
        for vals, label, color in zip(data, labels, colors):
            counts, _ = np.histogram(vals, bins=self.bins)
            if counts.sum() > 0:
                normalized = counts / counts.sum()
                smoothed = gaussian_filter1d(normalized.astype(float), sigma=5.0)
                ax.plot(bin_centers, smoothed, color=color, linewidth=2.5,
                        label=label, alpha=0.9)
                ax.fill_between(bin_centers, 0, smoothed, color=color, alpha=0.3)
                if not getattr(ax, "_vline_drawn", False):
                    ax.axvline(760, color='k', linestyle='--', linewidth=1, label='_vline')
                    ax._vline_drawn = True
        ax.set_xscale('log')
        ax.set_xlabel('W80 [km/s]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Frequency', fontsize=12, fontweight='bold')
        
        title_suffix = "AGN Classification" if self.agn_mode else "SIMBAD Classification"
        ax.set_title(f'Normalized W80 Trends by {title_suffix}',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)

    def update_statistics(self):
        if not self.w80_by_class:
            self.stats_label.setText("No data loaded")
            return
        selected = self.get_selected_classes()
        if not selected:
            self.stats_label.setText("No classes selected")
            return
        
        stats_text = []
        total_spaxels = 0
        
        for cls in selected:
            # Prepare a combined array for grouped selections (lists/tuples/ndarray)
            combined_vals = []

            if isinstance(cls, (list, tuple, np.ndarray)):
                for sub in cls:
                    key = str(sub)
                    if self.agn_mode:
                        arr = np.array(self.w80_by_class.get('_w80_by_agn_', {}).get(key, []))
                    else:
                        arr = np.array(self.w80_by_class.get(key, []))
                    if arr.size:
                        arr = arr[np.isfinite(arr) & (arr > 0)]
                        if arr.size:
                            combined_vals.extend(arr.tolist())
                label = ", ".join(map(str, cls))
            else:
                key = str(cls)
                if self.agn_mode:
                    arr = np.array(self.w80_by_class.get('_w80_by_agn_', {}).get(key, []))
                else:
                    arr = np.array(self.w80_by_class.get(key, []))
                if arr.size:
                    arr = arr[np.isfinite(arr) & (arr > 0)]
                    if arr.size:
                        combined_vals = arr.tolist()
                label = str(cls)

            vals = np.array(combined_vals)
            if vals.size > 0:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                median_val = np.median(vals)
                stats_text.append(f"{label}:")
                stats_text.append(f"  N = {len(vals)}")
                stats_text.append(f"  Mean = {mean_val:.1f} km/s")
                stats_text.append(f"  Std  = {std_val:.1f} km/s")
                stats_text.append(f"  Median = {median_val:.1f} km/s")
                stats_text.append("")
                total_spaxels += len(vals)
        
        stats_text.insert(0, f"Total Spaxels: {total_spaxels}")
        stats_text.insert(1, "")
        
        # Add AGN summary if in AGN mode
        if self.agn_mode and '_w80_means_by_agn_' in self.w80_by_class:
            means_agn = self.w80_by_class['_w80_means_by_agn_']
            n_agn = len(means_agn.get('AGN', []))
            n_nonagn = len(means_agn.get('Non-AGN', []))
            stats_text.insert(2, f"AGN Galaxies: {n_agn}")
            stats_text.insert(3, f"Non-AGN Galaxies: {n_nonagn}")
            stats_text.insert(4, "")
        
        self.stats_label.setText("\n".join(stats_text))

    def toggle_data_mode(self):
        """Toggle between showing individual spaxels and galaxy means."""
        if self.btn_toggle_data.isChecked():
            is_mean = False
            self.btn_toggle_data.setText("Show Individual Spaxels")
        else:
            is_mean = True
            self.btn_toggle_data.setText("Show Galaxy Means")
        self.update_plot(is_mean)

    # Event handlers
    def on_selection_changed(self):
        QTimer.singleShot(100, self._delayed_update)

    def _delayed_update(self):
        self.setup_bins()
        self.update_plot()
        self.update_statistics()

    def select_all(self):
        for i in range(self.class_list.count()):
            self.class_list.item(i).setSelected(True)

    def clear_all(self):
        self.class_list.clearSelection()

    def toggle_view_mode(self):
        self.trend_mode = self.btn_toggle_view.isChecked()
        self.btn_toggle_view.setText("Show Histogram" if self.trend_mode else "Show Trends")
        is_mean = not self.btn_toggle_data.isChecked()
        self.update_plot(is_mean=is_mean, miliqua_mode=getattr(self, 'miliqua_mode', False))

    def update_bins(self):
        self.setup_bins()
        self.update_plot()

    def export_plot(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Plot", "w80_distribution.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )
        if filename:
            try:
                self.figure.savefig(filename, dpi=300, bbox_inches='tight',
                                    facecolor='white', edgecolor='none')
                self.status_bar.showMessage(f"Plot exported to {filename}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export plot:\\n{str(e)}")


def main():
    # Initialize QApplication
    try:
        app = QApplication(sys.argv)
    except Exception as e:
        print(f"Failed to initialize Qt application: {e}")
        return 1

    app.setApplicationName(" ")
    app.setApplicationVersion(" ")
    app.setOrganizationName("Kolja Reuter")
    app.setApplicationDisplayName("W80 Histograms")

    try:
        if Path("icon.png").exists():
            app.setWindowIcon(QIcon("icon.png"))
    except Exception:
        pass

    # Quick checks
    required_dirs = ["Gaussian_fits", "KMOS3D_ALL"]
    missing = [d for d in required_dirs if not Path(d).exists()]
    if missing:
        error_msg = "Missing required directories: " + ", ".join(missing)
        print(error_msg)
        QMessageBox.critical(None, "Missing Data Directories",
                             error_msg + "\n\nExpected:\n- Gaussian_fits/ with *_voronoi_binned.fits\n- KMOS3D_ALL/ with matching .fits")
        return 1

    # Optional: warn if DB missing (SIMBAD still queried by coords but z/coords need DB)
    if not Path("k3d_fnlsp_table_v3.fits").exists():
        QMessageBox.warning(None, "Missing Database",
                            "Database file k3d_fnlsp_table_v3.fits not found.\n"
                            "Galaxy metadata (z, RA/Dec) may be unavailable.")
    
    # Optional: warn if AGN catalog missing
    if not Path("AGN_SAMPLE.fits").exists():
        QMessageBox.information(None, "AGN Catalog",
                               "AGN_SAMPLE.fits not found.\n"
                               "AGN classification will be unavailable, but SIMBAD classification will work normally.")

    window = ModernHistogramWidget()
    window.setMinimumSize(800, 600)
    window.show()

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())