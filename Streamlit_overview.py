from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
import streamlit as st
import os
import subprocess
import sys
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
from astropy.io import fits
import numpy as np
import warnings
import pandas as pd
import re
from tools import KMOS_readout as tools
import pickle
import matplotlib.pyplot as plt
from urllib.parse import quote
import logging
from scipy.ndimage import label

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Galaxy Explorer",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* Versteckt den bunten Ladebalken oben (stDecoration) */
        [data-testid="stDecoration"] {
            display: none;
        }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
    }
    .metric-box {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ff4b4b;
        margin-bottom: 10px;
    }
    /* 1. Hide the standard radio circles */
    div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    /* 2. Container styling: A simple line at the bottom */
    div[role="radiogroup"] {
        background-color: transparent;
        display: flex;
        gap: 24px;              /* Space between tabs */
        border-bottom: 1px solid #334155; /* Thin gray line across the whole width */
        padding-bottom: 0;
        margin-bottom: 20px;
    }

    /* 3. Label styling: Plain text */
    div[role="radiogroup"] label {
        background-color: transparent;
        color: #94A3B8;         /* Muted gray text */
        padding: 8px 4px;       /* Vertical padding */
        border-radius: 0;
        border-bottom: 3px solid transparent; /* Hidden underline by default */
        cursor: pointer;
        font-size: 1rem;
        transition: all 0.2s ease;
        margin-bottom: -1px;    /* Pulls active underline on top of the gray line */
    }

    /* Hover state */
    div[role="radiogroup"] label:hover {
        color: #E2E8F0;         /* Lighter gray on hover */
    }

    /* 4. Selected State: White text + Accent Underline */
    div[role="radiogroup"] label:has(input:checked) {
        color: white !important;
        font-weight: 600;
        border-bottom: 3px solid #FF4B4B !important; /* Streamlit Red (or change to #00CCFF for Cyan) */
    }
    </style>""", unsafe_allow_html=True)

# --- CONFIGURATION ---
DATA_PATH = "KMOS3D_ALL"
RESULTS_PATH = "Gaussian_fits"
PKL_PATH = "pkl_files"  # New path for pickle files
CROSSMATCH_PATH = "crossmatch.fits"  # Crossmatch catalog with survey links

# Ensure PKL directory exists
os.makedirs(PKL_PATH, exist_ok=True)

# --- 0. MATH & FITTING FUNCTIONS ---
def gaussian(x: np.ndarray, a: float, sigma: float, c: float) -> np.ndarray:
    """Gaussian function with numerical stability checks"""
    if sigma == 0: return np.zeros_like(x)
    arg = (x - c)**2 / (2 * sigma**2 + 1e-10)
    arg = np.clip(arg, 0, 50)
    return np.abs(a) * np.exp(-arg)

def linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Linear background model"""
    return a * x + b

def double_gaussian(x: np.ndarray, a1: float, b1: float, c1: float,
                    a2: float, b2: float, c2: float) -> np.ndarray:
    return gaussian(x, a1, b1, c1) + gaussian(x, a2, b2, c2)



def get_galaxies_sorted_by_w80(galaxy_names, results_path):
    """
    Calculates mean W80 for all galaxies and returns a sorted list of tuples:
    [(galaxy_name, mean_w80_value), ...]
    """
    # Use PKL_PATH instead of results_path for cache
    os.makedirs(PKL_PATH, exist_ok=True)  # Create directory if it doesn't exist
    cache_file = os.path.join(PKL_PATH, "sorted_w80_cache.pkl")
    
    # --- 1. CACHE CHECK ---
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                # Validate cache format: should be list of tuples (name, value)
                if isinstance(cached_data, list) and len(cached_data) > 0:
                    if isinstance(cached_data[0], (list, tuple)) and len(cached_data[0]) >= 2:
                        return cached_data
                # If validation fails, delete corrupted cache
                os.remove(cache_file)
        except Exception:
            # If cache is corrupted, delete it and recalculate
            try:
                os.remove(cache_file)
            except:
                pass

    galaxy_stats = []
    
    # --- 2. INITIALIZE PROGRESS BAR ---
    st.markdown("#### 📊 Sorting Galaxies by W80")
    progress_text = "Analyzing W80 maps and removing noise clusters..."
    progress_bar = st.progress(0, text=progress_text)

    for i, galaxy in enumerate(galaxy_names):
        file_path = os.path.join(results_path, f"{galaxy}_voronoi_binned.fits")
        mean_w80 = 0.0
        
        if os.path.exists(file_path):
            try:
                # Use your existing tool to read the W80 map
                w80_map = tools.master_map_w80(file_path)
                
                if w80_map is not None:
                    # --- 3. SPATIAL FILTERING (Remove floating pixels) ---
                    # Identify where W80 has valid positive values
                    valid_mask = np.isfinite(w80_map) & (w80_map > 10.0)
                    
                    # Label connected islands of pixels
                    labeled_array, num_features = label(valid_mask)
                    
                    # Cleaned mask: only keep islands with 5 or more pixels
                    cleaned_mask = np.zeros_like(valid_mask)
                    min_cluster_size = 5
                    
                    for feature_id in range(1, num_features + 1):
                        coords = np.where(labeled_array == feature_id)
                        if len(coords[0]) >= min_cluster_size:
                            cleaned_mask[coords] = True
                    
                    # Extract valid data from the cleaned area
                    valid_values = w80_map[cleaned_mask]
                    
                    if valid_values.size > 0:
                        mean_w80 = np.mean(valid_values)
                    else:
                        mean_w80 = 0.0 # No consistent galaxy structure found
                
            except Exception as e:
                print(f"Error processing {galaxy}: {e}")
                mean_w80 = 0.0

        galaxy_stats.append((galaxy, mean_w80))
        
        # --- 4. UPDATE PROGRESS BAR ---
        progress_bar.progress((i + 1) / len(galaxy_names), text=f"{progress_text} ({i+1}/{len(galaxy_names)})")

    # --- 5. CLEAN UP & SORTING ---
    progress_bar.empty()
    
    # Sort by value (descending), moving objects with 0.0 (noise/failed) to the bottom
    sorted_stats = sorted(galaxy_stats, key=lambda x: x[1], reverse=True)

    try:
        os.makedirs(PKL_PATH, exist_ok=True)  # Ensure directory exists
        with open(cache_file, "wb") as f:
            pickle.dump(sorted_stats, f)
    except Exception as e:
        st.error(f"Could not save W80 cache: {e}")

    return sorted_stats

@st.cache_data
def load_galaxy_list(path):
    if not os.path.exists(path): return []
    files = [f for f in os.listdir(path) if f.endswith(".fits") and "_voronoi_binned" not in f]
    return sorted([os.path.splitext(f)[0] for f in files])

@st.cache_data
def load_fits_header(file_path):
    try:
        with fits.open(file_path) as hdul:
            return dict(hdul[0].header)
    except Exception: return None

@st.cache_data(show_spinner=False)
def summed_aperture_spectrum(flux_cube, noise_cube, center_x, center_y, radius_px):
    """
    Calculates the TOTAL (Summed) flux within a circular aperture.
    """
    ny, nx = flux_cube.shape[1], flux_cube.shape[2]
    y_grid, x_grid = np.indices((ny, nx))
    
    # Create Aperture Mask
    dist_grid = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
    mask_2d = dist_grid <= radius_px
    
    # Broadcast mask to 3D (spectral axis)
    # We only sum pixels inside the mask
    flux_masked = flux_cube[:, mask_2d]
    noise_masked = noise_cube[:, mask_2d]
    
    # 1. Calculate Total Flux (Sum)
    # nansum treats NaNs as zero, which is safer than simple sum
    stacked_flux = np.nansum(flux_masked, axis=1)
    
    # 2. Calculate Propagated Noise (Quadrature Sum)
    # Error propagation for a sum: sqrt(sum(sigma^2))
    stacked_noise = np.sqrt(np.nansum(noise_masked**2, axis=1))
    
    return stacked_flux, stacked_noise

@st.cache_data(show_spinner=True)
def calculate_w80_map(results_file_path, fits_file_path):
    try:
        with fits.open(results_file_path) as hdul:
            def get_data(name): return hdul[name].data if name in hdul else np.zeros_like(hdul['A'].data)
            A, B, C = get_data('A'), get_data('B'), get_data('C')
            A1, B1, C1 = get_data('A1'), get_data('B1'), get_data('C1')
            A2, B2, C2 = get_data('A2'), get_data('B2'), get_data('C2')
            chi_single, chi_double = get_data('CHI1'), get_data('CHI2')

        with fits.open(fits_file_path) as hdul_raw:
            hdr = hdul_raw[1].header
            z_gal = tools.redshift(hdul_raw)  # Get the galaxy redshift
            n_spectral = hdul_raw[1].data.shape[0]
            
            # 1. Create REST-FRAME wavelength grid
            wave_obs = hdr['CRVAL3'] + (np.arange(n_spectral) - (hdr['CRPIX3'] - 1)) * hdr['CDELT3']
            wave_rest = wave_obs / (1 + z_gal)

        ny, nx = A.shape
        w80_map = np.full((ny, nx), np.nan)
        C_LIGHT, HA_REST = 299792.458, 0.65628 

        for i in range(ny):
            for j in range(nx):
                # 2. Skip if no signal detected
                if A[i,j] < 1e-10 and A1[i,j] < 1e-10: continue

                # 3. Decision logic: Which fit is better?
                use_double = abs(chi_double[i,j] - 1) < abs(chi_single[i,j] - 1)
                
                if not use_double:
                    model = gaussian(wave_rest, A[i,j], B[i,j], C[i,j])
                else:
                    model = gaussian(wave_rest, A1[i,j], B1[i,j], C1[i,j]) + \
                            gaussian(wave_rest, A2[i,j], B2[i,j], C2[i,j])

                # 4. Integrate on the rest-frame grid
                cumsum = np.cumsum(model)
                total = cumsum[-1]
                if total > 0:
                    l_idx = np.searchsorted(cumsum, 0.1 * total)
                    h_idx = np.searchsorted(cumsum, 0.9 * total)
                    h_idx = min(h_idx, len(wave_rest)-1)
                    
                    # Width in rest-frame microns
                    width_microns = wave_rest[h_idx] - wave_rest[l_idx]
                    
                    # Final physical velocity calculation
                    w80_kms = (width_microns / HA_REST) * C_LIGHT
                    w80_map[i,j] = w80_kms if w80_kms > 10 else np.nan

        return w80_map
    except Exception as e:
        st.error(f"W80 Calc Error: {e}")
        return None
    

# --- AGN & SIMBAD HELPER FUNCTIONS ---
# --- SIMBAD TYPE MAPPING ---
# Maps SIMBAD short codes to readable names
SIMBAD_OTYPE_MAPPING = {
    "AGN":    "Active Galaxy Nucleus",
    "QSO":    "Quasar",
    "Sy1":    "Seyfert 1",
    "Sy2":    "Seyfert 2",
    "Sy":     "Seyfert Galaxy",
    "rG":     "Radio Galaxy",
    "X":      "X-ray Source",
    "IR":     "Infra-Red Source",
    "UV":     "UV Source",
    "Mas":    "Maser",
    "SBG":    "Starburst Galaxy",
    "G":      "Galaxy",
    "ClG":    "Cluster of Galaxies",
    "GinPair": "Galaxy in Pair",
    "GinGroup": "Galaxy in Group",
    "LINER":  "LINER-type AGN",
    "Bla":    "Blazar",
    "BLLac":  "BL Lac Object",
    "EmG":    "Emission Line Galaxy",
    "Blue":   "Blue Object",
    "Radio":  "Radio Source",
    "HII":    "HII Region",
    "LSB":    "Low Surface Brightness Galaxy",
    "GiC":    "Galaxy in Cluster",
    "VisS":   "Visual Source",
    "Candidate_G": "Candidate Galaxy"
}

def translate_simbad_types(type_list):
    """
    Takes a LIST of codes (e.g. ['Sy1', 'rG']), translates them, 
    and returns a clean string.
    """
    translated = []
    for code in type_list:
        clean_code = code.strip()
        if not clean_code or clean_code == "--": continue
        
        # Translate or keep original
        readable = SIMBAD_OTYPE_MAPPING.get(clean_code, clean_code)
        translated.append(readable)
    
    # Return unique, sorted types joined by pipe
    # Using dict.fromkeys to preserve order but remove duplicates
    return " | ".join(list(dict.fromkeys(translated)))

# --- CLASSIFICATION HELPER FUNCTIONS ---
@st.cache_data
def load_agn_catalog_ids(path="AGN_SAMPLE.fits"):
    """Loads AGN catalog IDs from the provided FITS file."""
    try:
        if not os.path.exists(path):
            st.warning(f"AGN catalog not found: {path}")
            return []
        with fits.open(path) as hdulist:
            data = hdulist[1].data
            # Check available columns
            colnames = [col.name for col in hdulist[1].columns]
            
            # Try different possible column names
            id_col = None
            for possible_col in ['ID_TARGETED', 'ID', 'NAME', 'OBJECT']:
                if possible_col in colnames:
                    id_col = possible_col
                    break
            
            if id_col is None:
                st.warning(f"No ID column found in AGN catalog. Available: {colnames}")
                return []
            
            raw_ids = data[id_col]
            # Clean and normalize IDs
            agn_list = []
            for element in raw_ids:
                if isinstance(element, bytes):
                    element = element.decode('utf-8')
                # Normalize: strip whitespace, replace underscores
                clean_id = str(element).strip().replace('_', ' ')
                agn_list.append(clean_id)
            
            return agn_list
    except Exception as e:
        st.error(f"Error loading AGN catalog: {e}")
        return []

def is_agn_galaxy(galaxy_name, agn_list):
    """Checks if a galaxy is in the AGN list using flexible matching."""
    if not galaxy_name or not agn_list:
        return False
    
    # Normalize the input name
    name_normalized = galaxy_name.strip().replace('_', ' ')
    
    # Also create variants for COSMOS naming convention
    # COS4_00779 -> COSMOS 4 00779 or COS4 00779
    variants = [name_normalized]
    
    # Handle COS -> COSMOS conversion
    if name_normalized.startswith("COS"):
        # COS4 00779 -> COSMOS 4 00779
        cosmos_variant = re.sub(r'^COS(\d+)\s*', r'COSMOS \1 ', name_normalized)
        variants.append(cosmos_variant.strip())
        
        # Also try without space: COSMOS4 00779
        cosmos_variant2 = re.sub(r'^COS(\d+)\s*', r'COSMOS\1 ', name_normalized)
        variants.append(cosmos_variant2.strip())
    
    # Handle COSMOS -> COS conversion (reverse)
    if name_normalized.startswith("COSMOS"):
        cos_variant = re.sub(r'^COSMOS\s*(\d+)\s*', r'COS\1 ', name_normalized)
        variants.append(cos_variant.strip())
    
    # Check all variants against the AGN list
    for variant in variants:
        # Direct match
        if variant in agn_list:
            return True
        
        # Case-insensitive match
        variant_lower = variant.lower()
        for agn_id in agn_list:
            if agn_id.lower() == variant_lower:
                return True
            
            # Partial match (if the galaxy ID is contained in the AGN ID or vice versa)
            if variant_lower in agn_id.lower() or agn_id.lower() in variant_lower:
                # Make sure it's a significant match (not just a few characters)
                if len(variant) > 5 and len(agn_id) > 5:
                    return True
    
    return False

def get_simbad_link_from_header(fits_path):
    """
    Reads SIMBAD classification and URL from the FITS header instead of querying SIMBAD.
    Returns a clickable Markdown link.
    """
    try:
        with fits.open(fits_path) as hdul:
            header = hdul[0].header
            
            simbad_class = header.get('SIMBAD_CL', 'Not found')
            simbad_url = header.get('HIERARCH SIMBAD_URL', None)
            
            if simbad_class and simbad_class not in ['Not found', 'N/A', 'Invalid coordinates']:
                # Translate the class code to readable name
                readable_class = SIMBAD_OTYPE_MAPPING.get(simbad_class, simbad_class)
                
                if simbad_url and simbad_url != 'N/A':
                    return f"[{readable_class}]({simbad_url})"
                else:
                    return readable_class
            else:
                return "Not Found"
                
    except Exception as e:
        return f"Err: {str(e)[:15]}"

@st.cache_data
def load_crossmatch_links(crossmatch_path=CROSSMATCH_PATH):
    """
    Load the LINKS table from the crossmatch FITS file.
    Returns a DataFrame with KMOS3D_ID, matched flags, and URLs for all surveys.
    Returns None if the file doesn't exist or can't be read.
    """
    if not os.path.exists(crossmatch_path):
        return None
    try:
        with fits.open(crossmatch_path) as hdul:
            # Try LINKS extension first
            links_df = None
            for ext in hdul:
                if ext.name == 'LINKS':
                    links_df = pd.DataFrame(ext.data)
                    break
            
            if links_df is None:
                return None
            
            # Decode byte strings from FITS
            for col in links_df.columns:
                if links_df[col].dtype == object:
                    links_df[col] = links_df[col].apply(
                        lambda x: x.decode('utf-8').strip() if isinstance(x, bytes) else str(x).strip()
                    )
            
            # Clean up the KMOS3D_ID column for matching
            if 'KMOS3D_ID' in links_df.columns:
                links_df['KMOS3D_ID'] = links_df['KMOS3D_ID'].str.strip()
            
            return links_df
    except Exception as e:
        print(f"Error loading crossmatch: {e}")
        return None


def get_catalog_links_for_galaxy(galaxy_name, ra=None, dec=None, crossmatch_df=None):
    """
    Look up a galaxy in the crossmatch table and return its catalog links.
    
    Returns a dict: {survey_name: {'matched': bool, 'url': str}} 
    Returns empty dict if no match found.
    """
    if crossmatch_df is None or crossmatch_df.empty:
        return {}
    
    # --- 1. Try matching by KMOS3D_ID ---
    row = None
    if galaxy_name:
        # Clean the galaxy name for matching
        clean_name = galaxy_name.strip()
        mask = crossmatch_df['KMOS3D_ID'] == clean_name
        if mask.any():
            row = crossmatch_df[mask].iloc[0]
        else:
            # Fuzzy: try case-insensitive
            mask = crossmatch_df['KMOS3D_ID'].str.lower() == clean_name.lower()
            if mask.any():
                row = crossmatch_df[mask].iloc[0]
    
    # --- 2. Fallback: match by closest RA/DEC ---
    if row is None and ra is not None and dec is not None:
        try:
            df_ra = pd.to_numeric(crossmatch_df['RA'], errors='coerce')
            df_dec = pd.to_numeric(crossmatch_df['DEC'], errors='coerce')
            # Simple angular distance in arcsec (small-angle approx is fine for <3")
            cos_dec = np.cos(np.radians(dec))
            dist_arcsec = 3600 * np.sqrt(
                ((df_ra - ra) * cos_dec)**2 + (df_dec - dec)**2
            )
            min_idx = dist_arcsec.idxmin()
            if dist_arcsec[min_idx] < 5.0:  # 5 arcsec tolerance
                row = crossmatch_df.loc[min_idx]
        except Exception:
            pass
    
    if row is None:
        return {}
    
    # --- 3. Extract survey links ---
    # Survey display config: name → (matched_col, url_col, icon, display_name)
    survey_display = {
        'NED':     ('NED_MATCHED',     'NED_URL',     '🌐', 'NED'),
        'SDSS':    ('SDSS_MATCHED',    'SDSS_URL',    '🔴', 'SDSS DR16'),
        'WISE':    ('WISE_MATCHED',    'WISE_URL',    '🟡', 'AllWISE'),
        '2MASS':   ('2MASS_MATCHED',   '2MASS_URL',   '🟠', '2MASS'),
        'JWST':    ('JWST_MATCHED',    'JWST_URL',    '🔭', 'JWST/MAST'),
        # Deep field surveys
        'COS2020': ('COS2020_MATCHED', 'COS2020_URL', '🟣', 'COSMOS2020'),
        'VLA3GHZ': ('VLA3GHZ_MATCHED', 'VLA3GHZ_URL', '📻', 'VLA 3GHz'),
        'CCOSLEG': ('CCOSLEG_MATCHED', 'CCOSLEG_URL', '☢️', 'Chandra COSMOS'),
        'CDFS7MS': ('CDFS7MS_MATCHED', 'CDFS7MS_URL', '☢️', 'CDF-S 7Ms'),
        'VLACDFS': ('VLACDFS_MATCHED', 'VLACDFS_URL', '📻', 'VLA E-CDFS'),
        'XUDS':    ('XUDS_MATCHED',    'XUDS_URL',    '☢️', 'X-UDS Chandra'),
    }
    
    links = {}
    for survey, (match_col, url_col, icon, display_name) in survey_display.items():
        matched = False
        url = ''
        
        if match_col in row.index:
            val = row[match_col]
            matched = bool(val) if not isinstance(val, str) else val.lower() == 'true'
        
        if url_col in row.index:
            url = str(row[url_col]).strip()
            if url in ('', 'nan', 'None', '--'):
                url = ''
        
        links[survey] = {
            'matched': matched,
            'url': url,
            'icon': icon,
            'display_name': display_name,
        }
    
    # Add SIMBAD link (always present for every source)
    simbad_url = ''
    if 'SIMBAD_URL' in row.index:
        simbad_url = str(row['SIMBAD_URL']).strip()
        if simbad_url in ('', 'nan', 'None', '--'):
            simbad_url = ''
    links['SIMBAD'] = {
        'matched': True,  # always a coordinate lookup
        'url': simbad_url,
        'icon': '🔵',
        'display_name': 'SIMBAD',
    }
    
    return links


@st.cache_data
def load_crossmatch_full(crossmatch_path=CROSSMATCH_PATH):
    """
    Load full crossmatch data from all survey HDUs in the crossmatch FITS.
    Returns a merged DataFrame with WISE magnitudes, NED data, JWST info etc.
    keyed by KMOS3D_ID.
    """
    if not os.path.exists(crossmatch_path):
        return None
    try:
        merged = None
        with fits.open(crossmatch_path) as hdul:
            # Load from each survey HDU and merge by KMOS3D_ID
            for ext in hdul:
                if ext.name in ('PRIMARY', 'SUMMARY', 'LINKS'):
                    continue
                if not hasattr(ext, 'data') or ext.data is None:
                    continue
                try:
                    df = pd.DataFrame(ext.data)
                    # Decode byte strings
                    for col in df.columns:
                        if df[col].dtype == object:
                            df[col] = df[col].apply(
                                lambda x: x.decode('utf-8').strip() if isinstance(x, bytes) else str(x).strip()
                            )
                    
                    if ext.name in ('KMOS3D_S', 'KMOS3D_SOURCES', 'KMOS3D_SO'):  # handle FITS name truncation
                        merged = df.copy()
                    elif merged is not None and 'KMOS3D_ID' in df.columns:
                        # Strip _MATCH suffix to get clean survey prefix
                        survey_prefix = re.sub(r'_M(?:ATCH|ATC|AT|A)?$', '', ext.name) + '_'
                        rename_map = {}
                        for col in df.columns:
                            if col not in ('KMOS3D_ID', 'KMOS3D_RA', 'KMOS3D_DEC'):
                                rename_map[col] = survey_prefix + col
                        df_renamed = df.rename(columns=rename_map)
                        merged = merged.merge(
                            df_renamed.drop(columns=['KMOS3D_RA', 'KMOS3D_DEC'], errors='ignore'),
                            left_on='OBJECT', right_on='KMOS3D_ID', how='left'
                        )
                        if 'KMOS3D_ID' in merged.columns:
                            merged.drop(columns=['KMOS3D_ID'], inplace=True, errors='ignore')
                except Exception as e:
                    print(f"Error reading HDU {ext.name}: {e}")
                    continue
        
        if merged is not None:
            # Convert numeric columns
            for col in merged.columns:
                if col in ('OBJECT', 'OBJ_TARG', 'RADECSYS', 'OBSBAND', 'VERSION',
                           'INSTRUME', 'filename', 'filepath'):
                    continue
                try:
                    merged[col] = pd.to_numeric(merged[col], errors='ignore')
                except Exception:
                    pass
        
        return merged
    except Exception as e:
        print(f"Error loading full crossmatch: {e}")
        return None


@st.cache_data(show_spinner="Computing fossil scores...", ttl=3600)
def compute_fossil_scores(galaxy_list, sorted_galaxy_data, crossmatch_df,
                          agn_ids, results_path):
    """
    Compute a Fossil AGN Outflow Score for each galaxy.
    
    HARD EXCLUSION:
      Milliquas AGN OR SIMBAD AGN → Lit_AGN = True, Fossil_Score = 0
    
    Scoring (0–8 scale, higher = stronger fossil candidate):
    
    KINEMATIC EVIDENCE (0–3):
      W80 > 600 km/s → 3  |  400–600 → 2  |  200–400 → 1  |  <200 → 0
    
    WHAN ABSENCE (0–1):
      WHAN class is NOT AGN (sAGN/wAGN) → 1
    
    MID-IR FADING (0–2):
      WISE W1−W2 < 0.5 → 2  |  0.5–0.8 → 1  |  ≥ 0.8 (AGN-like) → 0
    
    MISMATCH BONUS (0–2):
      High W80 (≥400) AND low W1−W2 (<0.8) → 2
      Moderate W80 (200–400) AND low W1−W2 (<0.8) → 1
    
    Deep radio/X-ray detections are tracked as INFORMATIONAL flags
    (VLA 3GHz, VLA E-CDFS, CDF-S 7Ms, Chandra COSMOS, X-UDS)
    but do NOT contribute to the score because detection at these
    depths is expected for star-forming galaxies at z~1–2.
    
    Returns a DataFrame with ID, scores, and breakdown.
    """
    # Build W80 lookup from sorted galaxy data
    w80_lookup = {}
    for item in sorted_galaxy_data:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            w80_lookup[item[0]] = item[1]
    
    # Build WISE color lookups from crossmatch
    wise_w1w2 = {}
    wise_w2w3 = {}
    has_deep_radio = {}   # informational only
    has_deep_xray = {}    # informational only
    if crossmatch_df is not None:
        w1_col = next((c for c in crossmatch_df.columns if 'W1mag' in c), None)
        w2_col = next((c for c in crossmatch_df.columns if 'W2mag' in c), None)
        w3_col = next((c for c in crossmatch_df.columns if 'W3mag' in c), None)
        id_col = 'OBJECT' if 'OBJECT' in crossmatch_df.columns else None
        
        # Find deep radio match columns (VLA 3GHz for COSMOS, VLA E-CDFS for GOODS-S)
        radio_match_cols = [c for c in crossmatch_df.columns
                           if ('VLA3G' in c or 'VLAGS' in c) and 'MATCHED' in c]
        # Find deep X-ray match columns (Chandra COSMOS, CDF-S 7Ms, X-UDS)
        xray_match_cols = [c for c in crossmatch_df.columns
                          if ('CXCOS' in c or 'CDFS' in c or 'XUDS' in c) and 'MATCHED' in c]
        
        if id_col:
            for _, row in crossmatch_df.iterrows():
                gal_id = str(row[id_col]).strip()
                # WISE colors
                if w1_col and w2_col:
                    try:
                        w1 = float(row[w1_col])
                        w2 = float(row[w2_col])
                        if np.isfinite(w1) and np.isfinite(w2):
                            wise_w1w2[gal_id] = w1 - w2
                    except (ValueError, TypeError):
                        pass
                if w2_col and w3_col:
                    try:
                        w2_ = float(row[w2_col])
                        w3 = float(row[w3_col])
                        if np.isfinite(w2_) and np.isfinite(w3):
                            wise_w2w3[gal_id] = w2_ - w3
                    except (ValueError, TypeError):
                        pass
                # Deep radio detection (informational flag)
                for rc in radio_match_cols:
                    try:
                        val = row[rc]
                        if (bool(val) if not isinstance(val, str) else val.lower() == 'true'):
                            has_deep_radio[gal_id] = True
                    except Exception:
                        pass
                # Deep X-ray detection (informational flag)
                for xc in xray_match_cols:
                    try:
                        val = row[xc]
                        if (bool(val) if not isinstance(val, str) else val.lower() == 'true'):
                            has_deep_xray[gal_id] = True
                    except Exception:
                        pass
    
    # Load WHAN and SIMBAD info from result headers (batch)
    simbad_agn = {}
    whan_agn = {}
    for gal_name in galaxy_list:
        res_path = os.path.join(results_path, f"{gal_name}_voronoi_binned.fits")
        if os.path.exists(res_path):
            try:
                with fits.open(res_path) as hdul:
                    hdr = hdul[0].header
                    # SIMBAD
                    sc = hdr.get('SIMBAD_CL', '')
                    agn_kw = ['Active', 'Seyfert', 'Quasar', 'Blazar', 'BL Lac',
                              'LINER', 'AGN', 'Sy1', 'Sy2', 'QSO', 'rG', 'AG?']
                    simbad_agn[gal_name] = any(k.lower() in str(sc).lower() for k in agn_kw)
                    # WHAN
                    wc = str(hdr.get('WHAN_CLS', '')).upper()
                    whan_agn[gal_name] = 'SAGN' in wc or 'WAGN' in wc or 'AGN' in wc
            except Exception:
                pass
    
    # Score each galaxy
    rows = []
    for gal_name in galaxy_list:
        w80 = w80_lookup.get(gal_name, 0.0)
        w1w2 = wise_w1w2.get(gal_name, np.nan)
        
        # --- HARD EXCLUSION: Literature AGN ---
        is_mq = is_agn_galaxy(gal_name, agn_ids)
        is_simbad = simbad_agn.get(gal_name, False)
        is_lit_agn = is_mq or is_simbad
        
        is_whan = whan_agn.get(gal_name, False)
        is_radio = has_deep_radio.get(gal_name, False)
        is_xray = has_deep_xray.get(gal_name, False)
        
        if is_lit_agn:
            rows.append({
                'ID': gal_name,
                'W80': round(w80, 1),
                'W1_W2': round(w1w2, 3) if np.isfinite(w1w2) else np.nan,
                'Lit_AGN': True,
                'Milliquas': is_mq,
                'SIMBAD_AGN': is_simbad,
                'WHAN_AGN': is_whan,
                'DeepRadio': is_radio,
                'DeepXray': is_xray,
                'Kin_Score': 0,
                'WHAN_Score': 0,
                'MIR_Score': 0,
                'Mismatch': 0,
                'Fossil_Score': 0,
            })
            continue
        
        # --- Kinematic score (0–3) ---
        if w80 >= 600:
            kin_score = 3
        elif w80 >= 400:
            kin_score = 2
        elif w80 >= 200:
            kin_score = 1
        else:
            kin_score = 0
        
        # --- WHAN absence (0–1) ---
        whan_score = 0 if is_whan else 1
        
        # --- MID-IR fading (0–2) ---
        if np.isfinite(w1w2):
            if w1w2 < 0.5:
                mir_score = 2
            elif w1w2 < 0.8:
                mir_score = 1
            else:
                mir_score = 0
        else:
            mir_score = 1  # no WISE data = neutral
        
        # --- Mismatch bonus (0–2) ---
        mismatch = 0
        if np.isfinite(w1w2) and w1w2 < 0.8:
            if w80 >= 400:
                mismatch = 2
            elif w80 >= 200:
                mismatch = 1
        
        total = kin_score + whan_score + mir_score + mismatch
        
        rows.append({
            'ID': gal_name,
            'W80': round(w80, 1),
            'W1_W2': round(w1w2, 3) if np.isfinite(w1w2) else np.nan,
            'Lit_AGN': False,
            'Milliquas': is_mq,
            'SIMBAD_AGN': is_simbad,
            'WHAN_AGN': is_whan,
            'DeepRadio': is_radio,
            'DeepXray': is_xray,
            'Kin_Score': kin_score,
            'WHAN_Score': whan_score,
            'MIR_Score': mir_score,
            'Mismatch': mismatch,
            'Fossil_Score': total,
        })
    
    df = pd.DataFrame(rows)
    return df.sort_values('Fossil_Score', ascending=False).reset_index(drop=True)


def load_catalogs():
    """Loads and merges the physical properties and H-alpha catalogs."""
    try:
        # Paths to your specific catalog files
        cat_phys_path = "k3d_fnlsp_table_v3.fits"
        cat_ha_path = "k3d_fnlsp_table_hafits_v3.fits"
        
        if not os.path.exists(cat_phys_path) or not os.path.exists(cat_ha_path):
            return None

        # Load Physical Properties
        with fits.open(cat_phys_path) as hdul:
            df_phys = pd.DataFrame(hdul[1].data)
            # Handle byte-swapping for Big Endian FITS data if needed
            for col in df_phys.select_dtypes(include=['>f4', '>f8']):
                df_phys[col] = df_phys[col].astype(float)
        
        # Load H-alpha Properties
        with fits.open(cat_ha_path) as hdul:
            df_ha = pd.DataFrame(hdul[1].data)
            for col in df_ha.select_dtypes(include=['>f4', '>f8']):
                df_ha[col] = df_ha[col].astype(float)

        # Decode ID columns if they are bytes
        if df_phys['ID'].dtype == 'O': 
            df_phys['ID'] = df_phys['ID'].apply(lambda x: x.decode('utf-8').strip() if isinstance(x, bytes) else str(x).strip())
        if df_ha['ID'].dtype == 'O':
            df_ha['ID'] = df_ha['ID'].apply(lambda x: x.decode('utf-8').strip() if isinstance(x, bytes) else str(x).strip())

        # Merge tables on ID
        df_merged = pd.merge(df_phys, df_ha[['ID', 'SIG', 'SIG_ERR', 'FLUX_HA']], on='ID', how='left')

        # --- PRE-CALCULATE LOGS FOR PLOTTING ---
        # Log SFR
        df_merged['Log SFR'] = np.where(df_merged['SFR'] > 0, np.log10(df_merged['SFR']), np.nan)
        # Log Mass (Already LMSTAR usually, but ensure numeric)
        df_merged['Log Mass'] = pd.to_numeric(df_merged['LMSTAR'], errors='coerce')
        # Rename Z for clarity
        df_merged.rename(columns={'Z': 'Redshift'}, inplace=True)
        
        return df_merged
    except Exception as e:
        print(f"Catalog Load Error: {e}")
        return None

@st.cache_data
def load_raw_cube(file_path):
    try:
        with fits.open(file_path) as hdul:
            if len(hdul) >= 2 and hdul[1].header['NAXIS'] == 3:
                flux_cube = hdul[1].data
                header_flux = dict(hdul[1].header)
                if len(hdul) >= 3: noise_cube = hdul[2].data
                else: noise_cube = np.ones_like(flux_cube)
                return flux_cube, noise_cube, header_flux
    except Exception: return None, None, None
    return None, None, None

# --- SPECTRAL ANALYSIS HELPER FUNCTIONS ---
@st.cache_data(show_spinner=False)
def extract_wavelength_calibration(header, n_spectral):
    try:
        crval = header.get('CRVAL3', 0)
        cdelt = header.get('CDELT3', 0)
        crpix = header.get('CRPIX3', 1)
        indices = np.arange(n_spectral)
        return crval + (indices - (crpix - 1)) * cdelt
    except Exception: return np.arange(n_spectral)

@st.cache_data(show_spinner=False)
def weighted_stack_spectrum(flux_cube, noise_cube, center_x, center_y, radius_px):
    ny, nx = flux_cube.shape[1], flux_cube.shape[2]
    y_grid, x_grid = np.indices((ny, nx))
    dist_grid = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
    mask_2d = dist_grid <= radius_px
    mask_3d = np.broadcast_to(mask_2d, flux_cube.shape)
    
    safe_noise = np.where(noise_cube <= 0, np.inf, noise_cube)
    weights = 1.0 / (safe_noise**2)
    weights[~mask_3d] = 0
    weights[np.isnan(flux_cube)] = 0
    weights[np.isnan(weights)] = 0
    
    numerator = np.nansum(flux_cube * weights, axis=(1, 2))
    sum_weights = np.nansum(weights, axis=(1, 2))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        stacked_flux = np.divide(numerator, sum_weights)
        stacked_noise = np.sqrt(np.divide(1.0, sum_weights))
    return np.nan_to_num(stacked_flux), np.nan_to_num(stacked_noise)

@st.cache_data(show_spinner="Applying Redshift Correction & Cleaning Maps...")
def calculate_best_fit_maps(results_file_path, fits_file_path):
    try:
        # --- 1. DATA LOADING ---
        with fits.open(results_file_path) as hdul:
            def gd(n): return hdul[n].data if n in hdul else np.zeros_like(hdul['A'].data)
            A, B, C = gd('A'), gd('B'), gd('C')
            A1, B1, C1 = gd('A1'), gd('B1'), gd('C1')
            A2, B2, C2 = gd('A2'), gd('B2'), gd('C2')
            D, E = gd('D'), gd('E')
            chi1, chi2 = gd('CHI1'), gd('CHI2')

        with fits.open(fits_file_path) as hdul_raw:
            hdr = hdul_raw[1].header
            z_gal = tools.redshift(hdul_raw)  # Critical: Use for Rest-Frame shift
            n_spec = hdul_raw[1].data.shape[0]
            # Construct Rest-Frame Wavelength Grid
            wave_obs = hdr['CRVAL3'] + (np.arange(n_spec) - (hdr['CRPIX3'] - 1)) * hdr['CDELT3']
            wave_rest = wave_obs / (1 + z_gal) 

        # --- 2. INITIALIZE ---
        ny, nx = A.shape
        map_f, map_v, map_w = np.full((ny, nx), np.nan), np.full((ny, nx), np.nan), np.full((ny, nx), np.nan)
        HA_R, C_L = 0.65628, 299792.458

        # --- 3. SPATIAL NOISE FILTER ---
        raw_mask = (A > 1e-10) | (A1 > 1e-10)
        labeled, num_features = label(raw_mask)
        for fid in range(1, num_features + 1):
            coords = np.where(labeled == fid)
            if len(coords[0]) < 5: raw_mask[coords] = False 

        valid_pixels = np.where(raw_mask)

        # --- 4. LOOP & CALCULATE ---
        for i, j in zip(*valid_pixels):
            use_double = abs(chi2[i,j] - 1) < abs(chi1[i,j] - 1)
            
            if not use_double:
                model = gaussian(wave_rest, A[i,j], B[i,j], C[i,j])
                map_f[i,j] = A[i,j]
                # Velocity: Rest-frame center vs Rest-frame H-alpha
                map_v[i,j] = ((C[i,j] - HA_R) / HA_R) * C_L
            else:
                m1 = gaussian(wave_rest, A1[i,j], B1[i,j], C1[i,j])
                m2 = gaussian(wave_rest, A2[i,j], B2[i,j], C2[i,j])
                model = m1 + m2
                map_f[i,j] = A1[i,j] + A2[i,j]
                # Flux-weighted velocity
                v_cent = (A1[i,j]*C1[i,j] + A2[i,j]*C2[i,j]) / (A1[i,j] + A2[i,j])
                map_v[i,j] = ((v_cent - HA_R) / HA_R) * C_L

            # W80: Calculated on the rest-frame wavelength grid
            csum = np.cumsum(model)
            if csum[-1] > 0:
                l_idx = np.searchsorted(csum, 0.1 * csum[-1])
                h_idx = np.searchsorted(csum, 0.9 * csum[-1])
                h_idx = min(h_idx, len(wave_rest)-1)
                width_microns = wave_rest[h_idx] - wave_rest[l_idx]
                map_w[i,j] = (width_microns / HA_R) * C_L

        return {"Best Flux": map_f, "Best Velocity": map_v, "W80": map_w}
    except Exception as e:
        return None
        

from scipy.ndimage import gaussian_filter

def plot_interactive_map(data, title, cmap='plasma', selected_point=None, key_id=None, blur_sigma=0):
    """
    Creates an interactive Plotly Heatmap with clickable pixels.
    Uses invisible scatter points overlay for reliable click detection.
    """
    ny, nx = data.shape
    fig = go.Figure()
    
    # 1. Handle Data, Outliers & Blurring
    data_plot = data.copy()
    valid_mask = np.isfinite(data_plot) & (np.abs(data_plot) < 1e30)
    data_plot[~valid_mask] = np.nan
    
    if blur_sigma > 0:
        fill_val = np.nanmedian(data_plot)
        filled = data_plot.copy()
        filled[np.isnan(filled)] = fill_val
        data_plot = gaussian_filter(filled, sigma=blur_sigma)
        data_plot[~valid_mask] = np.nan

    # 2. Color Scaling
    valid_pixels = data_plot[np.isfinite(data_plot)]
    if len(valid_pixels) > 0:
        zmin, zmax = np.percentile(valid_pixels, [1, 99])
        rng = zmax - zmin
        zmin -= 0.05 * rng
        zmax += 0.05 * rng
    else:
        zmin, zmax = None, None
    
    # 3. Heatmap (visual layer)
    fig.add_trace(go.Heatmap(
        z=data_plot, 
        colorscale=cmap, 
        showscale=True,
        zmin=zmin, zmax=zmax,
        colorbar=dict(thickness=8, len=0.8, tickfont=dict(color='white', size=9), outlinewidth=0),
        hoverinfo='skip',  # Disable hover on heatmap
    ))
    
    # 4. INVISIBLE CLICKABLE SCATTER POINTS (key for click detection!)
    # Create grid of points over valid pixels only
    if np.any(valid_mask):
        y_indices, x_indices = np.where(valid_mask)
        z_values = data_plot[valid_mask]
        
        fig.add_trace(go.Scatter(
            x=x_indices,
            y=y_indices,
            mode='markers',
            marker=dict(size=12, color='rgba(0,0,0,0)', line=dict(width=0)),  # Invisible
            customdata=np.column_stack([x_indices, y_indices]),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Value: %{text:.3f}<extra></extra>',
            text=z_values,
            showlegend=False,
        ))
    
    # 5. Selected Point Marker (green cross)
    if selected_point:
        px, py = selected_point
        fig.add_trace(go.Scatter(
            x=[px], y=[py], mode='markers',
            marker=dict(color='#00FF00', size=14, symbol='x', line=dict(width=3, color='#00FF00')),
            showlegend=False, hoverinfo='skip'
        ))

    # 6. Auto-Zoom
    if np.any(valid_mask):
        ys, xs = np.where(valid_mask)
        pad = 2
        x_min = max(xs.min() - pad, -0.5)
        x_max = min(xs.max() + pad, nx - 0.5)
        y_min = max(ys.min() - pad, -0.5)
        y_max = min(ys.max() + pad, ny - 0.5)
    else:
        x_min, x_max = -0.5, nx - 0.5
        y_min, y_max = -0.5, ny - 0.5

    # 7. Clean Layout (minimal UI)
    # uirevision preserves zoom/pan state across reruns when the key stays the same
    fig.update_layout(
        title=dict(text=title, font=dict(color='white', size=13)),
        paper_bgcolor='black', 
        plot_bgcolor='black',  
        height=320, 
        margin=dict(l=5, r=5, t=35, b=5),
        xaxis=dict(visible=False, range=[x_min, x_max], uirevision=key_id),
        yaxis=dict(visible=False, range=[y_min, y_max], scaleanchor="x", scaleratio=1, uirevision=key_id),
        dragmode=False,  # Disable drag - just click
        uirevision=key_id,  # Preserve zoom/pan state across reruns
    )

    # 8. Render - minimal config, no toolbar clutter
    selection = st.plotly_chart(
        fig, 
        width='stretch',
        config={'displayModeBar': False},
        on_select="rerun",
        selection_mode="points",
        key=key_id
    )
    return selection

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


def get_elliptical_points(x_center, y_center, maj_axis, min_axis, angle_deg):
    a = min_axis / 2.0 
    b = maj_axis / 2.0
    phi = np.radians(angle_deg + 90)
    t = np.linspace(0, 2*np.pi, 100)
    x = a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi) + x_center
    y = a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi) + y_center
    return x, y

# --- MAIN APP LOGIC ---
galaxies = load_galaxy_list(DATA_PATH)
if not galaxies:
    st.error(f"No data found in {DATA_PATH}. Please ensure folders exist.")
    st.stop()

fits_file_path = os.path.join(DATA_PATH, galaxies[0] + ".fits")
results_file_path = os.path.join(RESULTS_PATH, galaxies[0] + "_voronoi_binned.fits")

header = load_fits_header(fits_file_path)
flux_cube_raw, noise_cube_raw, header_flux_raw = load_raw_cube(fits_file_path)
# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("🌌 Navigation")

# 1. Load basic list
raw_galaxies = load_galaxy_list(DATA_PATH)

# 2. Calculate W80 stats and sort (This is cached, so fast after first run)
# Make sure 'RESULTS_PATH' is defined in your config (e.g. "Gaussian_fits")
sorted_galaxy_data = get_galaxies_sorted_by_w80(raw_galaxies, RESULTS_PATH)

# 3. Create display labels for the Selectbox
# Format: "GalaxyName (Value km/s)"
# Handle case where cache might have unexpected format
display_options = {}
for item in sorted_galaxy_data:
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        name, val = item[0], item[1]
        display_options[f"{name}"] = name
    elif isinstance(item, str):
        # Fallback if item is just a string (galaxy name)
        display_options[f"{item}"] = item
    else:
        # Skip invalid entries
        continue

# Fallback if display_options is empty
if not display_options:
    display_options = {name: name for name in raw_galaxies}
# 4. Render the Selectbox with Two-Way Sync
# A. Initialize the "Source of Truth" state variable if missing
if "current_selection" not in st.session_state:
    # Default to the first available galaxy
    st.session_state.current_selection = list(display_options.keys())[0]

# B. Callback to update state when User manually changes the Sidebar
def update_selection_from_sidebar():
    st.session_state.current_selection = st.session_state.galaxy_selector_widget

# C. Calculate the correct index to show based on current state
options_list = list(display_options.keys())
try:
    current_index = options_list.index(st.session_state.current_selection)
except ValueError:
    current_index = 0

# D. Render the widget
# Note: We changed the key to 'galaxy_selector_widget' to avoid conflicts
selected_label = st.sidebar.selectbox(
    "Select Galaxy (Sorted by Mean W80):", 
    options=options_list,
    index=current_index,
    key="galaxy_selector_widget", 
    on_change=update_selection_from_sidebar
)
header_info = load_fits_header(os.path.join(DATA_PATH, display_options[selected_label] + ".fits"))

selected_galaxy_name = display_options[selected_label]

# --- CLASSIFICATION LOGIC START ---

results_fits_path = os.path.join(RESULTS_PATH, selected_galaxy_name + "_voronoi_binned.fits")
ra_val, dec_val = None, None
simbad_class = "Not found"
simbad_url = None
simbad_main_id = None
galaxy_id_from_header = None
# WHAN classification variables
whan_class = None
ew_ha = None
nii_ha = None

if os.path.exists(results_fits_path):
    try:
        with fits.open(results_fits_path) as hdul:
            header = hdul[0].header
            
            # Get coordinates from header
            ra_val = header.get('RA', None)
            if ra_val == 'UNKNOWN':
                ra_val = None
            dec_val = header.get('DEC', None)
            if dec_val == 'UNKNOWN':
                dec_val = None
                
            # Get Galaxy ID from header
            galaxy_id_from_header = header.get('GALAXY_ID', None)
            
            # Get SIMBAD info from header
            simbad_class = header.get('SIMBAD_CL', 'Not found')
            simbad_main_id = header.get('SIMBAD_ID', None)
            
            # Try to get URL from HIERARCH keyword
            try:
                simbad_url = header.get('HIERARCH SIMBAD_URL', None)
            except:
                simbad_url = None
            
            # If no URL stored but we have SIMBAD_ID, construct the direct link
            if (simbad_url is None or simbad_url == 'N/A') and simbad_main_id and simbad_main_id != 'N/A':
                safe_id = quote(simbad_main_id)
                simbad_url = f"https://simbad.cds.unistra.fr/simbad/sim-id?Ident={safe_id}"
            
            # Get WHAN classification from header
            whan_class = header.get('WHAN_CLS', None)
            if whan_class:
                whan_class = whan_class.strip()
            ew_ha = header.get('EW_HA', None)
            nii_ha = header.get('NII_HA', None)
            
    except Exception as e:
        print(f"Error reading header: {e}")

# Fallback to catalog if header didn't have coords
if ra_val is None:
    df_cat_sidebar = load_catalogs()
    if df_cat_sidebar is not None:
        match_row = df_cat_sidebar[df_cat_sidebar['ID'] == selected_galaxy_name.strip()]
        if not match_row.empty:
            ra_val = match_row.iloc[0]['RA']
            dec_val = match_row.iloc[0]['DEC']

# Fallback to raw FITS header
if ra_val is None and header_info:
    ra_val = header_info.get("RA")
    dec_val = header_info.get("DEC")

# 2. Get Milliquas/AGN Status
agn_ids = load_agn_catalog_ids("AGN_SAMPLE.fits")
check_name = galaxy_id_from_header if galaxy_id_from_header else selected_galaxy_name
is_milliquas = is_agn_galaxy(check_name, agn_ids)

# 3. Build SIMBAD Markdown link
if simbad_class and simbad_class not in ['Not found', 'N/A', 'Invalid coordinates', 'No classification']:
    readable_class = SIMBAD_OTYPE_MAPPING.get(simbad_class, simbad_class)
    
    if simbad_url and simbad_url != 'N/A':
        # Direct link to the SIMBAD object page
        simbad_markdown = f"[{readable_class}]({simbad_url})"
    else:
        simbad_markdown = readable_class
else:
    # No classification found - provide search link as fallback
    if ra_val is not None and dec_val is not None:
        search_url = f"https://simbad.cds.unistra.fr/simbad/sim-coo?Coord={ra_val}+{dec_val}&CooFrame=ICRS&Radius=2&Radius.unit=arcsec"
        simbad_markdown = f"[{simbad_class}]({search_url})"
    else:
        simbad_markdown = "Not Found"

# 4. Analyze SIMBAD Text for AGN Keywords
simbad_clean = simbad_class if simbad_class else ""

agn_keywords = [
    "Active", "Seyfert", "Quasar", "Blazar", "BL Lac", 
    "LINER", "Radio Galaxy", "AGN", "Sy1", "Sy2", "Sy", 
    "QSO", "Bla", "BLLac", "rG", "Candidate", "AG?"
]


simbad_is_active = any(keyword.lower() in simbad_clean.lower() for keyword in agn_keywords)

# 4b. Analyze WHAN Classification for AGN (Seyfert = AGN in WHAN diagram)
# WHAN classes: SF (Star Forming), sAGN (Strong AGN/Seyfert), wAGN (Weak AGN), 
#               Retired, Passive, or combinations
whan_is_agn = False
if whan_class:
    whan_upper = whan_class.upper()
    # Seyfert/AGN indicators in WHAN: sAGN, wAGN, or any class containing "AGN" or "SEYFERT"
    whan_is_agn = 'SAGN' in whan_upper or 'WAGN' in whan_upper or 'AGN' in whan_upper or 'SEYFERT' in whan_upper

# 5. Determine Overall Status (now includes WHAN)
is_agn_overall = is_milliquas or simbad_is_active 
AGN_text = 'AGN DETECTED!' if is_agn_overall else 'No AGN signs.'
AGN_text = 'CANDIDATE' if not is_agn_overall and (whan_is_agn) else AGN_text

# 6. Build WHAN display string
if whan_class:
    whan_display = f"❌ {whan_class}" if whan_is_agn else f"✅ {whan_class}"
else:
    whan_display = "❓ No data"

# 7. Display in Sidebar
milliquas_str = "❌ AGN" if is_milliquas else "✅ No AGN"
simbad_str = f"❌ {simbad_markdown}" if simbad_is_active else f"✅  {simbad_markdown}" if simbad_markdown != "Not Found" else "❌ Not Found"
display_content = f"""
**{AGN_text}**  
**Milliquas:**   {milliquas_str}  
**SIMBAD:**   {simbad_str}  
**WHAN:**   {whan_display}
"""

if is_agn_overall:
    st.sidebar.error(display_content, icon="🚨")
elif whan_is_agn:
    st.sidebar.warning(display_content, icon="⚠️")
else:
    st.sidebar.success(display_content, icon="⭐")

# --- CLASSIFICATION LOGIC END ---

# --- CATALOG LINKS BOX ---
crossmatch_df = load_crossmatch_links()
catalog_links = get_catalog_links_for_galaxy(
    galaxy_name=selected_galaxy_name,
    ra=ra_val,
    dec=dec_val,
    crossmatch_df=crossmatch_df,
)

if catalog_links:
    # Build HTML for the catalog links box
    matched_lines = []
    unmatched_lines = []
    
    for survey, info in catalog_links.items():
        icon = info['icon']
        name = info['display_name']
        url = info['url']
        matched = info['matched']
        
        if matched and url:
            matched_lines.append(
                f'<a href="{url}" target="_blank" '
                f'style="color:#58A6FF; text-decoration:none; display:block; '
                f'padding:3px 0; font-size:0.9em;">'
                f'{icon} {name} ↗</a>'
            )
        elif matched:
            matched_lines.append(
                f'<span style="color:#7EE787; display:block; padding:3px 0; '
                f'font-size:0.9em;">{icon} {name} ✓</span>'
            )
        else:
            unmatched_lines.append(name)
    
    n_matched = len(matched_lines)
    n_total = len(catalog_links)
    
    html_content = (
        f'<div style="background-color:#1a1a2e; border:1px solid #334155; '
        f'border-radius:8px; padding:12px; margin:8px 0;">'
        f'<div style="color:#E2E8F0; font-weight:600; font-size:0.95em; '
        f'margin-bottom:8px; border-bottom:1px solid #334155; padding-bottom:6px;">'
        f'📡 Catalog Crossmatch ({n_matched}/{n_total})</div>'
    )
    
    html_content += ''.join(matched_lines)
    
    if unmatched_lines:
        html_content += (
            f'<div style="color:#6B7280; font-size:0.8em; margin-top:6px; '
            f'padding-top:6px; border-top:1px solid #262630;">'
            f'No match: {", ".join(unmatched_lines)}</div>'
        )
    
    html_content += '</div>'
    st.sidebar.markdown(html_content, unsafe_allow_html=True)
else:
    st.sidebar.markdown(
        '<div style="background-color:#1a1a2e; border:1px solid #334155; '
        'border-radius:8px; padding:12px; margin:8px 0; color:#6B7280; '
        'font-size:0.9em;">'
        '📡 No crossmatch data available.<br>'
        '<span style="font-size:0.8em;">Run Crossmatch_generator.py first.</span>'
        '</div>',
        unsafe_allow_html=True
    )


selected_val = next((val for name, val in sorted_galaxy_data if name == display_options[selected_label]), None)
st.sidebar.markdown(f"**Mean W80:** {selected_val:.2f} km/s")

# Optionally show more info (e.g. number of galaxies, selected index)
st.sidebar.markdown(f"**Total Galaxies:** {len(sorted_galaxy_data)}")
st.sidebar.markdown(f"**Selected Index:** {list(display_options.keys()).index(selected_label)+1}")

# Print more information from the header of the fits-file
if header_info:
    st.sidebar.markdown("**Header Info:**")
    for key in ["OBJECT", "RA", "DEC", "Z", "INSTRUME", "DATE-OBS"]:
        if key in header_info:
            st.sidebar.markdown(f"- **{key}**: {header_info[key]}")

# UPDATE FILE PATHS FOR SELECTION
fits_file_path = os.path.join(DATA_PATH, selected_galaxy_name + ".fits")
results_file_path = os.path.join(RESULTS_PATH, selected_galaxy_name + "_voronoi_binned.fits")

# --- CRITICAL FIX START: RELOAD DATA FOR SELECTED GALAXY ---
header = load_fits_header(fits_file_path)
flux_cube_raw, noise_cube_raw, header_flux_raw = load_raw_cube(fits_file_path)

if flux_cube_raw is None:
    st.error(f"Could not load data for {selected_galaxy_name}")
    st.stop()
# --- CRITICAL FIX END ---

with fits.open(fits_file_path) as hdul:
    psf = tools.psf(hdul)


# --- MAIN CONTENT ---
# 1. Header with Toggle Button
title_col, toggle_col = st.columns([0.9, 0.1])
with title_col:
    st.title(f"Objekt: {selected_galaxy_name}")

# 2. State Management for Right Sidebar
if "show_tools" not in st.session_state:
    st.session_state.show_tools = True

def toggle_sidebar():
    st.session_state.show_tools = not st.session_state.show_tools

# 3. Dynamic Layout Creation
if st.session_state.show_tools:
    # Sidebar is OPEN: 75% Main, 25% Right Sidebar
    col_main, col_right = st.columns([3, 1], gap="medium")
else:
    # Sidebar is CLOSED: 100% Main
    col_main = st.container()
    col_right = None

    # Show "Open" button in the top right corner when closed
    with toggle_col:
        st.button("⏪ Tools", on_click=toggle_sidebar, help="Open Analysis Sidebar")

# --- LEFT COLUMN (MAIN CONTENT) ---
with col_main:
    # Your existing Tabs logic
    view_mode = st.radio(
    "Navigation", 
    ["🖼️ Flux Preview", "🗺️ Kinematic Maps", "📈 Correlations"], 
    horizontal=True, 
    label_visibility="collapsed"
)
    if view_mode == "🖼️ Flux Preview":
        if flux_cube_raw is not None:
            # 1. Load Data Context & Cosmology
            from astropy.cosmology import FlatLambdaCDM
            
            with fits.open(fits_file_path) as hdu:  
                z_val = tools.redshift(hdu)
                psf_maj, psf_min, psf_pa = extract_psf_parameters(hdu)

            # --- COSMOLOGY & SCALE CALCULATION ---
            # Standard KMOS pixel scale is 0.2 arcsec.
            pixel_scale = 0.2 
            
            if z_val > 0:
                cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
                kpc_per_arcsec = 1.0 / cosmo.arcsec_per_kpc_proper(z_val).value
                kpc_per_pixel = kpc_per_arcsec * pixel_scale
            else:
                kpc_per_pixel = None 

            # 2. Top Control Bar
            # Initialize control states in session_state if not present
            display_mode_key = f"display_mode_{selected_galaxy_name}"
            radius_key = f"radius_{selected_galaxy_name}"
            rest_frame_key = f"rest_frame_{selected_galaxy_name}"
            psf_key = f"psf_{selected_galaxy_name}"
            smooth_key = f"smooth_{selected_galaxy_name}"
            
            if display_mode_key not in st.session_state:
                st.session_state[display_mode_key] = "Flux"
            if radius_key not in st.session_state:
                st.session_state[radius_key] = 3.5
            if rest_frame_key not in st.session_state:
                st.session_state[rest_frame_key] = True
            if psf_key not in st.session_state:
                st.session_state[psf_key] = False
            if smooth_key not in st.session_state:
                st.session_state[smooth_key] = True
            
            c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4 = st.columns([1, 1, 0.8, 0.8])
            
            with c_ctrl1:
                display_mode = st.radio("Display Mode:", ["Flux", "Signal-to-Noise"], horizontal=True, label_visibility="collapsed", key=display_mode_key)
            with c_ctrl2:
                radius = st.slider("Aperture Radius (px)", 1.0, 10.0, step=0.1, key=radius_key)
            with c_ctrl3:
                show_rest_frame = st.toggle("Rest-Frame Spectrum", key=rest_frame_key)
            with c_ctrl4:
                use_psf = st.toggle("PSF Weighted Extraction", key=psf_key)
                smooth_view = st.toggle("Smooth Map", key=smooth_key)

            st.divider()

            # 3. Main Analysis Columns
            c_img, c_spec = st.columns([1, 2])
            
            # --- LEFT COLUMN: IMAGE ---
            # --- LEFT COLUMN: IMAGE ---
            with c_img:
                # 1. REVERTED TO MEDIAN (Cleaner for Visualization)
                # Summing the whole cube (white light) adds too much noise.
                # Median filters noise and shows the structure better.
                white_light = np.nanmedian(flux_cube_raw, axis=0)
                
                if display_mode == "Signal-to-Noise":
                    # Median S/N proxy (Signal / Noise per pixel)
                    noise_map = np.nanmedian(noise_cube_raw, axis=0)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        map_data = np.divide(white_light, noise_map, 
                                           out=np.zeros_like(white_light), 
                                           where=noise_map!=0)
                    cmap = 'Viridis'
                    unit_label = "Avg S/N (per px)"
                else:
                    # Flux Map
                    map_data = white_light
                    cmap = 'Magma'
                    unit_label = "Median Flux (10⁻¹⁷ W/m²/µm)"

                # Initialize Session State Keys for Coordinates
                flux_coord_key_x = f"flux_click_x_{selected_galaxy_name}"
                flux_coord_key_y = f"flux_click_y_{selected_galaxy_name}"
                
                # Initialize with center if not exists
                if flux_coord_key_x not in st.session_state:
                    st.session_state[flux_coord_key_x] = float(map_data.shape[1] // 2)
                if flux_coord_key_y not in st.session_state:
                    st.session_state[flux_coord_key_y] = float(map_data.shape[0] // 2)
                
                # Get current coordinates for drawing
                current_x = st.session_state[flux_coord_key_x]
                current_y = st.session_state[flux_coord_key_y]

                # Build Plotly Map
                fig_map = go.Figure()
                ny, nx = map_data.shape
                
                # Heatmap (visual only)
                fig_map.add_trace(go.Heatmap(
                    z=map_data,  
                    colorscale=cmap, 
                    showscale=False, 
                    zsmooth='best' if smooth_view else False,
                    hoverinfo='skip',
                ))
                
                # INVISIBLE CLICKABLE POINTS for click detection
                valid_mask = np.isfinite(map_data)
                if np.any(valid_mask):
                    y_idx, x_idx = np.where(valid_mask)
                    z_vals = map_data[valid_mask]
                    fig_map.add_trace(go.Scatter(
                        x=x_idx, y=y_idx, mode='markers',
                        marker=dict(size=10, color='rgba(0,0,0,0)'),
                        text=z_vals,
                        hovertemplate=f'X: %{{x}}<br>Y: %{{y}}<br>{unit_label}: %{{text:.2f}}<extra></extra>',
                        showlegend=False,
                    ))
                
                # Visual Guides: Aperture Circle
                fig_map.add_shape(type="circle",
                    xref="x", yref="y",
                    x0=current_x - radius, y0=current_y - radius,
                    x1=current_x + radius, y1=current_y + radius,
                    line_color="#00FF00", line_width=2
                )
                
                # Visual Guides: Center Crosshair
                fig_map.add_trace(go.Scatter(
                    x=[current_x], 
                    y=[current_y],
                    mode='markers',
                    marker=dict(symbol='cross', size=10, color='#00FF00'),
                    hoverinfo='skip', showlegend=False
                ))
                
                # Visual Guides: PSF Ellipse
                elli_color = 'rgba(0, 255, 0, 0.9)' if use_psf else 'rgba(255, 255, 0, 0.6)'
                elli_width = 2 if use_psf else 1
                elli_x, elli_y = get_elliptical_points(
                    current_x, current_y, 
                    psf_maj, psf_min, psf_pa
                )
                fig_map.add_trace(go.Scatter(
                    x=elli_x, y=elli_y, 
                    mode='lines', 
                    line=dict(color=elli_color, width=elli_width, dash='dot'), 
                    hoverinfo='skip', showlegend=False
                ))

                # --- ADD SCALE BAR (Maßstab) ---
                if kpc_per_pixel:
                    target_kpc = 5.0
                    bar_len_px = target_kpc / kpc_per_pixel
                    
                    if bar_len_px > map_data.shape[1] / 3:
                        target_kpc = 1.0
                        bar_len_px = target_kpc / kpc_per_pixel
                    
                    x_start = map_data.shape[1] - bar_len_px - 2
                    x_end = map_data.shape[1] - 2
                    y_pos = 2 

                    fig_map.add_shape(type="line",
                        x0=x_start, y0=y_pos, x1=x_end, y1=y_pos,
                        line=dict(color="white", width=4)
                    )
                    fig_map.add_annotation(
                        x=x_start + bar_len_px/2, y=y_pos + 1.5,
                        text=f"<b>{target_kpc:.0f} kpc</b>",
                        showarrow=False,
                        font=dict(color="white", size=12, shadow="1px 1px 2px black")
                    )

                # Layout (clean, minimal)
                # uirevision preserves zoom/pan state across reruns
                flux_map_key = f"flux_map_{selected_galaxy_name}"
                fig_map.update_layout(
                    width=350, height=350,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor='black', plot_bgcolor='black',
                    xaxis=dict(visible=False, uirevision=flux_map_key),
                    yaxis=dict(visible=False, scaleanchor="x", scaleratio=1, uirevision=flux_map_key),
                    dragmode=False,
                    showlegend=False,
                    uirevision=flux_map_key,  # Preserve zoom/pan state across reruns
                )

                # Render Interactive Chart
                flux_selection = st.plotly_chart(
                    fig_map, 
                    width='stretch',
                    config={'displayModeBar': False}, 
                    on_select="rerun",
                    selection_mode="points",
                    key=flux_map_key
                )

                # Precise Coordinate Input - these ARE the source of truth
                # Number inputs use session_state keys directly
                st.markdown("**Selected Coordinates:**")
                # --- 1. DATA LOADING HELPER FUNCTIONS ---

                # Handle Click Logic - Update the number input session state directly
                if flux_selection and "selection" in flux_selection:
                    pts = flux_selection.get("selection", {}).get("points", [])
                    if pts:
                        click_x = pts[0].get("x")
                        click_y = pts[0].get("y")
                        if click_x is not None and click_y is not None:
                            click_x = float(int(round(click_x)))
                            click_y = float(int(round(click_y)))
                            # Only update and rerun if coordinates actually changed
                            if click_x != st.session_state[flux_coord_key_x] or click_y != st.session_state[flux_coord_key_y]:
                                st.session_state[flux_coord_key_x] = click_x
                                st.session_state[flux_coord_key_y] = click_y
                                st.rerun()
                
                # Read current coordinates from session state (after potential click update)
                current_x = st.session_state[flux_coord_key_x]
                current_y = st.session_state[flux_coord_key_y]

            # --- RIGHT COLUMN: SPECTRUM ---
            with c_spec:
                wave_obs = extract_wavelength_calibration(header_flux_raw, flux_cube_raw.shape[0])
                
                # --- SPECTRUM EXTRACTION LOGIC ---
                if use_psf:
                    # 1. PSF-WEIGHTED EXTRACTION (Optimal Extraction)
                    ny, nx = flux_cube_raw.shape[1], flux_cube_raw.shape[2]
                    y_grid, x_grid = np.indices((ny, nx))
                    
                    sigma_maj = psf_maj / 2.35482
                    sigma_min = psf_min / 2.35482
                    theta = np.deg2rad(psf_pa + 90)
                    
                    dx = x_grid - current_x
                    dy = y_grid - current_y
                    x_rot = dx * np.cos(theta) + dy * np.sin(theta)
                    y_rot = -dx * np.sin(theta) + dy * np.cos(theta)
                    
                    psf_profile = np.exp(-0.5 * ((x_rot / sigma_min)**2 + (y_rot / sigma_maj)**2))
                    
                    # FIX 2: Removed "psf_profile[dist_grid > radius] = 0" 
                    # The PSF Extraction is now independent of the Aperture Circle radius.
                    
                    psf_sum = np.sum(psf_profile)
                    if psf_sum > 0: psf_profile /= psf_sum
                    
                    P = np.broadcast_to(psf_profile, flux_cube_raw.shape)
                    safe_noise = np.where(noise_cube_raw <= 0, np.inf, noise_cube_raw)
                    inv_var = 1.0 / (safe_noise**2)
                    
                    num = np.nansum(flux_cube_raw * P * inv_var, axis=(1,2))
                    denom = np.nansum((P**2) * inv_var, axis=(1,2))
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        spec_flux = np.divide(num, denom)
                        spec_noise = np.divide(1.0, np.sqrt(denom))
                    spec_flux = np.nan_to_num(spec_flux)
                    spec_noise = np.nan_to_num(spec_noise)
                    
                else:
                    # 2. STANDARD APERTURE MEAN
                    spec_flux, spec_noise = summed_aperture_spectrum(
                        flux_cube_raw, noise_cube_raw, 
                        current_x, current_y, radius
                    )

                # --- PLOTTING ---
                if show_rest_frame:
                    wave_plot = wave_obs / (1 + z_val)
                    x_axis_label = "Rest-Frame Wavelength (µm)"
                    line_factor = 1.0 
                else:
                    wave_plot = wave_obs
                    x_axis_label = "Observed Wavelength (µm)"
                    line_factor = (1 + z_val)

                fig_spec = go.Figure()
                
                y_label_text = "Flux (10⁻¹⁷ W/m²/µm)" if display_mode == "Flux" else "Signal-to-Noise"

                # 1. Flux Trace
                if display_mode == "Flux":
                    y_data = spec_flux 
                    fig_spec.add_trace(go.Scatter(
                        x=wave_plot, y=spec_flux, 
                        mode='lines', 
                        line=dict(color='#00CCFF', width=1.5), 
                        name='Flux',
                        hovertemplate='%{y:.2f} 10⁻¹⁷ W/m²/µm<extra></extra>'
                    ))
                    fig_spec.add_trace(go.Scatter(
                        x=np.concatenate([wave_plot, wave_plot[::-1]]),
                        y=np.concatenate([spec_flux + spec_noise, (spec_flux - spec_noise)[::-1]]),
                        fill='toself', 
                        fillcolor='rgba(128,128,128,0.2)', 
                        line=dict(color='rgba(0,0,0,0)'),
                        name='Noise (±1σ)',
                        hoverinfo='skip'
                    ))
                
                # 2. SNR Trace
                elif display_mode == "Signal-to-Noise":
                    with np.errstate(divide='ignore', invalid='ignore'):
                        snr = np.divide(spec_flux, spec_noise, out=np.zeros_like(spec_flux), where=spec_noise!=0)
                    y_data = snr
                    fig_spec.add_trace(go.Scatter(
                        x=wave_plot, y=snr, 
                        mode='lines', 
                        line=dict(color='#FFAA00', width=1.5), 
                        name='S/N Ratio',
                        hovertemplate='S/N: %{y:.2f}<extra></extra>'
                    ))

                # 3. Scaling
                valid_y = y_data[np.isfinite(y_data)]
                if len(valid_y) > 0:
                    y_min, y_max = np.min(valid_y), np.max(valid_y)
                    y_range_pixels = y_max - y_min
                    if y_range_pixels == 0: y_range_pixels = 1.0
                    y_range = [y_min - 0.1 * y_range_pixels, y_max + 0.1 * y_range_pixels]
                else:
                    y_range = None

                # 4. Lines
                lines = {'[NII]6548': 0.65480, 'Hα': 0.65628, '[NII]6583': 0.65834}
                for name, rest_lam in lines.items():
                    plot_lam = rest_lam * line_factor
                    if np.min(wave_plot) <= plot_lam <= np.max(wave_plot):
                        fig_spec.add_vline(x=plot_lam, line_width=1, line_dash="dash", line_color="rgba(0, 255, 0, 0.5)")
                        fig_spec.add_annotation(
                            x=plot_lam, y=1.02, 
                            xref="x", yref="paper", 
                            text=name, showarrow=False, 
                            font=dict(color="#00FF00", size=10),
                            yshift=5
                        )

                # --- DEFAULT ZOOM ON Hα AND [NII] LINES ---
                # Rest-frame wavelengths: [NII]6548=0.65480, Hα=0.65628, [NII]6583=0.65834
                # Default zoom range: ~0.648 - 0.665 µm (rest-frame), adjusted for redshift if needed
                ha_center_rest = 0.65628  # Hα rest wavelength in µm
                zoom_half_width = 0.008   # ±8nm around center (covers [NII] doublet + Hα)
                
                if show_rest_frame:
                    # Rest-frame: use rest wavelengths directly
                    default_x_range = [ha_center_rest - zoom_half_width, ha_center_rest + zoom_half_width]
                else:
                    # Observed frame: shift by (1 + z)
                    default_x_range = [(ha_center_rest - zoom_half_width) * (1 + z_val), 
                                       (ha_center_rest + zoom_half_width) * (1 + z_val)]

                fig_spec.update_layout(
                    height=450,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor='#0F172A', 
                    plot_bgcolor='#1E293B',
                    title=dict(text="Integrated Spectrum" + (" (PSF Weighted)" if use_psf else ""), font=dict(size=14, color='white')),
                    xaxis=dict(
                        title=x_axis_label, 
                        showgrid=True, gridcolor='#334155', 
                        tickfont=dict(color='#F1F5F9'),
                        zeroline=False,
                        range=default_x_range,  # Default zoom on Hα and [NII] lines
                        uirevision=f"spec_{selected_galaxy_name}_{show_rest_frame}"
                    ),
                    yaxis=dict(
                        title=y_label_text, 
                        showgrid=True, gridcolor='#334155', 
                        tickfont=dict(color='#F1F5F9'),
                        range=y_range,
                        uirevision=f"spec_{selected_galaxy_name}_{show_rest_frame}"
                    ),
                    legend=dict(
                        orientation="h", y=1.0, x=1.0, 
                        xanchor='right', yanchor='bottom',
                        font=dict(color='#F1F5F9')
                    ),
                    hovermode="x unified",
                    uirevision=f"spec_{selected_galaxy_name}_{show_rest_frame}",  # Preserve zoom/pan state
                )

                st.plotly_chart(fig_spec, width='stretch')

        else:
            st.error("Could not load flux data. Please check file paths.")

# --- TAB 2: KINEMATIC MAPS ---
    elif view_mode == "🗺️ Kinematic Maps":
        results_file_path = os.path.join(RESULTS_PATH, selected_galaxy_name + "_voronoi_binned.fits")

        if os.path.exists(results_file_path):
            try:
                # --- NEW STATE SYNC LOGIC ---
                # 1. Define distinct keys for the "Internal Value" vs the "Widget ID"
                base_key = f"kin_{selected_galaxy_name}"
                internal_x_key = f"{base_key}_val_x"
                internal_y_key = f"{base_key}_val_y"
                widget_x_key = f"{base_key}_widget_x"
                widget_y_key = f"{base_key}_widget_y"

                # 2. Initialize Internal Values if they don't exist
                if internal_x_key not in st.session_state:
                    mid_y, mid_x = 0, 0
                    if flux_cube_raw is not None:
                        mid_y, mid_x = flux_cube_raw.shape[1] // 2, flux_cube_raw.shape[2] // 2
                    st.session_state[internal_x_key] = float(mid_x)
                    st.session_state[internal_y_key] = float(mid_y)

                # 3. Define Callback to sync Widget -> Internal Value
                def sync_widget_to_internal():
                    st.session_state[internal_x_key] = st.session_state[widget_x_key]
                    st.session_state[internal_y_key] = st.session_state[widget_y_key]

                z_gal = 0.0
                try:
                    with fits.open(fits_file_path) as hdu: z_gal = tools.redshift(hdu)
                except: z_gal = header.get('Z', 0.0) if header else 0.0

                # 2. Load Results File
                with fits.open(results_file_path) as hdul_res:
                    map_extensions = [ext.name for ext in hdul_res if ext.header.get('NAXIS') == 2 and ext.name not in ['PRIMARY', '']]
                    
                    if map_extensions:
                        # --- CONTROLS ROW ---
                        # Initialize toggle states in session_state if not present
                        blur_key = f"blur_{selected_galaxy_name}"
                        comp_key = f"comp_{selected_galaxy_name}"
                        over_key = f"over_{selected_galaxy_name}"
                        
                        if blur_key not in st.session_state:
                            st.session_state[blur_key] = False
                        if comp_key not in st.session_state:
                            st.session_state[comp_key] = False
                        if over_key not in st.session_state:
                            st.session_state[over_key] = True
                        
                        c_info, c_blur, c_comp, c_over = st.columns([1.5, 0.8, 0.8, 0.9])
                        with c_info:
                            st.caption(f"📁 Source: `{os.path.basename(results_file_path)}` | z = {z_gal:.4f}")
                        with c_blur:
                            use_blur = st.toggle("Blur Maps", key=blur_key)
                            sigma_val = 1.0 if use_blur else 0.0
                        with c_comp:
                            show_components = st.toggle("Show Components", key=comp_key)
                        with c_over:
                            show_overlays = st.toggle("Mini Gaussians", key=over_key)

                        # Calculate Composite Maps for UI
                        best_maps = calculate_best_fit_maps(results_file_path, fits_file_path)

                        def get_label(ext):
                            label_map = {'A': 'Flux', 'B': 'Broadening', 'C': 'Velocity', 'D': 'Slope', 'E': 'Intercept'}
                            match = re.match(r"([A-Z]+)(.*)", ext)
                            if match:
                                base, suf = match.groups()
                                suf = suf.replace('_', ' ')
                                if base in label_map:
                                    n = label_map[base]
                                    if base in ['B', 'C']: n += " (km/s)"
                                    return n + suf if suf else n
                            return ext
                        
                        std_options = {get_label(ext): ext for ext in map_extensions}
                        std_options["⭐ Best Flux"] = "COMP_FLUX"
                        std_options["⭐ Best Velocity"] = "COMP_VEL"
                        std_options["⭐ W80"] = "COMP_W80"

                        HA_REST, C_LIGHT = 0.65628, 299792.458
                        def get_d(k):
                            """
                            Retrieves and masks data consistently based on the presence of signal (Flux).
                            """
                            if best_maps is None:
                                return np.zeros_like(hdul_res['A'].data)

                            # Use Flux as the indicator of valid fit area
                            flux_ref = best_maps["Best Flux"]
                            master_mask = np.isnan(flux_ref) | (flux_ref <= 1e-12)

                            if k == "COMP_FLUX":
                                data = flux_ref.copy()
                            elif k == "COMP_VEL":
                                data = best_maps["Best Velocity"].copy()
                            elif k == "COMP_W80":
                                data = best_maps["W80"].copy()
                            else:
                                # Standardize unit conversion for raw extension parameters
                                data = hdul_res[k].data.copy().astype(float) if k in hdul_res else np.zeros_like(flux_ref)
                                if k.startswith('B'): 
                                    data = (data / 0.65628) * 299792.458 # Sigma km/s
                                elif k.startswith('C') and not k.startswith('CHI'):
                                    data = ((data - 0.65628) / 0.65628) * 299792.458 # Center km/s

                            data[master_mask] = np.nan
                            return data
                        
                        # --- MAP GRID ---
                        m1, m2, m3 = st.columns(3)
                        cx = st.session_state[internal_x_key]
                        cy = st.session_state[internal_y_key]

                        with m1:
                            l1 = st.selectbox("Map 1", list(std_options.keys()), index=list(std_options.keys()).index("⭐ Best Flux") if "⭐ Best Flux" in std_options else 0, key=f"s1_{selected_galaxy_name}")
                            d1 = get_d(std_options[l1])
                            sel1 = plot_interactive_map(d1, l1, 'Plasma', (cx, cy), f"p1_{l1}", blur_sigma=sigma_val)

                        with m2:
                            l2 = st.selectbox("Map 2", list(std_options.keys()), index=list(std_options.keys()).index("⭐ W80") if "⭐ W80" in std_options else 0, key=f"s2_{selected_galaxy_name}")
                            d2 = get_d(std_options[l2])
                            
                            if l2 == "⭐ W80" and show_overlays:
                                ny, nx = d2.shape
                                fig_w80 = go.Figure()
                                
                                # 1. Zoom & Scaling
                                valid_mask = np.isfinite(d2) & (np.abs(d2) < 1e30) 
                                if np.any(valid_mask):
                                    ys, xs = np.where(valid_mask)
                                    pad = 2 
                                    x_range = [max(xs.min() - pad, -0.5), min(xs.max() + pad, nx - 0.5)]
                                    y_range = [max(ys.min() - pad, -0.5), min(ys.max() + pad, ny - 0.5)]
                                    zmin, zmax = np.nanpercentile(d2[valid_mask], [1, 99])
                                else:
                                    x_range, y_range = [-0.5, nx - 0.5], [-0.5, ny - 0.5]
                                    zmin, zmax = None, None

                                # 2. Heatmap (visual)
                                fig_w80.add_trace(go.Heatmap(
                                    z=d2, colorscale='Oranges', zmin=zmin, zmax=zmax, showscale=True,
                                    colorbar=dict(thickness=8, len=0.8, tickfont=dict(color='white', size=9), outlinewidth=0),
                                    hoverinfo='skip'
                                ))
                                
                                # 3. INVISIBLE CLICKABLE POINTS
                                if np.any(valid_mask):
                                    y_idx, x_idx = np.where(valid_mask)
                                    z_vals = d2[valid_mask]
                                    fig_w80.add_trace(go.Scatter(
                                        x=x_idx, y=y_idx, mode='markers',
                                        marker=dict(size=10, color='rgba(0,0,0,0)'),
                                        text=z_vals,
                                        hovertemplate='X: %{x}<br>Y: %{y}<br>W80: %{text:.1f} km/s<extra></extra>',
                                        showlegend=False,
                                    ))

                                # 4. Mini Gaussian Overlays
                                valid_indices = np.argwhere(valid_mask & (d2 > 0))
                                if valid_indices.size > 0:
                                    max_draw = 1000 
                                    step = max(1, len(valid_indices) // max_draw)
                                    A_s, B_s, C_s = hdul_res[1].data, hdul_res[4].data, hdul_res[7].data
                                    chi1_map, chi2_map = hdul_res[12].data, hdul_res[13].data
                                    A1, B1, C1 = hdul_res[2].data, hdul_res[5].data, hdul_res[8].data
                                    A2, B2, C2 = hdul_res[3].data, hdul_res[6].data, hdul_res[9].data

                                    for k in range(0, len(valid_indices), step):
                                        i, j = valid_indices[k]
                                        use_single = abs(1 - chi1_map[i, j]) < abs(1 - chi2_map[i, j])
                                        if use_single:
                                            amp, sig, off = A_s[i,j], B_s[i,j], C_s[i,j]
                                            if not np.isfinite([amp, sig, off]).all(): continue
                                            x_wave = np.linspace(off - sig*3, off + sig*3, 25)
                                            y_vals = np.abs(amp) * np.exp(-0.5 * ((x_wave - off) / (sig + 1e-10))**2)
                                        else:
                                            amp1, sig1, off1 = A1[i,j], B1[i,j], C1[i,j]
                                            amp2, sig2, off2 = A2[i,j], B2[i,j], C2[i,j]
                                            if not np.isfinite([amp1, sig1, off1, amp2, sig2, off2]).all(): continue
                                            dom_sig = max(sig1, sig2)
                                            dom_off = off1 if sig1 > sig2 else off2
                                            x_wave = np.linspace(dom_off - dom_sig*3, dom_off + dom_sig*3, 25)
                                            y_vals = (np.abs(amp1) * np.exp(-0.5 * ((x_wave - off1) / (sig1 + 1e-10))**2) + 
                                                    np.abs(amp2) * np.exp(-0.5 * ((x_wave - off2) / (sig2 + 1e-10))**2))
                                        if np.nanmax(y_vals) > 0:
                                            y_norm = (y_vals / np.nanmax(y_vals)) * 0.6
                                            fig_w80.add_trace(go.Scatter(
                                                x=np.linspace(-0.4, 0.4, len(x_wave)) + j,
                                                y=y_norm + i - 0.3,
                                                mode='lines', line=dict(color='black', width=0.5),
                                                hoverinfo='skip', showlegend=False
                                            ))

                                # 5. Green Crosshair
                                fig_w80.add_trace(go.Scatter(
                                    x=[cx], y=[cy], mode='markers',
                                    marker=dict(color='#00FF00', size=14, symbol='x', line=dict(width=3, color='#00FF00')),
                                    showlegend=False, hoverinfo='skip'
                                ))

                                # 6. Layout (clean)
                                # uirevision preserves zoom/pan state across reruns
                                w80_map_key = f"p2_w80_fixed_{selected_galaxy_name}"
                                fig_w80.update_layout(
                                    height=320, paper_bgcolor='black', plot_bgcolor='black',
                                    margin=dict(l=5, r=5, t=35, b=5), 
                                    title=dict(text=l2, font=dict(color='white', size=13)),
                                    xaxis=dict(visible=False, range=x_range, uirevision=w80_map_key),
                                    yaxis=dict(visible=False, range=y_range, scaleanchor="x", scaleratio=1, uirevision=w80_map_key),
                                    dragmode=False,
                                    uirevision=w80_map_key,  # Preserve zoom/pan state across reruns
                                )
                                
                                sel2 = st.plotly_chart(fig_w80, width='stretch', 
                                    config={'displayModeBar': False}, on_select="rerun", 
                                    selection_mode="points", key=w80_map_key)
                            else:
                                sel2 = plot_interactive_map(d2, l2, 'Plasma', (cx, cy), f"p2_{l2}", blur_sigma=sigma_val)

                        with m3:
                            l3 = st.selectbox("Map 3", list(std_options.keys()), index=list(std_options.keys()).index("⭐ Best Velocity") if "⭐ Best Velocity" in std_options else 0, key=f"s3_{selected_galaxy_name}")
                            k3 = std_options[l3]
                            d3 = get_d(k3)
                            cm3 = 'Spectral_r' if any(x in l3.upper() for x in ['VELOCITY', 'VEL', 'C']) else 'Plasma'
                            sel3 = plot_interactive_map(d3, l3, cm3, (cx, cy), f"p3_{l3}", blur_sigma=sigma_val)

                        # --- COORDINATE INPUTS - These are the source of truth ---
                        c_in1, c_in2 = st.columns(2)
                        max_x = float(flux_cube_raw.shape[2] - 1) if flux_cube_raw is not None else 100.0
                        max_y = float(flux_cube_raw.shape[1] - 1) if flux_cube_raw is not None else 100.0

                        with c_in1: 
                            # Use 'value' to set the state, but a different 'key' for the widget
                            st.number_input("X", 0.0, max_x, step=1.0, 
                                            value=float(st.session_state[internal_x_key]),
                                            key=widget_x_key, 
                                            on_change=sync_widget_to_internal)
                        with c_in2: 
                            st.number_input("Y", 0.0, max_y, step=1.0, 
                                            value=float(st.session_state[internal_y_key]),
                                            key=widget_y_key, 
                                            on_change=sync_widget_to_internal)

                        # --- UPDATED CLICK HANDLING ---
                        def get_click(sel):
                            if sel and "selection" in sel:
                                pts = sel.get("selection", {}).get("points", [])
                                if pts:
                                    x, y = pts[0].get("x"), pts[0].get("y")
                                    if x is not None and y is not None:
                                        return (int(round(x)), int(round(y)))
                            return None
                        
                        new_click = get_click(sel1) or get_click(sel2) or get_click(sel3)
                        if new_click:
                            click_x, click_y = new_click
                            # Update INTERNAL keys. The number_input will pick up the change
                            # via the 'value' parameter on the next rerun.
                            if click_x != st.session_state[internal_x_key] or click_y != st.session_state[internal_y_key]:
                                st.session_state[internal_x_key] = float(click_x)
                                st.session_state[internal_y_key] = float(click_y)
                                st.rerun()
                        #st.markdown(f"#### 🔎 Pixel Analysis (X={tcx}, Y={tcy})")
                        tcx = int(round(st.session_state[internal_x_key]))
                        tcy = int(round(st.session_state[internal_y_key]))
                        
                        try:
                            # 1. Aggregate Spectra (Voronoi Logic)
                            raw_spec, raw_noise = None, None
                            if 'BIN_NUM' in hdul_res:
                                bin_map = hdul_res['BIN_NUM'].data
                                bin_id = bin_map[tcy, tcx]
                                mask = (bin_map == bin_id)
                                if np.sum(mask) > 0:
                                    raw_spec = np.nansum(flux_cube_raw[:, mask] * psf[mask], axis=1)
                                    raw_noise = np.sqrt(np.nansum((noise_cube_raw[:, mask] * psf[mask])**2, axis=1))
                            
                            if raw_spec is None:
                                raw_spec = flux_cube_raw[:, tcy, tcx] * psf[tcy, tcx]
                                raw_noise = noise_cube_raw[:, tcy, tcx] * psf[tcy, tcx]

                            # 2. Setup Wavelengths
                            wave_obs = extract_wavelength_calibration(header_flux_raw, len(raw_spec))
                            wave_rest = wave_obs / (1 + z_gal)
                            wave_smooth = np.linspace(wave_rest.min(), wave_rest.max(), len(wave_rest) * 10)

                            # 3. Parameter Extraction
                            slope, intercept = float(hdul_res[10].data[tcy, tcx]), float(hdul_res[11].data[tcy, tcx])
                            a_s, s_s, o_s = float(hdul_res[1].data[tcy, tcx]), float(hdul_res[4].data[tcy, tcx]), float(hdul_res[7].data[tcy, tcx])
                            chi1, chi2 = float(hdul_res[12].data[tcy, tcx]), float(hdul_res[13].data[tcy, tcx])
                            a1, s1, o1 = float(hdul_res[2].data[tcy, tcx]), float(hdul_res[5].data[tcy, tcx]), float(hdul_res[8].data[tcy, tcx])
                            a2, s2, o2 = float(hdul_res[3].data[tcy, tcx]), float(hdul_res[6].data[tcy, tcx]), float(hdul_res[9].data[tcy, tcx])
                            an1, sn1, on1 = float(hdul_res[16].data[tcy, tcx]), float(hdul_res[17].data[tcy, tcx]), float(hdul_res[18].data[tcy, tcx])
                            an2, sn2, on2 = float(hdul_res[19].data[tcy, tcx]), float(hdul_res[20].data[tcy, tcx]), float(hdul_res[21].data[tcy, tcx])

                            # 4. Model Components
                            m_lin_smooth = linear(wave_smooth, slope, intercept)
                            
                            # Handle NII: Check if parameters are valid (not NaN and not zero amplitude)
                            def safe_gaussian(wave, a, s, o):
                                """Returns gaussian or zeros if parameters are invalid"""
                                if not np.isfinite([a, s, o]).all() or a == 0 or s == 0:
                                    return np.zeros_like(wave)
                                return gaussian(wave, a, s, o)
                            
                            nii1_smooth = safe_gaussian(wave_smooth, an1, sn1, on1)
                            nii2_smooth = safe_gaussian(wave_smooth, an2, sn2, on2)
                            nii_comp_smooth = nii1_smooth + nii2_smooth
                            
                            # Total Models (Sum of H-alpha + NII + Continuum)
                            # NII will be zero-array if invalid, so fits still display correctly
                            y_total_single = gaussian(wave_smooth, a_s, s_s, o_s) + nii_comp_smooth + m_lin_smooth
                            y_total_double = (gaussian(wave_smooth, a1, s1, o1) + gaussian(wave_smooth, a2, s2, o2)) + nii_comp_smooth + m_lin_smooth

                            # Best fit selection for residuals
                            nii1_raw = safe_gaussian(wave_rest, an1, sn1, on1)
                            nii2_raw = safe_gaussian(wave_rest, an2, sn2, on2)
                            nii_comp_raw = nii1_raw + nii2_raw
                            m_lin_raw = linear(wave_rest, slope, intercept)
                            if chi1 < chi2:
                                y_best_raw = gaussian(wave_rest, a_s, s_s, o_s) + nii_comp_raw + m_lin_raw
                                best_label, b_color = "Single Gaussian", "#FF4B4B"
                            else:
                                y_best_raw = (gaussian(wave_rest, a1, s1, o1) + gaussian(wave_rest, a2, s2, o2)) + nii_comp_raw + m_lin_raw
                                best_label, b_color = "Double Gaussian", "#00FF00"

                            res_sigma = np.divide(raw_spec - y_best_raw, raw_noise, out=np.zeros_like(raw_spec), where=raw_noise > 0)
                            
                            # Store residuals in session state for sidebar access
                            st.session_state['current_res_sigma'] = res_sigma.copy()

                            # 5. Y-Axis Zoom
                            ha_window = (wave_rest >= 0.652) & (wave_rest <= 0.662)
                            local_peak = np.nanmax(raw_spec[ha_window]) if np.any(ha_window) else 1.0
                            y_max = local_peak * 1.3
                            y_min = np.nanmin(raw_spec[ha_window]) - (local_peak * 0.1) if np.any(ha_window) else -0.1

                            # 6. Plotting with SHARED X-AXIS
                            st.markdown(f"**Best Model:** <span style='color:{b_color};'>{best_label}</span> | $\\chi^2_1$: `{chi1:.2f}` | $\\chi^2_2$: `{chi2:.2f}`", unsafe_allow_html=True)
                            
                            # Create subplots with shared x-axis (spectrum on top, residuals below)
                            fig_combined = make_subplots(
                                rows=2, cols=1,
                                shared_xaxes=True,  # Share x-axis between spectrum and residuals
                                vertical_spacing=0.02,  # Small gap between plots
                                row_heights=[0.7, 0.3],  # Spectrum gets 70%, residuals 30%
                            )
                            
                            # --- TOP SUBPLOT: Spectrum ---
                            # Noise Shading
                            mask = np.isfinite(raw_spec) & np.isfinite(raw_noise)
                            fig_combined.add_trace(go.Scatter(
                                x=np.concatenate([wave_rest[mask], wave_rest[mask][::-1]]),
                                y=np.concatenate([(raw_spec + raw_noise)[mask], (raw_spec - raw_noise)[mask][::-1]]),
                                fill='toself', fillcolor='rgba(128, 128, 128, 0.2)', line=dict(color='rgba(0,0,0,0)'), name='Noise (±1σ)', hoverinfo='skip'
                            ), row=1, col=1)

                            # Data and Main Fit Lines (Always visible)
                            fig_combined.add_trace(go.Scatter(x=wave_rest, y=raw_spec, mode='lines', line=dict(color='white', width=1.5), name='Data'), row=1, col=1)
                            fig_combined.add_trace(go.Scatter(x=wave_smooth, y=y_total_single, line=dict(color='#FF4B4B', width=2.5), name='Single Fit (Total)', opacity=1.0 if chi1 < chi2 else 0.4), row=1, col=1)
                            fig_combined.add_trace(go.Scatter(x=wave_smooth, y=y_total_double, line=dict(color='#00FF00', width=2.5), name='Double Fit (Total)', opacity=1.0 if chi2 <= chi1 else 0.4), row=1, col=1)

                            if show_components:
                                # Detailed Components (Dashed)
                                fig_combined.add_trace(go.Scatter(x=wave_smooth, y=gaussian(wave_smooth, a_s, s_s, o_s) + m_lin_smooth, line=dict(color='#FF4B4B', width=1, dash='dot'), name='Hα Single Comp'), row=1, col=1)
                                fig_combined.add_trace(go.Scatter(x=wave_smooth, y=gaussian(wave_smooth, a1, s1, o1) + m_lin_smooth, line=dict(color='#00FF00', width=1, dash='dot'), name='Hα Double C1'), row=1, col=1)
                                fig_combined.add_trace(go.Scatter(x=wave_smooth, y=gaussian(wave_smooth, a2, s2, o2) + m_lin_smooth, line=dict(color='#00FF00', width=1, dash='dot'), name='Hα Double C2'), row=1, col=1)
                                fig_combined.add_trace(go.Scatter(x=wave_smooth, y=nii_comp_smooth + m_lin_smooth, line=dict(color='#00CCFF', width=1, dash='dash'), name='[NII] Component'), row=1, col=1)

                            # Annotations for emission lines (top subplot only)
                            lines = {"[NII]6548": 0.65480, "Hα": 0.65628, "[NII]6583": 0.65834}
                            for name, val in lines.items():
                                fig_combined.add_vline(x=val, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)", row=1, col=1)
                                fig_combined.add_annotation(x=val, y=y_max*0.9, text=name, showarrow=False, font=dict(color="#00CCFF", size=10), row=1, col=1)

                            # --- BOTTOM SUBPLOT: Residuals ---
                            fig_combined.add_trace(go.Scatter(x=wave_rest, y=res_sigma, mode='lines', line=dict(color='#FFFF00', width=1), name='Residuals', showlegend=False), row=2, col=1)
                            
                            # Horizontal reference lines for residuals
                            for lv, cl in [(0, 'white'), (3, 'red'), (-3, 'red')]:
                                fig_combined.add_hline(y=lv, line_dash="dash", line_color=cl, line_width=1, row=2, col=1)

                            # Update layout for combined figure
                            fig_combined.update_layout(
                                height=550,  # Combined height
                                template="plotly_dark",
                                margin=dict(t=10, b=40, l=60, r=10),
                                legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
                                uirevision=f"spec_res_{selected_galaxy_name}_{tcx}_{tcy}",  # Preserve zoom
                            )
                            
                            # Update x-axes (shared, so only bottom one gets label)
                            fig_combined.update_xaxes(range=[0.652, 0.662], row=1, col=1, showticklabels=False)
                            fig_combined.update_xaxes(range=[0.652, 0.662], row=2, col=1, title_text="Rest Wavelength (µm)")
                            
                            # Update y-axes
                            fig_combined.update_yaxes(title_text="Flux", range=[y_min, y_max], row=1, col=1)
                            fig_combined.update_yaxes(title_text="Res/σ", range=[-5, 5], row=2, col=1)
                            
                            st.plotly_chart(fig_combined, width='stretch')

                        except Exception as ex: st.error(f"Fit extraction error: {ex}")

            except Exception as e: st.error(f"Tab Error: {e}")
        else:
            st.warning(f"Results file not found: {results_file_path}")

    elif view_mode == "📈 Correlations":
        st.subheader("Interactive Galaxy Properties")
        
        # Sub-navigation for correlation views
        corr_view = st.radio(
            "Correlation View",
            ["🔬 Properties", "🌡️ WISE Diagnostics", "🦴 Fossil Candidates"],
            horizontal=True,
            label_visibility="collapsed",
            key="corr_sub_view"
        )
        
        # =====================================================================
        # SUB-VIEW 1: Galaxy Properties (existing scatter plot)
        # =====================================================================
        if corr_view == "🔬 Properties":
            df_catalog = load_catalogs()
            
            if df_catalog is not None:
                # Fix negative redshift values (set to 0)
                if 'Redshift' in df_catalog.columns:
                    df_catalog.loc[df_catalog['Redshift'] < 0, 'Redshift'] = 0
                
                base_cols = {
                    'Stellar Mass (LMSTAR)': 'LMSTAR',
                    'SFR': 'SFR',
                    'Redshift': 'Redshift',
                    'Velocity Disp. (σ)': 'SIG',
                    'Hα Flux': 'FLUX_HA',
                    'V-Band (Rest)': 'RF_V',
                    'Dust (Av)': 'SED_AV'
                }
                
                c_ctrl1, c_ctrl2, c_ctrl3 = st.columns(3)
                with c_ctrl1:
                    x_axis_label = st.selectbox("X-Axis", list(base_cols.keys()), index=0, key='qp_x')
                    x_log = st.checkbox("Log X", value=False, key='qp_x_log')
                with c_ctrl2:
                    y_axis_label = st.selectbox("Y-Axis", list(base_cols.keys()), index=1, key='qp_y')
                    y_log = st.checkbox("Log Y", value=True, key='qp_y_log')
                with c_ctrl3:
                    c_axis_label = st.selectbox("Color Scale", list(base_cols.keys()), index=2, key='qp_c')
                    c_log = st.checkbox("Log Color", value=False, key='qp_c_log')

                x_col = base_cols[x_axis_label]
                y_col = base_cols[y_axis_label]
                c_col = base_cols[c_axis_label]
                
                plot_df = df_catalog.dropna(subset=[x_col, y_col]).copy()
                
                def get_data_and_label(df, col, label, is_log):
                    data = df[col]
                    final_label = label
                    if is_log:
                        vals = np.where(data > 0, data, np.nan)
                        vals = np.log10(vals)
                        data = pd.Series(vals, index=df.index)
                        final_label = f"Log({label})"
                    return data, final_label

                x_data, x_title = get_data_and_label(plot_df, x_col, x_axis_label, x_log)
                y_data, y_title = get_data_and_label(plot_df, y_col, y_axis_label, y_log)
                c_data, c_title = get_data_and_label(plot_df, c_col, c_axis_label, c_log)
                
                clean_id = selected_galaxy_name.strip()
                current_gal_idx = plot_df[plot_df['ID'] == clean_id].index
                if len(current_gal_idx) == 0 and 'FILE' in plot_df.columns:
                     current_gal_idx = plot_df[plot_df['FILE'].astype(str).str.contains(clean_id, regex=False, na=False)].index

                fig_main = go.Figure()
                fig_main.add_trace(go.Scatter(
                    x=x_data, y=y_data,
                    mode='markers',
                    marker=dict(size=8, color=c_data, colorscale='Viridis', showscale=True, opacity=0.7,
                                colorbar=dict(title=c_title)),
                    text=plot_df['ID'],
                    customdata=plot_df['ID'],
                    name='Galaxies',
                    hovertemplate='<b>%{text}</b><br>' + x_title + ': %{x:.3f}<br>' + y_title + ': %{y:.3f}<extra></extra>'
                ))
                
                if len(current_gal_idx) > 0:
                    idx = current_gal_idx[0]
                    gx = x_data.loc[idx]
                    gy = y_data.loc[idx]
                    if np.isfinite(gx) and np.isfinite(gy):
                        fig_main.add_trace(go.Scatter(
                            x=[gx], y=[gy], mode='markers',
                            marker=dict(size=25, color='rgba(0,0,0,0)', line=dict(color='#FF00FF', width=3)),
                            name='Selected', hoverinfo='skip'
                        ))
                        fig_main.add_trace(go.Scatter(
                            x=[gx], y=[gy], mode='markers',
                            marker=dict(size=10, color='#FF00FF', symbol='cross'),
                            showlegend=False, hoverinfo='skip'
                        ))

                fig_main.update_layout(
                    height=600, template="plotly_dark",
                    margin=dict(l=80, r=20, t=40, b=60),
                    xaxis_title=x_title, yaxis_title=y_title,
                    hovermode='closest', dragmode=False,
                    legend=dict(orientation="v", y=0.98, x=0.02, xanchor='left', yanchor='middle',
                                font=dict(color='#F1F5F9')),
                    uirevision="corr_plot_interactive",
                )
                
                event = st.plotly_chart(
                    fig_main, width='stretch',
                    config={'displayModeBar': False},
                    on_select="rerun", selection_mode="points",
                    key="corr_plot_interactive"
                )
                
                if event and "selection" in event:
                    pts = event.get("selection", {}).get("points", [])
                    if pts:
                        clicked_id = pts[0].get("customdata")
                        if clicked_id and clicked_id != selected_galaxy_name:
                            for label, galaxy_name in display_options.items():
                                if galaxy_name == clicked_id:
                                    st.session_state.current_selection = label
                                    st.rerun()
                                    break

        # =====================================================================
        # SUB-VIEW 2: WISE Color-Color Diagnostics
        # =====================================================================
        elif corr_view == "🌡️ WISE Diagnostics":
            xmatch_full = load_crossmatch_full()
            
            if xmatch_full is not None:
                # Find WISE magnitude columns
                w1_col = next((c for c in xmatch_full.columns if 'W1mag' in c), None)
                w2_col = next((c for c in xmatch_full.columns if 'W2mag' in c), None)
                w3_col = next((c for c in xmatch_full.columns if 'W3mag' in c), None)
                w4_col = next((c for c in xmatch_full.columns if 'W4mag' in c), None)
                
                if w1_col and w2_col and w3_col:
                    # Compute colors
                    wise_df = xmatch_full[['OBJECT', 'RA', 'DEC']].copy()
                    wise_df['W1_W2'] = pd.to_numeric(xmatch_full[w1_col], errors='coerce') - \
                                       pd.to_numeric(xmatch_full[w2_col], errors='coerce')
                    wise_df['W2_W3'] = pd.to_numeric(xmatch_full[w2_col], errors='coerce') - \
                                       pd.to_numeric(xmatch_full[w3_col], errors='coerce')
                    if w4_col:
                        wise_df['W3_W4'] = pd.to_numeric(xmatch_full[w3_col], errors='coerce') - \
                                           pd.to_numeric(xmatch_full[w4_col], errors='coerce')
                    
                    # Add W80 data for color coding
                    w80_map = {item[0]: item[1] for item in sorted_galaxy_data
                               if isinstance(item, (list, tuple)) and len(item) >= 2}
                    wise_df['W80'] = wise_df['OBJECT'].map(w80_map)
                    
                    # Filter to valid data
                    plot_wise = wise_df.dropna(subset=['W1_W2', 'W2_W3']).copy()
                    
                    if len(plot_wise) > 0:
                        # --- Controls ---
                        cw1, cw2 = st.columns([2, 1])
                        with cw1:
                            wise_color_by = st.selectbox(
                                "Color by", ["W80 (km/s)", "W1−W2", "None"],
                                key="wise_color_sel"
                            )
                        with cw2:
                            show_wedge = st.checkbox("Show AGN wedges", value=True, key="wise_wedge")
                        
                        # Color data
                        if wise_color_by == "W80 (km/s)" and 'W80' in plot_wise.columns:
                            color_data = plot_wise['W80']
                            cbar_title = "W80 (km/s)"
                            colorscale = 'Hot'
                        elif wise_color_by == "W1−W2":
                            color_data = plot_wise['W1_W2']
                            cbar_title = "W1−W2"
                            colorscale = 'RdYlBu_r'
                        else:
                            color_data = None
                            cbar_title = None
                            colorscale = None
                        
                        # --- Build WISE figure ---
                        fig_wise = go.Figure()
                        
                        # AGN demarcation regions (drawn first so they're behind)
                        if show_wedge:
                            # Stern+2012: W1-W2 >= 0.8 horizontal line
                            fig_wise.add_hline(
                                y=0.8, line_dash="dash", line_color="#FF4B4B",
                                line_width=2, opacity=0.8,
                                annotation_text="Stern+12 (W1−W2=0.8)",
                                annotation_position="top left",
                                annotation_font=dict(color="#FF4B4B", size=11)
                            )
                            
                            # Mateos+2012 AGN wedge (approximate parametric boundaries)
                            # Upper boundary
                            w2w3_wedge = np.linspace(1.5, 5.5, 50)
                            upper_w1w2 = 0.315 * w2w3_wedge - 0.222
                            lower_w1w2 = 0.315 * w2w3_wedge - 0.796
                            
                            # Vertical left boundary at W2-W3 ~ 2.0
                            fig_wise.add_trace(go.Scatter(
                                x=np.concatenate([w2w3_wedge, w2w3_wedge[::-1]]),
                                y=np.concatenate([upper_w1w2, lower_w1w2[::-1]]),
                                fill='toself',
                                fillcolor='rgba(255,75,75,0.08)',
                                line=dict(color='rgba(255,75,75,0.4)', width=1, dash='dot'),
                                name='Mateos+12 Wedge',
                                hoverinfo='skip'
                            ))
                        
                        # Main scatter
                        marker_dict = dict(
                            size=9,
                            opacity=0.8,
                            line=dict(width=0.5, color='rgba(255,255,255,0.3)')
                        )
                        if color_data is not None:
                            marker_dict['color'] = color_data
                            marker_dict['colorscale'] = colorscale
                            marker_dict['showscale'] = True
                            marker_dict['colorbar'] = dict(title=cbar_title, thickness=15)
                        else:
                            marker_dict['color'] = '#00CCFF'
                        
                        fig_wise.add_trace(go.Scatter(
                            x=plot_wise['W2_W3'], y=plot_wise['W1_W2'],
                            mode='markers',
                            marker=marker_dict,
                            text=plot_wise['OBJECT'],
                            customdata=plot_wise['OBJECT'],
                            name='KMOS3D',
                            hovertemplate=(
                                '<b>%{text}</b><br>'
                                'W2−W3: %{x:.2f}<br>'
                                'W1−W2: %{y:.2f}<br>'
                                '<extra></extra>'
                            )
                        ))
                        
                        # Highlight selected galaxy
                        cur_wise = plot_wise[plot_wise['OBJECT'] == selected_galaxy_name]
                        if not cur_wise.empty:
                            cx = cur_wise.iloc[0]['W2_W3']
                            cy = cur_wise.iloc[0]['W1_W2']
                            if np.isfinite(cx) and np.isfinite(cy):
                                fig_wise.add_trace(go.Scatter(
                                    x=[cx], y=[cy], mode='markers',
                                    marker=dict(size=22, color='rgba(0,0,0,0)',
                                                line=dict(color='#FF00FF', width=3)),
                                    name='Selected', hoverinfo='skip'
                                ))
                                fig_wise.add_trace(go.Scatter(
                                    x=[cx], y=[cy], mode='markers',
                                    marker=dict(size=8, color='#FF00FF', symbol='cross'),
                                    showlegend=False, hoverinfo='skip'
                                ))
                        
                        fig_wise.update_layout(
                            height=600, template="plotly_dark",
                            margin=dict(l=80, r=20, t=40, b=60),
                            xaxis_title="W2 − W3 (Vega mag)",
                            yaxis_title="W1 − W2 (Vega mag)",
                            hovermode='closest', dragmode=False,
                            legend=dict(orientation="h", y=1.02, x=0.5, xanchor='center',
                                        font=dict(color='#F1F5F9', size=11)),
                            xaxis=dict(range=[-0.5, 6.5]),
                            yaxis=dict(range=[-0.5, 2.5]),
                        )
                        
                        # Click to select galaxy
                        wise_event = st.plotly_chart(
                            fig_wise, width='stretch',
                            config={'displayModeBar': False},
                            on_select="rerun", selection_mode="points",
                            key="wise_plot_interactive"
                        )
                        
                        if wise_event and "selection" in wise_event:
                            pts = wise_event.get("selection", {}).get("points", [])
                            if pts:
                                clicked_id = pts[0].get("customdata")
                                if clicked_id and clicked_id != selected_galaxy_name:
                                    for label, galaxy_name in display_options.items():
                                        if galaxy_name == clicked_id:
                                            st.session_state.current_selection = label
                                            st.rerun()
                                            break
                        
                        # --- Statistics summary ---
                        n_total = len(plot_wise)
                        n_stern = int((plot_wise['W1_W2'] >= 0.8).sum())
                        
                        # Check Mateos wedge membership
                        in_mateos = (
                            (plot_wise['W1_W2'] >= 0.315 * plot_wise['W2_W3'] - 0.796) &
                            (plot_wise['W1_W2'] <= 0.315 * plot_wise['W2_W3'] - 0.222) &
                            (plot_wise['W2_W3'] >= 1.5)
                        )
                        n_mateos = int(in_mateos.sum())
                        
                        if n_total > 0:
                            st.markdown(
                                f"**{n_total}** galaxies with WISE colors · "
                                f"**{n_stern}** ({100*n_stern/n_total:.0f}%) above Stern+12 · "
                                f"**{n_mateos}** ({100*n_mateos/n_total:.0f}%) in Mateos+12 wedge"
                            )
                        
                        # Highlight fossil candidates in text
                        if 'W80' in plot_wise.columns:
                            # Exclude literature AGN from fossil count
                            agn_ids_wise = load_agn_catalog_ids("AGN_SAMPLE.fits")
                            is_lit = plot_wise['OBJECT'].apply(
                                lambda g: is_agn_galaxy(g, agn_ids_wise))
                            fossil_mask = (
                                (plot_wise['W1_W2'] < 0.8) &
                                (plot_wise['W80'] >= 400) &
                                (~is_lit)
                            )
                            n_fossil = int(fossil_mask.sum())
                            if n_fossil > 0:
                                st.info(
                                    f"🦴 **{n_fossil} potential fossil outflow candidates**: "
                                    f"W80 ≥ 400 km/s but W1−W2 < 0.8 (below AGN threshold), "
                                    f"excluding literature AGN"
                                )
                    else:
                        st.warning("No valid WISE color data found in crossmatch.")
                else:
                    st.warning("WISE magnitude columns not found in crossmatch file.")
            else:
                st.info(
                    "No crossmatch data available. Run `Crossmatch_generator.py` first to "
                    "generate `crossmatch.fits`."
                )

        # =====================================================================
        # SUB-VIEW 3: Fossil AGN Candidate Scoring
        # =====================================================================
        elif corr_view == "🦴 Fossil Candidates":
            st.markdown(
                "Galaxies scored by likelihood of hosting **fossil AGN outflows** — "
                "kinematic disturbance without current AGN activity."
            )
            
            # Load required data
            xmatch_full = load_crossmatch_full()
            agn_ids_fossil = load_agn_catalog_ids("AGN_SAMPLE.fits")
            galaxy_list = [item[0] if isinstance(item, (list, tuple)) else item
                           for item in sorted_galaxy_data]
            
            fossil_df = compute_fossil_scores(
                galaxy_list=galaxy_list,
                sorted_galaxy_data=sorted_galaxy_data,
                crossmatch_df=xmatch_full,
                agn_ids=agn_ids_fossil,
                results_path=RESULTS_PATH,
            )
            
            if fossil_df is not None and len(fossil_df) > 0:
                
                # Separate lit AGN from scorable galaxies early
                fossil_non_agn = fossil_df[fossil_df['Lit_AGN'] == False].reset_index(drop=True)
                n_excluded = int(fossil_df['Lit_AGN'].sum())
                n_candidates = len(fossil_non_agn)
                
                # --- Top: Score distribution plot (non-AGN only) ---
                col_chart, col_legend = st.columns([3, 1])
                
                with col_chart:
                    # Color each bar by score tier
                    def score_color(s):
                        if s >= 6: return '#FF4B4B'   # strong fossil
                        if s >= 4: return '#FF8C00'   # likely fossil
                        if s >= 3: return '#FFD700'   # possible (threshold)
                        return '#4A5568'               # unlikely
                    
                    bar_colors = fossil_non_agn['Fossil_Score'].apply(score_color)
                    
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Bar(
                        x=fossil_non_agn['ID'], y=fossil_non_agn['Fossil_Score'],
                        marker_color=bar_colors,
                        text=fossil_non_agn['Fossil_Score'],
                        textposition='outside',
                        textfont=dict(size=8, color='gray'),
                        hovertemplate=(
                            '<b>%{x}</b><br>'
                            'Fossil Score: %{y}/8<br>'
                            '<extra></extra>'
                        ),
                    ))
                    
                    # Highlight selected galaxy
                    sel_row = fossil_non_agn[fossil_non_agn['ID'] == selected_galaxy_name]
                    if not sel_row.empty:
                        fig_dist.add_annotation(
                            x=selected_galaxy_name,
                            y=sel_row.iloc[0]['Fossil_Score'] + 0.5,
                            text="▼", showarrow=False,
                            font=dict(color='#FF00FF', size=16)
                        )
                    
                    fig_dist.update_layout(
                        height=280, template="plotly_dark",
                        margin=dict(l=40, r=10, t=10, b=80),
                        xaxis=dict(tickangle=-45, tickfont=dict(size=7), title=None),
                        yaxis=dict(title="Fossil Score", range=[0, 9],
                                   dtick=1, gridcolor='#334155'),
                        bargap=0.15, showlegend=False,
                    )
                    st.plotly_chart(fig_dist, use_container_width=True,
                                   config={'displayModeBar': False})
                    
                    if n_excluded > 0:
                        st.caption(f"⛔ {n_excluded} literature AGN excluded from chart")
                
                with col_legend:
                    st.markdown("""
                    **Score Legend**
                    
                    <span style='color:#FF4B4B'>■</span> **6–8** Strong fossil  
                    <span style='color:#FF8C00'>■</span> **4–5** Likely fossil  
                    <span style='color:#FFD700'>■</span> **3** Possible  
                    <span style='color:#4A5568'>■</span> **0–2** Unlikely
                    
                    ---
                    **Scoring (0–8):**
                    
                    ⛔ *Lit. AGN → 0*  
                    Milliquas OR SIMBAD
                    
                    *Kinematics (0–3)*  
                    W80 strength
                    
                    *WHAN (0–1)*  
                    Not sAGN/wAGN
                    
                    *MIR Fading (0–2)*  
                    W1−W2 below AGN
                    
                    *Mismatch (0–2)*  
                    High W80 + faded MIR
                    """, unsafe_allow_html=True)
                
                # --- Scatter: W80 vs Fossil Score ---
                st.markdown("---")
                fig_scatter = go.Figure()
                
                # Literature AGN are hard-excluded (score = 0)
                lit_agn = fossil_df[fossil_df['Lit_AGN'] == True]
                non_agn = fossil_df[fossil_df['Lit_AGN'] == False]
                
                # Non-AGN galaxies (fossil candidates live here)
                fig_scatter.add_trace(go.Scatter(
                    x=non_agn['W80'], y=non_agn['Fossil_Score'],
                    mode='markers',
                    marker=dict(size=9, color='#00CCFF', opacity=0.8,
                                line=dict(width=0.5, color='white')),
                    text=non_agn['ID'], name='No lit. AGN',
                    hovertemplate='<b>%{text}</b><br>W80: %{x:.0f} km/s<br>Score: %{y}/8<extra></extra>'
                ))
                # Literature AGN (excluded, shown as X markers at score 0)
                fig_scatter.add_trace(go.Scatter(
                    x=lit_agn['W80'], y=lit_agn['Fossil_Score'],
                    mode='markers',
                    marker=dict(size=9, color='#FF4B4B', symbol='x', opacity=0.5,
                                line=dict(width=1.5, color='#FF4B4B')),
                    text=lit_agn['ID'], name='Lit. AGN (excluded)',
                    hovertemplate='<b>%{text}</b><br>W80: %{x:.0f} km/s<br>⛔ Literature AGN<extra></extra>'
                ))
                
                # Highlight selected
                sel_fossil = fossil_df[fossil_df['ID'] == selected_galaxy_name]
                if not sel_fossil.empty:
                    sx = sel_fossil.iloc[0]['W80']
                    sy = sel_fossil.iloc[0]['Fossil_Score']
                    fig_scatter.add_trace(go.Scatter(
                        x=[sx], y=[sy], mode='markers',
                        marker=dict(size=22, color='rgba(0,0,0,0)',
                                    line=dict(color='#FF00FF', width=3)),
                        name='Selected', hoverinfo='skip'
                    ))
                
                # Fossil zone annotation
                w80_max = fossil_df['W80'].max()
                if not np.isfinite(w80_max) or w80_max < 200:
                    w80_max = 800  # fallback
                fig_scatter.add_shape(
                    type="rect", x0=200, x1=w80_max * 1.05,
                    y0=3, y1=8.5,
                    fillcolor="rgba(255,75,75,0.06)", line=dict(color="rgba(255,75,75,0.3)", dash="dot"),
                )
                w80_median = fossil_df['W80'].median()
                if not np.isfinite(w80_median):
                    w80_median = 400
                fig_scatter.add_annotation(
                    x=max(400, w80_median),
                    y=8.2, text="🦴 Fossil Zone (≥3/8)",
                    showarrow=False, font=dict(color='#FF8C00', size=12)
                )
                
                fig_scatter.update_layout(
                    height=450, template="plotly_dark",
                    margin=dict(l=60, r=20, t=30, b=60),
                    xaxis_title="Mean W80 (km/s)",
                    yaxis_title="Fossil Score (0–8)",
                    yaxis=dict(range=[-0.5, 9], dtick=1, gridcolor='#334155'),
                    hovermode='closest', dragmode=False,
                    legend=dict(orientation="h", y=1.05, x=0.5, xanchor='center',
                                font=dict(color='#F1F5F9', size=11)),
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True,
                               config={'displayModeBar': False})
                
                # --- Detailed table for top candidates ---
                st.markdown(f"#### Top Fossil Candidates  \n"
                            f"*{n_candidates} scored · {n_excluded} literature AGN excluded*")
                
                if n_candidates > 0:
                    slider_max = min(30, n_candidates)
                    slider_min = min(5, slider_max)
                    top_n = st.slider("Show top N", slider_min, slider_max,
                                      min(10, slider_max), key="fossil_top_n")
                    
                    top_df = fossil_non_agn.head(top_n).copy()
                    
                    # Format for display
                    display_cols = ['ID', 'Fossil_Score', 'W80', 'W1_W2',
                                    'Kin_Score', 'WHAN_Score', 'MIR_Score',
                                    'Mismatch', 'DeepRadio', 'DeepXray']
                    display_df = top_df[display_cols].copy()
                    display_df.columns = ['Galaxy', 'Score', 'W80', 'W1−W2',
                                          'Kin', 'WHAN', 'MIR', 'Mismatch', 'Radio', 'X-ray']
                    
                    # Style: highlight the selected galaxy row
                    def highlight_selected(row):
                        if row['Galaxy'] == selected_galaxy_name:
                            return ['background-color: #2d1b4e'] * len(row)
                        return [''] * len(row)
                    
                    styled = display_df.style.apply(highlight_selected, axis=1).format({
                        'W80': '{:.0f}',
                        'W1−W2': '{:.3f}',
                        'Score': '{:.0f}',
                    })
                    
                    st.dataframe(styled, use_container_width=True, hide_index=True, height=400)
                
                # Selected galaxy breakdown (shown for all, including excluded AGN)
                sel_fossil = fossil_df[fossil_df['ID'] == selected_galaxy_name]
                if not sel_fossil.empty:
                    s = sel_fossil.iloc[0]
                    if s['Lit_AGN']:
                        agn_sources = []
                        if s['Milliquas']:
                            agn_sources.append('Milliquas')
                        if s['SIMBAD_AGN']:
                            agn_sources.append('SIMBAD')
                        st.error(f"**{selected_galaxy_name}** — ⛔ **Literature AGN** "
                                 f"({' + '.join(agn_sources)}) → excluded from scoring")
                    else:
                        w1w2_str = f"(W1−W2={s['W1_W2']:.2f})" if np.isfinite(s['W1_W2']) else "(no WISE)"
                        radio_flag = " 📻 deep detected" if s['DeepRadio'] else ""
                        xray_flag = " ☢️ deep detected" if s['DeepXray'] else ""
                        st.markdown(f"""
                        ---
                        **{selected_galaxy_name}** — Fossil Score: **{int(s['Fossil_Score'])}/8**  
                        Kinematics: {int(s['Kin_Score'])}/3 (W80={s['W80']:.0f}) · 
                        WHAN: {int(s['WHAN_Score'])}/1 · 
                        MIR Fading: {int(s['MIR_Score'])}/2 {w1w2_str} · 
                        Mismatch: {int(s['Mismatch'])}/2{radio_flag}{xray_flag}
                        """)
            else:
                st.warning("Could not compute fossil scores. Check that W80 data and crossmatch file are available.")


# --- RIGHT COLUMN (SIMULATED SIDEBAR) ---
if st.session_state.show_tools and col_right:
    with col_right:
        # MAIN SIDEBAR CONTAINER
        # We use a main container to hold the sidebar together
        with st.container(border=False, gap="medium"):
            
            # --- A. HEADER ---
            # Using columns to push the close button to the far right
            head_col1, head_col2 = st.columns([0.85, 0.15])
            with head_col1:
                st.markdown("### 🛠️ Controls")
                st.caption("Analysis & Configuration")
            with head_col2:
                # 'type="tertiary"' makes the button subtle (no border until hovered)
                st.button("⏩", key="close_btn", on_click=toggle_sidebar, help="Collapse Sidebar", type="tertiary")
            
# --- B. TABBED ANALYSIS PANELS ---
            # Initialize session state for sidebar tab
            if "sidebar_tab" not in st.session_state:
                st.session_state.sidebar_tab = "Context"
            
            # Tab selector
            sidebar_tab = st.radio(
                "Panel", 
                ["📍 Context", "📊 WHAN", "📈 Histogram", "📉 Residuals"],
                horizontal=True,
                label_visibility="collapsed",
                key="sidebar_tab_selector"
            )
            
            # --- TAB 1: CONTEXT (Original XKCD Plot) ---
            if sidebar_tab == "📍 Context":
                with st.container(border=True):
                    st.markdown("##### 📍 Context")
                    
                    df_cat = load_catalogs()
                    
                    if df_cat is not None:
                        # 1. Determine Columns & Scale from Session State
                        base_cols = {
                            'Stellar Mass (LMSTAR)': 'LMSTAR',
                            'SFR': 'SFR',
                            'Redshift': 'Redshift',
                            'Velocity Disp. (σ)': 'SIG',
                            'Hα Flux': 'FLUX_HA',
                            'V-Band (Rest)': 'RF_V',
                            'Dust (Av)': 'SED_AV'
                        }

                        # Get Selections (Defaults if not set)
                        x_lbl = st.session_state.get('qp_x', 'Stellar Mass (LMSTAR)')
                        y_lbl = st.session_state.get('qp_y', 'SFR')
                        x_is_log = st.session_state.get('qp_x_log', False)
                        y_is_log = st.session_state.get('qp_y_log', True) # Default SFR to Log
                        
                        x_key = base_cols.get(x_lbl, 'LMSTAR')
                        y_key = base_cols.get(y_lbl, 'SFR')

                        # 2. Filter & Transform Data
                        # We create a mini dataframe for plotting
                        ms_df = df_cat[[x_key, y_key, 'ID', 'FILE']].copy()
                        
                        # Apply Log if requested
                        if x_is_log:
                            ms_df[x_key] = np.log10(np.where(ms_df[x_key] > 0, ms_df[x_key], np.nan))
                            x_lbl = f"Log({x_lbl})"
                        
                        if y_is_log:
                            ms_df[y_key] = np.log10(np.where(ms_df[y_key] > 0, ms_df[y_key], np.nan))
                            y_lbl = f"Log({y_lbl})"
                            
                        ms_df = ms_df.dropna()

                        # 3. Locate Current Galaxy
                        clean_id = selected_galaxy_name.strip()
                        cur = ms_df[ms_df['ID'] == clean_id]
                        if cur.empty and 'FILE' in ms_df.columns:
                            cur = ms_df[ms_df['FILE'].astype(str).str.contains(clean_id, regex=False, na=False)]

                        # 4. DRAW XKCD PLOT
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning)
                            warnings.filterwarnings("ignore", category=RuntimeWarning)
                            warnings.filterwarnings("ignore")
                            # 4. DRAW XKCD PLOT

                            logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

                            with plt.xkcd():
                                fig_x, ax_x = plt.subplots(figsize=(4, 3))
                                fig_x.patch.set_alpha(0.0)
                                ax_x.patch.set_alpha(0.0)
                                
                                # Plot All
                                ax_x.scatter(ms_df[x_key], ms_df[y_key], s=10, c='gray', alpha=0.3, edgecolors='none')
                                
                                # Plot Selected
                                if not cur.empty:
                                    cx, cy = cur.iloc[0][x_key], cur.iloc[0][y_key]
                                    # Check boundaries (in case log made it NaN or infinite)
                                    if np.isfinite(cx) and np.isfinite(cy):
                                        ax_x.scatter(cx, cy, s=200, facecolors='none', edgecolors='red', linewidth=2)
                                        ax_x.annotate('THIS ONE', xy=(cx, cy), xytext=(cx, cy + (ms_df[y_key].max() - ms_df[y_key].min())*0.1),
                                                    arrowprops=dict(arrowstyle='->', color='red'), color='red', fontsize=10)
                                
                                # Labels
                                ax_x.set_xlabel(x_lbl, fontsize=8, color='gray')
                                ax_x.set_ylabel(y_lbl, fontsize=8, color='gray')
                                
                                # Styling
                                ax_x.spines['right'].set_color('none')
                                ax_x.spines['top'].set_color('none')
                                ax_x.tick_params(colors='gray', labelsize=7)
                                ax_x.spines['bottom'].set_color('gray')
                                ax_x.spines['left'].set_color('gray')

                                plt.tight_layout()
                                st.pyplot(fig_x, width='stretch', transparent=True)
                                plt.close(fig_x)
                    else:
                        st.caption("No Catalog Data")

            # --- TAB 2: WHAN DIAGRAM ---
            elif sidebar_tab == "📊 WHAN":
                with st.container(border=True):
                    st.markdown("##### 📊 WHAN Diagram")
                    
                    # Collect WHAN data from all galaxies
                    @st.cache_data(ttl=3600)
                    def load_whan_data_all(results_path, galaxy_list):
                        """Load WHAN parameters (EW_HA, NII_HA) from all galaxy headers."""
                        whan_data = []
                        for gal_name in galaxy_list:
                            res_path = os.path.join(results_path, f"{gal_name}_voronoi_binned.fits")
                            if os.path.exists(res_path):
                                try:
                                    with fits.open(res_path) as hdul:
                                        hdr = hdul[0].header
                                        ew = hdr.get('EW_HA', None)
                                        nii = hdr.get('NII_HA', None)
                                        whan_cls = hdr.get('WHAN_CLS', 'Unknown')
                                        if ew is not None and nii is not None:
                                            if nii > 0:
                                                log_nii_ha = np.log10(nii)
                                            else:
                                                log_nii_ha = np.nan
                                            whan_data.append({
                                                'ID': gal_name,
                                                'EW_HA': float(ew),
                                                'log_NII_HA': log_nii_ha,
                                                'WHAN_CLS': whan_cls.strip() if whan_cls else 'Unknown'
                                            })
                                except Exception:
                                    pass
                        return pd.DataFrame(whan_data) if whan_data else None
                    
                    # Get galaxy list from sorted data
                    galaxy_list = [item[0] if isinstance(item, (list, tuple)) else item for item in sorted_galaxy_data]
                    whan_df = load_whan_data_all(RESULTS_PATH, galaxy_list)
                    
                    if whan_df is not None and len(whan_df) > 0:
                        # Color map for WHAN classes
                        color_map = {
                            'SF': 'cyan',
                            'sAGN': 'red',
                            'wAGN': 'orange',
                            'Retired': 'gray',
                            'Passive': 'lightgray',
                            'Unknown': 'white'
                        }
                        
                        # Draw XKCD style WHAN diagram
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
                            
                            with plt.xkcd():
                                fig_whan, ax_whan = plt.subplots(figsize=(5, 4))
                                fig_whan.patch.set_alpha(0.0)
                                ax_whan.patch.set_alpha(0.0)
                                
                                # Plot all galaxies by class with colors
                                for cls_name, color in color_map.items():
                                    cls_df = whan_df[whan_df['WHAN_CLS'] == cls_name]
                                    if not cls_df.empty:
                                        ax_whan.scatter(cls_df['log_NII_HA'], cls_df['EW_HA'], 
                                                       s=25, c=color, alpha=0.6, edgecolors='none', label=cls_name)
                                
                                # Highlight current galaxy
                                cur_whan = whan_df[whan_df['ID'] == selected_galaxy_name]
                                if not cur_whan.empty:
                                    cx, cy = cur_whan.iloc[0]['log_NII_HA'], cur_whan.iloc[0]['EW_HA']
                                    if np.isfinite(cx) and np.isfinite(cy):
                                        ax_whan.scatter(cx, cy, s=200, facecolors='none', edgecolors='yellow', linewidth=3, zorder=10)
                                        ax_whan.annotate('YOU', xy=(cx, cy), xytext=(cx + 0.15, cy * 1.5),
                                                        arrowprops=dict(arrowstyle='->', color='yellow'), 
                                                        color='yellow', fontsize=9, fontweight='bold')
                                
                                # Add classification boundary lines
                                ax_whan.axvline(x=-0.4, color='white', linestyle='--', alpha=0.4, linewidth=1)
                                ax_whan.axhline(y=3, color='white', linestyle='--', alpha=0.4, linewidth=1)
                                ax_whan.axhline(y=6, color='white', linestyle=':', alpha=0.3, linewidth=1)
                                
                                # Region labels
                                ax_whan.text(-1.3, 40, 'SF', color='cyan', fontsize=9, alpha=0.8)
                                ax_whan.text(0.3, 40, 'sAGN', color='red', fontsize=9, alpha=0.8)
                                ax_whan.text(0.3, 4.2, 'wAGN', color='orange', fontsize=9, alpha=0.8)
                                ax_whan.text(0.3, 1.2, 'Ret.', color='gray', fontsize=8, alpha=0.8)
                                ax_whan.text(-1.3, 1.2, 'Pas.', color='gray', fontsize=8, alpha=0.8)
                                
                                # Styling
                                ax_whan.set_yscale('log')
                                ax_whan.set_xlim(-1.8, 0.8)
                                ax_whan.set_ylim(0.3, 150)
                                ax_whan.set_xlabel('log([NII]/Hα)', fontsize=9, color='gray')
                                ax_whan.set_ylabel('EW(Hα) [Å]', fontsize=9, color='gray')
                                ax_whan.spines['right'].set_color('none')
                                ax_whan.spines['top'].set_color('none')
                                ax_whan.tick_params(colors='gray', labelsize=8)
                                ax_whan.spines['bottom'].set_color('gray')
                                ax_whan.spines['left'].set_color('gray')
                                
                                plt.tight_layout()
                                st.pyplot(fig_whan, use_container_width=True, transparent=True)
                                plt.close(fig_whan)
                        
                        # Show current galaxy class
                        if not cur_whan.empty:
                            st.caption(f"**{selected_galaxy_name}**: {cur_whan.iloc[0]['WHAN_CLS']}")
                    else:
                        st.caption("No WHAN data available")

            # --- TAB 3: MAP HISTOGRAM ---
            elif sidebar_tab == "📈 Histogram":
                with st.container(border=True):
                    results_file_path_hist = os.path.join(RESULTS_PATH, selected_galaxy_name + "_voronoi_binned.fits")
                    
                    if os.path.exists(results_file_path_hist):
                        try:
                            with fits.open(results_file_path_hist) as hdul_hist:
                                # Get available map extensions
                                map_ext_hist = [ext.name for ext in hdul_hist if ext.name and ext.data is not None and len(getattr(ext.data, 'shape', [])) == 2]
                                
                                # Composite options first
                                hist_options = {
                                    "⭐ Best Flux": "COMP_FLUX",
                                    "⭐ Best Velocity": "COMP_VEL", 
                                    "⭐ W80": "COMP_W80"
                                }
                                for ext in map_ext_hist:
                                    if ext not in ['PRIMARY', 'BIN_NUM']:
                                        hist_options[ext] = ext
                                
                                # Controls row
                                c1, c2 = st.columns([2, 1])
                                with c1:
                                    selected_hist_map = st.selectbox("Map", list(hist_options.keys()), key="hist_map_selector", label_visibility="collapsed")
                                with c2:
                                    log_y = st.toggle("Log", key="hist_log_y", help="Log scale Y-axis")
                                
                                # Zoom slider
                                clip_pct = st.slider("Clip %", 90, 100, 98, key="hist_clip", help="Percentile clipping")
                                
                                # Get map data
                                map_key = hist_options[selected_hist_map]
                                
                                if map_key.startswith("COMP_"):
                                    best_maps_hist = calculate_best_fit_maps(results_file_path_hist, fits_file_path)
                                    if best_maps_hist:
                                        hist_data = best_maps_hist.get({"COMP_FLUX": "Best Flux", "COMP_VEL": "Best Velocity", "COMP_W80": "W80"}.get(map_key))
                                    else:
                                        hist_data = None
                                else:
                                    if map_key in hdul_hist:
                                        hist_data = hdul_hist[map_key].data.copy().astype(float)
                                        if map_key.startswith('B'):
                                            hist_data = (hist_data / 0.65628) * 299792.458
                                        elif map_key.startswith('C') and not map_key.startswith('CHI'):
                                            hist_data = ((hist_data - 0.65628) / 0.65628) * 299792.458
                                    else:
                                        hist_data = None
                                
                                if hist_data is not None:
                                    flat_data = hist_data.flatten()
                                    valid_data = flat_data[np.isfinite(flat_data) & (np.abs(flat_data) < 1e30)]
                                    
                                    if len(valid_data) > 0:
                                        # Apply clipping
                                        p_low = (100 - clip_pct) / 2
                                        p_high = 100 - p_low
                                        lo, hi = np.percentile(valid_data, [p_low, p_high])
                                        clipped = valid_data[(valid_data >= lo) & (valid_data <= hi)]
                                        
                                        mean_val, median_val = np.mean(clipped), np.median(clipped)
                                        
                                        fig_hist = go.Figure()
                                        fig_hist.add_trace(go.Histogram(x=clipped, nbinsx=35, marker_color='#00CCFF', opacity=0.75))
                                        fig_hist.add_vline(x=mean_val, line_color="yellow", line_width=1.5)
                                        fig_hist.add_vline(x=median_val, line_dash="dash", line_color="lime", line_width=1.5)
                                        
                                        fig_hist.update_layout(
                                            height=180, margin=dict(l=35, r=5, t=5, b=30),
                                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1E293B',
                                            xaxis=dict(gridcolor='#334155', tickfont=dict(color='gray', size=8)),
                                            yaxis=dict(type="log" if log_y else "linear", gridcolor='#334155', tickfont=dict(color='gray', size=8)),
                                            showlegend=False
                                        )
                                        st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
                                        
                                        st.caption(f"μ={mean_val:.1f} | med={median_val:.1f} | σ={np.std(clipped):.1f} | n={len(clipped)}")
                                    else:
                                        st.caption("No valid data")
                                else:
                                    st.caption("Could not load map")
                        except Exception as e:
                            st.caption(f"Error: {e}")
                    else:
                        st.caption("No results file")

            # --- TAB 4: RESIDUAL DISTRIBUTION ---
            elif sidebar_tab == "📉 Residuals":
                with st.container(border=True):
                    if 'current_res_sigma' in st.session_state:
                        res_sigma = st.session_state['current_res_sigma']
                        clean_res = res_sigma[np.isfinite(res_sigma)]
                        
                        if len(clean_res) > 0:
                            # Controls
                            c1, c2 = st.columns([1, 1])
                            with c1:
                                x_range = st.slider("±σ", 3, 10, 5, key="res_range")
                            with c2:
                                log_y_res = st.toggle("Log", key="res_log_y")
                            
                            # Clip outliers
                            lo, hi = np.percentile(clean_res, [1, 99])
                            clipped = clean_res[(clean_res >= lo) & (clean_res <= hi)]
                            
                            mean_res, std_res = np.mean(clipped), np.std(clipped)
                            
                            fig_res = go.Figure()
                            fig_res.add_trace(go.Histogram(x=clipped, nbinsx=30, marker_color='#FFFF00', opacity=0.75))
                            fig_res.add_vline(x=0, line_color="white", line_width=1.5)
                            fig_res.add_vline(x=mean_res, line_dash="dot", line_color="cyan", line_width=1.5)
                            
                            fig_res.update_layout(
                                height=180, margin=dict(l=35, r=5, t=5, b=30),
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1E293B',
                                xaxis=dict(range=[-x_range, x_range], gridcolor='#334155', tickfont=dict(color='gray', size=8)),
                                yaxis=dict(type="log" if log_y_res else "linear", gridcolor='#334155', tickfont=dict(color='gray', size=8)),
                                showlegend=False
                            )
                            st.plotly_chart(fig_res, use_container_width=True, config={'displayModeBar': False})
                            
                            # Compact stats with quality indicator
                            q_col = "lime" if 0.8 < std_res < 1.2 else "orange" if 0.5 < std_res < 1.5 else "red"
                            st.caption(f"μ={mean_res:.2f} | σ=<span style='color:{q_col}'>{std_res:.2f}</span> | n={len(clipped)}", unsafe_allow_html=True)
                        else:
                            st.caption("No valid data")
                    else:
                        st.caption("Select a pixel in Kinematic Maps first")


            # --- C. MODULES (Card Style) ---
            '''with st.container(border=True):
                st.markdown("##### 🚀 Launchers")
                
                def run_script(script_name, label):
                    if os.path.exists(script_name):
                        try:
                            subprocess.Popen([sys.executable, script_name, fits_file_path])
                            st.toast(f"🚀 {label} launched!", icon="✅")
                        except Exception as e: st.error(f"Failed: {e}")
                    else: st.error(f"Script '{script_name}' not found.")

                # Using columns for buttons to make them look like a grid or full width
                # Here we stick to full width for readability
                if st.button("📊 Stacked Spectra", width='stretch', help="Open Spectral Stacking GUI"): 
                    run_script("GUI_stacked_spectra.py", "Stacked Spectra")
                    
                if st.button("🌀 Kinematik GUI", width='stretch', help="Open W80 & Velocity Analysis"): 
                    run_script("w80_gui copy.py", "Kinematik GUI")
                    
                if st.button("🔭 Ergebnisse GUI", width='stretch', help="View Final Results"): 
                    run_script("program_runner.py", "Ergebnisse GUI")
            '''
            st.write("") # Small spacer
            # Only works if your streamlit version is >= 1.30
            st.html("""
                <div style='
                background-color: #0E1117; 
                padding: 15px; 
                border-radius: 8px; 
                text-align: center; 
                font-size: 0.8em;
                color: #888;
                margin-top: 10px;
            '>
                <strong style='color: #EEE;'>Master Thesis</strong><br>
                
                <a href="https://github.com/koljareuter" target="_blank" style="text-decoration: none; color: inherit;">
                    <span style='letter-spacing: 1px; color: #BBB;'>Kolja Reuter</span>
                </a>
                <br><span style='letter-spacing: 1px; color: #BBB;'>supervised by<br>
                
                <a href="https://wwwstaff.ari.uni-heidelberg.de/dwylezalek" target="_blank" style="text-decoration: none; color: inherit;">
                    <span style='letter-spacing: 1px; color: #BBB;'>Prof. Dr. Dominika Wylezalek</span>
                </a>
            </div>
            """)