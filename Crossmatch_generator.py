#!/usr/bin/env python3
"""
KMOS3D Multi-Survey Crossmatch Tool (v4 - Deep Field)
======================================================
Crossmatches KMOS3D FITS files with multiple surveys.

Key improvements from v3:
- Removed shallow all-sky surveys (Gaia DR3, FIRST, ROSAT) that are
  irrelevant for z=0.7-2.7 galaxies
- Added field-aware deep surveys:
    COSMOS:  COSMOS2020, VLA-COSMOS 3GHz (rms~2.3μJy), Chandra COSMOS Legacy
    GOODS-S: CDF-S 7Ms X-ray (deepest ever), VLA E-CDFS 1.4GHz (rms~7μJy)
    UDS:     X-UDS Chandra
- Auto-detects KMOS3D field from filename prefix (COS4/GS4/U4)
- Only queries field-specific surveys for galaxies in that field

Requirements:
    pip install astropy astroquery numpy

Usage:
    python Crossmatch_generator.py /path/to/KMOS3D_ALL/ output_crossmatch.fits

Author: Improved version
"""

import os
import sys
import warnings
from glob import glob
from datetime import datetime
from urllib.parse import quote as urlquote
import numpy as np

warnings.filterwarnings('ignore')

try:
    from astropy.io import fits
    from astropy.table import Table, vstack, Column
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from astroquery.vizier import Vizier
    from astroquery.mast import Observations
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install astropy astroquery numpy")
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

MATCH_RADIUS_ARCSEC = 3.0

# ==== ALL-SKY / broad-coverage surveys (queried for ALL galaxies) ====
SURVEYS = {
    'NED': {
        'catalog': 'NED',
        'ra_col': 'RA',
        'dec_col': 'DEC',
        'id_col': 'Object Name',
        'prefix': 'NED_',
        'key_cols': ['Type', 'Redshift', 'Mag'],
        'coverage': 'all_sky',
        'url_template': 'https://ned.ipac.caltech.edu/byname?objname={id}',
        'url_id_col': 'Object Name',
    },
    'WISE': {
        'catalog': 'II/328/allwise',
        'ra_col': 'RAJ2000',
        'dec_col': 'DEJ2000',
        'id_col': 'AllWISE',
        'prefix': 'WISE_',
        'key_cols': ['W1mag', 'W2mag', 'W3mag', 'W4mag'],
        'coverage': 'all_sky',
        'url_template': 'https://vizier.cds.unistra.fr/viz-bin/VizieR-5?-source=II/328/allwise&-c={ra}+{dec}&-c.rs=3',
        'url_id_col': None,
    },
    'SDSS': {
        'catalog': 'V/154/sdss16',
        'ra_col': 'RA_ICRS',
        'dec_col': 'DE_ICRS',
        'id_col': 'objID',
        'prefix': 'SDSS_',
        'key_cols': ['gmag', 'rmag', 'imag', 'zmag', 'zsp'],
        'coverage': 'all_sky',
        'url_template': 'https://skyserver.sdss.org/dr16/en/tools/explore/summary.aspx?ra={ra}&dec={dec}',
        'url_id_col': None,
    },
    '2MASS': {
        'catalog': 'II/246/out',
        'ra_col': 'RAJ2000',
        'dec_col': 'DEJ2000',
        'id_col': '_2MASS',
        'prefix': '2MASS_',
        'key_cols': ['Jmag', 'Hmag', 'Kmag'],
        'coverage': 'all_sky',
        'url_template': 'https://vizier.cds.unistra.fr/viz-bin/VizieR-5?-source=II/246/out&-c={ra}+{dec}&-c.rs=3',
        'url_id_col': None,
    },
    'JWST': {
        'catalog': 'MAST',
        'ra_col': 's_ra',
        'dec_col': 's_dec',
        'id_col': 'obs_id',
        'prefix': 'JWST_',
        'key_cols': ['instrument_name', 'filters', 't_exptime', 'proposal_id',
                     'target_name', 'dataproduct_type'],
        'coverage': 'all_sky',
        'url_template': 'https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html?searchQuery={ra}+{dec}',
        'url_id_col': None,
    },
}

# ==== DEEP FIELD-SPECIFIC surveys (queried only for galaxies in that field) ====
# Field assignment from filename: COS4→COSMOS, GS4→GOODS_S, U4→UDS
FIELD_SURVEYS = {
    # ======================== COSMOS FIELD ========================
    'COS2020': {
        'catalog': 'J/ApJS/258/11/classic',   # COSMOS2020 (Weaver+2022)
        'ra_col': 'RAJ2000', 'dec_col': 'DEJ2000', 'id_col': 'ID',
        'prefix': 'COS20_',
        'key_cols': ['zPDF', 'Mass', 'SFR'],
        'field': 'COSMOS',
        'url_template': 'https://vizier.cds.unistra.fr/viz-bin/VizieR-5?-source=J/ApJS/258/11/classic&-c={ra}+{dec}&-c.rs=3',
        'url_id_col': None,
        'description': 'COSMOS2020 Classic (Weaver+2022)',
    },
    'VLA3GHZ': {
        'catalog': 'J/A+A/602/A1/table1',     # VLA-COSMOS 3GHz (Smolčić+2017)
        'ra_col': 'RAJ2000', 'dec_col': 'DEJ2000', 'id_col': 'ID',
        'prefix': 'VLA3G_',
        'key_cols': ['Stot', 'e_Stot', 'Speak', 'rms'],
        'field': 'COSMOS',
        'match_radius': 2.0,
        'url_template': 'https://vizier.cds.unistra.fr/viz-bin/VizieR-5?-source=J/A+A/602/A1&-c={ra}+{dec}&-c.rs=5',
        'url_id_col': None,
        'description': 'VLA-COSMOS 3GHz (Smolčić+2017) rms~2.3μJy',
    },
    'CCOSLEG': {
        'catalog': 'J/ApJ/819/62/table5',     # Chandra COSMOS Legacy (Civano+2016)
        'ra_col': 'RAJ2000', 'dec_col': 'DEJ2000', 'id_col': 'CID',
        'prefix': 'CXCOS_',
        'key_cols': ['Flux', 'Flux2', 'HR'],
        'field': 'COSMOS',
        'match_radius': 5.0,  # Chandra PSF degrades off-axis
        'url_template': 'https://vizier.cds.unistra.fr/viz-bin/VizieR-5?-source=J/ApJ/819/62/table5&-c={ra}+{dec}&-c.rs=5',
        'url_id_col': None,
        'description': 'Chandra COSMOS Legacy X-ray (Civano+2016)',
    },
    # ======================== GOODS-S FIELD ========================
    'CDFS7MS': {
        'catalog': 'J/ApJS/228/2',       # CDF-S 7Ms (Luo+2017) — let VizieR resolve table
        'ra_col': 'RAJ2000', 'dec_col': 'DEJ2000', 'id_col': 'XID',
        'prefix': 'CDFS_',
        'key_cols': ['FB', 'SB', 'HB', 'z'],
        'field': 'GOODS_S',
        'match_radius': 5.0,  # Chandra PSF degrades off-axis
        'url_template': 'https://vizier.cds.unistra.fr/viz-bin/VizieR-5?-source=J/ApJS/228/2&-c={ra}+{dec}&-c.rs=5',
        'url_id_col': None,
        'description': 'CDF-S 7Ms X-ray (Luo+2017) deepest X-ray',
    },
    'VLACDFS': {
        'catalog': 'J/ApJS/205/13/table3',    # VLA E-CDFS 1.4GHz (Miller+2013)
        'ra_col': 'RAJ2000', 'dec_col': 'DEJ2000', 'id_col': 'ID',
        'prefix': 'VLAGS_',
        'key_cols': ['Sp', 'Sint', 'Maj'],
        'field': 'GOODS_S',
        'match_radius': 3.0,
        'url_template': 'https://vizier.cds.unistra.fr/viz-bin/VizieR-5?-source=J/ApJS/205/13&-c={ra}+{dec}&-c.rs=5',
        'url_id_col': None,
        'description': 'VLA E-CDFS 1.4GHz (Miller+2013) rms~7.4μJy',
    },
    # ======================== UDS FIELD ========================
    'XUDS': {
        'catalog': 'J/ApJS/236/48/xuds',    # X-UDS Chandra (Kocevski+2018) — VizieR merged t4-6
        'ra_col': 'RAJ2000', 'dec_col': 'DEJ2000', 'id_col': 'XID',
        'prefix': 'XUDS_',
        'key_cols': ['FB', 'SB', 'HB', 'Lx'],
        'field': 'UDS',
        'match_radius': 5.0,  # Chandra PSF degrades off-axis
        'url_template': 'https://vizier.cds.unistra.fr/viz-bin/VizieR-5?-source=J/ApJS/236/48/xuds&-c={ra}+{dec}&-c.rs=5',
        'url_id_col': None,
        'description': 'X-UDS Chandra (Kocevski+2018)',
    },
}

# Combined dict of ALL surveys for iteration in output functions
ALL_SURVEYS = {**SURVEYS, **FIELD_SURVEYS}


# ============================================================================
# FITS File Reader
# ============================================================================

def read_kmos3d_fits(filepath):
    """Extract key information from a KMOS3D FITS file."""
    info = {'filepath': filepath, 'filename': os.path.basename(filepath)}
    
    try:
        with fits.open(filepath) as hdul:
            hdr = hdul[0].header
            
            # Core identifiers
            info['OBJECT'] = hdr.get('OBJECT', '')
            info['OBJ_TARG'] = hdr.get('OBJ_TARG', '')
            info['RA'] = hdr.get('RA', np.nan)
            info['DEC'] = hdr.get('DEC', np.nan)
            info['EQUINOX'] = hdr.get('EQUINOX', 2000)
            info['RADECSYS'] = hdr.get('RADECSYS', 'FK5')
            
            # Observation metadata
            info['OBSBAND'] = hdr.get('OBSBAND', hdr.get('ESO INS FILT1 NAME', ''))
            info['EXPTIME'] = hdr.get('EXPTIME', np.nan)
            info['NEXP'] = hdr.get('NEXP', 0)
            info['VERSION'] = hdr.get('VERSION', '')
            info['INSTRUME'] = hdr.get('INSTRUME', 'KMOS')
            
            # Spectral resolution info - try multiple header variants
            info['RES_MIN'] = hdr.get('HIERARCH ESO K3D RES MIN', 
                                       hdr.get('RES_MIN', np.nan))
            info['RES_MAX'] = hdr.get('HIERARCH ESO K3D RES MAX',
                                       hdr.get('RES_MAX', np.nan))
            
            # Data cube dimensions
            for ext_idx in range(1, len(hdul)):
                if hdul[ext_idx].data is not None and len(hdul[ext_idx].data.shape) >= 2:
                    hdr1 = hdul[ext_idx].header
                    info['NAXIS1'] = hdr1.get('NAXIS1', 0)
                    info['NAXIS2'] = hdr1.get('NAXIS2', 0)
                    info['NAXIS3'] = hdr1.get('NAXIS3', 0)
                    info['CDELT3'] = hdr1.get('CDELT3', np.nan)
                    info['CRVAL3'] = hdr1.get('CRVAL3', np.nan)
                    break
                    
    except Exception as e:
        print(f"  Warning: Error reading {filepath}: {e}")
        info['READ_ERROR'] = str(e)
    
    return info


def detect_field(filename):
    """Determine KMOS3D field from filename prefix: COS4→COSMOS, GS4→GOODS_S, U4→UDS."""
    base = os.path.basename(filename).upper()
    if base.startswith('COS'):
        return 'COSMOS'
    elif base.startswith('GS'):
        return 'GOODS_S'
    elif base.startswith('U'):
        return 'UDS'
    return 'UNKNOWN'


def load_kmos3d_folder(folder_path):
    """Load all FITS files from KMOS3D_ALL folder."""
    fits_files = glob(os.path.join(folder_path, '*.fits'))
    fits_files += glob(os.path.join(folder_path, '**/*.fits'), recursive=True)
    fits_files = list(set(fits_files))
    
    print(f"Found {len(fits_files)} FITS files in {folder_path}")
    
    if not fits_files:
        raise ValueError(f"No FITS files found in {folder_path}")
    
    all_info = []
    n_valid = 0
    for i, fpath in enumerate(sorted(fits_files)):
        if (i + 1) % 50 == 0:
            print(f"  Reading file {i+1}/{len(fits_files)}...")
        info = read_kmos3d_fits(fpath)
        
        # Validate coordinates
        if np.isfinite(info.get('RA', np.nan)) and np.isfinite(info.get('DEC', np.nan)):
            n_valid += 1
        
        all_info.append(info)
    
    print(f"  Valid coordinates: {n_valid}/{len(fits_files)}")
    
    # Convert to Table
    table = Table()
    all_keys = set()
    for info in all_info:
        all_keys.update(info.keys())
    
    for key in sorted(all_keys):
        values = [info.get(key, None) for info in all_info]
        sample = [v for v in values if v is not None]
        if not sample:
            continue
        if isinstance(sample[0], (int, np.integer)):
            dtype = np.int64
            values = [v if v is not None else -999 for v in values]
        elif isinstance(sample[0], (float, np.floating)):
            dtype = np.float64
            values = [v if v is not None else np.nan for v in values]
        else:
            dtype = str
            values = [str(v) if v is not None else '' for v in values]
        
        table[key] = Column(values, dtype=dtype)
    
    # Filter to only valid coordinates
    valid_mask = np.isfinite(table['RA']) & np.isfinite(table['DEC'])
    if np.sum(~valid_mask) > 0:
        print(f"  WARNING: Removing {np.sum(~valid_mask)} sources with invalid coordinates")
        table = table[valid_mask]
    
    # Detect field for each galaxy from filename
    table['FIELD'] = [detect_field(fn) for fn in table['filename']]
    field_counts = {}
    for f in table['FIELD']:
        field_counts[f] = field_counts.get(f, 0) + 1
    print(f"  Field breakdown: {dict(field_counts)}")
    
    return table


# ============================================================================
# Survey Query Functions
# ============================================================================

from astroquery.ipac.ned import Ned

def query_ned_catalog(coords, radius_arcsec):
    """Specialized query for NED using astroquery.ipac.ned."""
    print(f"  Querying NED...")
    all_results = []
    
    n_errors = 0
    for i, coord in enumerate(coords):
        if (i + 1) % 100 == 0 or (i + 1) == len(coords):
            print(f"    Queried {i+1}/{len(coords)} positions... ({len(all_results)} matches)")
        try:
            # NED query_region returns an astropy table
            result = Ned.query_region(coord, radius=radius_arcsec * u.arcsec)
            if result and len(result) > 0:
                # Strip unrecognized units that cause copy()/vstack() to fail
                for col in result.colnames:
                    try:
                        hash(result[col].unit)
                    except TypeError:
                        result[col].unit = None
                
                # NED results are already sorted by proximity
                row = result[:1].copy()
                # Calculate separation
                row['SEP_ARCSEC'] = coord.separation(
                    SkyCoord(row['RA'], row['DEC'], unit='deg')
                ).arcsec
                row['_query_idx'] = i
                all_results.append(row)
        except Exception as e:
            n_errors += 1
            if n_errors <= 3:
                print(f"    NED error at idx {i}: {type(e).__name__}: {str(e)[:80]}")
            elif n_errors == 4:
                print(f"    (Suppressing further NED error messages...)")
            continue
    
    if n_errors > 0:
        print(f"    Total NED errors: {n_errors}/{len(coords)} (normal for positions with no NED entry)")
    
    if all_results:
        try:
            return vstack(all_results)
        except Exception as e:
            print(f"    Warning: Could not stack NED results: {e}")
            return all_results[0] if all_results else None
    return None


def query_jwst_mast(coords, radius_arcsec):
    """
    Query MAST for JWST observations near each position.
    
    Returns a table with one row per KMOS3D source (the closest JWST observation),
    including key observation metadata.
    """
    print(f"  Querying JWST via MAST archive...")
    all_results = []
    n_errors = 0
    
    for i, coord in enumerate(coords):
        if (i + 1) % 100 == 0 or (i + 1) == len(coords):
            print(f"    Queried {i+1}/{len(coords)} positions... ({len(all_results)} matches)")
        try:
            # Query MAST for JWST observations within the search radius
            obs_table = Observations.query_criteria(
                coordinates=coord,
                radius=radius_arcsec * u.arcsec,
                obs_collection='JWST',
            )
            
            if obs_table is not None and len(obs_table) > 0:
                # Sort by angular distance to pick the closest observation
                obs_coords = SkyCoord(ra=obs_table['s_ra'], dec=obs_table['s_dec'], unit='deg')
                seps = coord.separation(obs_coords).arcsec
                obs_table['SEP_ARCSEC'] = seps
                closest_idx = np.argmin(seps)
                row = obs_table[closest_idx:closest_idx + 1].copy()
                
                # Add summary columns: total number of JWST obs at this position
                row['N_JWST_OBS'] = len(obs_table)
                
                # Collect unique instruments and filters across all obs
                all_instruments = ','.join(sorted(set(
                    str(v) for v in obs_table['instrument_name'] if str(v).strip()
                )))
                all_filters = ','.join(sorted(set(
                    str(v) for v in obs_table['filters'] if str(v).strip()
                )))
                row['ALL_INSTRUMENTS'] = all_instruments[:200]  # truncate for safety
                row['ALL_FILTERS'] = all_filters[:200]
                
                row['_query_idx'] = i
                all_results.append(row)
                
        except Exception as e:
            n_errors += 1
            if n_errors <= 3:
                print(f"    JWST/MAST error at idx {i}: {type(e).__name__}: {str(e)[:80]}")
            elif n_errors == 4:
                print(f"    (Suppressing further MAST error messages...)")
            continue
    
    if n_errors > 0:
        print(f"    Total MAST errors: {n_errors}/{len(coords)}")
    
    if all_results:
        try:
            return vstack(all_results)
        except Exception as e:
            print(f"    Warning: Could not stack JWST results: {e}")
            return all_results[0] if all_results else None
    return None


def safe_get_value(value):
    """Safely extract value from potentially masked data."""
    if value is None:
        return None
    if hasattr(value, 'mask'):
        if value.mask:
            return None
        return value.data
    if value is np.ma.masked:
        return None
    try:
        if np.ma.is_masked(value):
            return None
    except:
        pass
    return value


def query_vizier_catalog(coords, catalog_id, radius_arcsec, survey_name):
    """
    Query a Vizier catalog with proper error handling.
    
    Returns a table with _query_idx and SEP_ARCSEC columns.
    """
    print(f"  Querying {survey_name} ({catalog_id})...")
    
    # Configure Vizier - get ALL columns plus separation
    v = Vizier(columns=['**', '+_r'], row_limit=1, timeout=120)
    
    # Verify catalog exists using a non-positional query (won't miss sparse catalogs)
    try:
        v_test = Vizier(catalog=catalog_id, row_limit=1, timeout=60)
        test_result = v_test.query_constraints()
        if test_result and len(test_result) > 0:
            tbl_name = test_result[0].meta.get('name', catalog_id)
            print(f"    VizieR table resolved: {tbl_name} ({len(test_result[0])} test rows)")
        else:
            # Non-positional query returned nothing — catalog ID is likely wrong
            print(f"    WARNING: VizieR catalog '{catalog_id}' not found — check table name!")
            print(f"    Skipping this survey. Verify at: https://vizier.cds.unistra.fr/viz-bin/VizieR?-source={catalog_id}")
            return None
    except Exception as e:
        # Network errors etc. — don't abort, just warn and proceed
        print(f"    WARNING: VizieR test query for {catalog_id} raised: {str(e)[:100]}")
        print(f"    Proceeding with queries anyway...")
    
    all_results = []
    n_errors = 0
    n_queries = len(coords)
    
    for i, coord in enumerate(coords):
        if (i + 1) % 100 == 0 or (i + 1) == n_queries:
            print(f"    Queried {i+1}/{n_queries} positions... ({len(all_results)} matches)")
        
        try:
            result = v.query_region(coord, radius=radius_arcsec * u.arcsec, 
                                    catalog=catalog_id)
            if result and len(result) > 0:
                tbl = result[0]
                if len(tbl) > 0:
                    # Take only closest match
                    row = tbl[:1].copy()
                    
                    # Rename _r to SEP_ARCSEC for clarity
                    if '_r' in row.colnames:
                        row.rename_column('_r', 'SEP_ARCSEC')
                    
                    # Add query index
                    row['_query_idx'] = i
                    all_results.append(row)
                    
        except Exception as e:
            n_errors += 1
            if n_errors <= 5:
                print(f"    Error at idx {i}: {str(e)[:80]}")
            elif n_errors == 6:
                print(f"    (Suppressing further error messages...)")
    
    if n_errors > 0:
        print(f"    Total errors: {n_errors}/{n_queries}")
    
    if all_results:
        try:
            # Handle potential column type mismatches
            result_table = vstack(all_results)
            return result_table
        except Exception as e:
            print(f"    Warning: Could not stack results: {e}")
            # Try to return first result at least
            return all_results[0] if all_results else None
    return None


def query_all_surveys(kmos_table, radius_arcsec=MATCH_RADIUS_ARCSEC):
    """Query all configured surveys (all-sky + field-specific)."""
    # Validate coordinates
    valid_mask = np.isfinite(kmos_table['RA']) & np.isfinite(kmos_table['DEC'])
    if np.sum(~valid_mask) > 0:
        print(f"  WARNING: {np.sum(~valid_mask)} sources have invalid coordinates")
    
    coords = SkyCoord(ra=kmos_table['RA'] * u.deg, 
                      dec=kmos_table['DEC'] * u.deg, 
                      frame='icrs')
    
    # Print coordinate range for debugging
    print(f"\n  KMOS3D coordinate range:")
    print(f"    RA:  {np.nanmin(kmos_table['RA']):.2f} to {np.nanmax(kmos_table['RA']):.2f} deg")
    print(f"    DEC: {np.nanmin(kmos_table['DEC']):.2f} to {np.nanmax(kmos_table['DEC']):.2f} deg")
    print(f"    N sources: {len(coords)}")
    print()
    
    survey_results = {}
    
    # --- 1. All-sky surveys: query ALL galaxies ---
    print("=" * 40)
    print("  ALL-SKY SURVEYS")
    print("=" * 40)
    for survey_name, config in SURVEYS.items():
        try:
            survey_radius = config.get('match_radius', radius_arcsec)
            
            if survey_name == 'NED':
                result = query_ned_catalog(coords, survey_radius)
            elif survey_name == 'JWST':
                result = query_jwst_mast(coords, survey_radius)
            else:
                result = query_vizier_catalog(
                    coords, config['catalog'], survey_radius, survey_name
                )
            
            if result is not None and len(result) > 0:
                print(f"  {survey_name}: Found {len(result)} matches")
                cols_preview = [c for c in result.colnames if not c.startswith('_')][:6]
                print(f"    Key columns: {cols_preview}")
                if 'SEP_ARCSEC' in result.colnames:
                    seps = result['SEP_ARCSEC']
                    valid_seps = seps[~np.ma.getmaskarray(seps)]
                    if len(valid_seps) > 0:
                        print(f"    Median separation: {np.median(valid_seps):.2f} arcsec")
                survey_results[survey_name] = result
            else:
                print(f"  {survey_name}: No matches found")
                
        except Exception as e:
            print(f"  {survey_name}: Query failed - {e}")
    
    # --- 2. Field-specific deep surveys ---
    print()
    print("=" * 40)
    print("  DEEP FIELD-SPECIFIC SURVEYS")
    print("=" * 40)
    
    # Pre-build field masks (indices into kmos_table)
    field_col = kmos_table['FIELD'] if 'FIELD' in kmos_table.colnames else None
    
    for survey_name, config in FIELD_SURVEYS.items():
        target_field = config['field']
        
        # Select only galaxies in the matching field
        if field_col is not None:
            field_mask = np.array([str(f) == target_field for f in field_col])
        else:
            field_mask = np.ones(len(kmos_table), dtype=bool)
        
        n_field = int(np.sum(field_mask))
        if n_field == 0:
            print(f"  {survey_name} ({target_field}): No galaxies in this field, skipping")
            continue
        
        field_coords = coords[field_mask]
        # Track original indices so we can map back results
        field_indices = np.where(field_mask)[0]
        
        print(f"  {survey_name} ({target_field}): Querying {n_field} galaxies...")
        
        try:
            survey_radius = config.get('match_radius', radius_arcsec)
            result = query_vizier_catalog(
                field_coords, config['catalog'], survey_radius, survey_name
            )
            
            if result is not None and len(result) > 0:
                # Remap _query_idx from field-local to global indices
                for i in range(len(result)):
                    local_idx = result['_query_idx'][i]
                    result['_query_idx'][i] = field_indices[local_idx]
                
                print(f"  {survey_name}: Found {len(result)} matches")
                cols_preview = [c for c in result.colnames if not c.startswith('_')][:6]
                print(f"    Key columns: {cols_preview}")
                if 'SEP_ARCSEC' in result.colnames:
                    seps = result['SEP_ARCSEC']
                    valid_seps = seps[~np.ma.getmaskarray(seps)]
                    if len(valid_seps) > 0:
                        print(f"    Median separation: {np.median(valid_seps):.2f} arcsec")
                survey_results[survey_name] = result
            else:
                print(f"  {survey_name}: No matches found")
                
        except Exception as e:
            print(f"  {survey_name}: Query failed - {e}")
    
    return survey_results


# ============================================================================
# Crossmatch and Merge
# ============================================================================

def merge_survey_results(kmos_table, survey_results):
    """Merge survey results into the KMOS3D table."""
    merged = kmos_table.copy()
    
    for survey_name, config in ALL_SURVEYS.items():
        prefix = config['prefix']
        
        if survey_name not in survey_results:
            # Add flag column for missing surveys
            merged[prefix + 'MATCHED'] = np.zeros(len(merged), dtype=bool)
            continue
        
        result_table = survey_results[survey_name]
        
        # Initialize all columns from result (except _query_idx)
        for col in result_table.colnames:
            if col == '_query_idx':
                continue
            new_col = prefix + col
            dtype = result_table[col].dtype
            
            # Handle different data types
            if np.issubdtype(dtype, np.integer):
                try:
                    col_data = result_table[col]
                    if hasattr(col_data, 'filled'):
                        col_data = col_data.filled(0)
                    max_val = np.max(np.abs(col_data))
                    if max_val > 2**62:
                        merged[new_col] = np.full(len(merged), '', dtype='U30')
                    else:
                        merged[new_col] = np.full(len(merged), -999, dtype=np.int64)
                except (OverflowError, ValueError, TypeError):
                    merged[new_col] = np.full(len(merged), '', dtype='U30')
            elif np.issubdtype(dtype, np.floating):
                merged[new_col] = np.full(len(merged), np.nan, dtype=np.float64)
            else:
                max_len = 50
                try:
                    max_len = max(len(str(v)) for v in result_table[col] if v is not None)
                except:
                    pass
                merged[new_col] = np.full(len(merged), '', dtype=f'U{max(max_len, 50)}')
        
        # Add MATCHED column
        merged[prefix + 'MATCHED'] = np.zeros(len(merged), dtype=bool)
        
        # Fill in matched values
        for row in result_table:
            idx = row['_query_idx']
            merged[prefix + 'MATCHED'][idx] = True
            
            for col in result_table.colnames:
                if col == '_query_idx':
                    continue
                new_col = prefix + col
                
                try:
                    val = safe_get_value(row[col])
                    if val is None:
                        continue
                    
                    # Convert large integers to string
                    if isinstance(val, (int, np.integer)):
                        try:
                            int_val = int(val)
                            if abs(int_val) > 2**62:
                                val = str(int_val)
                        except (OverflowError, ValueError):
                            val = str(val)
                    
                    merged[new_col][idx] = val
                    
                except (ValueError, TypeError, OverflowError) as e:
                    try:
                        str_val = str(row[col])
                        if str_val not in ('--', 'nan', ''):
                            merged[new_col][idx] = str_val
                    except:
                        pass
    
    return merged


# ============================================================================
# Database Link Generation
# ============================================================================

def generate_database_links(merged_table):
    """
    Generate direct URLs to each database entry for every matched source.
    Adds a {PREFIX}URL column for each survey.
    """
    print("  Generating database links...")
    
    for survey_name, config in ALL_SURVEYS.items():
        prefix = config['prefix']
        matched_col = prefix + 'MATCHED'
        url_col = prefix + 'URL'
        template = config.get('url_template', '')
        url_id_col = config.get('url_id_col')
        
        if not template:
            continue
        
        # Initialize URL column
        merged_table[url_col] = np.full(len(merged_table), '', dtype='U300')
        
        if matched_col not in merged_table.colnames:
            continue
        
        n_links = 0
        for i in range(len(merged_table)):
            if not merged_table[matched_col][i]:
                continue
            
            ra = merged_table['RA'][i]
            dec = merged_table['DEC'][i]
            
            # Build URL: either from matched object ID or from coordinates
            if url_id_col and (prefix + url_id_col) in merged_table.colnames:
                obj_id = str(merged_table[prefix + url_id_col][i]).strip()
                if obj_id and obj_id not in ('', '--', 'nan'):
                    url = template.format(
                        id=urlquote(obj_id),
                        ra=f"{ra:.6f}",
                        dec=f"{dec:+.6f}",
                    )
                else:
                    url = template.format(
                        ra=f"{ra:.6f}",
                        dec=f"{dec:+.6f}",
                        id='',
                    )
            else:
                url = template.format(
                    ra=f"{ra:.6f}",
                    dec=f"{dec:+.6f}",
                )
            
            merged_table[url_col][i] = url[:300]
            n_links += 1
        
        if n_links > 0:
            print(f"    {survey_name}: {n_links} links generated")
    
    # Also add a general SIMBAD coordinate lookup link for every source
    merged_table['SIMBAD_URL'] = np.full(len(merged_table), '', dtype='U300')
    for i in range(len(merged_table)):
        ra = merged_table['RA'][i]
        dec = merged_table['DEC'][i]
        merged_table['SIMBAD_URL'][i] = (
            f"https://simbad.u-strasbg.fr/simbad/sim-coo"
            f"?Coord={ra:.6f}+{dec:+.6f}&CooFrame=FK5&CooEpoch=2000"
            f"&CooEqui=2000&CooDefinedFrames=none&Radius=3&Radius.unit=arcsec"
        )
    print(f"    SIMBAD: {len(merged_table)} coordinate lookup links generated")
    
    return merged_table


# ============================================================================
# Output FITS Creation
# ============================================================================

def create_output_fits(merged_table, output_path, kmos_folder):
    """Create a well-organized multi-extension FITS file."""
    print(f"\nCreating output FITS file: {output_path}")
    
    hdu_list = fits.HDUList()
    
    # ----- HDU 0: Primary header -----
    primary = fits.PrimaryHDU()
    primary.header['TITLE'] = 'KMOS3D Multi-Survey Crossmatch Catalog'
    primary.header['AUTHOR'] = 'Crossmatch_generator_v3.py'
    primary.header['DATE'] = datetime.now().isoformat()
    primary.header['KMOSPATH'] = str(kmos_folder)[:68]  # Truncate if too long
    primary.header['NSOURCES'] = len(merged_table)
    primary.header['MATCHRAD'] = (MATCH_RADIUS_ARCSEC, 'Crossmatch radius [arcsec]')
    
    # Survey match counts
    for survey_name, config in ALL_SURVEYS.items():
        prefix = config['prefix']
        matched_col = prefix + 'MATCHED'
        if matched_col in merged_table.colnames:
            n_matched = int(np.sum(merged_table[matched_col]))
        else:
            n_matched = 0
        key_name = f'N_{survey_name[:6]}'
        primary.header[key_name] = (n_matched, f'{survey_name} matches')
    
    hdu_list.append(primary)
    
    # ----- HDU 1: Core KMOS3D data (NAMED 'KMOS3D_SOURCES' for visualization) -----
    kmos_cols = ['OBJECT', 'OBJ_TARG', 'RA', 'DEC', 'EQUINOX', 'RADECSYS',
                 'OBSBAND', 'EXPTIME', 'NEXP', 'VERSION', 'INSTRUME',
                 'RES_MIN', 'RES_MAX', 'NAXIS1', 'NAXIS2', 'NAXIS3',
                 'CDELT3', 'CRVAL3', 'FIELD', 'filename', 'filepath']
    
    kmos_table_out = Table()
    for col in kmos_cols:
        if col in merged_table.colnames:
            kmos_table_out[col] = merged_table[col]
    
    # Use 'KMOS3D_SOURCES' name for compatibility with visualization script
    kmos_hdu = fits.BinTableHDU(kmos_table_out, name='KMOS3D_SOURCES')
    hdu_list.append(kmos_hdu)
    
    # ----- HDU 2+: Survey-specific tables -----
    for survey_name, config in ALL_SURVEYS.items():
        prefix = config['prefix']
        
        survey_table = Table()
        survey_table['KMOS3D_ID'] = merged_table['OBJECT']
        survey_table['KMOS3D_RA'] = merged_table['RA']
        survey_table['KMOS3D_DEC'] = merged_table['DEC']
        
        # Get all columns for this survey
        for col in merged_table.colnames:
            if col.startswith(prefix):
                new_name = col.replace(prefix, '')
                survey_table[new_name] = merged_table[col]
        
        # FITS EXTNAME supports up to 68 chars (not limited to 8)
        ext_name = f'{survey_name}_MATCH'
        survey_hdu = fits.BinTableHDU(survey_table, name=ext_name)
        survey_hdu.header['CATALOG'] = config['catalog']
        survey_hdu.header['MATCHRAD'] = MATCH_RADIUS_ARCSEC
        
        matched_col = prefix + 'MATCHED'
        if matched_col in merged_table.colnames:
            survey_hdu.header['NMATCHED'] = int(np.sum(merged_table[matched_col]))
        else:
            survey_hdu.header['NMATCHED'] = 0
            
        hdu_list.append(survey_hdu)
    
    # ----- Summary table -----
    summary_data = {
        'SURVEY': [],
        'CATALOG_ID': [],
        'N_SEARCHED': [],
        'N_MATCHED': [],
        'MATCH_PCT': []
    }
    
    for survey_name, config in ALL_SURVEYS.items():
        prefix = config['prefix']
        matched_col = prefix + 'MATCHED'
        
        # For field-specific surveys, count only galaxies in that field
        if survey_name in FIELD_SURVEYS and 'FIELD' in merged_table.colnames:
            target_field = FIELD_SURVEYS[survey_name]['field']
            field_mask = np.array([str(f) == target_field for f in merged_table['FIELD']])
            n_searched = int(np.sum(field_mask))
        else:
            n_searched = len(merged_table)
        
        if matched_col in merged_table.colnames:
            n_matched = int(np.sum(merged_table[matched_col]))
        else:
            n_matched = 0
        
        summary_data['SURVEY'].append(survey_name)
        summary_data['CATALOG_ID'].append(config['catalog'])
        summary_data['N_SEARCHED'].append(n_searched)
        summary_data['N_MATCHED'].append(n_matched)
        summary_data['MATCH_PCT'].append(round(100 * n_matched / n_searched, 2) if n_searched > 0 else 0)
    
    summary_hdu = fits.BinTableHDU(Table(summary_data), name='SUMMARY')
    hdu_list.append(summary_hdu)
    
    # ----- LINKS table: all database URLs in one place -----
    links_table = Table()
    links_table['KMOS3D_ID'] = merged_table['OBJECT']
    links_table['RA'] = merged_table['RA']
    links_table['DEC'] = merged_table['DEC']
    
    if 'SIMBAD_URL' in merged_table.colnames:
        links_table['SIMBAD_URL'] = merged_table['SIMBAD_URL']
    
    for survey_name, config in ALL_SURVEYS.items():
        prefix = config['prefix']
        url_col = prefix + 'URL'
        matched_col = prefix + 'MATCHED'
        if url_col in merged_table.colnames:
            links_table[f'{survey_name}_URL'] = merged_table[url_col]
        if matched_col in merged_table.colnames:
            links_table[f'{survey_name}_MATCHED'] = merged_table[matched_col]
    
    links_hdu = fits.BinTableHDU(links_table, name='LINKS')
    links_hdu.header['COMMENT'] = 'Direct URLs to database entries for each matched source'
    hdu_list.append(links_hdu)
    
    # Write output
    hdu_list.writeto(output_path, overwrite=True)
    print(f"  Saved to: {output_path}")
    print(f"  Total HDUs: {len(hdu_list)}")
    
    # Print structure
    print("\n  Output FITS structure:")
    for i, hdu in enumerate(hdu_list):
        if hasattr(hdu, 'data') and hdu.data is not None:
            print(f"    [{i}] {hdu.name}: {len(hdu.data) if hasattr(hdu.data, '__len__') else 'N/A'} rows")
        else:
            print(f"    [{i}] {hdu.name}: Primary")


# ============================================================================
# Main
# ============================================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python Crossmatch_generator_v3.py /path/to/KMOS3D_ALL/ output.fits")
        print("\nCrossmatches KMOS3D with deep-field surveys:")
        print("  All-sky: NED, WISE, SDSS, 2MASS, JWST/MAST")
        print("  COSMOS:  COSMOS2020, VLA-COSMOS 3GHz, Chandra COSMOS Legacy")
        print("  GOODS-S: CDF-S 7Ms X-ray, VLA E-CDFS 1.4GHz")
        print("  UDS:     X-UDS Chandra")
        print("\nFeatures:")
        print("  - Crossmatches against 7 surveys including JWST archive")
        print("  - Generates direct database URLs for every matched source")
        print("  - Proper separation column handling (SEP_ARCSEC)")
        print("  - Compatible output format for visualization script")
        sys.exit(1)
    
    kmos_folder = sys.argv[1]
    output_file = sys.argv[2]
    
    print("=" * 60)
    print("KMOS3D Multi-Survey Crossmatch Tool v3")
    print("=" * 60)
    print(f"Input folder: {kmos_folder}")
    print(f"Output file: {output_file}")
    print(f"Match radius: {MATCH_RADIUS_ARCSEC} arcsec")
    print()
    
    # Load KMOS3D data
    print("Step 1: Loading KMOS3D FITS files...")
    kmos_table = load_kmos3d_folder(kmos_folder)
    print(f"  Loaded {len(kmos_table)} sources with valid coordinates")
    print()
    
    # Query surveys
    print("Step 2: Querying external surveys...")
    survey_results = query_all_surveys(kmos_table)
    print()
    
    # Merge results
    print("Step 3: Merging crossmatch results...")
    merged = merge_survey_results(kmos_table, survey_results)
    print(f"  Final table has {len(merged.colnames)} columns")
    print()
    
    # Generate database links
    print("Step 4: Generating database links...")
    merged = generate_database_links(merged)
    print()
    
    # Create output
    print("Step 5: Creating output FITS file...")
    create_output_fits(merged, output_file, kmos_folder)
    
    # Print summary
    print()
    print("=" * 60)
    print("CROSSMATCH SUMMARY")
    print("=" * 60)
    for survey_name, config in ALL_SURVEYS.items():
        prefix = config['prefix']
        matched_col = prefix + 'MATCHED'
        if matched_col in merged.colnames:
            n_matched = int(np.sum(merged[matched_col]))
            pct = 100 * n_matched / len(merged) if len(merged) > 0 else 0
            print(f"  {survey_name:12s}: {n_matched:4d}/{len(merged)} ({pct:5.1f}%)")
    print("=" * 60)


if __name__ == '__main__':
    main()