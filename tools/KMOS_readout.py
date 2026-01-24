# Kolja Reuter
# Tools to work with the data from the KMOS IFU Survey
# 17-10-2024

# Utf-8 encoding

# Useful imports

import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from astropy import units as u
#from skimage import restoration
from photutils.isophote import EllipseGeometry, Ellipse
from scipy.optimize import curve_fit
from astropy.table import Table
import matplotlib.pyplot as plt
from tools import Fitting_Voronoi as fittingvoronoi

import re

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

# Determine the redshift of the galaxy
def redshift(hdulist1, FILE = 'k3d_fnlsp_table_v3.fits', testdata = False):
    '''Determine the redshift of the galaxy from the FITS file header and a reference FITS table.
    Parameters: hdulist1: astropy.io.fits.HDUList'''
    if testdata:
        return 0
    hdulist = fits.open(FILE)
    
    filename = hdulist1.filename()
    data = hdulist[1].data
    # Convert the FITS data table to a structured array and then to a pandas DataFrame
    structured_array = np.array(data).byteswap().newbyteorder()
    df = pd.DataFrame(structured_array)
    # Try to extract the ID from the filename in a robust way
    # Remove 'KMOS3D\' or 'KMOS3D/' or 'KMOS3D_ALL\' or 'KMOS3D_ALL/' from the filename
    ID = re.sub(r'(KMOS3D_ALL|KMOS3D)[\\/]+', '', filename)
    ID = re.sub(r'Gaussian_fits[\\/]+', '', ID)
    ID = re.sub(r'_voronoi_binned\.fits$', '', ID)
    # Remove trailing '.fits' if present
    ID = re.sub(r'\.fits$', '', ID)
    ID = str(ID) + '.fits'
    # Clean both the ID and the dataframe column
    # Convert bytes to strings first
    df['FILE'] = df['FILE'].str.decode('utf-8')

    # Now you can do string operations
    ID = ID.strip()
    matching_rows = df[df['FILE'].str.strip() == ID]
    if len(matching_rows) == 0:
        raise ValueError('Error: Filename not found in the database. Check the whitespaces', ID, df['FILE'][0])


    df_filtered = matching_rows
    #print(len(matching_rows), df_filtered)
    z = df_filtered ['Z'].iloc[0]
    if z < 0:
        z = df_filtered ['Z_TARGETED'].iloc[0]

    #print('redshift: ', z)
    return(z)

# Determine the wavelength of the Map
def get_wavelength(slice , hdulist , pr=False, testdata = False):
    z = redshift(hdulist, testdata = testdata)
    try:
        wavelength = (hdulist[1].header['CRVAL3'] * u.micrometer + hdulist[1].header['CDELT3'] * slice *  u.micrometer)/ (z+1)
    except:
        print('hehe')
        wavelength = (hdulist.header['CRVAL3'] * u.micrometer + hdulist.header['CDELT3'] * slice *  u.micrometer)/ (z+1)

    if testdata:
        wavelength = 1.0 * u.micrometer *slice 
        return wavelength
    if pr:
        wavelength_max = (hdulist[1].header['CRVAL3'] * u.micrometer + hdulist[1].header['CDELT3'] * 2047 *  u.micrometer)/ (z+1)
        wavelength_min = (hdulist[1].header['CRVAL3'] * u.micrometer)/ (z +1)
        print('The slice corresponds to wavelength '+ str(wavelength))
        print('The max wavelength is: ' + str(wavelength_max))
        print('The min wavelength is: ' + str(wavelength_min))
    return wavelength


# Get the Slice for a given redshift
def get_slice(wavelength, hdulist, pr=False, testdata = False):
    z = redshift(hdulist)
    if testdata:
        slice = wavelength / (1.0 * u.micrometer)   
    slice = (wavelength * (z + 1) - hdulist[1].header['CRVAL3'] * u.micrometer) / (hdulist[1].header['CDELT3'] * u.micrometer)
    slice = int(slice.value)
    if pr:
        print(slice)
    return slice

# Plot the wavelength map for a given wavelength and galaxy
def plot_wavelength_map(wavelength, filename = r'KNMOS3D\\U4_36227_YJ.fits'):
    hdulist = fits.open(filename)
    slice = int(get_slice(wavelength, filename, pr=True))
    data = hdulist[1].data
    slice_data = data[slice, :, :]
    hdulist.close()
    return slice_data

'''# Richardson Lucy Deconvolution
def richardson_lucy(image, psf, iterations=30):
    image[image < 0] = 0
    psf[psf < 0] = 0
    image = np.nan_to_num(image)
    psf = np.ones(psf.shape)
    deconvolved_RL = restoration.richardson_lucy(image, psf, num_iter=iterations)
    return deconvolved_RL

def determine_PSF_MASK(filename = r'KNMOS3D\\U4_36227_YJ.fits'):
    hdulist = fits.open(filename)
    # Extract the necessary information from the header
    x0 = hdulist[4].header['CRPIX1']
    y0 = hdulist[4].header['CRPIX2']
    a = hdulist[4].header['HIERARCH ESO K3D PSF GAUSS FWHM_MIN']
    b = hdulist[4].header['HIERARCH ESO K3D PSF GAUSS FWHM_MAJ']
    theta = 1

    # Define the elliptic Gaussian function
    def elliptic_gaussian(x, y):
        x_diff = x - x0
        y_diff = y - y0
        x_rot = x_diff * np.cos(theta) - y_diff * np.sin(theta)
        y_rot = x_diff * np.sin(theta) + y_diff * np.cos(theta)
        exponent = -((x_rot / a) ** 2 + (y_rot / b) ** 2)
        return np.exp(exponent)

    # Test the elliptic Gaussian function
    x = np.linspace(0, 17, 17)
    y = np.linspace(0, 16, 16)
    X, Y = np.meshgrid(x, y)
    Z = elliptic_gaussian(X, Y)
    return Z


def deconvolute_maps(map1, map2):
    # Flatten the maps to 1D arrays for deconvolution
    map1_flat = map1.flatten()
    map2_flat = map2.flatten()

    # Perform deconvolution
    deconvolved, remainder = deconvolve(map1_flat, map2_flat)
    
    # Ensure the deconvolved result is the correct size
    if deconvolved.size >= map1.size:
        deconvolved_map = deconvolved[:map1.size].reshape(map1.shape)
    else:
        # Pad the deconvolved result with zeros if it's smaller than the original map size
        padded_deconvolved = np.zeros(map1.size)
        padded_deconvolved[:deconvolved.size] = deconvolved
        deconvolved_map = padded_deconvolved.reshape(map1.shape)

    return deconvolved_map'''

def find_non_nan_mask_psf(data):
    # Mask to filter out NaN values
    mask = ~np.isnan(data)

    # Get the indices where the data is not NaN
    indices = np.argwhere(mask)

    # Find the bounding box of the non-NaN region
    min_row, min_col = indices.min(axis=0)
    max_row, max_col = indices.max(axis=0)

    # Extract the subarray that contains the non-NaN values
    zoomed_data = data[min_row:max_row+1, min_col:max_col+1]

    return zoomed_data

def psf(hdulist):
    ref_pixel_flux_x = hdulist[1].header['CRPIX1']
    ref_pixel_flux_y = hdulist[1].header['CRPIX2']
    ref_pixel_psf_x = hdulist[4].header['CRPIX1']
    ref_pixel_psf_y = hdulist[4].header['CRPIX2']
    delta_x = ref_pixel_flux_x - ref_pixel_psf_x
    delta_y = ref_pixel_flux_y - ref_pixel_psf_y

    psf_raw = hdulist[4].data
    psf = np.zeros_like(hdulist[1].data [1800])
    for i in range(hdulist[1].data[1800].shape[0]):
        for j in range(hdulist[1].data[1800].shape[1]):
            if delta_x%1 == 0 and delta_y%1 == 0:
                    psf[i][j] = psf_raw[i - delta_y][j - delta_x]
            elif delta_x%1 == 0.5 and delta_y%1 == 0.5:
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


def sersic_profile(r, I_e, r_e, n):
    #Sersic profile function.
    # r is the radius, I_e is the intensity at the effective radius, r_e is the effective radius, n is the Sersic index
    b_n = 2 * n - 0.327
    return I_e * np.exp(-b_n * ((r / r_e) ** (1 / n) - 1))

def fit_sersic_profile(isophote_list, debug=True):
    # Define the radius and intensity arrays
    radii = []
    intensities = []
    if debug:
        print(f"Fitting {len(isophote_list)} isophotes")
        print(f"SMA: {[iso.sma for iso in isophote_list]}")
        print(f"INTENS: {[iso.intens for iso in isophote_list]}")
    # Extract the radii and intensities from the isophote list
    for isophote in isophote_list:
        radii.append(isophote.sma)
        intensities.append(isophote.intens)
    
    # Define the initial guess for the Sersic profile parameters
    I_e_guess = intensities[0]
    r_e_guess = radii[1]
    n_guess = 4
    
    # Fit the Sersic profile
    try:
        popt, pcov = curve_fit(sersic_profile, radii, intensities, p0=[I_e_guess, r_e_guess, n_guess])
        if debug:
            print(f"Fit parameters: {popt}")
            return popt

    except Exception as e:
        print(f"Error in fit_sersic_profile: {e}")
        return [I_e_guess, r_e_guess, n_guess]
    
def fit_isophotes(data_raw,psf, debug=False):
    map = psf * data_raw
    # Set the negative values to NaN
    map[np.isnan(map) | (map <= 0)] = -9999
    try:
        # Define the geometry of the ellipse
        # Define the geometry of the ellipse based on the map's center and size
        y_center, x_center = np.array(map.shape) // 2
        sma = min(map.shape) // 4  # Semi-major axis length as a quarter of the smallest dimension
        geometry = EllipseGeometry(x0=x_center, y0=y_center, sma=sma, eps=0.1, pa=0)
        
        # Create an Ellipse instance with the defined geometry
        ellipse = Ellipse(map, geometry)
        
        # Fit the isophotes
        isophote_list = ellipse.fit_image(step=0.5, maxsma=10, integrmode='median')
        isophote_list = [iso for iso in isophote_list if iso.intens >= 0]
        if debug:   
           print(f"Fitted {len(isophote_list)} isophotes")
        return isophote_list
    except Exception as e:
        if debug:
            print(f"Error in fit_isophotes: {e}")
        return []
    
def sersic_profile(r, I_e, r_e, n):
    #Sersic profile function.
    # r is the radius, I_e is the intensity at the effective radius, r_e is the effective radius, n is the Sersic index
    b_n = 2 * n - 0.327
    return I_e * np.exp(-b_n * ((r / r_e) ** (1 / n) - 1))

def fit_sersic_profile(isophote_list, debug=False):
    # Define the radius and intensity arrays
    radii = []
    intensities = []
    if debug:
        print(f"Fitting {len(isophote_list)} isophotes")
        print(f"SMA: {[iso.sma for iso in isophote_list]}")
        print(f"INTENS: {[iso.intens for iso in isophote_list]}")
    # Extract the radii and intensities from the isophote list
    for isophote in isophote_list:
        radii.append(isophote.sma)
        intensities.append(isophote.intens)
    
    # Define the initial guess for the Sersic profile parameters
    I_e_guess = intensities[0]
    r_e_guess = radii[1]
    n_guess = 4
    
    # Fit the Sersic profile
    try:
        popt, pcov = curve_fit(sersic_profile, radii, intensities, p0=[I_e_guess, r_e_guess, n_guess])
        if debug:
            print(f"Fit parameters: {popt}")
            return popt

    except Exception as e:
        if debug:
            print(f"Error in fit_sersic_profile: {e}")
        return [I_e_guess, r_e_guess, n_guess]
    
def processbar(i, total):
    # Display a process bar
    print(f'\rProcessing {i}/{total}', end='')

def save_all_isophotes_in_pd_dataframe(filename, debug=False, today = 'datum_vergessen'):
    hdulist = fits.open(filename)
    print(f'Processing {filename}', hdulist[1].shape[0])
    frame = pd.DataFrame(columns=['WAVELENGTH','SMA', 'INTENS', 'PA', 'EPS'], dtype=float)
    for i in range(hdulist[1].shape[0]):
        wavelength = get_wavelength(i, hdulist)
        isophote_list = fit_isophotes(hdulist[1].data[i], psf(hdulist), debug=debug)

        for isophote in isophote_list:
            frame = pd.concat([frame, pd.DataFrame([{'WAVELENGTH':wavelength,'SMA': isophote.sma, 'INTENS': isophote.intens, 'PA': isophote.pa, 'EPS': isophote.eps}], index=[i])])
        processbar(i, hdulist[1].shape[0])
    frame.to_csv(f'isophotes/{filename}_isophote_fitting{today}.csv', index=True, sep=',', index_label='Index')
    hdulist.close()
    return frame


from scipy.ndimage import label

def process_data_cube(hdulist, fwhm=3.5):
    """Extract and prepare flux, noise, and wavelength arrays."""
    flux_cube = hdulist[1].data.copy()
    noise_cube = hdulist[2].data.copy()
    psf_value = psf(hdulist)
    flux_cube *= psf_value
    noise_cube *= psf_value
    wavelength = get_wavelength(np.arange(0, 2048), hdulist).value
    return wavelength, flux_cube, noise_cube

def master_map_w80(filename):
    hdulist = fits.open(filename)
    
    # 1. LOAD DATA
    single_popt = {
        'A': np.array(hdulist['A'].data),
        'B': np.array(hdulist['B'].data),
        'C': np.array(hdulist['C'].data),
    }
    double_popt = {
        'A1': np.array(hdulist['A1'].data), 'B1': np.array(hdulist['B1'].data), 'C1': np.array(hdulist['C1'].data),
        'A2': np.array(hdulist['A2'].data), 'B2': np.array(hdulist['B2'].data), 'C2': np.array(hdulist['C2'].data),
    }
    chi = {'chi_single': hdulist['chi1'].data, 'chi_double': hdulist['chi2'].data}
    hdulist.close()

    # 2. GET REDSHIFT & WAVELENGTH
    filename1 = filename.replace('Gaussian_fits', 'KMOS3D_ALL').replace('_voronoi_binned.fits', '.fits')
    with fits.open(filename1) as hdul_raw:
        # Crucial: Get the redshift to move from observed to rest-frame
        z_gal = redshift(hdul_raw) 
        wavelength, _, _ = process_data_cube(hdul_raw) # Assuming this is your helper

    # 3. INITIALIZE MAPS
    ny, nx = single_popt['A'].shape
    width_single_map = np.full((ny, nx), np.nan)
    width_double_map = np.full((ny, nx), np.nan)
    
    # Constants
    HALPHA_REST = 0.65628
    C_LIGHT = 299792.458 

    # 4. PIXEL LOOP
    for i in range(ny):
        for j in range(nx):
            # Skip background (Optimization)
            if single_popt['A'][i,j] < 1e-10 and double_popt['A1'][i,j] < 1e-10:
                continue
            
            # Reconstruct Models
            s_func = gaussian(wavelength, single_popt['A'][i,j], single_popt['B'][i,j], single_popt['C'][i,j])
            d_func = double_gaussian(wavelength, double_popt['A1'][i,j], double_popt['B1'][i,j], double_popt['C1'][i,j], 
                                                        double_popt['A2'][i,j], double_popt['B2'][i,j], double_popt['C2'][i,j])
            
            def get_w80_microns(func):
                csum = np.cumsum(func)
                if csum[-1] <= 0: return np.nan
                low = np.searchsorted(csum, 0.1 * csum[-1])
                high = np.searchsorted(csum, 0.9 * csum[-1])
                high = min(high, len(wavelength)-1)
                return wavelength[high] - wavelength[low]

            w_s_obs = get_w80_microns(s_func)
            w_d_obs = get_w80_microns(d_func)

            # CORRECT CONVERSION: 
            # 1. Divide by (1+z) to get the Rest-Frame width in Microns
            # 2. Then apply the Doppler formula
            if np.isfinite(w_s_obs):
                width_single_map[i,j] = (w_s_obs / (1 + z_gal)) / HALPHA_REST * C_LIGHT
            if np.isfinite(w_d_obs):
                width_double_map[i,j] = (w_d_obs / (1 + z_gal)) / HALPHA_REST * C_LIGHT

    # 5. BEST FIT SELECTION
    mask = np.abs(chi['chi_single'] - 1) < np.abs(chi['chi_double'] - 1)
    w80_master = np.where(mask, width_single_map, width_double_map)

    # 6. SPATIAL FILTER (Remove the "Floating Pixels")
    valid_mask = np.isfinite(w80_master) & (w80_master > 10)
    labeled, num_feat = label(valid_mask)
    for feat_id in range(1, num_feat + 1):
        coords = np.where(labeled == feat_id)
        if len(coords[0]) < 5: # Threshold for consistent area
            w80_master[coords] = np.nan

    return w80_master