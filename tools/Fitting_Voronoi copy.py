# -*- coding: utf-8 -*-
from fileinput import filename
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from vorbin.voronoi_2d_binning import voronoi_2d_binning # type: ignore

try:
    from tools import KMOS_readout as tools
except ImportError:
    import KMOS_readout as tools

Halpha = 0.656281
onepix = 0.00028

def gaussian(x, a, sigma, c):
    return abs(a) * np.exp(- (x - c)**2 / (2 * sigma**2))

def linear(x, a, b):
    return a * x + b

def gaussian_with_linear_offset(x, a, sigma, c, d, e):
    return gaussian(x, a, sigma, c) + linear(x, d, e)

def double_gaussian(x, a1, b1, c1, a2, b2, c2):
    # Ensure the two Gaussians overlap at least a little bit
    # Force the centers to be within 2 sigma of each other
    min_dist = 0.5 * (abs(b1) + abs(b2))
    if abs(c1 - c2) >  min_dist:
        # Move c2 closer to c1
        c2 = c1
    
    return gaussian(x, a1, b1, c1) + gaussian(x, a2, b2, c2)

def double_gaussian_nii(x, a1, b1, c1, a2, b2, c2):
    return gaussian(x, a1, b1, c1) + gaussian(x, a2, b2, c2)

def double_gaussian_with_linear_offset(x, a1, b1, c1, a2, b2, c2, d, e):
    return double_gaussian(x, a1, b1, c1, a2, b2, c2) + linear(x, d, e)



def process_data_cube(hdulist, fwhm=3.5):
    flux_cube = hdulist[1].data.copy()
    noise_cube = hdulist[2].data.copy()
    psf = tools.psf(hdulist)

    flux_cube *= psf
    noise_cube *= psf

    wavelength = tools.get_wavelength(np.arange(0, 2048), hdulist).value

    
    
    # Smooth the cube with a Gaussian kernel
    # Not necessary spacial due to the fact that the spacial cleaning is done via psf-multiplication

    sigma_pix = fwhm / (2 * np.sqrt(2 * np.log(2)))
    sigma_spectral = hdulist[1].header.get('CDELT3', 1.0)

    flux_smooth = flux_cube # gaussian_filter(flux_cube, sigma=(sigma_spectral, 0, 0))
    noise_smooth = noise_cube#gaussian_filter(noise_cube, sigma=(sigma_spectral, 0, 0))

    return wavelength, flux_smooth, noise_smooth
    '''
    return wavelength, flux_cube, noise_cube
    '''

def background(x, y, sigma=None):
    mask = (x < 0.65) | (x > 0.662)
    x_linear = x[mask].copy()
    y_linear = y[mask].copy()
    if sigma is not None:
        sigma = sigma[mask]

    # Sigma-clipping

    if sigma is not None:
        residuals = y_linear - linear(x_linear, *(np.polyfit(x_linear, y_linear, 1)))
        std = np.std(residuals)
        clip_mask = np.abs(residuals) < 3 * std
        x_linear = x_linear[clip_mask]
        y_linear = y_linear[clip_mask]
        sigma = sigma[clip_mask]
    else:
        residuals = y_linear - linear(x_linear, *(np.polyfit(x_linear, y_linear, 1)))
        std = np.std(residuals)
        clip_mask = np.abs(residuals) < 3 * std
        x_linear = x_linear[clip_mask]
        y_linear = y_linear[clip_mask]

    popt, pcov = curve_fit(linear, x_linear, y_linear, p0=[y[0]-y[-1], np.mean(y)], sigma=sigma)

    return popt, pcov

def gaussian_fit(x, y, error, sigma=None, give_pcov=False):
    if sigma is None:
        sigma = error

    # Background subtraction
    bg_popt, bg_pcov = background(x, y, sigma)
    mask = (x > Halpha - 0.01) & (x < Halpha + 0.01) & np.isfinite(y) & np.isfinite(error) & ~np.isnan(y) & ~np.isnan(error)
    x_fit = x[mask]
    y_corrected = y[mask]
    error_fit = error[mask]
    
    nii_popt, qual_nii = nii_fit(x_fit, y_corrected, error_fit)
    red_chi2_with_nii = np.inf
    chi2_with_nii = np.inf
    
    if ~np.isnan(nii_popt[0]):
        y_fit = y_corrected - double_gaussian_with_linear_offset(x_fit, *nii_popt)
        try:
            popt_onlygauss, pcov_onlygauss = curve_fit(
                gaussian, x_fit, y_fit,
                p0=[max(np.abs(y_fit)), onepix*2, Halpha],
                bounds=([np.mean(np.abs(y_fit)), onepix/4, 0.655],
                        [np.max(np.abs(y_fit)), onepix*20, 0.6575]),
                sigma=error_fit, maxfev=5000
            )
            popt_with_nii = np.append(popt_onlygauss, bg_popt)
            model = gaussian(x_fit, *popt_onlygauss)
            residuals = (y_fit - model) / error_fit 
            chi2_with_nii = np.sum((residuals**2))
            dof_with_nii = max(len(x_fit) - len(popt_onlygauss), 1)
            red_chi2_with_nii = chi2_with_nii / dof_with_nii
            
            # IMPROVED S/N: Use integrated flux over noise in fitting region
            # This is more robust than peak/mean_error
            integrated_flux = np.trapz(model, x_fit)
            noise_in_line = np.sqrt(np.sum(error_fit**2)) * (x_fit[1] - x_fit[0])  # Integrated noise
            snr_with_nii = integrated_flux / noise_in_line
            
        except Exception as e:
            print(Exception, e)
            return np.full(5, np.nan), [np.nan, np.inf]

    y_fit = y_corrected - linear(x_fit, *bg_popt)
    try:
        popt_onlygauss, pcov_onlygauss = curve_fit(
            gaussian, x_fit, y_fit,
            p0=[max(np.abs(y_fit)), onepix*2, Halpha],
            bounds=([np.mean(y_fit), onepix/4, 0.6555],
                    [np.max(np.abs(y_fit)), onepix*20, 0.6575]),
            sigma=error_fit, maxfev=5000
        )
        popt_with_bg = np.append(popt_onlygauss, bg_popt)
        pcov_with_bg = np.zeros((len(popt_with_bg), len(popt_with_bg))) #### Repair here
        model = gaussian(x_fit, *popt_onlygauss)
        residuals = (y_fit - model) / error_fit 
        chi2_with_bg = np.sum((residuals**2))
        dof_with_bg = max(len(x_fit) - len(popt_onlygauss), 1)
        red_chi2_with_bg = chi2_with_bg / dof_with_bg
        
        # IMPROVED S/N: Use integrated flux over noise
        integrated_flux = np.trapz(model, x_fit)
        noise_in_line = np.sqrt(np.sum(error_fit**2)) * (x_fit[1] - x_fit[0])
        snr_with_bg = integrated_flux / noise_in_line
        
    except Exception as e:
        print(Exception, e)
        return np.full(5, np.nan), [np.nan, np.inf]
    
    if red_chi2_with_bg < red_chi2_with_nii:
        if give_pcov:
            return popt_with_bg, pcov_with_bg, [snr_with_bg, red_chi2_with_bg], False
        return popt_with_bg, [snr_with_bg, red_chi2_with_bg], False
    else:
        if give_pcov:
            return popt_with_nii, [snr_with_nii, red_chi2_with_nii], True
        return popt_with_nii, [snr_with_nii, red_chi2_with_nii], True


def nii_fit(x, y, error, p0=None, sigma=None):
    if p0 is None:
        p0 = [abs(np.mean(y)), onepix*2, 0.6583, 1/3*abs(np.mean(y)), onepix*2, 0.6548]
    if sigma is None:
        sigma = error

    mask = (x > Halpha - 0.1) & (x < Halpha + 0.2) & np.isfinite(y) & np.isfinite(error) & ~np.isnan(y) & ~np.isnan(error)
    x_fit = x[mask]
    y_corrected = y[mask]
    error_fit = error[mask]

    popt_bg, pcov_bg = background(x, y, sigma)
    y_fit = y_corrected - linear(x_fit, *popt_bg)

    if len(x_fit) < 3:
        print('Not enough data points for fitting')
        return np.full(8, np.nan), [np.nan, np.inf]

    try:
        popt_onlynii, pcov_onlynii = curve_fit(
            double_gaussian_nii, x_fit, y_fit, p0=p0,
            bounds=([0, onepix/4, 0.658, 0, onepix/4, 0.654],
            [max(np.abs((y_fit))), onepix*3, 0.659, 1/3*max(np.abs(y_fit)), onepix*3, 0.655]),
            sigma=error_fit, maxfev=5000
        )
        model = double_gaussian(x_fit, *popt_onlynii)
        residuals = (y_fit - model) / error_fit
        chi2 = np.sum((residuals ** 2))
        dof = max(len(x_fit) - len(popt_onlynii), 1)
        red_chi2 = chi2 / dof
        
        # IMPROVED S/N: Use integrated flux for both NII lines
        integrated_flux = np.trapz(model, x_fit)
        noise_in_line = np.sqrt(np.sum(error_fit**2)) * (x_fit[1] - x_fit[0])
        snr = integrated_flux / noise_in_line
        
        popt = np.append(popt_onlynii, popt_bg)

        # Adjusted thresholds for integrated S/N (typically higher than peak S/N)
        snr_threshold = 5  # Increased from 3
        red_chi2_max = 10
        if snr < snr_threshold or red_chi2 > red_chi2_max:
            popt[:] = 0, onepix*2, 0.6583, 0, onepix*2, 0.6548, popt_bg[0], popt_bg[1]

        return popt, [snr, red_chi2]
    except RuntimeError:
        return np.full(8, np.nan), [np.nan, np.inf]


def double_gaussian_fit(x, y, error, sigma=None, give_pcov=False):
    if sigma is None:
        sigma = error

    # Background subtraction
    bg_popt, bg_pcov = background(x, y, sigma)
    mask = (x > Halpha - 0.01) & (x < Halpha + 0.01) & np.isfinite(y) & np.isfinite(error) & ~np.isnan(y) & ~np.isnan(error)
    x_fit = x[mask]
    y_corrected = y[mask]
    error_fit = error[mask]
    nii_popt, qual_nii = nii_fit(x_fit, y_corrected, error_fit)

    best_popt_with_nii = None
    best_chi2_with_nii = np.inf
    best_popt_with_bg = None
    best_chi2_with_bg = np.inf
    best_dof_with_nii = 1
    best_dof_with_bg = 1
    best_snr_with_nii = 0
    best_snr_with_bg = 0

    if ~np.isnan(nii_popt[0]):
        y_fit = y_corrected - double_gaussian_with_linear_offset(x_fit, *nii_popt)

        if len(x_fit) < 5 or np.sum(np.isfinite(y_fit)) < len(y_fit) - 5:
            print('Repair here')
            return np.full(8, np.nan), [np.nan, np.inf], False

        for b2 in np.linspace(onepix+1e-5, onepix*15, 1):
            initial = [np.max(np.abs(y_fit)), onepix*2, Halpha, 
                      0.7 * np.max(np.abs(y_fit)), b2, Halpha]
            lower = [np.mean(np.abs(y_fit))/2, onepix/4, 0.6555,
                    np.mean(np.abs(y_fit))/2, onepix/4, 0.6555]
            upper = [np.max(np.abs(y_fit)), onepix*20, 0.675,
                    np.max(np.abs(y_fit)), onepix*20, 0.675]

            initial_model = double_gaussian(x_fit, *initial)
            if not np.all(np.isfinite((y_fit - initial_model) / error_fit)):
                continue

            try:
                popt_onlydouble, pcov_onlydouble = curve_fit(
                    double_gaussian, x_fit, y_fit,
                    p0=initial, bounds=(lower, upper), sigma=error_fit,
                    maxfev=50000
                )
            except RuntimeError as e:
                continue

            model = double_gaussian(x_fit, *popt_onlydouble)
            residuals = (y_fit - model) / error_fit 
            chi2 = np.sum((residuals ** 2))
            dof = max(len(x_fit) - len(popt_onlydouble), 1)

            if chi2 < best_chi2_with_nii:
                best_chi2_with_nii = chi2
                best_dof_with_nii = dof
                best_popt_with_nii = np.append(popt_onlydouble, bg_popt)
                
                # IMPROVED S/N: Use integrated flux for double Gaussian
                integrated_flux = np.trapz(model, x_fit)
                noise_in_line = np.sqrt(np.sum(error_fit**2)) * (x_fit[1] - x_fit[0])
                best_snr_with_nii = integrated_flux / noise_in_line

    # Same improvements for background-only fitting
    y_fit = y_corrected - linear(x_fit, *bg_popt)

    for b2 in np.linspace(onepix/2+1e-5, onepix*15, 1):
        initial = [np.max(np.abs(y_fit)), onepix*2, Halpha, 
                  0.7 * np.max(np.abs(y_fit)), b2, Halpha]
        lower = [0, onepix/4, 0.6555, 0, onepix/4, 0.6555]
        upper = [np.max(np.abs(y_fit)), onepix*25, 0.675,
                np.max(np.abs(y_fit)), onepix*25, 0.675]
        
        initial_model = double_gaussian(x_fit, *initial)
        if not np.all(np.isfinite((y_fit - initial_model) / error_fit)):
            continue

        try:
            popt_onlydouble, pcov_onlydouble = curve_fit(
                double_gaussian, x_fit, y_fit,
                p0=initial, bounds=(lower, upper), sigma=error_fit,
                maxfev=50000
            )
        except RuntimeError:
            continue

        model = double_gaussian(x_fit, *popt_onlydouble)
        residuals = (y_fit - model) / error_fit 
        chi2 = np.sum((residuals ** 2))
        dof = max(len(x_fit) - len(popt_onlydouble), 1)

        if chi2 < best_chi2_with_bg:
            best_chi2_with_bg = chi2
            best_dof_with_bg = dof
            best_popt_with_bg = np.append(popt_onlydouble, bg_popt)
            
            # IMPROVED S/N: Use integrated flux
            integrated_flux = np.trapz(model, x_fit)
            noise_in_line = np.sqrt(np.sum(error_fit**2)) * (x_fit[1] - x_fit[0])
            best_snr_with_bg = integrated_flux / noise_in_line

    if (best_popt_with_nii is None) and (best_popt_with_bg is None):
        print('something is wrong here')
        return np.full(8, np.nan), [np.nan, np.inf], False

    if best_chi2_with_nii/best_dof_with_nii < best_chi2_with_bg/best_dof_with_bg: 
        if give_pcov:
            return best_popt_with_nii, None, [best_snr_with_nii, best_chi2_with_nii/best_dof_with_nii], True
        return best_popt_with_nii, [best_snr_with_nii, best_chi2_with_nii/best_dof_with_nii], True
    else:
        if give_pcov:
            return best_popt_with_bg, None, [best_snr_with_bg, best_chi2_with_bg/best_dof_with_bg], False
        return best_popt_with_bg, [best_snr_with_bg, best_chi2_with_bg/best_dof_with_bg], False

def nii_fit_and_save(hdulist, snr_threshold=3):
    wavelength, cube, error_cube = process_data_cube(hdulist)
    psf = tools.psf(hdulist)
    results = []

    for i in range(cube.shape[1]):
        for j in range(cube.shape[2]):
            if psf[i, j] == 0:
                print(f"Skipping pixel ({i}, {j}) due to zero PSF")
                results.append({
                    'pixel_x': i, 'pixel_y': j,
                    'a1_nii': np.nan, 'b1_nii': np.nan, 'c1_nii': np.nan,'a2_nii': np.nan, 'b2_nii': np.nan, 'c2_nii': np.nan,'d': np.nan, 'e': np.nan, 
                    'snr_nii': np.nan, 'chi_nii': np.nan
                    })
                continue

            flux = cube[:, i, j] 
            err = error_cube[:, i, j]

            mask = ~np.isnan(flux) & ~np.isnan(err)

            if np.sum(mask) < 2:
                #print(f"Not enough data points for fitting at pixel ({i}, {j}, {np.sum(mask)}) ")
                results.append({
                    'pixel_x': i, 'pixel_y': j,
                    'a1_nii': np.nan, 'b1_nii': np.nan, 'c1_nii': np.nan,'a2_nii': np.nan, 'b2_nii': np.nan, 'c2_nii': np.nan, 'd': np.nan, 'e': np.nan, 
                    'snr_nii': np.nan, 'chi_nii': np.nan
                    })
                continue


            popt, quality = nii_fit(wavelength[mask], flux[mask], err[mask])

            snr, chi2 = quality
            if snr < snr_threshold:
                #print(f"Low SNR at pixel ({i}, {j}): {snr}")
                popt[:] = np.nan

            results.append({
                'pixel_x': i, 'pixel_y': j,
                'a1_nii': popt[0], 'b1_nii': popt[1], 'c1_nii': popt[2],'a2':popt[3], 'b2':popt[4] ,'c2': popt[5],'d':popt[6], 'e':popt[7],
                'snr_nii': snr, 'chi_nii': chi2
            })

    return pd.DataFrame(results)

def single_gaussian_fit_and_save(hdulist):
    wavelength, cube, error_cube = process_data_cube(hdulist)
    psf = tools.psf(hdulist)
    fit_results = []

    for i in range(cube.shape[1]):
        for j in range(cube.shape[2]):
            if psf[i, j] == 0:
                fit_results.append({**dict.fromkeys(['a','b','c','d','e','son1','chi1'], np.nan), 'pixel_x': i, 'pixel_y': j})
                continue

            flux = cube[:, i, j]
            err = error_cube[:, i, j]
            mask = ~np.isnan(flux) & ~np.isnan(err)

            if np.sum(mask) < 5:
                fit_results.append({**dict.fromkeys(['a','b','c','d','e','son1','chi1'], np.nan), 'pixel_x': i, 'pixel_y': j})
                continue

            popt, quality, nii_used = gaussian_fit(wavelength[mask], flux[mask], err[mask])
            if np.isnan(popt).any():
                continue

            fit_results.append({
                'pixel_x': i, 'pixel_y': j,
                'a': popt[0], 'b': popt[1], 'c': popt[2],
                'd': popt[3], 'e': popt[4],
                'son1': quality[0], 'chi1': quality[1]
            })

    return pd.DataFrame(fit_results)

def double_gaussian_fit_and_save(hdulist):
    wavelength, cube, error_cube = process_data_cube(hdulist)
    psf = tools.psf(hdulist)
    fit_results = []

    for i in range(cube.shape[1]):
        for j in range(cube.shape[2]):
            if psf[i, j] == 0:
                fit_results.append({**dict.fromkeys(['a1','b1','c1','a2','b2','c2','d','e','son2','chi2'], np.nan), 'pixel_x': i, 'pixel_y': j})
                continue

            flux = cube[:, i, j]
            err = error_cube[:, i, j]
            mask = ~np.isnan(flux) & ~np.isnan(err)

            if np.sum(mask) < 5:
                fit_results.append({**dict.fromkeys(['a1','b1','c1','a2','b2','c2','d','e','son2','chi2'], np.nan), 'pixel_x': i, 'pixel_y': j})
                continue

            result = double_gaussian_fit(wavelength[mask], flux[mask], err[mask])
            if result is None or any(r is None for r in result):
                continue
            popt,  quality = result
            if np.isnan(popt).any():
                continue
            fit_results.append({
                'pixel_x': i, 'pixel_y': j,
                'a1': popt[0], 'b1': popt[1], 'c1': popt[2],
                'a2': popt[3], 'b2': popt[4], 'c2': popt[5],
                'd': popt[6], 'e': popt[7],
                'son2': quality[0], 'chi2': quality[1]
            })

    return pd.DataFrame(fit_results)

def voronoi_binned_maps(filename, target_snr=5):
    filename1 = filename.replace('Gaussian_fits', 'KMOS3D_ALL')
    filename1 = filename1.replace('_voronoi_binned.fits', '.fits')

    hdulist = fits.open(filename)
    single_results = single_gaussian_fit_and_save(hdulist)
    shape = hdulist[1].data.shape[1:]
    hdulist.close()

    # Create empty maps
    a_map = np.full(shape, np.nan)
    son_map = np.full(shape, np.nan)
    chi_map = np.full(shape, np.nan)

    for _, row in single_results.iterrows():
        a_map[int(row.pixel_x), int(row.pixel_y)] = row.a
        son_map[int(row.pixel_x), int(row.pixel_y)] = row.son1
        chi_map[int(row.pixel_x), int(row.pixel_y)] = row.chi1

    hdulist1 = fits.open(filename1)
    wavelength, signal_cube, error_cube = process_data_cube(hdulist1)
    
    # IMPROVED: Use broader wavelength range for better S/N estimate
    mask_wave = (wavelength > Halpha - 0.005) & (wavelength < Halpha + 0.005)
    
    # IMPROVED: Use RMS noise instead of mean, and integrated signal
    noise_flat = np.sqrt(np.nanmean(error_cube[mask_wave, :, :]**2, axis=0)).flatten()
    # Use integrated signal over the wavelength range instead of just max
    signal_flat = np.nansum(np.maximum(signal_cube[mask_wave, :, :], 0), axis=0).flatten()
    
    hdulist1.close()
    x_flat = np.repeat(np.arange(shape[0]), shape[1])
    y_flat = np.tile(np.arange(shape[1]), shape[0])

    # Ensure noise_flat is positive and finite
    valid_noise = np.isfinite(noise_flat) & (noise_flat > 0)
    mask = np.isfinite(signal_flat) & valid_noise & ~np.isnan(son_map.flatten()) 
    x_flat = x_flat[mask]
    y_flat = y_flat[mask]
    signal_flat = signal_flat[mask]
    noise_flat = noise_flat[mask]

    # Debug: Print S/N statistics
    input_snr = signal_flat / noise_flat
    print(f"Input S/N range: {np.nanmin(input_snr):.2f} - {np.nanmax(input_snr):.2f}")
    print(f"Median input S/N: {np.nanmedian(input_snr):.2f}")

    try:
        bin_num, x_bin, y_bin, XBAR, YBAR, sn_bin, bin_pix, Scale = voronoi_2d_binning(
            x_flat, y_flat, signal_flat, noise_flat,
            target_snr, wvt=False, plot=False, quiet=True
        )
        
        print(f"Binned S/N range: {np.nanmin(sn_bin):.2f} - {np.nanmax(sn_bin):.2f}")
        print(f"Median binned S/N: {np.nanmedian(sn_bin):.2f}")
        
        bin_num_2d = np.full(shape, np.nan)
        bin_num_2d_flat = np.full(np.prod(shape), np.nan)
        bin_num_2d_flat[np.flatnonzero(mask)] = bin_num
        bin_num_2d = bin_num_2d_flat.reshape(shape)
    except Exception as e:
        print("Voronoi binning failed:", e)
        bin_num_2d = np.full(shape, np.nan)

    # Initialize final maps
    voronoi_maps = {key: np.full(shape, np.nan) for key in [
        'a', 'b', 'c', 'd', 'e', 'son1', 'chi1',
        'a1', 'b1', 'c1', 'a2', 'b2', 'c2', 'chi2', 'son2',
        'a1_nii', 'b1_nii', 'c1_nii',
        'a2_nii', 'b2_nii', 'c2_nii', 'snr_nii', 'chi_nii', 'bin_num',
        'StoN','NII_single_used', 'NII_double_used'
    ]}

    # Load datacube
    hdulist = fits.open(filename1)
    flux_cube = hdulist[1].data
    error_cube = hdulist[2].data
    psf = tools.psf(hdulist)
    wavelength = tools.get_wavelength(np.arange(flux_cube.shape[0]), hdulist).value

    # Fill maps from results using stacked spectra per Voronoi bin
    for idx, bin_id in enumerate(np.unique(bin_num_2d[~np.isnan(bin_num_2d)])):
        bin_mask = bin_num_2d == bin_id
        bin_indices = np.where(bin_mask)

        # Stack spectra for all pixels in the bin
        stacked_flux = np.nansum([flux_cube[:, i, j] * psf[i, j] for i, j in zip(*bin_indices)], axis=0)
        stacked_error = np.sqrt(np.nansum([(error_cube[:, i, j] * psf[i, j])**2 for i, j in zip(*bin_indices)], axis=0))

        valid = ~np.isnan(stacked_flux) & ~np.isnan(stacked_error) & (stacked_error > 0)

        popt_single, quality_single, Nii_single_used = gaussian_fit(wavelength[valid], stacked_flux[valid], stacked_error[valid])
        popt_double, quality_double, Nii_double_used = double_gaussian_fit(wavelength[valid], stacked_flux[valid], stacked_error[valid])
        popt_nii, quality_nii = nii_fit(wavelength[valid], stacked_flux[valid], stacked_error[valid])
        son1, chi1 = quality_single
        son2, chi2 = quality_double
        son_nii, chi_nii = quality_nii
        voronoi_maps['bin_num'][bin_mask] = bin_id
        voronoi_maps['StoN'][bin_mask] = sn_bin[idx]
        voronoi_maps['NII_single_used'][bin_mask] = Nii_single_used
        voronoi_maps['NII_double_used'][bin_mask] = Nii_double_used

        # Fill the Voronoi maps with the fitted parameters

        for key, val in zip(['a', 'b', 'c', 'd', 'e', 'son1', 'chi1'], list(popt_single) + [son1, chi1]):
            voronoi_maps[key][bin_mask] = val
        for key, val in zip(['a1', 'b1', 'c1', 'a2', 'b2', 'c2','d', 'e', 'son2', 'chi2'], list(popt_double) + [son2, chi2]):
            voronoi_maps[key][bin_mask] = val
        for key, val in zip(['a1_nii', 'b1_nii', 'c1_nii','a2_nii', 'b2_nii', 'c2_nii','d', 'e', 'snr_nii', 'chi_nii'], list(popt_nii) + [son_nii, chi_nii]):
            voronoi_maps[key][bin_mask] = val
    
    hdulist.close()
    df_voronoi = {key: voronoi_maps[key] for key in voronoi_maps}
    return df_voronoi
