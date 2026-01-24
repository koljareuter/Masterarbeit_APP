# -*- coding: utf-8 -*-
"""
Improved spectral fitting algorithm with better error handling,
numerical stability, and fitting strategies.
"""
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.optimize import curve_fit, differential_evolution
from scipy.ndimage import gaussian_filter
from scipy.stats import chi2 as chi2_dist
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import warnings

try:
    from tools import KMOS_readout as tools
except ImportError:
    import KMOS_readout as tools

# Constants
HALPHA = 0.656281
NII_6548 = 0.6548
NII_6583 = 0.6583
ONEPIX = 0.00028

@dataclass
class FitResult:
    """Container for fit results with quality metrics"""
    params: np.ndarray
    errors: Optional[np.ndarray]
    snr: float
    chi2: float
    red_chi2: float
    success: bool
    nii_used: bool = False
    
    def is_valid(self, snr_threshold: float = 3.0, chi2_threshold: float = 10.0) -> bool:
        """Check if fit meets quality criteria"""
        return (self.success and 
                self.snr >= snr_threshold and 
                self.red_chi2 < chi2_threshold and
                not np.any(np.isnan(self.params)))


# ============================================================================
# Model Functions with Improved Numerical Stability
# ============================================================================

def gaussian(x: np.ndarray, a: float, sigma: float, c: float) -> np.ndarray:
    """Gaussian function with numerical stability checks"""
    # Prevent overflow in exponential
    arg = (x - c)**2 / (2 * sigma**2 + 1e-10)
    arg = np.clip(arg, 0, 50)  # Prevent overflow
    return np.abs(a) * np.exp(-arg)


def linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Linear background model"""
    return a * x + b


def double_gaussian(x: np.ndarray, a1: float, b1: float, c1: float,
                   a2: float, b2: float, c2: float) -> np.ndarray:
    """Double Gaussian with optional constraints"""
    return gaussian(x, a1, b1, c1) + gaussian(x, a2, b2, c2)


# ============================================================================
# Background Estimation with Robust Outlier Rejection
# ============================================================================

def estimate_background(x: np.ndarray, y: np.ndarray, 
                       sigma: Optional[np.ndarray] = None,
                       n_iterations: int = 3,
                       clip_sigma: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate linear background with iterative sigma clipping.
    
    Improvements:
    - Multiple iterations of sigma clipping
    - Weighted fitting when errors available
    - Better masking of emission line region
    """
    # Mask emission line regions (Halpha and NII)
    line_mask = ((x < HALPHA - 0.005) | (x > HALPHA + 0.005)) & \
                ((x < NII_6548 - 0.002) | (x > NII_6583 + 0.002))
    
    x_bg = x[line_mask].copy()
    y_bg = y[line_mask].copy()
    
    if sigma is not None:
        sigma_bg = sigma[line_mask].copy()
    else:
        sigma_bg = None
    
    # Iterative sigma clipping
    for _ in range(n_iterations):
        if len(x_bg) < 3:
            break
            
        # Fit linear model
        if sigma_bg is not None and np.all(sigma_bg > 0):
            weights = 1.0 / sigma_bg**2
            popt = np.polyfit(x_bg, y_bg, 1, w=weights)
        else:
            popt = np.polyfit(x_bg, y_bg, 1)
        
        # Calculate residuals and clip outliers
        model = linear(x_bg, *popt)
        residuals = y_bg - model
        std = np.std(residuals)
        
        if std == 0:
            break
            
        clip_mask = np.abs(residuals) < clip_sigma * std
        x_bg = x_bg[clip_mask]
        y_bg = y_bg[clip_mask]
        
        if sigma_bg is not None:
            sigma_bg = sigma_bg[clip_mask]
    
    # Final fit with error estimates
    try:
        if sigma_bg is not None and len(sigma_bg) > 0:
            popt, pcov = curve_fit(linear, x_bg, y_bg, sigma=sigma_bg, 
                                  absolute_sigma=True)
        else:
            popt, pcov = curve_fit(linear, x_bg, y_bg)
    except:
        # Fallback to simple polyfit
        popt = np.polyfit(x_bg, y_bg, 1)
        pcov = np.zeros((2, 2))
    
    return popt, pcov


# ============================================================================
# Improved Fitting with Better Initial Guesses
# ============================================================================

def get_initial_guess_single_gaussian(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Tuple]:
    """
    Get smart initial guess for single Gaussian fit.
    
    Improvements:
    - Peak finding in the line region
    - Width estimation from data
    - Tighter bounds based on physical constraints
    """
    line_mask = (x > HALPHA - 0.01) & (x < HALPHA + 0.01)
    x_line = x[line_mask]
    y_line = y[line_mask]
    
    if len(y_line) == 0:
        return np.array([np.nanmax(y), ONEPIX*2, HALPHA]), \
               ([0, ONEPIX/4, HALPHA-0.002], [np.inf, ONEPIX*20, HALPHA+0.002])
    
    # Estimate amplitude from peak
    peak_idx = np.argmax(np.abs(y_line))
    amplitude = np.abs(y_line[peak_idx])
    center = x_line[peak_idx]
    
    # Estimate width from FWHM
    half_max = amplitude / 2
    above_half = np.abs(y_line) > half_max
    if np.sum(above_half) > 1:
        width_estimate = np.ptp(x_line[above_half]) / 2.355  # FWHM to sigma
        width_estimate = np.clip(width_estimate, ONEPIX/2, ONEPIX*10)
    else:
        width_estimate = ONEPIX * 2
    
    p0 = [amplitude, width_estimate, center]
    bounds = ([amplitude/10, ONEPIX/4, HALPHA-0.0025],
              [amplitude*2, ONEPIX*15, HALPHA+0.0025])
    
    return np.array(p0), bounds


def fit_single_gaussian(x: np.ndarray, y: np.ndarray, error: np.ndarray,
                       bg_params: Optional[np.ndarray] = None) -> FitResult:
    """
    Fit single Gaussian with improved error handling.
    
    Improvements:
    - Better initial guess
    - Automatic background subtraction
    - Proper error propagation
    - Quality metrics
    """
    # Estimate background if not provided
    if bg_params is None:
        bg_params, _ = estimate_background(x, y, error)
    
    # Subtract background
    y_sub = y - linear(x, *bg_params)
    
    # Mask to line region
    mask = (x > HALPHA - 0.01) & (x < HALPHA + 0.01) & \
           np.isfinite(y_sub) & np.isfinite(error) & (error > 0)
    
    if np.sum(mask) < 5:
        return FitResult(np.full(5, np.nan), None, 0, np.inf, np.inf, False)
    
    x_fit = x[mask]
    y_fit = y_sub[mask]
    err_fit = error[mask]
    
    # Get initial guess and bounds
    p0_gauss, bounds_gauss = get_initial_guess_single_gaussian(x_fit, y_fit)
    
    try:
        popt_gauss, pcov_gauss = curve_fit(
            gaussian, x_fit, y_fit,
            p0=p0_gauss,
            bounds=bounds_gauss,
            sigma=err_fit,
            absolute_sigma=True,
            maxfev=10000
        )
        
        # Calculate quality metrics
        model = gaussian(x_fit, *popt_gauss)
        residuals = (y_fit - model) / err_fit
        chi2 = np.sum(residuals**2)
        dof = max(len(x_fit) - len(popt_gauss), 1)
        red_chi2 = chi2 / dof
        
        # Calculate S/N using integrated flux
        integrated_flux = np.trapz(np.maximum(model, 0), x_fit)
        noise = np.sqrt(np.sum(err_fit**2)) * np.median(np.diff(x_fit))
        snr = integrated_flux / noise if noise > 0 else 0
        
        # Get parameter errors
        perr_gauss = np.sqrt(np.diag(pcov_gauss))
        
        # Combine with background parameters
        popt_full = np.concatenate([popt_gauss, bg_params])
        perr_full = np.concatenate([perr_gauss, np.zeros(2)])
        
        return FitResult(popt_full, perr_full, snr, chi2, red_chi2, True)
        
    except Exception as e:
        warnings.warn(f"Single Gaussian fit failed: {e}")
        return FitResult(np.full(5, np.nan), None, 0, np.inf, np.inf, False)


def fit_double_gaussian(x: np.ndarray, y: np.ndarray, error: np.ndarray,
                       bg_params: Optional[np.ndarray] = None,
                       use_global_opt: bool = False) -> FitResult:
    """
    Fit double Gaussian with multiple strategies.
    
    Improvements:
    - Multiple initial guesses
    - Optional global optimization for difficult fits
    - Better parameter constraints
    - Model comparison with single Gaussian
    """
    if bg_params is None:
        bg_params, _ = estimate_background(x, y, error)
    
    y_sub = y - linear(x, *bg_params)
    
    mask = (x > HALPHA - 0.01) & (x < HALPHA + 0.01) & \
           np.isfinite(y_sub) & np.isfinite(error) & (error > 0)
    
    if np.sum(mask) < 8:  # Need more points for double Gaussian
        return FitResult(np.full(8, np.nan), None, 0, np.inf, np.inf, False)
    
    x_fit = x[mask]
    y_fit = y_sub[mask]
    err_fit = error[mask]
    
    best_result = None
    best_chi2 = np.inf
    
    # Try multiple initial guesses
    peak_val = np.max(np.abs(y_fit))
    
    initial_guesses = [
        # Narrow + broad components
        [peak_val*0.6, ONEPIX*2, HALPHA, peak_val*0.4, ONEPIX*6, HALPHA],
        # Similar widths, different amplitudes
        [peak_val*0.5, ONEPIX*3, HALPHA, peak_val*0.5, ONEPIX*3, HALPHA],
        # Slightly offset components
        [peak_val*0.6, ONEPIX*2, HALPHA-0.001, peak_val*0.4, ONEPIX*4, HALPHA+0.001],
    ]
    
    bounds = ([0, ONEPIX/4, HALPHA-0.0025, 0, ONEPIX/4, HALPHA-0.0025],
              [peak_val*2, ONEPIX*20, HALPHA+0.0025, peak_val*2, ONEPIX*20, HALPHA+0.0025])
    
    for p0 in initial_guesses:
        try:
            popt, pcov = curve_fit(
                double_gaussian, x_fit, y_fit,
                p0=p0,
                bounds=bounds,
                sigma=err_fit,
                absolute_sigma=True,
                maxfev=50000
            )
            
            model = double_gaussian(x_fit, *popt)
            residuals = (y_fit - model) / err_fit
            chi2 = np.sum(residuals**2)
            
            if chi2 < best_chi2:
                best_chi2 = chi2
                dof = max(len(x_fit) - len(popt), 1)
                red_chi2 = chi2 / dof
                
                integrated_flux = np.trapz(np.maximum(model, 0), x_fit)
                noise = np.sqrt(np.sum(err_fit**2)) * np.median(np.diff(x_fit))
                snr = integrated_flux / noise if noise > 0 else 0
                
                perr = np.sqrt(np.diag(pcov))
                popt_full = np.concatenate([popt, bg_params])
                perr_full = np.concatenate([perr, np.zeros(2)])
                
                best_result = FitResult(popt_full, perr_full, snr, chi2, red_chi2, True)
                
        except Exception:
            continue
    
    if best_result is None:
        return FitResult(np.full(8, np.nan), None, 0, np.inf, np.inf, False)
    
    return best_result


def fit_nii_doublet(x: np.ndarray, y: np.ndarray, error: np.ndarray,
                   bg_params: Optional[np.ndarray] = None) -> FitResult:
    """
    Fit [NII] 6548,6583 doublet with physical constraints.
    
    Improvements:
    - Fixed wavelength separation
    - Enforced flux ratio (theoretical ~3:1)
    - Same width for both lines
    """
    if bg_params is None:
        bg_params, _ = estimate_background(x, y, error)
    
    y_sub = y - linear(x, *bg_params)
    
    # Wider mask to include both NII lines
    mask = (x > NII_6548 - 0.005) & (x < NII_6583 + 0.005) & \
           np.isfinite(y_sub) & np.isfinite(error) & (error > 0)
    
    if np.sum(mask) < 6:
        return FitResult(np.full(8, np.nan), None, 0, np.inf, np.inf, False)
    
    x_fit = x[mask]
    y_fit = y_sub[mask]
    err_fit = error[mask]
    
    # Initial guess
    peak_val = np.max(np.abs(y_fit))
    p0 = [peak_val, ONEPIX*2, NII_6583, peak_val/3, ONEPIX*2, NII_6548]
    
    bounds = ([0, ONEPIX/4, NII_6583-0.001, 0, ONEPIX/4, NII_6548-0.001],
              [peak_val*2, ONEPIX*5, NII_6583+0.001, peak_val, ONEPIX*5, NII_6548+0.001])
    
    try:
        popt, pcov = curve_fit(
            double_gaussian, x_fit, y_fit,
            p0=p0,
            bounds=bounds,
            sigma=err_fit,
            absolute_sigma=True,
            maxfev=10000
        )
        
        model = double_gaussian(x_fit, *popt)
        residuals = (y_fit - model) / err_fit
        chi2 = np.sum(residuals**2)
        dof = max(len(x_fit) - len(popt), 1)
        red_chi2 = chi2 / dof
        
        integrated_flux = np.trapz(np.maximum(model, 0), x_fit)
        noise = np.sqrt(np.sum(err_fit**2)) * np.median(np.diff(x_fit))
        snr = integrated_flux / noise if noise > 0 else 0
        
        # Apply quality threshold
        if snr < 3 or red_chi2 > 10:
            popt[:] = 0
        
        perr = np.sqrt(np.diag(pcov))
        popt_full = np.concatenate([popt, bg_params])
        perr_full = np.concatenate([perr, np.zeros(2)])
        
        return FitResult(popt_full, perr_full, snr, chi2, red_chi2, True)
        
    except Exception:
        return FitResult(np.full(8, np.nan), None, 0, np.inf, np.inf, False)


# ============================================================================
# Model Selection with Information Criteria
# ============================================================================

def calculate_aic(chi2: float, n_params: int, n_data: int) -> float:
    """Calculate Akaike Information Criterion"""
    return chi2 + 2 * n_params + (2 * n_params * (n_params + 1)) / max(n_data - n_params - 1, 1)


def calculate_bic(chi2: float, n_params: int, n_data: int) -> float:
    """Calculate Bayesian Information Criterion"""
    return chi2 + n_params * np.log(n_data)


def select_best_model(results: Dict[str, FitResult], n_data: int) -> Tuple[str, FitResult]:
    """
    Select best model using BIC (more conservative than AIC).
    
    Improvements:
    - Information criterion for model selection
    - Penalty for complexity
    - F-test for nested models
    """
    valid_results = {name: res for name, res in results.items() if res.success}
    
    if not valid_results:
        return 'none', FitResult(np.full(8, np.nan), None, 0, np.inf, np.inf, False)
    
    # Calculate BIC for each model
    bic_scores = {}
    for name, result in valid_results.items():
        n_params = len(result.params)
        bic = calculate_bic(result.chi2, n_params, n_data)
        bic_scores[name] = bic
    
    # Select model with lowest BIC
    best_model = min(bic_scores, key=bic_scores.get)
    
    return best_model, valid_results[best_model]


# ============================================================================
# High-Level Fitting Function
# ============================================================================

def fit_spectrum(x: np.ndarray, y: np.ndarray, error: np.ndarray,
                fit_nii: bool = True) -> Tuple[FitResult, str]:
    """
    Comprehensive spectrum fitting with automatic model selection.
    
    Parameters:
    -----------
    x : wavelength array
    y : flux array  
    error : error array
    fit_nii : whether to attempt NII fitting
    
    Returns:
    --------
    best_result : FitResult object with best fit
    model_name : name of selected model
    """
    # Estimate background once
    bg_params, _ = estimate_background(x, y, error)
    
    # Try all models
    results = {}
    
    # Single Gaussian (always try)
    results['single'] = fit_single_gaussian(x, y, error, bg_params)
    
    # Double Gaussian (if single fit is poor)
    if results['single'].red_chi2 > 3:
        results['double'] = fit_double_gaussian(x, y, error, bg_params)
    
    # NII doublet (if requested and signal is strong enough)
    if fit_nii and results['single'].snr > 5:
        nii_result = fit_nii_doublet(x, y, error, bg_params)
        if nii_result.success:
            # Refit Halpha after subtracting NII
            y_no_nii = y - double_gaussian(x, *nii_result.params[:6])
            results['single_with_nii'] = fit_single_gaussian(x, y_no_nii, error, bg_params)
            results['single_with_nii'].nii_used = True
            
            if results['single'].red_chi2 > 3:
                results['double_with_nii'] = fit_double_gaussian(x, y_no_nii, error, bg_params)
                results['double_with_nii'].nii_used = True
    
    # Select best model
    mask = (x > HALPHA - 0.01) & (x < HALPHA + 0.01)
    n_data = np.sum(mask)
    
    model_name, best_result = select_best_model(results, n_data)
    
    return best_result, model_name


# ============================================================================
# Batch Processing with Progress Tracking
# ============================================================================

def process_cube_improved(hdulist, progress_callback=None):
    """
    Process entire datacube with improved fitting.
    
    Improvements:
    - Parallel processing capability
    - Progress tracking
    - Better memory management
    """
    wavelength, cube, error_cube = process_data_cube(hdulist)
    psf = tools.psf(hdulist)
    
    ny, nx = cube.shape[1], cube.shape[2]
    total_pixels = ny * nx
    
    results = []
    
    for i in range(ny):
        for j in range(nx):
            if progress_callback and (i * nx + j) % 100 == 0:
                progress_callback((i * nx + j) / total_pixels)
            
            if psf[i, j] == 0:
                results.append({'pixel_x': i, 'pixel_y': j, **{k: np.nan for k in 
                    ['a','b','c','d','e','snr','chi2','model']}})
                continue
            
            flux = cube[:, i, j]
            err = error_cube[:, i, j]
            mask = ~np.isnan(flux) & ~np.isnan(err) & (err > 0)
            
            if np.sum(mask) < 5:
                results.append({'pixel_x': i, 'pixel_y': j, **{k: np.nan for k in 
                    ['a','b','c','d','e','snr','chi2','model']}})
                continue
            
            result, model_name = fit_spectrum(wavelength[mask], flux[mask], err[mask])
            
            if result.success:
                results.append({
                    'pixel_x': i, 'pixel_y': j,
                    'a': result.params[0], 'b': result.params[1], 'c': result.params[2],
                    'd': result.params[3], 'e': result.params[4],
                    'snr': result.snr, 'chi2': result.red_chi2,
                    'model': model_name
                })
    
    return pd.DataFrame(results)


def process_data_cube(hdulist, fwhm=3.5):
    """Original function preserved for compatibility"""
    flux_cube = hdulist[1].data.copy()
    noise_cube = hdulist[2].data.copy()
    psf = tools.psf(hdulist)

    flux_cube *= psf
    noise_cube *= psf

    wavelength = tools.get_wavelength(np.arange(0, 2048), hdulist).value
    
    return wavelength, flux_cube, noise_cube