#!/usr/bin/env python
"""
Set specified pixels to NaN in all HDUs of a FITS file.

Usage:
    python set_fits_pixels_nan.py input.fits output.fits --pixels "X:1 Y:5-8" "X:2 Y:6-"
"""

import argparse
import numpy as np
from astropy.io import fits


def parse_pixel_spec(spec, max_dim):
    """
    Parse a pixel specification like '5-8' or '6-' into a range.
    
    Args:
        spec: String like '5', '5-8', or '6-'
        max_dim: Maximum dimension size for open-ended ranges
    
    Returns:
        List of indices (0-based)
    """
    spec = spec.strip()
    
    if '-' in spec:
        parts = spec.split('-')
        start = int(parts[0])
        if parts[1] == '':
            end = max_dim
        else:
            end = int(parts[1])
        return list(range(start, end + 1))
    else:
        return [int(spec)]


def parse_pixel_string(pixel_str):
    """
    Parse a pixel specification string like 'X:1 Y:5-8'.
    
    Returns:
        Tuple of (x_spec, y_spec) as strings
    """
    parts = pixel_str.upper().split()
    x_spec = None
    y_spec = None
    
    for part in parts:
        if part.startswith('X:'):
            x_spec = part[2:]
        elif part.startswith('Y:'):
            y_spec = part[2:]
    
    return x_spec, y_spec


def set_pixels_to_nan(hdulist, pixel_specs):
    """
    Set specified pixels to NaN for all image HDUs.
    
    Args:
        hdulist: An opened FITS HDUList
        pixel_specs: List of pixel specification strings
    """
    for hdu in hdulist:
        if hdu.data is None:
            continue
        
        if not isinstance(hdu.data, np.ndarray):
            continue
        
        if hdu.data.ndim < 2:
            continue
        
        # Convert to float if needed (can't have NaN in integer arrays)
        if not np.issubdtype(hdu.data.dtype, np.floating):
            hdu.data = hdu.data.astype(np.float64)
        
        shape = hdu.data.shape
        ny, nx = shape[-2], shape[-1]  # Last two dimensions are Y, X
        
        for spec in pixel_specs:
            x_spec, y_spec = parse_pixel_string(spec)
            
            if x_spec is None or y_spec is None:
                print(f"Warning: Invalid spec '{spec}', skipping")
                continue
            
            x_indices = parse_pixel_spec(x_spec, nx - 1)
            y_indices = parse_pixel_spec(y_spec, ny - 1)
            
            for x in x_indices:
                for y in y_indices:
                    if 0 <= x < nx and 0 <= y < ny:
                        if hdu.data.ndim == 2:
                            hdu.data[y, x] = np.nan
                        else:
                            # For 3D+ arrays, set all slices
                            hdu.data[..., y, x] = np.nan
                        print(f"Set pixel (X={x}, Y={y}) to NaN in HDU '{hdu.name}'")


def main():
    parser = argparse.ArgumentParser(
        description='Set specified pixels to NaN in FITS file HDUs'
    )
    parser.add_argument('input', help='Input FITS file')
    parser.add_argument('output', help='Output FITS file')
    parser.add_argument(
        '--pixels', '-p',
        nargs='+',
        required=True,
        help='Pixel specifications, e.g., "X:1 Y:5-8" "X:2 Y:6-"'
    )
    
    args = parser.parse_args()
    
    # Read entire file into memory to allow same input/output file
    hdulist = fits.open(args.input)
    
    # Force loading ALL data into memory before closing the file
    for hdu in hdulist:
        if hdu.data is not None:
            hdu.data = np.array(hdu.data)  # Force copy into memory
    
    # Close the input file
    hdulist._file.close()
    
    # Now safely modify the data
    set_pixels_to_nan(hdulist, args.pixels)
    
    # Write to output (safe even if same as input)
    hdulist.writeto(args.output, overwrite=True)
    
    print(f"\nSaved modified FITS to: {args.output}")


if __name__ == '__main__':
    main()