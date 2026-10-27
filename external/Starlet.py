import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter

"""
Implementation based on https://github.com/broxtronix/pymultiscale/blob/master/pymultiscale/starlet.py
"""

# Define the B-spline kernels
B_SPLINE_KERNELS = {
    'B1': np.array([1/4, 2/4, 1/4], dtype=np.float64),
    'B2': np.array([1/8, 3/8, 3/8, 1/8], dtype=np.float64),
    'B3': np.array([1/16, 4/16, 6/16, 4/16, 1/16], dtype=np.float64),
    'B4': np.array([1/32, 5/32, 10/32, 10/32, 5/32, 1/32], dtype=np.float64),
    'B5': np.array([1/64, 6/64, 15/64, 20/64, 15/64, 6/64, 1/64], dtype=np.float64)
}

def _upsample_kernel_1d(base_kernel, step):
    """Upsamples a 1D kernel by inserting 'step-1' zeros between its elements."""
    if step == 1:
        return base_kernel
    len_base = len(base_kernel)
    len_upsampled = len_base + (len_base - 1) * (step - 1)
    upsampled_kernel = np.zeros(len_upsampled, dtype=base_kernel.dtype)
    upsampled_kernel[0::step] = base_kernel
    return upsampled_kernel

def _convolve_separable_3d(image_3d_in, base_kernel, scale_step_j):
    """Convolves a 3D image with an upsampled B-spline kernel."""
    kernel_1d_upsampled = _upsample_kernel_1d(base_kernel, scale_step_j)
    smoothed = convolve1d(image_3d_in.astype(np.float64), kernel_1d_upsampled, axis=0, mode='mirror')
    smoothed = convolve1d(smoothed, kernel_1d_upsampled, axis=1, mode='mirror')
    smoothed = convolve1d(smoothed, kernel_1d_upsampled, axis=2, mode='mirror')
    return smoothed



def starlet_transform_3d(image_3d, num_scales, filter_type='bspline', filter_param='B3', gen2=False, verbose=False):
    """
    Performs a 3D Starlet transform using either B-spline or Gaussian filters.

    Args:
        image_3d (np.ndarray): The 3D input image.
        num_scales (int): The number of wavelet scales.
        filter_type (str): Type of filter, 'bspline' or 'gaussian'.
        filter_param: The parameter for the chosen filter.
                      For 'bspline': a string like 'B3' (default).
                      For 'gaussian': a float for the base_sigma (e.g., 1.0).
        gen2 (bool): If True, computes the 2nd generation Starlet transform.
    """
    # --- Input Validation ---
    if filter_type == 'bspline':
        if filter_param not in B_SPLINE_KERNELS:
            raise ValueError(f"Unknown kernel: '{filter_param}'. Available options are {list(B_SPLINE_KERNELS.keys())}")
        base_kernel = B_SPLINE_KERNELS[filter_param]
    elif filter_type == 'gaussian':
        if not isinstance(filter_param, (float, int)) or filter_param <= 0:
            raise ValueError("For 'gaussian' filter_type, filter_param must be a positive number for base_sigma.")
        base_sigma = float(filter_param)
    else:
        raise ValueError(f"Unknown filter_type: '{filter_type}'. Options are 'bspline', 'gaussian'.")

    current_coarse_scale = image_3d.astype(np.float64, copy=True)
    wavelet_scales_list = []

    for scale_idx in range(num_scales):
        # Apply the selected convolution method
        if filter_type == 'bspline':
            scale_step_j = 2**scale_idx
            smoothed_scale_once = _convolve_separable_3d(current_coarse_scale, base_kernel, scale_step_j)
        else: # gaussian
            sigma_j = base_sigma * (2**scale_idx)
            smoothed_scale_once = gaussian_filter(current_coarse_scale, sigma=sigma_j, mode='mirror')

        # Calculate detail coefficients
        if gen2:
            if filter_type == 'bspline':
                scale_step_j = 2**scale_idx
                smoothed_scale_twice = _convolve_separable_3d(smoothed_scale_once, base_kernel, scale_step_j)
            else: # gaussian
                sigma_j = base_sigma * (2**scale_idx)
                smoothed_scale_twice = gaussian_filter(smoothed_scale_once, sigma=sigma_j, mode='mirror')
            detail_coefficients = current_coarse_scale - smoothed_scale_twice
        else:
            detail_coefficients = current_coarse_scale - smoothed_scale_once
            
        wavelet_scales_list.append(detail_coefficients)
        current_coarse_scale = smoothed_scale_once
    
    return wavelet_scales_list, current_coarse_scale

def inverse_starlet_transform_3d(wavelet_scales_list, coarse_residual, filter_type='bspline', filter_param='B3', gen2=False):
    """
    Reconstructs a 3D image from its Starlet components (B-spline or Gaussian).
    """
    if not wavelet_scales_list:
        return coarse_residual.astype(np.float64, copy=True)

    if filter_type == 'bspline':
        if filter_param not in B_SPLINE_KERNELS:
            raise ValueError(f"Unknown kernel: '{filter_param}'.")
        base_kernel = B_SPLINE_KERNELS[filter_param]
    elif filter_type == 'gaussian':
        if not isinstance(filter_param, (float, int)) or filter_param <= 0:
            raise ValueError("For 'gaussian' filter_type, filter_param must be a positive number for base_sigma.")
        base_sigma = float(filter_param)
    else:
        raise ValueError(f"Unknown filter_type: '{filter_type}'.")

    reconstructed_image = coarse_residual.astype(np.float64, copy=True)
    num_scales = len(wavelet_scales_list)

    for scale_idx in reversed(range(num_scales)):
        current_wavelet_plane = wavelet_scales_list[scale_idx]
        
        if gen2:
            if filter_type == 'bspline':
                current_scale_step_j = 2**scale_idx
                smoothed_coarser_scale = _convolve_separable_3d(reconstructed_image, base_kernel, current_scale_step_j)
            else: # gaussian
                sigma_j = base_sigma * (2**scale_idx)
                smoothed_coarser_scale = gaussian_filter(reconstructed_image, sigma=sigma_j, mode='mirror')
            reconstructed_image = current_wavelet_plane + smoothed_coarser_scale
        else:
            # Gen1 reconstruction is identical for both
            reconstructed_image = current_wavelet_plane + reconstructed_image
    
    return reconstructed_image

