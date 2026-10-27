import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from scipy.special import erf
from typing import List, Dict
from Starlet import starlet_transform_3d


def _sigmoid(x, center, tightness=10.0):
    """
    Helper function (a sigmoid) to smooth-out the windows
    """
    tightness = float(tightness)
    return 1.0 / (1.0 + np.exp(-tightness * (x - center)))

def visualize_component_mask_product(kspace_component_cornered, component_name, angular_mask_cornered, mask_name, N_shape, contrast, slice_dim=0, slice_idx=None, cmap_component='bwr_r', cmap_mask='viridis'):
    """
    Visualizes a k-space component, thed angular mask, and their product.
    """
    N_shape = np.array(N_shape)
    if slice_idx is None:
        slice_idx = N_shape[slice_dim] // 2

    component_shifted = kspace_component_cornered
    mask_shifted = angular_mask_cornered
    product_shifted = kspace_component_cornered * angular_mask_cornered

    if slice_dim == 0:
        slice_comp, slice_mask, slice_prod = component_shifted[slice_idx, :, :], mask_shifted[slice_idx, :, :], product_shifted[slice_idx, :, :]
        xlabel, ylabel, view_name = "kz", "ky", f"Sagittal (kx={slice_idx})"
    elif slice_dim == 1:
        slice_comp, slice_mask, slice_prod = component_shifted[:, slice_idx, :], mask_shifted[:, slice_idx, :], product_shifted[:, slice_idx, :]
        xlabel, ylabel, view_name = "kz", "kx", f"Coronal (ky={slice_idx})"
    else:
        slice_comp, slice_mask, slice_prod = component_shifted[:, :, slice_idx], mask_shifted[:, :, slice_idx], product_shifted[:, :, slice_idx]
        xlabel, ylabel, view_name = "ky", "kx", f"Axial (kz={slice_idx})"

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fig.suptitle(f"'{component_name}' with '{mask_name}' - {view_name}", fontsize=16)

    im1 = axes[0].imshow(slice_comp, cmap=cmap_component, aspect='auto',vmin = contrast[0],vmax = contrast[1])
    axes[0].set_title(component_name); axes[0].set_xlabel(xlabel); axes[0].set_ylabel(ylabel)
    fig.colorbar(im1, ax=axes[0], shrink=0.75, pad=0.02)

    im2 = axes[1].imshow(slice_mask, cmap='grey', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title(mask_name); axes[1].set_xlabel(xlabel); axes[1].set_ylabel(ylabel)
    fig.colorbar(im2, ax=axes[1], shrink=0.75, pad=0.02)

    #comp_min, comp_max = np.min(slice_comp), np.max(slice_comp)
    im3 = axes[2].imshow(slice_prod, cmap=cmap_component, aspect='auto', vmin=contrast[0], vmax=contrast[1])
    axes[2].set_title("Product"); axes[2].set_xlabel(xlabel); axes[2].set_ylabel(ylabel)
    fig.colorbar(im3, ax=axes[2], shrink=0.75, pad=0.02)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()


def create_annulus_dipolelet_windows(D_qsm, thresholds, tightness_factor=None):
    """
    Creates smooth, non-overlapping angular windows based on increasing thresholds of |D(k)|.
    The output dictionary is explicitly ordered: Core, Annuli (inner to outer), Outer.
    """
  
    if np.size(thresholds) == 0:
        return {'all_pass': np.ones_like(D_qsm, dtype=float)}

    abs_Dk = np.abs(D_qsm)
    sorted_thresholds = sorted(thresholds)
    
    smooth_disks = []
    for i, delta_i in enumerate(sorted_thresholds):
        if tightness_factor is None:
            width = (sorted_thresholds[i+1] - delta_i) if (i == 0 and len(sorted_thresholds) > 1) else \
                    (delta_i - sorted_thresholds[i-1]) if i > 0 else delta_i
            tf = 20.0 / (width + 1e-9)
        else:
            tf = tightness_factor
        disk_i = 1.0 - _sigmoid(abs_Dk, center=delta_i, tightness=tf)
        smooth_disks.append(disk_i)

    windows = {}
    windows[f'A_core_d0'] = smooth_disks[0]
    
    for i in range(len(sorted_thresholds) - 1):
        annulus_i = smooth_disks[i+1] - smooth_disks[i]
        windows[f'A_annulus_d{i}_d{i+1}'] = np.maximum(annulus_i, 0)
        
    windows[f'A_outer_d{len(sorted_thresholds)-1}'] = 1.0 - smooth_disks[-1]

    core_keys = [k for k in windows.keys() if 'core' in k]
    annulus_keys = sorted([k for k in windows.keys() if 'annulus' in k])
    outer_keys = [k for k in windows.keys() if 'outer' in k]

    ordered_windows = {}
    for key in core_keys + annulus_keys + outer_keys:
        ordered_windows[key] = windows[key]

    sum_all = np.zeros_like(D_qsm, dtype=float)
    for w in ordered_windows.values():
        sum_all += w
    sum_all[sum_all < 1e-9] = 1e-9
    
    normalized_windows = {name: w / sum_all for name, w in ordered_windows.items()}
    return normalized_windows


def _smooth_bump_cutoff(x, m, M):
    """
    Calculates the smooth bump function which is 1 for |x|<=m, 0 for |x|>=M,
    and transitions smoothly in between.
    """
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=np.float64)
    abs_x = np.abs(x)

    # Region 1: Plateau (|x| <= m)
    mask_le_m = abs_x <= m
    y[mask_le_m] = 1.0

    # Region 2: Smooth Transition (m < |x| < M)
    mask_between = (abs_x > m) & (abs_x < M)
    if np.any(mask_between):
        val = abs_x[mask_between]
        m_val = m if np.isscalar(m) else m[mask_between]
        M_val = M if np.isscalar(M) else M[mask_between]
        
        # Ensure M_val > m_val to avoid division by zero
        if np.any(M_val <= m_val):
            return y

        ratio_sq = (val - m_val)**2 / (M_val - m_val)**2
        denom = 1.0 - ratio_sq
        denom[denom <= 0] = np.finfo(float).eps
        y[mask_between] = np.e * np.exp(-1.0 / denom)
    
    return y

def create_annulus_dipolelet_windows_cutoff(D_qsm, thresholds,width_factor=2.0):
    """
    Creates smooth, non-overlapping angular windows using the smooth_bump_cutoff function.
    """
    if np.size(thresholds) == 0:
        return {'all_pass': np.ones_like(D_qsm, dtype=float)}

    abs_Dk = np.abs(D_qsm)
    sorted_thresholds = sorted(thresholds)
    
    smooth_disks = []

    for i, delta_i in enumerate(sorted_thresholds):
        # Determine the base transition width
        if len(sorted_thresholds) == 1:
            base_width = delta_i
        elif i == 0:
            base_width = (sorted_thresholds[i+1] - delta_i)
        else:
            base_width = (delta_i - sorted_thresholds[i-1])

        # Increase the width
        width = base_width * width_factor 
        
        # Define boundaries based on the wider transition
        m = delta_i - width / 2.0
        M = delta_i + width / 2.0
        m = max(m, 0)
        
        disk_i = _smooth_bump_cutoff(abs_Dk, m, M)
        smooth_disks.append(disk_i)

    windows = {}
    windows[f'A_core_d0'] = smooth_disks[0]
    
    for i in range(len(sorted_thresholds) - 1):
        annulus_i = smooth_disks[i+1] - smooth_disks[i]
        windows[f'A_annulus_d{i}_d{i+1}'] = np.maximum(annulus_i, 0)
        
    windows[f'A_outer_d{len(sorted_thresholds)-1}'] = 1.0 - smooth_disks[-1]

    core_keys = [k for k in windows.keys() if 'core' in k]
    annulus_keys = sorted([k for k in windows.keys() if 'annulus' in k])
    outer_keys = [k for k in windows.keys() if 'outer' in k]

    ordered_windows = {}
    for key in core_keys + annulus_keys + outer_keys:
        ordered_windows[key] = windows[key]

    sum_all = np.sum(list(ordered_windows.values()), axis=0)
    sum_all[sum_all < 1e-9] = 1e-9
    
    normalized_windows = {name: w / sum_all for name, w in ordered_windows.items()}
    return normalized_windows


def _gaussian_erf_window(x, center, width):
    """
    Creates a smooth step function using the error function (erf).
    The 'width' parameter controls the steepness of the transition.
    """
    # A smaller width results in a steeper transition
    sigma = width / 2.0
    # Add epsilon to avoid division by zero if width is zero
    return 0.5 * (1 + erf((x - center) / (np.sqrt(2) * sigma + 1e-9)))

def create_annulus_dipolelet_windows_gaussian(D_qsm, thresholds, width_factor=1.0):
    """
    Creates smooth, non-overlapping angular windows using the Gaussian erf function.
    """
    if np.size(thresholds) == 0:
        return {'all_pass': np.ones_like(D_qsm, dtype=float)}

    abs_Dk = np.abs(D_qsm)
    sorted_thresholds = sorted(thresholds)
    
    smooth_disks = []
    for i, delta_i in enumerate(sorted_thresholds):
        # Determine the base transition width from neighboring thresholds
        if len(sorted_thresholds) == 1:
            base_width = delta_i
        elif i == 0:
            base_width = (sorted_thresholds[i+1] - delta_i)
        else:
            base_width = (delta_i - sorted_thresholds[i-1])

        # The effective transition width is scaled by the factor
        transition_width = base_width * width_factor
        
        # Create the disk (1 at center, 0 at edges) using the erf window
        disk_i = 1.0 - _gaussian_erf_window(abs_Dk, center=delta_i, width=transition_width)
        smooth_disks.append(disk_i)

    windows = {}
    windows[f'A_core_d0'] = smooth_disks[0]
    
    for i in range(len(sorted_thresholds) - 1):
        annulus_i = smooth_disks[i+1] - smooth_disks[i]
        windows[f'A_annulus_d{i}_d{i+1}'] = np.maximum(annulus_i, 0)
        
    windows[f'A_outer_d{len(sorted_thresholds)-1}'] = 1.0 - smooth_disks[-1]

    core_keys = [k for k in windows.keys() if 'core' in k]
    annulus_keys = sorted([k for k in windows.keys() if 'annulus' in k])
    outer_keys = [k for k in windows.keys() if 'outer' in k]

    ordered_windows = {key: windows[key] for key in core_keys + annulus_keys + outer_keys}

    sum_all = np.sum(list(ordered_windows.values()), axis=0)
    sum_all[sum_all < 1e-9] = 1e-9
    
    normalized_windows = {name: w / sum_all for name, w in ordered_windows.items()}
    return normalized_windows


def _radial_frequency_grid(shape, voxel_size=(1.0, 1.0, 1.0)):
    """
    Build |xi| on the centered FFT grid.
    Frequencies are in cycles per 'voxel_size' unit along each axis.
    """
    nx, ny, nz = shape
    dx, dy, dz = voxel_size
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    fz = np.fft.fftfreq(nz, d=dz)
    FX, FY, FZ = np.meshgrid(np.fft.fftshift(fx),
                             np.fft.fftshift(fy),
                             np.fft.fftshift(fz),
                             indexing='ij')
    R = np.sqrt(FX**2 + FY**2 + FZ**2)
    return R

def build_flat_radial_bump(shape, rho0=0.02, rho1=0.40, voxel_size=(1.0, 1.0, 1.0)):
    """
    Flat-at-zero + Gaussian tail radial bump:
        R(|xi|) = exp(-(rho0/|xi|)^2) * exp(-( |xi|/rho1 )^2)
    rho0: controls flatness near the origin (cycles per unit); make it ~ 1–2 bins.
    rho1: controls high-frequency decay (as a fraction of Nyquist radius ~ 0.866 in 3D if voxel_size=(1,1,1)).
    """
    R = _radial_frequency_grid(shape, voxel_size=voxel_size)
    eps = 1e-12
    return np.exp(-(rho0 / (R + eps))**2) * np.exp(- (R / rho1)**2)


    
# def parabolic_dipolelet_decomposition(
#     image_3d,
#     D_qsm,
#     num_radial_scales,
#     base_angular_bands,
#     delta0,
#     transform,
#     # Radial window
#     starlet_filter_type='bspline', #gaussian
#     starlet_filter_param='B3', #if gaussian, select width here!
#     # Angular window
#     window_type='sigmoid',
#     tightness_factor=20.0,
#     cutoff_width_factor=1.0,
#     gaussian_width_factor=1.0,
#     reverse_scaling=False,
#     scaling_law='dyadic' #curvelet (parabolic scaling), static (meaning base bands are the same in every level)
# ):
#     """
#     Performs decomposition with explicit control over the starlet transform.
#     """
    
#     starlet_coeffs, coarse_approx = transform(
#         image_3d, 
#         num_scales=num_radial_scales, 
#         filter_type=starlet_filter_type, 
#         filter_param=starlet_filter_param,
#         verbose=False
#     )

#     segmented_starlet_coeffs = []

#     # --- Process each starlet scale (w_j) ---
#     for j, w_j in enumerate(starlet_coeffs):
#         w_j_kspace = fftshift(fftn(w_j))
        
#         # Determine number of angular bands for this scale
#         if scaling_law == 'static':
#             num_angular_bands_j = base_angular_bands
#         else: # dyadic or curvelet
#             exponent = j if reverse_scaling else num_radial_scales - 1 - j
#             scaling_factor = 2**exponent if scaling_law == 'dyadic' else 2**(exponent / 2.0)
#             num_angular_bands_j = max(1, int(np.round(base_angular_bands * scaling_factor)))
            
#         print(f"Scale w{j+1}: Using {num_angular_bands_j} angular bands (window: {window_type}).")
        
#         # Determine thresholds for this scale
#         num_thresholds = num_angular_bands_j - 1
#         thresholds_j = []
#         if num_thresholds >= 1:
#             thresholds_j = np.logspace(np.log10(delta0), np.log10(1/3), num_thresholds)

#         # Choose which window function to call
#         if window_type == 'cutoff':
#             angular_windows_j = create_annulus_dipolelet_windows_cutoff(D_qsm, thresholds_j, width_factor=cutoff_width_factor)
#         elif window_type == 'sigmoid':
#             angular_windows_j = create_annulus_dipolelet_windows(D_qsm, thresholds_j, tightness_factor=tightness_factor)
#         elif window_type == 'gaussian':
#             angular_windows_j = create_annulus_dipolelet_windows_gaussian(D_qsm, thresholds_j, width_factor=gaussian_width_factor)
#         else:
#             raise ValueError("window_type must be 'sigmoid', 'cutoff', or 'gaussian'")
            
#         scale_j_segments = {name: np.real(ifftn(ifftshift(w_j_kspace * mask))) for name, mask in angular_windows_j.items()}
#         segmented_starlet_coeffs.append(scale_j_segments)

# # --- Process the coarse scale (c_J) ---
# coarse_kspace = fftshift(fftn(coarse_approx))

# # (Bands and thresholds for the coarse scale)
# if scaling_law == 'static':
#     num_coarse_bands = base_angular_bands
# else: # dyadic or curvelet
#     coarse_exponent = num_radial_scales if reverse_scaling else 0
#     scaling_factor = 2**coarse_exponent if scaling_law == 'dyadic' else 2**(coarse_exponent / 2.0)
#     num_coarse_bands = max(1, int(np.round(base_angular_bands * scaling_factor)))

# print(f"Coarse scale: Using {num_coarse_bands} angular bands (window: {window_type}).")

# num_coarse_thresholds = num_coarse_bands - 1
# coarse_thresholds = []
# if num_coarse_thresholds >= 1:
#     coarse_thresholds = np.logspace(np.log10(delta0), np.log10(1/3), num_coarse_thresholds)

# # Which window to call (for coarse scale)
# if window_type == 'cutoff':
#     angular_windows_coarse = create_annulus_dipolelet_windows_cutoff(D_qsm, coarse_thresholds, width_factor=cutoff_width_factor)
# elif window_type == 'sigmoid':
#     angular_windows_coarse = create_annulus_dipolelet_windows(D_qsm, coarse_thresholds, tightness_factor=tightness_factor)
# elif window_type == 'gaussian':
#     angular_windows_coarse = create_annulus_dipolelet_windows_gaussian(D_qsm, coarse_thresholds, width_factor=gaussian_width_factor)


# segmented_coarse_approx = {name: np.real(ifftn(ifftshift(coarse_kspace * mask))) for name, mask in angular_windows_coarse.items()}
# WE DONT DECOMPOSE THE COARSE SCALE NOW (UNCOMENT ABOVE FOR DECOMPOSING)
    #coarse_kspace = fftshift(fftn(coarse_approx))

    # segmented_coarse_approx = { 'coarse_full': coarse_approx }
    # return segmented_starlet_coeffs, segmented_coarse_approx

# def parabolic_dipolelet_decomposition(
#     image_3d,
#     D_qsm,
#     num_radial_scales,
#     base_angular_bands,
#     delta0,
#     transform,
#     return_windows=False,
#     starlet_filter_type='bspline',
#     starlet_filter_param='B3',
#     window_type='sigmoid',
#     tightness_factor=20.0,
#     cutoff_width_factor=1.0,
#     gaussian_width_factor=1.0,
#     reverse_scaling=False,
#     scaling_law='dyadic'
# ):
#     """
#     Performs decomposition and can optionally return the generated k-space windows.
#     """
    
#     starlet_coeffs, coarse_approx = transform(
#         image_3d, 
#         num_scales=num_radial_scales, 
#         filter_type=starlet_filter_type, 
#         filter_param=starlet_filter_param,
#         verbose=False
#     )


#     segmented_coeffs = []
#     windows_list = []

#     # --- Process each starlet scale (w_j) ---
#     for j, w_j in enumerate(starlet_coeffs):
#         w_j_kspace = fftshift(fftn(w_j))
        
#         # Determine number of angular bands for this scale
#         if scaling_law == 'static':
#             num_angular_bands_j = base_angular_bands
#         else: # dyadic or curvelet
#             exponent = j if reverse_scaling else num_radial_scales - 1 - j
#             scaling_factor = 2**exponent if scaling_law == 'dyadic' else 2**(exponent / 2.0)
#             num_angular_bands_j = max(1, int(np.round(base_angular_bands * scaling_factor)))
        
#         # Determine thresholds for this scale
#         num_thresholds = num_angular_bands_j - 1
#         thresholds_j = []
#         if num_thresholds >= 1:
#             thresholds_j = np.logspace(np.log10(delta0), np.log10(1/3), num_thresholds)

#         # Generate the windows
#         if window_type == 'cutoff':
#             angular_windows_j = create_annulus_dipolelet_windows_cutoff(D_qsm, thresholds_j, width_factor=cutoff_width_factor)
#         elif window_type == 'sigmoid':
#             angular_windows_j = create_annulus_dipolelet_windows(D_qsm, thresholds_j, tightness_factor=tightness_factor)
#         elif window_type == 'gaussian':
#             angular_windows_j = create_annulus_dipolelet_windows_gaussian(D_qsm, thresholds_j, width_factor=gaussian_width_factor)
#         else:
#             raise ValueError("window_type must be 'sigmoid', 'cutoff', or 'gaussian'")
        
#         # if return_windows:
#         #     starlet_windows_list.append(angular_windows_j)
#         if return_windows:
#             combined_windows_j = {}
#             for key, angular_mask in angular_windows_j.items():
#                 # Multiply the radial filter by the angular filter
#                 combined_windows_j[key] = radial_kspace_filters[j] * angular_mask
#             windows_list.append(combined_windows_j)

#         # Apply the windows to the k-space data
#         scale_j_segments = {name: np.real(ifftn(ifftshift(w_j_kspace * mask))) for name, mask in angular_windows_j.items()}
#         segmented_coeffs.append(scale_j_segments)

#     # Coarse scale is not decomposed
#     segmented_coarse = {'coarse_full': coarse_approx}
    
#     if return_windows:
#         return segmented_coeffs, segmented_coarse, windows_list
#     else:
        
#         return segmented_coeffs, segmented_coarse


def parabolic_dipolelet_decomposition(
    image_3d,
    D_qsm,
    num_radial_scales,
    base_angular_bands,
    delta0,
    transform,
    return_windows=False,
    starlet_filter_type='bspline',
    starlet_filter_param='B3',
    window_type='sigmoid',
    tightness_factor=20.0,
    cutoff_width_factor=1.0,
    gaussian_width_factor=1.0,
    reverse_scaling=False,
    scaling_law='dyadic'
):
    """
    The infamous Dipole-lets
    """
    
    starlet_coeffs, coarse_approx = transform(
        image_3d, 
        num_scales=num_radial_scales, 
        filter_type=starlet_filter_type, 
        filter_param=starlet_filter_param,
        verbose=False
    )

    combined_windows_list = []
    if return_windows:
        delta_image = np.zeros(image_3d.shape, dtype=np.float32)
        center_vox = tuple(s // 2 for s in image_3d.shape)
        delta_image[center_vox] = 1.0
        
        starlet_impulse_responses, _ = transform(
            delta_image,
            num_scales=num_radial_scales,
            filter_type=starlet_filter_type,
            filter_param=starlet_filter_param
        )
        
        radial_kspace_filters = [
            np.abs(fftshift(fftn(ifftshift(response)))) 
            for response in starlet_impulse_responses
        ]
    
    
    segmented_coeffs = []
    for j, w_j in enumerate(starlet_coeffs):
        w_j_kspace = fftshift(fftn(w_j))
        
        if scaling_law == 'static':
            num_angular_bands_j = base_angular_bands
        else:
            exponent = j if reverse_scaling else num_radial_scales - 1 - j
            scaling_factor = 2**exponent if scaling_law == 'dyadic' else 2**(exponent / 2.0)
            num_angular_bands_j = max(1, int(np.round(base_angular_bands * scaling_factor)))
        
        num_thresholds = num_angular_bands_j - 1
        thresholds_j = []
        if num_thresholds >= 1:
            thresholds_j = np.logspace(np.log10(delta0), np.log10(1/3), num_thresholds)

        if window_type == 'sigmoid':
            angular_windows_j = create_annulus_dipolelet_windows(D_qsm, thresholds_j, tightness_factor=tightness_factor)
        elif window_type == 'gaussian':
            angular_windows_j = create_annulus_dipolelet_windows_gaussian(D_qsm, thresholds_j, width_factor=gaussian_width_factor)
        else:
            raise ValueError("Invalid window_type")

        if return_windows:
            combined_windows_j = {}
            for key, angular_mask in angular_windows_j.items():
                combined_windows_j[key] = radial_kspace_filters[j] * angular_mask
            combined_windows_list.append(combined_windows_j)

        scale_j_segments = {name: np.real(ifftn(ifftshift(w_j_kspace * mask))) for name, mask in angular_windows_j.items()}
        segmented_coeffs.append(scale_j_segments)

    segmented_coarse = {'coarse_full': coarse_approx}
    
    if return_windows:
        return segmented_coeffs, segmented_coarse, combined_windows_list
    else:
        return segmented_coeffs, segmented_coarse

def dipolelet_wrapper(
    segmented_coeffs: List[Dict[str, np.ndarray]], 
    segmented_coarse: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Takes the output of a parabolic dipolelet decomposition and stacks all
    image components into a single 4D NumPy array.

    Args:
        segmented_coeffs (List[Dict]): A list where each item is a dictionary
            of the angular components for a given starlet scale.
        segmented_coarse (Dict): A dictionary of the angular components for
            the coarse scale.

    Returns:
        np.ndarray: A 4D array where the last dimension contains all the
            stacked components, sorted by scale and then by component name.
    """
    all_components = []
    
    for scale_dict in segmented_coeffs:
        # Sort by component name to ensure consistent ordering
        for component_name in sorted(scale_dict.keys()):
            all_components.append(scale_dict[component_name])
            
    # Add the coarse scale components, also sorted
    for component_name in sorted(segmented_coarse.keys()):
        all_components.append(segmented_coarse[component_name])
        
    # Stack along a new last axis
    stacked_components = np.stack(all_components, axis=-1)
    
    return stacked_components

def inverse_parabolic_dipolelet_transform(segmented_coeffs, segmented_coarse):
    """
    Reconstructs the final image by summing all detail and coarse components

    """
    # Get the shape from the first component
    first_key = list(segmented_coeffs[0].keys())[0]
    final_image = np.zeros_like(segmented_coeffs[0][first_key])

    # Sum all detail coefficients
    for scale_segments in segmented_coeffs:
        for component in scale_segments.values():
            final_image += component

    # Add the coarse approximation
    for component in segmented_coarse.values():
        final_image += component
        
    return final_image

def _frequency_grids(shape, voxel_size=(1.0, 1.0, 1.0)):
    """
    Build centered FFT frequency grids FX, FY, FZ (cycles per voxel_size unit).
    """
    nx, ny, nz = shape
    dx, dy, dz = voxel_size
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    fz = np.fft.fftfreq(nz, d=dz)
    FX, FY, FZ = np.meshgrid(
        np.fft.fftshift(fx),
        np.fft.fftshift(fy),
        np.fft.fftshift(fz),
        indexing='ij'
    )
    return FX, FY, FZ


def parabolic_dipolelet_decomposition_vanishing_dipoleaware(
    image_3d,
    D_qsm,
    num_radial_scales,
    base_angular_bands,
    delta0,
    transform,
    return_windows=False,
    starlet_filter_type='bspline',
    starlet_filter_param='B3',
    window_type='sigmoid',
    tightness_factor=20.0,
    cutoff_width_factor=1.0,  # kept for API compatibility
    gaussian_width_factor=1.0,
    reverse_scaling=False,
    scaling_law='dyadic',
    # === NEW OPTIONS (first-order + dipole-aware upgrades) ===
    Lz_first_order=1,
    LD_dipole_aware=1,
    apply_first_order=True,
    apply_dipole_aware=True,
    voxel_size=(1.0, 1.0, 1.0)
):
    """
    Parabolic dipolelet decomposition with:
      - First-order upgrade: (i 2π k_z)^Lz_first_order multiplier (vanishing moments / derivative along z).
      - Dipole-aware upgrade: |D_qsm(k)|^LD_dipole_aware multiplier on OUTER bands (vanishing on the magic cone).

    The front part of the signature is identical to `parabolic_dipolelet_decomposition`.
    New options are appended at the end and are fully optional.

    Args (new):
        Lz_first_order (int): order of z-derivative multiplier (i 2π k_z)^Lz_first_order.
        LD_dipole_aware (int): power for dipole-aware factor |D_qsm|^LD_dipole_aware on outer bands.
        apply_first_order (bool): if False, disable the (i 2π k_z)^Lz multiplier.
        apply_dipole_aware (bool): if False, disable the |D_qsm|^LD multiplier.
        voxel_size (tuple): voxel size used to build frequency grids (fx,fy,fz).
    """

    # --- Starlet radial decomposition (unchanged structure) ---
    starlet_coeffs, coarse_approx = transform(
        image_3d,
        num_scales=num_radial_scales,
        filter_type=starlet_filter_type,
        filter_param=starlet_filter_param,
        verbose=False
    )

    combined_windows_list = []
    if return_windows:
        # Impulse response to visualize radial filters (same as your original code)
        delta_image = np.zeros(image_3d.shape, dtype=np.float32)
        center_vox = tuple(s // 2 for s in image_3d.shape)
        delta_image[center_vox] = 1.0

        starlet_impulse_responses, _ = transform(
            delta_image,
            num_scales=num_radial_scales,
            filter_type=starlet_filter_type,
            filter_param=starlet_filter_param
        )

        radial_kspace_filters = [
            np.abs(fftshift(fftn(ifftshift(response))))
            for response in starlet_impulse_responses
        ]

    # --- Build frequency multipliers for upgrades ---
    shape = image_3d.shape
    FX, FY, FZ = _frequency_grids(shape, voxel_size=voxel_size)

    # First-order upgrade: (i 2π k_z)^Lz
    if apply_first_order and Lz_first_order > 0:
        Pz_factor = (1j * 2.0 * np.pi * FZ) ** Lz_first_order
    else:
        Pz_factor = np.ones(shape, dtype=np.complex64)

    # Dipole-aware upgrade: |D_qsm|^LD on outer bands
    if apply_dipole_aware and LD_dipole_aware > 0:
        D_abs = np.abs(D_qsm)
        D_factor = D_abs ** LD_dipole_aware
    else:
        D_factor = np.ones_like(D_qsm, dtype=np.float32)

    # --- Angular splitting + modified masks ---
    segmented_coeffs = []
    for j, w_j in enumerate(starlet_coeffs):
        # k-space for this radial scale
        w_j_kspace = fftshift(fftn(w_j))

        # Scaling law for number of angular bands (unchanged)
        if scaling_law == 'static':
            num_angular_bands_j = base_angular_bands
        else:
            exponent = j if reverse_scaling else num_radial_scales - 1 - j
            if scaling_law == 'dyadic':
                scaling_factor = 2 ** exponent
            else:
                # parabolic-ish scaling law (as in your original code)
                scaling_factor = 2 ** (exponent / 2.0)
            num_angular_bands_j = max(1, int(np.round(base_angular_bands * scaling_factor)))

        # Thresholds in |D(k)| for this scale
        num_thresholds = num_angular_bands_j - 1
        thresholds_j = []
        if num_thresholds >= 1:
            thresholds_j = np.logspace(np.log10(delta0), np.log10(1/3), num_thresholds)

        # Angular windows (same choices as original)
        if window_type == 'sigmoid':
            angular_windows_j = create_annulus_dipolelet_windows(
                D_qsm, thresholds_j, tightness_factor=tightness_factor
            )
        elif window_type == 'gaussian':
            angular_windows_j = create_annulus_dipolelet_windows_gaussian(
                D_qsm, thresholds_j, width_factor=gaussian_width_factor
            )
        else:
            raise ValueError("Invalid window_type")

        scale_j_segments = {}
        if return_windows:
            combined_windows_j = {}

        # Apply angular masks + multipliers in k-space
        for name, mask in angular_windows_j.items():
            # Promote to complex to allow complex multipliers
            full_mask = mask.astype(np.complex128)

            # Dipole-aware upgrade: apply |D(k)|^LD ONLY to outer windows
            if apply_dipole_aware and LD_dipole_aware > 0 and ('outer' in name):
                full_mask = full_mask * D_factor

            # First-order upgrade: (i 2π k_z)^Lz, applied to all bands if enabled
            if apply_first_order and Lz_first_order > 0:
                full_mask = full_mask * Pz_factor

            # Segment in k-space and go back to image (real part)
            component_spatial = np.real(ifftn(ifftshift(w_j_kspace * full_mask)))
            scale_j_segments[name] = component_spatial

            # For window visualization: magnitude of the full k-space window
            if return_windows:
                combined_windows_j[name] = radial_kspace_filters[j] * np.abs(full_mask)

        segmented_coeffs.append(scale_j_segments)
        if return_windows:
            combined_windows_list.append(combined_windows_j)

    segmented_coarse = {'coarse_full': coarse_approx}

    if return_windows:
        return segmented_coeffs, segmented_coarse, combined_windows_list
    else:
        return segmented_coeffs, segmented_coarse
