import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy.ndimage import convolve1d

def dipole_kernel(N, FOV):
    

    N = np.array(N)
    FOV = np.array(FOV)

    kx_vec = np.arange(-np.floor(N[0]/2), np.ceil(N[0]/2))
    ky_vec = np.arange(-np.floor(N[1]/2), np.ceil(N[1]/2))
    kz_vec = np.arange(-np.floor(N[2]/2), np.ceil(N[2]/2))

    # Scale coordinates to have proper physical units (e.g., 1/mm)
    kx_vec = kx_vec / FOV[0]
    ky_vec = ky_vec / FOV[1]
    kz_vec = kz_vec / FOV[2]

    Kx, Ky, Kz = np.meshgrid(kx_vec, ky_vec, kz_vec, indexing='ij')

    K2 = Kx**2 + Ky**2 + Kz**2
    
    K2[K2 == 0] = np.finfo(float).eps

    kernel_centered = (1.0/3.0) - (Kz**2) / K2
    
    kernel_centered[K2 == np.finfo(float).eps] = 0.0

    kernel_cornered = np.fft.fftshift(kernel_centered)

    return kernel_cornered

def discrete_dipole_kernel(N, FOV):
    """
    Generates a discrete (finite-difference) dipole kernel in k-space.

    Same input signature as `dipole_kernel(N, FOV)` for drop-in compatibility.
    Derived from the discrete second-derivative (3-point stencil) approximation
    of the dipole operator on the voxel grid: each axis' frequency response term
    is weighted by 1/voxel_size_i^2 (voxel_size = FOV/N), which is required for
    anisotropic voxels. For isotropic voxels this 1/voxel_size^2 factor is a
    common scalar across x/y/z and cancels out of the x/(x+y+z) ratio, reducing
    to the plain cos(2*pi*K/N) form.
    """

    N = np.array(N)
    FOV = np.array(FOV)
    voxel_size = FOV / N

    kx_vec = np.arange(-np.floor(N[0]/2), np.ceil(N[0]/2))
    ky_vec = np.arange(-np.floor(N[1]/2), np.ceil(N[1]/2))
    kz_vec = np.arange(-np.floor(N[2]/2), np.ceil(N[2]/2))
    Kx, Ky, Kz = np.meshgrid(kx_vec, ky_vec, kz_vec, indexing='ij')

    term_x = (1 - np.cos(2 * np.pi * Kx / N[0])) / voxel_size[0]**2
    term_y = (1 - np.cos(2 * np.pi * Ky / N[1])) / voxel_size[1]**2
    term_z = (1 - np.cos(2 * np.pi * Kz / N[2])) / voxel_size[2]**2

    numerator = term_z
    denominator = term_x + term_y + term_z

    # Handle division by zero at the k-space origin (k=0)
    denominator[denominator == 0] = np.finfo(float).eps

    kernel_centered = (1/3) - (numerator / denominator)

    center_idx = (int(np.floor(N[0]/2)), int(np.floor(N[1]/2)), int(np.floor(N[2]/2)))
    kernel_centered[center_idx] = 0.0

    kernel_cornered = np.fft.ifftshift(kernel_centered)

    return kernel_cornered

def imagesc3d2(img, pos=None, fig_num=1, rot_deg=[0, 0, 0], cmap='gray',scale_fig=None,aspect = None,interp=None, contrast = [-0.1, 0.1], avg_size=0, title_fig=" "):
    """

    Display different views of the 3D data.

    Parameters:
    img: 3D data to be displayed.
    pos: vector defining the point at the center of the three projection planes.
    fig_num: figure number.
    rot_deg: rotation of each subfigure.
    scale_fig: data range to be displayed (black and white points).
    avg_size: downsample factor in voxels (0 to avoid subsampling).
    title_fig: title.
    
    Example:
    imagesc3d2(chi_cosmos, pos=size(chi_cosmos) // 2, fig_num=1, rot_deg=[90, 90, -90], scale_fig=[-0.10, 0.14], title_fig='COSMOS')
    
    """
    if np.linalg.norm(np.imag(img)) > 0:
        img = np.abs(img)

    if pos is None:

        pos = np.round(np.array(img.shape) / 2).astype(int)

    if scale_fig is None:
        scale_fig = [np.min(img), np.max(img), np.min(img), np.max(img)]
    
    def compute_avg_size(avg_size):
        if avg_size > 0:
            if avg_size % 2:
                Lavg_size = (avg_size - 1) // 2
                Ravg_size = (avg_size - 1) // 2
            else:
                Lavg_size = (avg_size // 2) - 1
                Ravg_size = avg_size // 2
            return Lavg_size, Ravg_size
        return 0, 0
    
    Lavg_size, Ravg_size = compute_avg_size(avg_size)

    # Obtain the sections of the img matrix
    section_x = np.squeeze(np.mean(img[pos[0] - Lavg_size:pos[0] + 1 + Ravg_size, :, :], axis=0))
    section_y = np.squeeze(np.mean(img[:, pos[1]- Lavg_size:pos[1] + 1 + Ravg_size, :], axis=1))
    section_z = np.mean(img[:, :, pos[2]- Lavg_size:pos[2] + 1 + Ravg_size], axis=2)

    # Add padding so that the sections have the same dimensions
    max_dim = max(section_x.shape[0], section_x.shape[1], section_y.shape[0], section_y.shape[1])
    section_x = np.pad(section_x, ((0, max_dim - section_x.shape[0]), (0, max_dim - section_x.shape[1])), mode='constant', constant_values=0)
    section_y = np.pad(section_y, ((0, max_dim - section_y.shape[0]), (0, max_dim - section_y.shape[1])), mode='constant', constant_values=0)
    
    # Apply rotation to the sections
    rotated_section_x = rotate(section_x, angle=rot_deg[0], reshape=False)
    rotated_section_y = rotate(section_y, angle=rot_deg[1], reshape=False)
    rotated_section_z = rotate(section_z, angle=rot_deg[2], reshape=False)
    
    # Create a figure with three subplots
    plt.figure(1, figsize=(12, 4), dpi=200, facecolor='black')

    plt.subplot(131)
    plt.imshow(rotated_section_x, cmap=cmap, vmin = contrast[0], vmax = contrast[1], extent=scale_fig, origin='upper',aspect = aspect,interpolation=interp)
    plt.axis('image')
    #plt.xlim(-0.12, 0.12)
    #plt.ylim(-0.12, -0.03)
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(rotated_section_y, cmap=cmap, vmin = contrast[0], vmax = contrast[1], extent=scale_fig, origin='upper',aspect = aspect,interpolation=interp)
    plt.axis('image')
    #plt.xlim(-0.12, 0.12)
    #plt.ylim(-0.12, -0.03)
    plt.axis('off')
    plt.title(title_fig, color='w', fontsize=24)

    plt.subplot(133)
    plt.imshow(rotated_section_z, cmap=cmap, vmin = contrast[0], vmax = contrast[1], extent=scale_fig, origin='upper',aspect=aspect,interpolation=interp)
    plt.axis('image')
    plt.axis('off')


    plt.tight_layout()
    plt.show()
    
    
def compute_rmse(chi_recon, chi_true):
    """
    
    Calculates the L2 norm of the difference between two images.
    
    """
    rmse = 100 * np.linalg.norm(chi_recon - chi_true) / np.linalg.norm(chi_true)
    
    return rmse