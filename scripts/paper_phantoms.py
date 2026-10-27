"""create_geometric_phantom copied VERBATIM from utils/Dipole-lets_Paper_Plots_3.ipynb (cell 3).

Quirk kept on purpose so the experiments match the paper notebook: it unpacks `z, y, x = np.indices(...)` but applies
center[0] to x, so the array axes seen by the caller are permuted (see artifacts.py docstring).
"""

import numpy as np


def create_geometric_phantom(
    matrix_size, 
    shape_type='sphere', 
    size=(20, 20, 20), 
    center=None, 
    rotation_angles=(0, 0, 0), 
    intensity=1.0
):
    """
    Generates a 3D geometric phantom (sphere, cube, or cylinder) with specified
    size, position, and rotation.

    Args:
        matrix_size (tuple): The shape of the output 3D array (e.g., (160, 160, 160)).
        shape_type (str): 'sphere', 'cube', or 'cylinder'.
        size (tuple or int): Dimensions of the shape.
                             - For 'sphere': (radius,)
                             - For 'cube': (side_x, side_y, side_z)
                             - For 'cylinder': (radius, height)
        center (tuple, optional): The (x, y, z) center of the shape. 
                                  Defaults to the matrix center.
        rotation_angles (tuple, optional): Euler angles (phi, theta, psi) in degrees
                                           for rotation around Z, Y, X axes respectively.
        intensity (float, optional): The value to fill the shape with.

    Returns:
        np.ndarray: A 3D array containing the geometric phantom.
    """
    if center is None:
        center = tuple(n // 2 for n in matrix_size)

    # 1. Create a coordinate grid centered at the specified location
    z, y, x = np.indices(matrix_size)
    x_c = x - center[0]
    y_c = y - center[1]
    z_c = z - center[2]
    
    # Flatten the coordinates for matrix multiplication
    coords = np.vstack([x_c.ravel(), y_c.ravel(), z_c.ravel()])

    # 2. Create the 3D rotation matrix from Euler angles
    phi, theta, psi = np.deg2rad(rotation_angles)
    
    # Rotation matrix for Z-axis (phi)
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0],
                   [np.sin(phi),  np.cos(phi), 0],
                   [0,            0,           1]])
    
    # Rotation matrix for Y-axis (theta)
    Ry = np.array([[np.cos(theta),  0, np.sin(theta)],
                   [0,              1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])

    # Rotation matrix for X-axis (psi)
    Rx = np.array([[1, 0,             0],
                   [0, np.cos(psi), -np.sin(psi)],
                   [0, np.sin(psi),  np.cos(psi)]])

    # Combined rotation matrix (Z-Y-X convention)
    R = Rz @ Ry @ Rx
    
    # 3. Apply the inverse rotation to the coordinate grid
    # (This is equivalent to rotating the object itself)
    rotated_coords = R.T @ coords
    x_rot = rotated_coords[0, :].reshape(matrix_size)
    y_rot = rotated_coords[1, :].reshape(matrix_size)
    z_rot = rotated_coords[2, :].reshape(matrix_size)

    # 4. Define the shape using its geometric equation on the rotated grid
    mask = np.zeros(matrix_size, dtype=bool)
    if shape_type.lower() == 'sphere':
        radius = size[0]
        mask = (x_rot**2 + y_rot**2 + z_rot**2) <= radius**2
        
    elif shape_type.lower() == 'cube':
        sx, sy, sz = size[0]/2, size[1]/2, size[2]/2
        mask = (np.abs(x_rot) <= sx) & (np.abs(y_rot) <= sy) & (np.abs(z_rot) <= sz)

    elif shape_type.lower() == 'cylinder':
        radius, height = size[0], size[1]/2
        # A cylinder aligned along the Z-axis in its own coordinate system
        mask = (x_rot**2 + y_rot**2 <= radius**2) & (np.abs(z_rot) <= height)
        
    else:
        raise ValueError("shape_type must be 'sphere', 'cube', or 'cylinder'")

    # 5. Create the final phantom
    phantom = np.zeros(matrix_size, dtype=np.float64)
    phantom[mask] = intensity
    
    return phantom
