"""
Utility functions for the continuum robot trajectory tracking system.

Converted from MATLAB files:
  - rodRotMat.m
  - computeRotationMatrix.m
  - makeHomogeneousTransform.m
  - generate_trajectories_inputs.m
"""

import numpy as np


def rod_rot_mat(k, theta):
    """
    Rodrigues' rotation formula.

    Computes the 3x3 rotation matrix for a rotation of angle `theta`
    about the unit axis `k`.

    Parameters
    ----------
    k : array_like, shape (3,)
        Rotation axis (will be normalized).
    theta : float
        Rotation angle in radians.

    Returns
    -------
    R : ndarray, shape (3, 3)
        Rotation matrix.

    Converted from: rodRotMat.m
    """
    k = np.asarray(k, dtype=float).flatten()
    k = k / np.linalg.norm(k)

    K = np.array([
        [    0,   -k[2],  k[1]],
        [ k[2],      0,  -k[0]],
        [-k[1],   k[0],     0 ]
    ])

    R = (np.cos(theta) * np.eye(3)
         + (1 - np.cos(theta)) * np.outer(k, k)
         + np.sin(theta) * K)
    return R


def compute_rotation_matrix(p_i, theta_i):
    """
    Compute rotation matrix via Rodrigues' formula about axis = [1,0,0] x p_i.

    Parameters
    ----------
    p_i : array_like, shape (3,)
        Position vector [x_i, y_i, z_i].
    theta_i : float
        Rotation angle in radians.

    Returns
    -------
    R_i : ndarray, shape (3, 3)
        Rotation matrix.

    Converted from: computeRotationMatrix.m
    """
    p_i = np.asarray(p_i, dtype=float).flatten()
    a = np.array([1.0, 0.0, 0.0])

    # Compute the rotation axis (cross product)
    w = np.cross(a, p_i)

    # Handle the degenerate case where axis is zero
    norm_w = np.linalg.norm(w)
    if norm_w < np.finfo(float).eps:
        return np.eye(3)

    # Unit rotation axis
    k = w / norm_w

    # Skew-symmetric matrix of k
    K = np.array([
        [    0,   -k[2],  k[1]],
        [ k[2],      0,  -k[0]],
        [-k[1],   k[0],     0 ]
    ])

    # Rodrigues' formula
    R_i = (np.cos(theta_i) * np.eye(3)
           + (1 - np.cos(theta_i)) * np.outer(k, k)
           + np.sin(theta_i) * K)
    return R_i


def make_homogeneous_transform(R_i, p_i):
    """
    Build a 4x4 homogeneous transformation matrix.

    Parameters
    ----------
    R_i : ndarray, shape (3, 3)
        Rotation matrix.
    p_i : array_like, shape (3,)
        Position vector.

    Returns
    -------
    H_i : ndarray, shape (4, 4)
        Homogeneous transform [R_i, p_i; 0 0 0 1].

    Converted from: makeHomogeneousTransform.m
    """
    R_i = np.asarray(R_i, dtype=float)
    p_i = np.asarray(p_i, dtype=float).flatten()

    assert R_i.shape == (3, 3), "R_i must be 3x3"
    assert p_i.shape == (3,), "p_i must be length 3"

    H_i = np.eye(4)
    H_i[:3, :3] = R_i
    H_i[:3, 3] = p_i
    return H_i


def generate_trajectories_inputs(num_sec, alpha_bounds, lv_bounds, num_points=1000):
    """
    Generate trajectory parameters for multiple sections.

    Returns trajectories as matrices where each column corresponds to a section.

    Parameters
    ----------
    num_sec : int
        Number of sections.
    alpha_bounds : array_like, shape (num_sec, 2)
        Each row is [lower, upper] bounds for alpha of that section.
    lv_bounds : array_like, shape (num_sec, 2)
        Each row is [lower, upper] bounds for lv of that section.
    num_points : int, optional
        Number of points in each trajectory (default: 1000).

    Returns
    -------
    alpha_traj : ndarray, shape (num_points, num_sec)
        Alpha trajectory for each section (columns).
    lv_traj : ndarray, shape (num_points, num_sec)
        Cable extension trajectory for each section (columns).

    Converted from: generate_trajectories_inputs.m
    """
    alpha_bounds = np.asarray(alpha_bounds, dtype=float)
    lv_bounds = np.asarray(lv_bounds, dtype=float)

    if alpha_bounds.shape != (num_sec, 2):
        raise ValueError(
            f"alpha_bounds must be a {num_sec}x2 matrix with [lower,upper] bounds for each section"
        )
    if lv_bounds.shape != (num_sec, 2):
        raise ValueError(
            f"lv_bounds must be a {num_sec}x2 matrix with [lower,upper] bounds for each section"
        )

    alpha_traj = np.zeros((num_points, num_sec))
    lv_traj = np.zeros((num_points, num_sec))

    for i in range(num_sec):
        alpha_lower = alpha_bounds[i, 0]
        alpha_upper = alpha_bounds[i, 1]
        lv_lower = lv_bounds[i, 0]
        lv_upper = lv_bounds[i, 1]

        alpha_traj[:, i] = np.linspace(alpha_lower, alpha_upper, num_points)
        lv_traj[:, i] = np.linspace(lv_lower, lv_upper, num_points)

    return alpha_traj, lv_traj
