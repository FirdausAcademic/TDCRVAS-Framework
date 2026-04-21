"""
Forward kinematics functions for the continuum robot.

Converted from MATLAB files:
  - forwardKinematicsSingleSection.m
  - forwardKinematicsCC.m
  - forwardKinematicsMultiSection.m
  - genForwardKinMultiSec.m
"""

import numpy as np
from utils import (
    compute_rotation_matrix,
    make_homogeneous_transform,
    rod_rot_mat,
)


def forward_kinematics_single_section(Li, rd, li, alpha_i):
    """
    Compute the local tip position p_i for a single continuum section.

    Parameters
    ----------
    Li : float
        Nominal (undeformed) arc-length of section i.
    rd : float
        Disk-radius parameter.
    li : float
        Current cable extension delta_l_i.
    alpha_i : float
        Bending-plane angle alpha_i [rad].

    Returns
    -------
    p_i : ndarray, shape (3,)
        Tip position vector in the local frame.

    Converted from: forwardKinematicsSingleSection.m
    """
    # Compute bending angle
    theta_i = li / rd

    # Compute radius of curvature
    R_i = Li * rd / li

    # Forward kinematics formula
    p_i = np.array([
        R_i * np.sin(theta_i),
        R_i * (1 - np.cos(theta_i)) * np.cos(alpha_i),
        R_i * (1 - np.cos(theta_i)) * np.sin(alpha_i),
    ])
    return p_i


def forward_kinematics_cc(L, r_d, n, del_lv, alpha, return_curve=False):
    """
    Continuum-link kinematics (tip and optional full curve).

    Parameters
    ----------
    L : float
        Total backbone length.
    r_d : float
        Disk radius.
    n : int
        Number of disks.
    del_lv : float
        Change in virtual cable length [m].
    alpha : float
        Bending-plane angle w.r.t. y-axis [rad].
    return_curve : bool, optional
        If True, also return the full backbone curve points (default: False).

    Returns
    -------
    If return_curve is False:
        tip : ndarray, shape (3,)
            Tip position [x, y, z].
        theta : float
            Total bend angle [rad].
    If return_curve is True:
        curve_pts : ndarray, shape (M, 3)
            Backbone curve points.
        tip : ndarray, shape (3,)
            Tip position [x, y, z].
        theta : float
            Total bend angle [rad].

    Converted from: forwardKinematicsCC.m
    """
    # Common preprocessing
    del_lv = abs(del_lv)
    del_lv_elem = del_lv / (n - 1)
    L_elem = L / (n - 1)

    # Radius of curvature
    R = (L_elem * r_d) / del_lv_elem

    # Total bending angle
    theta_bar = del_lv_elem / r_d
    theta = theta_bar * (n - 1)

    # Tip coordinate
    x_tip = R * np.sin(theta)
    rho = R - R * np.cos(theta)
    y_tip = rho * np.cos(alpha)
    z_tip = rho * np.sin(alpha)
    tip = np.array([x_tip, y_tip, z_tip])

    if not return_curve:
        return tip, theta

    # Generate full curve
    FCM_markerDensity = 150
    theta_i = np.linspace(0, theta, FCM_markerDensity + 1)

    cA = np.cos(alpha)
    sA = np.sin(alpha)

    x = R * np.sin(theta_i)
    rho_arr = R - R * np.cos(theta_i)
    y = rho_arr * cA
    z = rho_arr * sA
    curve_pts = np.column_stack([x, y, z])

    return curve_pts, tip, theta


def forward_kinematics_multi_section(Ls, rd, lv, alpha):
    """
    Compute tip pose of an n-section continuum robot by chaining
    single-section forward kinematics.

    Parameters
    ----------
    Ls : array_like, shape (n,)
        Undeformed arc-lengths [L1, ..., Ln].
    rd : float
        Disk-radius parameter.
    lv : array_like, shape (n,)
        Cable extensions [lv1, ..., lvn].
    alpha : array_like, shape (n,)
        Bending-plane angles [alpha1, ..., alphan].

    Returns
    -------
    p_tip : ndarray, shape (3,)
        Position of the final tip in the base frame.
    R_tip : ndarray, shape (3, 3)
        Orientation (rotation matrix) of the final tip in the base frame.
    H_all : ndarray, shape (4, 4, n)
        Array of homogeneous transforms for each section.

    Converted from: forwardKinematicsMultiSection.m
    """
    Ls = np.asarray(Ls, dtype=float).flatten()
    lv = np.asarray(lv, dtype=float).flatten()
    alpha = np.asarray(alpha, dtype=float).flatten()
    n_sec = len(Ls)

    H_all = np.zeros((4, 4, n_sec))
    H = np.eye(4)

    for i in range(n_sec):
        # 1) local tip position p_i
        p_i = forward_kinematics_single_section(Ls[i], rd, lv[i], alpha[i])

        # 2) bending angle theta_i
        theta_i = lv[i] / rd

        # 3) rotation about axis [1,0,0] x p_i
        R_i = compute_rotation_matrix(p_i, theta_i)

        # 4) build homogeneous transform H_i
        H_i = make_homogeneous_transform(R_i, p_i)
        H_all[:, :, i] = H_i

        # 5) accumulate into global transform
        H = H @ H_i

    # Extract final tip pose
    p_tip = H[:3, 3]
    R_tip = H[:3, :3]
    return p_tip, R_tip, H_all


def gen_forward_kin_multi_sec(Ls, r_d, Ns, del_lvs, alphas):
    """
    Generate forward kinematics for a multi-section continuum robot,
    returning backbone curve points, section tips, and frame data.

    Parameters
    ----------
    Ls : array_like, shape (n,)
        Section lengths [m].
    r_d : float or array_like
        Disk radii [m]. Scalar or per-section.
    Ns : array_like, shape (n,)
        Number of disks per section.
    del_lvs : array_like, shape (n,)
        Cable extensions [m].
    alphas : array_like, shape (n,)
        Bending-plane angles [rad].

    Returns
    -------
    SecCords : ndarray, shape (M, 3)
        All backbone curve points concatenated.
    SecTips : ndarray, shape (3, n)
        Tip positions for each section in world frame.
    Xaxis : ndarray, shape (3, n+1)
        X-axis directions for base + each section frame.
    Yaxis : ndarray, shape (3, n+1)
        Y-axis directions for base + each section frame.
    Zaxis : ndarray, shape (3, n+1)
        Z-axis directions for base + each section frame.
    PosFrames : ndarray, shape (3, n+1)
        Frame origins for base + each section.

    Converted from: genForwardKinMultiSec.m
    """
    Ls = np.asarray(Ls, dtype=float).flatten()
    Ns = np.asarray(Ns, dtype=int).flatten()
    del_lvs = np.asarray(del_lvs, dtype=float).flatten()
    alphas = np.asarray(alphas, dtype=float).flatten()
    n_sec = len(Ls)

    if np.isscalar(r_d) or (hasattr(r_d, '__len__') and len(np.asarray(r_d).flatten()) == 1):
        r_d_arr = np.full(n_sec, float(np.asarray(r_d).flatten()[0]))
    else:
        r_d_arr = np.asarray(r_d, dtype=float).flatten()

    # Preallocate outputs
    SecCords = np.empty((0, 3))
    SecTips = np.zeros((3, n_sec))
    Xaxis = np.zeros((3, n_sec + 1))
    Yaxis = np.zeros((3, n_sec + 1))
    Zaxis = np.zeros((3, n_sec + 1))
    PosFrames = np.zeros((3, n_sec + 1))

    # Base frame axes/origin
    Xaxis[:, 0] = [1, 0, 0]
    Yaxis[:, 0] = [0, 1, 0]
    Zaxis[:, 0] = [0, 0, 1]
    PosFrames[:, 0] = [0, 0, 0]

    Tbase = np.eye(4)

    for i in range(n_sec):
        # 1) compute this section in its own local frame
        curve_local, tip_local, theta = forward_kinematics_cc(
            Ls[i], r_d_arr[i], Ns[i], del_lvs[i], alphas[i], return_curve=True
        )

        # 2) lift to homogeneous and map into world
        n_pts = curve_local.shape[0]
        hom_L = np.vstack([curve_local.T, np.ones((1, n_pts))])  # 4 x nPts
        hom_W = Tbase @ hom_L  # 4 x nPts
        SecCords = np.vstack([SecCords, hom_W[:3, :].T])  # append N x 3

        # 3) build the local bend transform
        #    axis in local frame = cross([1,0,0], tip_local)
        axis_L = np.cross(np.array([1, 0, 0]), tip_local)
        if np.linalg.norm(axis_L) < np.finfo(float).eps:
            axis_L = np.array([0, 0, 1.0])
        axis_L = axis_L / np.linalg.norm(axis_L)
        R_loc = rod_rot_mat(axis_L, theta)

        # translation in local frame = tip_local
        H_sect_local = np.eye(4)
        H_sect_local[:3, :3] = R_loc
        H_sect_local[:3, 3] = tip_local

        # 4) update cumulative world frame
        Tbase = Tbase @ H_sect_local

        # 5) record new frame's axes & origin
        Xaxis[:, i + 1] = Tbase[:3, 0]
        Yaxis[:, i + 1] = Tbase[:3, 1]
        Zaxis[:, i + 1] = Tbase[:3, 2]
        PosFrames[:, i + 1] = Tbase[:3, 3]

        # Record this section's tip in world frame
        SecTips[:, i] = Tbase[:3, 3]

    return SecCords, SecTips, Xaxis, Yaxis, Zaxis, PosFrames
