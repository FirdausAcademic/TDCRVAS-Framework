"""
Jacobian functions for the continuum robot.

Converted from MATLAB files:
  - localsingleSecTipJacobian.m
  - partialHI.m
  - genMultiSecJacobianFixedBase.m
  - compoundJacobianMultiSection.m
"""

import numpy as np
from kinematics import forward_kinematics_single_section, forward_kinematics_multi_section
from utils import compute_rotation_matrix, make_homogeneous_transform


def local_single_sec_tip_jacobian(Li, rd, lv, alpha):
    """
    Compute the 3x2 Jacobian of the tip of a single continuum section.

    J maps [d(lv)/dt; d(alpha)/dt] -> d(p_tip)/dt

    Parameters
    ----------
    Li : float
        Nominal (undeformed) arc-length of section i.
    rd : float
        Disk-radius parameter.
    lv : float
        Current cable extension delta_l_i.
    alpha : float
        Bending-plane angle alpha_i [rad].

    Returns
    -------
    J : ndarray, shape (3, 2)
        Jacobian matrix.

    Converted from: localsingleSecTipJacobian.m
    """
    # Compute intermediate variables
    theta = lv / rd
    Ri = Li * rd / lv

    # Common term for the (2,1) and (3,1) entries
    common = -Ri / lv * (1 - np.cos(theta)) + (Ri / rd) * np.sin(theta)

    # Assemble Jacobian
    J = np.zeros((3, 2))

    # dp/dlv
    J[0, 0] = -Ri / lv * np.sin(theta) + (Ri / rd) * np.cos(theta)
    J[1, 0] = common * np.cos(alpha)
    J[2, 0] = common * np.sin(alpha)

    # dp/dalpha
    J[0, 1] = 0.0
    J[1, 1] = -Ri * (1 - np.cos(theta)) * np.sin(alpha)
    J[2, 1] = Ri * (1 - np.cos(theta)) * np.cos(alpha)

    return J


def partial_hi(Li, rd, lv, alpha):
    """
    Compute dH_i/dlv and dH_i/dalpha for one continuum section.

    Uses local_single_sec_tip_jacobian to get J_i = [dp/dlv, dp/dalpha].

    Parameters
    ----------
    Li : float
        Undeformed arc-length of section i.
    rd : float
        Disk-radius parameter.
    lv : float
        Current cable extension delta_l_i.
    alpha : float
        Bending-plane angle alpha_i [rad].

    Returns
    -------
    dH_dlv : ndarray, shape (4, 4)
        Partial derivative of H_i w.r.t. lv.
    dH_dalpha : ndarray, shape (4, 4)
        Partial derivative of H_i w.r.t. alpha.

    Converted from: partialHI.m
    """
    # Fixed world axis
    a = np.array([1.0, 0.0, 0.0])

    # 1) tip position and Jacobian
    p_i = forward_kinematics_single_section(Li, rd, lv, alpha)
    J = local_single_sec_tip_jacobian(Li, rd, lv, alpha)
    dp_dlv = J[:, 0]
    dp_dalpha = J[:, 1]

    # 2) build rotation axis and unit axis
    w = np.cross(a, p_i)
    nw = np.linalg.norm(w)
    if nw < np.finfo(float).eps:
        k = np.array([1.0, 0.0, 0.0])
    else:
        k = w / nw

    # 3) Rodrigues parameters
    theta = lv / rd
    dtheta_dlv = 1.0 / rd

    K = np.array([
        [    0,   -k[2],  k[1]],
        [ k[2],      0,  -k[0]],
        [-k[1],   k[0],     0 ]
    ])

    # 4) dR/dtheta (treating k constant)
    dR_dtheta = (-np.sin(theta) * np.eye(3)
                 + np.sin(theta) * np.outer(k, k)
                 + np.cos(theta) * K)

    # 5) dw/dlv and dw/dalpha
    dw_dlv = np.cross(a, dp_dlv)
    dw_dalpha = np.cross(a, dp_dalpha)

    # 6) dk/dlv and dk/dalpha
    if nw < np.finfo(float).eps:
        dk_dlv = np.zeros(3)
        dk_dalpha = np.zeros(3)
    else:
        P = np.eye(3) - np.outer(k, k)
        dk_dlv = (P @ dw_dlv) / nw
        dk_dalpha = (P @ dw_dalpha) / nw

    # 7) precompute dR/dk_j for j=0,1,2
    dR_dkj = np.zeros((3, 3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = 1.0
        dR_dkj[:, :, j] = ((1 - np.cos(theta)) * (np.outer(e, k) + np.outer(k, e))
                            + np.sin(theta) * np.array([
                                [    0,   -e[2],  e[1]],
                                [ e[2],      0,  -e[0]],
                                [-e[1],   e[0],     0 ]
                            ]))

    # 8) assemble dR/dlv and dR/dalpha
    dR_dlv = dR_dtheta * dtheta_dlv
    dR_dalpha = np.zeros((3, 3))
    for j in range(3):
        dR_dlv = dR_dlv + dR_dkj[:, :, j] * dk_dlv[j]
        dR_dalpha = dR_dalpha + dR_dkj[:, :, j] * dk_dalpha[j]

    # 9) build the 4x4 partial-transform matrices
    dH_dlv = np.zeros((4, 4))
    dH_dlv[:3, :3] = dR_dlv
    dH_dlv[:3, 3] = dp_dlv

    dH_dalpha = np.zeros((4, 4))
    dH_dalpha[:3, :3] = dR_dalpha
    dH_dalpha[:3, 3] = dp_dalpha

    return dH_dlv, dH_dalpha


def gen_multi_sec_jacobian_fixed_base(Ls, rd, lv, alpha):
    """
    Compute the 3x(2n) Jacobian for an n-section continuum robot
    with the base fixed.

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
    J : ndarray, shape (3, 2n)
        Jacobian mapping [dot_lv; dot_alpha] -> dot_p_tip (in base frame).

    Converted from: genMultiSecJacobianFixedBase.m
    """
    Ls = np.asarray(Ls, dtype=float).flatten()
    lv = np.asarray(lv, dtype=float).flatten()
    alpha = np.asarray(alpha, dtype=float).flatten()
    n = len(Ls)

    # 1) Get each section's H_i and the tip in the (fixed) base frame
    _, _, H_all = forward_kinematics_multi_section(Ls, rd, lv, alpha)

    # 2) Build prefix and suffix chains starting from identity
    P = np.zeros((4, 4, n + 1))
    P[:, :, 0] = np.eye(4)
    for i in range(n):
        P[:, :, i + 1] = P[:, :, i] @ H_all[:, :, i]

    S = np.zeros((4, 4, n + 2))
    S[:, :, n + 1] = np.eye(4)  # extra slot for boundary
    S[:, :, n] = np.eye(4)
    for i in range(n - 1, -1, -1):
        S[:, :, i] = H_all[:, :, i] @ S[:, :, i + 1]

    # 3) Shape-motion Jacobian: two columns per section
    J = np.zeros((3, 2 * n))

    # Sections 1..n-1: full dH_i/dlv_i, dH_i/dalpha_i
    for i in range(n - 1):
        dH_dlv, dH_dalpha = partial_hi(Ls[i], rd, lv[i], alpha[i])
        T_l = P[:, :, i] @ dH_dlv @ S[:, :, i + 1]
        T_alpha = P[:, :, i] @ dH_dalpha @ S[:, :, i + 1]

        J[:, 2 * i] = T_l[:3, 3]
        J[:, 2 * i + 1] = T_alpha[:3, 3]

    # Section n: only dp_n/dlv_n, dp_n/dalpha_n (no dR_n terms)
    Jn = local_single_sec_tip_jacobian(Ls[n - 1], rd, lv[n - 1], alpha[n - 1])
    dp_dlv_n = Jn[:, 0]
    dp_dalpha_n = Jn[:, 1]

    # The rotation prefix R_prefix = P(1:3,1:3,n)
    R_prefix = P[:3, :3, n - 1]

    J[:, 2 * (n - 1)] = R_prefix @ dp_dlv_n
    J[:, 2 * (n - 1) + 1] = R_prefix @ dp_dalpha_n

    return J


def compound_jacobian_multi_section(Ls, rd, lv, alpha):
    """
    Compute the compound (3n x 2n) Jacobian for all intermediate tip
    velocities of an n-section continuum robot with fixed base.

    Parameters
    ----------
    Ls : array_like, shape (n,)
        Undeformed arc-lengths [L1, ..., Ln].
    rd : float
        Disk-radius parameter.
    lv : array_like, shape (n,)
        Current cable extensions [lv1, ..., lvn].
    alpha : array_like, shape (n,)
        Bending-plane angles [alpha1, ..., alphan].

    Returns
    -------
    J_comp : ndarray, shape (3n, 2n)
        Compound Jacobian such that
        [dot_p_1; ...; dot_p_n] = J_comp * [dot_lv; dot_alpha].

    Converted from: compoundJacobianMultiSection.m
    """
    Ls = np.asarray(Ls, dtype=float).flatten()
    lv = np.asarray(lv, dtype=float).flatten()
    alpha = np.asarray(alpha, dtype=float).flatten()
    n = len(Ls)

    J_comp = np.zeros((3 * n, 2 * n))

    for k in range(n):
        # Compute Jacobian for the k-th tip using the first k+1 sections
        Ls_k = Ls[:k + 1]
        lv_k = lv[:k + 1]
        alpha_k = alpha[:k + 1]
        J_k_partial = gen_multi_sec_jacobian_fixed_base(Ls_k, rd, lv_k, alpha_k)  # 3 x 2(k+1)

        # Place in the k-th block row: columns 0 to 2(k+1)-1 get J_k_partial
        row_start = 3 * k
        col_end = 2 * (k + 1)
        J_comp[row_start:row_start + 3, :col_end] = J_k_partial

    return J_comp
