"""
Main trajectory tracking script for a multi-section continuum robot
with fixed base and closed-loop control.

Features a tkinter GUI for inputting Ls (section lengths) and Ns (number of disks).

Converted from: Main_Trjectory_traking_code_for_anySec_fixed_base.m
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from utils import generate_trajectories_inputs
from kinematics import gen_forward_kin_multi_sec
from jacobians import compound_jacobian_multi_section


# =============================================================================
# Simulation function (runs in a separate thread)
# =============================================================================
def run_simulation(Ls, r_d, Ns, alpha_bounds, lv_bounds, status_var=None):
    """



    Run the full trajectory tracking simulation and generate all plots.

    Parameters
    ----------
    Ls : ndarray
        Section lengths [m].
    r_d : float
        Disk radius [m].
    Ns : ndarray of int
        Number of disks per section.
    status_var : tk.StringVar or None
        Optional tkinter StringVar to update GUI status label.
    """

    def update_status(msg):
        print(msg)
        if status_var is not None:
            status_var.set(msg)

    num_sec = len(Ls)
    num_points = 1000

    update_status("Generating trajectory inputs...")
    alphas, del_lvs = generate_trajectories_inputs(num_sec, alpha_bounds, lv_bounds, num_points)

    # =========================================================================
    # 2) Run the multi-section FK routine (single configuration)
    # =========================================================================
    update_status("Computing initial FK and Jacobian...")

    SecCords, SecTips, Xaxis, Yaxis, Zaxis, PosFrames = \
        gen_forward_kin_multi_sec(Ls, r_d, Ns, del_lvs[0, :], alphas[0, :])

    J = compound_jacobian_multi_section(Ls, r_d, del_lvs[0, :], alphas[0, :])

    # =========================================================================
    # 3) Plot the backbone curve
    # =========================================================================
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(SecCords[:, 0], SecCords[:, 1], SecCords[:, 2],
            'o', markersize=3, linewidth=1.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Multi-Section Continuum Robot Backbone')

    # 4) Overlay each section's local frame axes
    n_sec_frames = Xaxis.shape[1] - 1
    scale = 0.05
    for i in range(n_sec_frames + 1):
        O = PosFrames[:, i]
        ax.quiver(O[0], O[1], O[2],
                  scale * Xaxis[0, i], scale * Xaxis[1, i], scale * Xaxis[2, i],
                  color='r', linewidth=0.8)
        ax.quiver(O[0], O[1], O[2],
                  scale * Yaxis[0, i], scale * Yaxis[1, i], scale * Yaxis[2, i],
                  color='g', linewidth=0.8)
        ax.quiver(O[0], O[1], O[2],
                  scale * Zaxis[0, i], scale * Zaxis[1, i], scale * Zaxis[2, i],
                  color='k', linewidth=0.8)

    # =========================================================================
    # 5) Loop through all trajectory points for FK
    # =========================================================================
    SecTips_all = np.zeros((num_points, 3, num_sec))

    for i in range(num_points):
        current_lv = del_lvs[i, :]
        current_alpha = alphas[i, :]

        _, sec_tips_i, _, _, _, _ = \
            gen_forward_kin_multi_sec(Ls, r_d, Ns, current_lv, current_alpha)

        SecTips_all[i, :, :] = sec_tips_i

        if (i + 1) % 100 == 0:
            update_status(f"FK trajectory: {i + 1}/{num_points}")

    # =========================================================================
    # 6) Plot 3D trajectories of all coordinate sets
    # =========================================================================
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')

    colors = plt.cm.tab10(np.linspace(0, 1, max(num_sec, 10)))[:num_sec]
    set_labels = [f'Set {s + 1}' for s in range(num_sec)]

    for s in range(num_sec):
        x = SecTips_all[:, 0, s]
        y = SecTips_all[:, 1, s]
        z = SecTips_all[:, 2, s]

        ax2.plot(x, y, z, color=colors[s], linewidth=2, label=set_labels[s])
        ax2.scatter(x[0], y[0], z[0], s=100, color=colors[s],
                    edgecolors='k', linewidths=1.5, zorder=5)
        ax2.scatter(x[-1], y[-1], z[-1], s=100, color=colors[s],
                    marker='s', edgecolors='k', linewidths=1.5, zorder=5)

    ax2.set_xlabel(r'$X$', fontsize=14)
    ax2.set_ylabel(r'$Y$', fontsize=14)
    ax2.set_zlabel(r'$Z$', fontsize=14)
    ax2.set_title('3D Trajectories of All Coordinate Sets', fontsize=16)
    ax2.legend(fontsize=12, loc='best')

    # =========================================================================
    # 7) Compute velocity via finite differences
    # =========================================================================
    dt = 1.0 / 1000.0
    velocity_all = np.zeros((num_points - 1, 3, num_sec))

    for s in range(num_sec):
        for dim in range(3):
            position_data = SecTips_all[:, dim, s]
            velocity_all[:, dim, s] = np.diff(position_data) / dt

    update_status("Velocity calculation completed.")
    for s in range(num_sec):
        vel_mag = np.linalg.norm(velocity_all[:, :, s], axis=1)
        print(f"  Set {s + 1} - Velocity magnitude range: [{vel_mag.min():.4f}, {vel_mag.max():.4f}] units/s")

    # =========================================================================
    # 8) Closed-loop control
    # =========================================================================
    lv_initial = del_lvs[0, :]
    alphas_initial = alphas[0, :]

    v = np.zeros(2 * num_sec)
    v[0::2] = lv_initial
    v[1::2] = alphas_initial

    x_tracking = []
    TdcrSec_list = []
    Time = [0.0]
    time_temp = 0.0

    num_time_steps = num_points - 1

    PosFrames_all_ctrl = [None] * num_time_steps
    Xaxis_all_ctrl = [None] * num_time_steps
    Yaxis_all_ctrl = [None] * num_time_steps
    Zaxis_all_ctrl = [None] * num_time_steps

    v_history = np.zeros((2 * num_sec, num_time_steps + 1))
    v_history[:, 0] = v

    for k in range(num_time_steps):
        current_lv = v_history[0::2, k]
        current_alpha = v_history[1::2, k]

        J_ctrl = compound_jacobian_multi_section(Ls, r_d, current_lv, current_alpha)

        TdcrSecCurrent, SecTipsFeedback, Xax, Yax, Zax, PosF = \
            gen_forward_kin_multi_sec(Ls, r_d, Ns, current_lv, current_alpha)

        x = SecTipsFeedback.flatten(order='F')
        xd = np.concatenate([SecTips_all[k, :, s] for s in range(num_sec)])
        vel = np.concatenate([velocity_all[k, :, s] for s in range(num_sec)])

        err = xd - x
        x_tracking.append(x.copy())
        TdcrSec_list.append(TdcrSecCurrent.copy())

        PosFrames_all_ctrl[k] = PosF
        Xaxis_all_ctrl[k] = Xax
        Yaxis_all_ctrl[k] = Yax
        Zaxis_all_ctrl[k] = Zax

        J_pinv = np.linalg.pinv(J_ctrl)
        v_history[:, k + 1] = v_history[:, k] + J_pinv @ (vel + 30 * err) * dt

        time_temp += dt
        Time.append(time_temp)

        if (k + 1) % 100 == 0:
            update_status(f"Control loop: {k + 1}/{num_time_steps}")

    x_tracking = np.array(x_tracking).T
    Time = np.array(Time)

    # =========================================================================
    # 9) Plot reference vs tracked trajectories (3D)
    # =========================================================================
    num_track_points = x_tracking.shape[1]

    fig3 = plt.figure(figsize=(12, 9))
    ax3 = fig3.add_subplot(111, projection='3d')

    for s in range(num_sec):
        x_ref = SecTips_all[:, 0, s]
        y_ref = SecTips_all[:, 1, s]
        z_ref = SecTips_all[:, 2, s]
        ax3.plot(x_ref, y_ref, z_ref, color=colors[s], linestyle='-', linewidth=2,
                 label=f'Section {s + 1} Ref')
        ax3.scatter(x_ref[0], y_ref[0], z_ref[0], s=100, color=colors[s],
                    marker='o', edgecolors='k', linewidths=1.5)
        ax3.scatter(x_ref[-1], y_ref[-1], z_ref[-1], s=100, color=colors[s],
                    marker='s', edgecolors='k', linewidths=1.5)

        row_start = 3 * s
        x_track = x_tracking[row_start, :]
        y_track = x_tracking[row_start + 1, :]
        z_track = x_tracking[row_start + 2, :]
        ax3.plot(x_track, y_track, z_track, color=colors[s], linestyle='--', linewidth=4,
                 label=f'Section {s + 1} Track')
        ax3.scatter(x_track[0], y_track[0], z_track[0], s=100, color=colors[s],
                    marker='^', edgecolors='k', linewidths=1.5)
        ax3.scatter(x_track[-1], y_track[-1], z_track[-1], s=100, color=colors[s],
                    marker='d', edgecolors='k', linewidths=1.5)

    ax3.set_xlabel(r'$X$ (m)', fontsize=14)
    ax3.set_ylabel(r'$Y$ (m)', fontsize=14)
    ax3.set_zlabel(r'$Z$ (m)', fontsize=14)
    ax3.set_title('3D Reference and Tracked Trajectories for All Sections', fontsize=16)
    ax3.legend(fontsize=11, loc='best')
    ax3.view_init(elev=30, azim=45)

    # =========================================================================
    # 10) Time-series plotting: reference vs tracked coordinates
    # =========================================================================
    time_ref = Time[:num_points]
    time_track = Time[:num_track_points]

    y_labels = []
    for s in range(num_sec):
        for coord in range(3):
            y_labels.append(rf'$\zeta_{{{s + 1}}}({coord + 1})$')

    fig4, axes = plt.subplots(num_sec, 3, figsize=(12, 2.5 * num_sec))
    if num_sec == 1:
        axes = axes.reshape(1, -1)

    for s in range(num_sec):
        for coord in range(3):
            ax_sub = axes[s, coord]

            ref_data = SecTips_all[:, coord, s]
            ax_sub.plot(time_ref, ref_data, color=[0, 0.4470, 0.7410],
                        linestyle='-', linewidth=2, label='Desired Trajectory')

            row_idx = 3 * s + coord
            track_data = x_tracking[row_idx, :]
            ax_sub.plot(time_track, track_data, color=[0.8500, 0.3250, 0.0980],
                        linestyle='--', linewidth=2, label='Actual Trajectory')

            ax_sub.set_xlim([Time.min(), Time.max()])
            ax_sub.set_xlabel('Time (s)', fontsize=10)
            ax_sub.set_ylabel(y_labels[3 * s + coord] + ' (mm)', fontsize=10)
            ax_sub.grid(True, which='both', linestyle=':', alpha=0.7)

    axes[0, 0].legend(['Desired Trajectory', 'Actual Trajectory'],
                       loc='upper right', fontsize=9)
    fig4.tight_layout()

    # =========================================================================
    # 11) 3D Segmented plotting of continuum robot with frames
    # =========================================================================
    ntdcr_plot = 4
    n_temp = len(TdcrSec_list)
    indices = np.round(np.linspace(0, n_temp - 1, ntdcr_plot)).astype(int)

    fig5 = plt.figure(figsize=(10, 8))
    ax5 = fig5.add_subplot(111, projection='3d')

    for idx_i in range(ntdcr_plot):
        time_k = indices[idx_i]
        TdcrSecPlot = TdcrSec_list[time_k]

        N = TdcrSecPlot.shape[0]
        points_per_sec = N // num_sec

        if N % num_sec != 0:
            print(f"Warning: TdcrSec rows ({N}) not evenly divisible by numSec ({num_sec}).")
            points_per_sec = N // num_sec

        Xax = Xaxis_all_ctrl[time_k]
        Yax = Yaxis_all_ctrl[time_k]
        Zax = Zaxis_all_ctrl[time_k]
        PosF = PosFrames_all_ctrl[time_k]

        colors_sec = plt.cm.tab10(np.linspace(0, 1, max(num_sec, 10)))[:num_sec]

        ax5.plot(SecTips_all[:, 0, num_sec - 1],
                 SecTips_all[:, 1, num_sec - 1],
                 SecTips_all[:, 2, num_sec - 1],
                 color=[0, 0.4470, 0.7410], linestyle='-', linewidth=2)
        ax5.plot(x_tracking[-3, :], x_tracking[-2, :], x_tracking[-1, :],
                 color=[0.8500, 0.3250, 0.0980], linestyle='--', linewidth=3)

        for sec in range(num_sec):
            idx_start = sec * points_per_sec
            idx_end = min((sec + 1) * points_per_sec, N)
            seg_data = TdcrSecPlot[idx_start:idx_end, :]

            ax5.plot(seg_data[:, 0], seg_data[:, 1], seg_data[:, 2],
                     'o', color=colors_sec[sec], markersize=3, linewidth=1.5)

        n_sec_frames = Xax.shape[1] - 1
        scale_frame = 0.05
        for frame_idx in range(n_sec_frames + 1):
            O = PosF[:, frame_idx]
            ax5.quiver(O[0], O[1], O[2],
                       scale_frame * Xax[0, frame_idx], scale_frame * Xax[1, frame_idx], scale_frame * Xax[2, frame_idx],
                       color='r', linewidth=0.8)
            ax5.quiver(O[0], O[1], O[2],
                       scale_frame * Yax[0, frame_idx], scale_frame * Yax[1, frame_idx], scale_frame * Yax[2, frame_idx],
                       color='g', linewidth=0.8)
            ax5.quiver(O[0], O[1], O[2],
                       scale_frame * Zax[0, frame_idx], scale_frame * Zax[1, frame_idx], scale_frame * Zax[2, frame_idx],
                       color='b', linewidth=0.8)

    ax5.set_xlabel(r'$\zeta_1$ (m)', fontsize=14)
    ax5.set_ylabel(r'$\zeta_2$ (m)', fontsize=14)
    ax5.set_zlabel(r'$\zeta_3$ (m)', fontsize=14)
    ax5.view_init(elev=30, azim=45)

    update_status("Simulation complete! Showing plots...")
    plt.show()


# =============================================================================
# GUI Application
# =============================================================================
class TrajectoryTrackingGUI:
    """Tkinter GUI for inputting robot parameters and launching the simulation."""

    def __init__(self, root):
        self.root = root
        self.root.title("Continuum Robot Trajectory Tracking")
        self.root.resizable(True, True)

        # --- Style ---
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 10), foreground="blue")

        # --- Main frame ---
        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        ttk.Label(main_frame, text="Continuum Robot Trajectory Tracking",
                  style="Title.TLabel").pack(pady=(0, 15))

        # --- Number of sections ---
        sec_frame = ttk.LabelFrame(main_frame, text="Robot Parameters", padding=10)
        sec_frame.pack(fill=tk.X, pady=(0, 10))

        row_frame = ttk.Frame(sec_frame)
        row_frame.pack(fill=tk.X)

        ttk.Label(row_frame, text="Number of Sections:", style="Header.TLabel").pack(side=tk.LEFT, padx=(0, 10))
        self.num_sec_var = tk.IntVar(value=5)
        self.num_sec_spin = ttk.Spinbox(row_frame, from_=1, to=20, textvariable=self.num_sec_var,
                                         width=5, font=("Segoe UI", 11))
        self.num_sec_spin.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(row_frame, text="Disk Radius (m):", style="Header.TLabel").pack(side=tk.LEFT, padx=(10, 10))
        self.r_d_var = tk.DoubleVar(value=0.005)
        self.r_d_entry = ttk.Entry(row_frame, textvariable=self.r_d_var, width=10, font=("Segoe UI", 11))
        self.r_d_entry.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(row_frame, text="Set Sections", command=self.generate_section_fields).pack(side=tk.LEFT, padx=10)

# Robot Parameters - Per Section (ls, ns)
        self.robot_frame = ttk.LabelFrame(main_frame, text="Robot Parameters - Per Section", padding=10)
        self.robot_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 5))

        self.robot_canvas = tk.Canvas(self.robot_frame, height=200)
        robot_scrollbar = ttk.Scrollbar(self.robot_frame, orient=tk.VERTICAL, command=self.robot_canvas.yview)
        self.robot_scroll_frame = ttk.Frame(self.robot_canvas)

        self.robot_scroll_frame.bind(
            "<Configure>",
            lambda e: self.robot_canvas.configure(scrollregion=self.robot_canvas.bbox("all"))
        )
        self.robot_canvas.create_window((0, 0), window=self.robot_scroll_frame, anchor="nw")
        self.robot_canvas.configure(yscrollcommand=robot_scrollbar.set)
        self.robot_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        robot_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # VAS FK Parameters - Per Section (alphas, lv)
        self.vas_frame = ttk.LabelFrame(main_frame, text="VAS FK Parameters - Per Section", padding=10)
        self.vas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.vas_canvas = tk.Canvas(self.vas_frame, height=250)
        vas_scrollbar = ttk.Scrollbar(self.vas_frame, orient=tk.VERTICAL, command=self.vas_canvas.yview)
        self.vas_scroll_frame = ttk.Frame(self.vas_canvas)

        self.vas_scroll_frame.bind(
            "<Configure>",
            lambda e: self.vas_canvas.configure(scrollregion=self.vas_canvas.bbox("all"))
        )
        self.vas_canvas.create_window((0, 0), window=self.vas_scroll_frame, anchor="nw")
        self.vas_canvas.configure(yscrollcommand=vas_scrollbar.set)
        self.vas_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vas_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Storage lists for all parameters
        self.ls_entries = []
        self.ns_entries = []
        self.alpha_mins = []
        self.alpha_maxs = []
        self.lv_mins = []
        self.lv_maxs = []

        # --- Buttons ---
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(5, 5))

        self.run_btn = ttk.Button(btn_frame, text="▶  Run Simulation", command=self.on_run)
        self.run_btn.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(btn_frame, text="Reset Defaults", command=self.reset_defaults).pack(side=tk.LEFT)

        # --- Status bar ---
        self.status_var = tk.StringVar(value="Ready. Set sections and click Run.")
        ttk.Label(main_frame, textvariable=self.status_var, style="Status.TLabel").pack(fill=tk.X, pady=(5, 0))

        # Generate default fields
        self.generate_section_fields()

    def generate_section_fields(self):
        """Generate per-section fields for Robot Parameters (Ls, Ns) and VAS FK Parameters."""
        # Clear robot frame
        for widget in self.robot_scroll_frame.winfo_children():
            widget.destroy()
        self.ls_entries.clear()
        self.ns_entries.clear()
        # Clear VAS frame
        for widget in self.vas_scroll_frame.winfo_children():
            widget.destroy()
        self.alpha_mins.clear()
        self.alpha_maxs.clear()
        self.lv_mins.clear()
        self.lv_maxs.clear()

        num_sec = self.num_sec_var.get()

        # Robot frame headers (3 columns)
        ttk.Label(self.robot_scroll_frame, text='Section', font=('Segoe UI', 10, 'bold')).grid(row=0, column=0, padx=5, pady=3)
        ttk.Label(self.robot_scroll_frame, text='Ls (m)', font=('Segoe UI', 10, 'bold')).grid(row=0, column=1, padx=5, pady=3)
        ttk.Label(self.robot_scroll_frame, text='Ns', font=('Segoe UI', 10, 'bold')).grid(row=0, column=2, padx=5, pady=3)

        # VAS FK frame headers (5 columns)
        ttk.Label(self.vas_scroll_frame, text='Section', font=('Segoe UI', 10, 'bold')).grid(row=0, column=0, padx=5, pady=3)
        ttk.Label(self.vas_scroll_frame, text='Alpha Min (rad)', font=('Segoe UI', 10, 'bold')).grid(row=0, column=1, padx=5, pady=3)
        ttk.Label(self.vas_scroll_frame, text=\"Alpha Max (rad)\", font=(\"Segoe UI\", 10, \"bold\")).grid(row=0, column=2, padx=5, pady=3)
        ttk.Label(self.vas_scroll_frame, text=\"LV Min (m)\", font=(\"Segoe UI\", 10, \"bold\")).grid(row=0, column=3, padx=5, pady=3)
        ttk.Label(self.vas_scroll_frame, text=\"LV Max (m)\", font=(\"Segoe UI\", 10, \"bold\")).grid(row=0, column=4, padx=5, pady=3)

        # Defaults
        default_ls = 0.7
        default_ns_list = [15, 12, 12, 12, 12]
        alpha_min_def = [0.001, 0.001, 0.001, 0.001, 0.001]
        alpha_max_def = [1 * np.pi, 1 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi]
        lv_min_def = [0.0001, 0.0002, 0.0003, 0.0001, 0.0004]
        lv_max_def = [0.002, 0.003, 0.005, 0.002, 0.006]

        for i in range(num_sec):
            # Robot Parameters row
            ttk.Label(self.robot_scroll_frame, text=f\"  {i + 1}\", font=(\"Segoe UI\", 10)).grid(row=i + 1, column=0, padx=5, pady=2)
            ls_var = tk.DoubleVar(value=default_ls)
            ls_entry = ttk.Entry(self.robot_scroll_frame, textvariable=ls_var, width=12, font=(\"Segoe UI\", 10))
            ls_entry.grid(row=i + 1, column=1, padx=5, pady=2)
            self.ls_entries.append(ls_var)

            default_ns = default_ns_list[i] if i < len(default_ns_list) else 12
            ns_var = tk.IntVar(value=default_ns)
            ns_entry = ttk.Entry(self.robot_scroll_frame, textvariable=ns_var, width=12, font=(\"Segoe UI\", 10))
            ns_entry.grid(row=i + 1, column=2, padx=5, pady=2)
            self.ns_entries.append(ns_var)

            # VAS FK Parameters row
            ttk.Label(self.vas_scroll_frame, text=f\"  {i + 1}\", font=(\"Segoe UI\", 10)).grid(row=i + 1, column=0, padx=5, pady=2)

            alpha_min_val = alpha_min_def[i] if i < len(alpha_min_def) else 0.001
            alpha_min_var = tk.DoubleVar(value=alpha_min_val)
            alpha_min_entry = ttk.Entry(self.vas_scroll_frame, textvariable=alpha_min_var, width=12, font=(\"Segoe UI\", 10))
            alpha_min_entry.grid(row=i + 1, column=1, padx=5, pady=2)
            self.alpha_mins.append(alpha_min_var)

            alpha_max_val = alpha_max_def[i] if i < len(alpha_max_def) else 2 * np.pi
            alpha_max_var = tk.DoubleVar(value=alpha_max_val)
            alpha_max_entry = ttk.Entry(self.vas_scroll_frame, textvariable=alpha_max_var, width=12, font=(\"Segoe UI\", 10))
            alpha_max_entry.grid(row=i + 1, column=2, padx=5, pady=2)
            self.alpha_maxs.append(alpha_max_var)

            lv_min_val = lv_min_def[i] if i < len(lv_min_def) else 0.0001
            lv_min_var = tk.DoubleVar(value=lv_min_val)
            lv_min_entry = ttk.Entry(self.vas_scroll_frame, textvariable=lv_min_var, width=12, font=(\"Segoe UI\", 10))
            lv_min_entry.grid(row=i + 1, column=3, padx=5, pady=2)
            self.lv_mins.append(lv_min_var)

            lv_max_val = lv_max_def[i] if i < len(lv_max_def) else 0.002
            lv_max_var = tk.DoubleVar(value=lv_max_val)
            lv_max_entry = ttk.Entry(self.vas_scroll_frame, textvariable=lv_max_var, width=12, font=(\"Segoe UI\", 10))
            lv_max_entry.grid(row=i + 1, column=4, padx=5, pady=2)
            self.lv_maxs.append(lv_max_var)

        self.status_var.set(f"Configured {num_sec} sections. Adjust values and click Run.")

    def reset_defaults(self):
        """Reset to default 5-section configuration."""
        self.num_sec_var.set(5)
        self.r_d_var.set(0.005)
        self.generate_section_fields()

    def on_run(self):
        """Validate inputs and launch the simulation."""
        try:
            num_sec = self.num_sec_var.get()
            r_d = self.r_d_var.get()

            if num_sec < 1:
                messagebox.showerror("Input Error", "Number of sections must be at least 1.")
                return
            if r_d <= 0:
                messagebox.showerror("Input Error", "Disk radius must be positive.")
                return

            Ls = np.array([entry.get() for entry in self.ls_entries], dtype=float)
            Ns = np.array([entry.get() for entry in self.ns_entries], dtype=int)

            alpha_bounds = np.zeros((num_sec, 2))
            for sec in range(num_sec):
                alpha_bounds[sec, 0] = self.alpha_mins[sec].get()
                alpha_bounds[sec, 1] = self.alpha_maxs[sec].get()

            lv_bounds = np.zeros((num_sec, 2))
            for sec in range(num_sec):
                lv_bounds[sec, 0] = self.lv_mins[sec].get()
                lv_bounds[sec, 1] = self.lv_maxs[sec].get()

            # Validate bounds
            for sec in range(num_sec):
                if alpha_bounds[sec, 0] >= alpha_bounds[sec, 1]:
                    messagebox.showerror("Input Error", f"Section {sec+1}: Alpha Min must be < Alpha Max.")
                    return
                if lv_bounds[sec, 0] >= lv_bounds[sec, 1]:
                    messagebox.showerror("Input Error", f"Section {sec+1}: LV Min must be < LV Max.")
                    return

            if np.any(Ls <= 0):
                messagebox.showerror("Input Error", "All section lengths (Ls) must be positive.")
                return
            if np.any(Ns < 2):
                messagebox.showerror("Input Error", "All disk counts (Ns) must be at least 2.")
                return


        except (tk.TclError, ValueError) as e:
            messagebox.showerror("Input Error", f"Invalid input value: {e}")
            return

        # Disable run button during simulation
        self.run_btn.config(state=tk.DISABLED)
        self.status_var.set("Starting simulation...")

        # Run simulation in a background thread so GUI stays responsive
        def run_thread():
            try:
                run_simulation(Ls, r_d, Ns, alpha_bounds, lv_bounds, self.status_var)
            except Exception as e:
                print(f"Simulation error: {e}")
                self.status_var.set(f"Error: {e}")
                messagebox.showerror("Simulation Error", str(e))
            finally:
                self.run_btn.config(state=tk.NORMAL)

        thread = threading.Thread(target=run_thread, daemon=True)
        thread.start()


# =============================================================================
# Entry point
# =============================================================================
if __name__ == '__main__':
    root = tk.Tk()
    app = TrajectoryTrackingGUI(root)
    root.mainloop()
