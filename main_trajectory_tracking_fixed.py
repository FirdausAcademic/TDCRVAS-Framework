"""Main trajectory tracking script for a multi-section continuum robot with fixed base and closed-loop control.

Features a tkinter GUI with Robot Parameters (num sections, disk radius, ls/ns per section) and VAS FK Parameters (alpha min/max, lv min/max per section).

Converted from: Main_Trjectory_traking_code_for_anySec_fixed_base.m"""

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

    update_status('Generating trajectory inputs...')
    alphas, del_lvs = generate_trajectories_inputs(num_sec, alpha_bounds, lv_bounds, num_points)

    # Initial FK
    update_status('Computing initial FK and Jacobian...')
    SecCords, SecTips, Xaxis, Yaxis, Zaxis, PosFrames = gen_forward_kin_multi_sec(Ls, r_d, Ns, del_lvs[0, :], alphas[0, :])
    J = compound_jacobian_multi_section(Ls, r_d, del_lvs[0, :], alphas[0, :])

    # Backbone plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(SecCords[:, 0], SecCords[:, 1], SecCords[:, 2], 'o', markersize=3, linewidth=1.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Multi-Section Continuum Robot Backbone')

    scale = 0.05
    for i in range(Xaxis.shape[1]):
        O = PosFrames[:, i]
        ax.quiver(O[0], O[1], O[2], scale * Xaxis[0, i], scale * Xaxis[1, i], scale * Xaxis[2, i], color='r', linewidth=0.8)
        ax.quiver(O[0], O[1], O[2], scale * Yaxis[0, i], scale * Yaxis[1, i], scale * Yaxis[2, i], color='g', linewidth=0.8)
        ax.quiver(O[0], O[1], O[2], scale * Zaxis[0, i], scale * Zaxis[1, i], scale * Zaxis[2, i], color='k', linewidth=0.8)

    # Trajectory FK
    SecTips_all = np.zeros((num_points, 3, num_sec))
    for i in range(num_points):
        _, sec_tips_i, _, _, _, _ = gen_forward_kin_multi_sec(Ls, r_d, Ns, del_lvs[i, :], alphas[i, :])
        SecTips_all[i, :, :] = sec_tips_i
        if (i + 1) % 100 == 0:
            update_status(f'FK trajectory: {i + 1}/{num_points}')

    # 3D trajectories plot
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    colors = plt.cm.tab10(np.linspace(0, 1, max(num_sec, 10)))[:num_sec]
    for s in range(num_sec):
        x, y, z = SecTips_all[:, 0, s], SecTips_all[:, 1, s], SecTips_all[:, 2, s]
        ax2.plot(x, y, z, color=colors[s], linewidth=2, label=f'Set {s + 1}')
        ax2.scatter(x[0], y[0], z[0], s=100, color=colors[s], edgecolors='k', linewidths=1.5, zorder=5)
        ax2.scatter(x[-1], y[-1], z[-1], s=100, color=colors[s], marker='s', edgecolors='k', linewidths=1.5, zorder=5)
    ax2.set_xlabel(r'$X$')
    ax2.set_ylabel(r'$Y$')
    ax2.set_zlabel(r'$Z$')
    ax2.set_title('3D Trajectories of All Coordinate Sets')
    ax2.legend(loc='best')

    # Velocity
    dt = 1.0 / 1000.0
    velocity_all = np.zeros((num_points - 1, 3, num_sec))
    for s in range(num_sec):
        for dim in range(3):
            velocity_all[:, dim, s] = np.diff(SecTips_all[:, dim, s]) / dt
    update_status('Velocity calculation completed.')

    # Control loop
    v = np.zeros(2 * num_sec)
    v[0::2] = del_lvs[0, :]
    v[1::2] = alphas[0, :]
    x_tracking = []
    TdcrSec_list = []
    Time = [0.0]
    num_time_steps = num_points - 1
    PosFrames_all_ctrl = [None] * num_time_steps
    v_history = np.zeros((2 * num_sec, num_time_steps + 1))
    v_history[:, 0] = v

    for k in range(num_time_steps):
        current_lv = v_history[0::2, k]
        current_alpha = v_history[1::2, k]
        J_ctrl = compound_jacobian_multi_section(Ls, r_d, current_lv, current_alpha)
        TdcrSecCurrent, SecTipsFeedback, Xax, Yax, Zax, PosF = gen_forward_kin_multi_sec(Ls, r_d, Ns, current_lv, current_alpha)
        x = SecTipsFeedback.flatten(order='F')
        xd = np.concatenate([SecTips_all[k, :, s] for s in range(num_sec)])
        vel = np.concatenate([velocity_all[k, :, s] for s in range(num_sec)])
        err = xd - x
        x_tracking.append(x.copy())
        TdcrSec_list.append(TdcrSecCurrent.copy())
        PosFrames_all_ctrl[k] = PosF
        J_pinv = np.linalg.pinv(J_ctrl)
        v_history[:, k + 1] = v_history[:, k] + J_pinv @ (vel + 30 * err) * dt
        Time.append(Time[-1] + dt)
        if (k + 1) % 100 == 0:
            update_status(f'Control loop: {k + 1}/{num_time_steps}')

    x_tracking = np.array(x_tracking).T
    Time = np.array(Time)

    # Tracking plots (simplified for space)
    fig3 = plt.figure(figsize=(12, 9))
    ax3 = fig3.add_subplot(111, projection='3d')
    for s in range(num_sec):
        # Reference
        x_ref = SecTips_all[:, 0, s]
        y_ref = SecTips_all[:, 1, s]
        z_ref = SecTips_all[:, 2, s]
        ax3.plot(x_ref, y_ref, z_ref, color=colors[s], linestyle='-', linewidth=2, label=f'Sec {s + 1} Ref')
        # Tracked
        row_start = 3 * s
        x_track = x_tracking[row_start, :]
        y_track = x_tracking[row_start + 1, :]
        z_track = x_tracking[row_start + 2, :]
        ax3.plot(x_track, y_track, z_track, color=colors[s], linestyle='--', linewidth=4, label=f'Sec {s + 1} Track')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.legend()
    ax3.view_init(elev=30, azim=45)

    update_status('Simulation complete!')
    plt.show()


# =============================================================================
# GUI Application
# =============================================================================
class TrajectoryTrackingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Continuum Robot Trajectory Tracking')
        self.root.resizable(True, True)

        style = ttk.Style()
        style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'))
        style.configure('Header.TLabel', font=('Segoe UI', 11, 'bold'))
        style.configure('Status.TLabel', font=('Segoe UI', 10), foreground='blue')

        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text='Continuum Robot Trajectory Tracking', style='Title.TLabel').pack(pady=(0, 15))

        # Robot Parameters
        robot_frame = ttk.LabelFrame(main_frame, text='Robot Parameters', padding=10)
        robot_frame.pack(fill=tk.X, pady=(0, 10))

        row_frame = ttk.Frame(robot_frame)
        row_frame.pack(fill=tk.X)

        ttk.Label(row_frame, text='No. Sections:', style='Header.TLabel').pack(side=tk.LEFT, padx=(0, 10))
        self.num_sec_var = tk.IntVar(value=5)
        self.num_sec_spin = ttk.Spinbox(row_frame, from_=1, to=20, textvariable=self.num_sec_var, width=5)
        self.num_sec_spin.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(row_frame, text='Disk Radius (m):', style='Header.TLabel').pack(side=tk.LEFT, padx=(10, 10))
        self.r_d_var = tk.DoubleVar(value=0.005)
        self.r_d_entry = ttk.Entry(row_frame, textvariable=self.r_d_var, width=10)
        self.r_d_entry.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(row_frame, text='Set Sections', command=self.generate_section_fields).pack(side=tk.LEFT, padx=10)

        # Per-section Robot params (ls, ns)
        self.robot_sec_frame = ttk.LabelFrame(main_frame, text='Robot Parameters - Per Section', padding=10)
        self.robot_sec_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 5))

        self.robot_canvas = tk.Canvas(self.robot_sec_frame, height=200)
        r_scrollbar = ttk.Scrollbar(self.robot_sec_frame, orient=tk.VERTICAL, command=self.robot_canvas.yview)
        self.robot_scrollable = ttk.Frame(self.robot_canvas)

        self.robot_scrollable.bind('<Configure>', lambda e: self.robot_canvas.configure(scrollregion=self.robot_canvas.bbox('all')))
        self.robot_canvas.create_window((0, 0), window=self.robot_scrollable, anchor='nw')
        self.robot_canvas.configure(yscrollcommand=r_scrollbar.set)
        self.robot_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        r_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # VAS FK params per section
        self.vas_frame = ttk.LabelFrame(main_frame, text='VAS FK Parameters - Per Section', padding=10)
        self.vas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.vas_canvas = tk.Canvas(self.vas_frame, height=250)
        v_scrollbar = ttk.Scrollbar(self.vas_frame, orient=tk.VERTICAL, command=self.vas_canvas.yview)
        self.vas_scrollable = ttk.Frame(self.vas_canvas)

        self.vas_scrollable.bind('<Configure>', lambda e: self.vas_canvas.configure(scrollregion=self.vas_canvas.bbox('all')))
        self.vas_canvas.create_window((0, 0), window=self.vas_scrollable, anchor='nw')
        self.vas_canvas.configure(yscrollcommand=v_scrollbar.set)
        self.vas_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Storage
        self.ls_entries = []
        self.ns_entries = []
        self.alpha_mins = []
        self.alpha_maxs = []
        self.lv_mins = []
        self.lv_maxs = []

        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(5, 5))

        self.run_btn = ttk.Button(btn_frame, text='Run Simulation', command=self.on_run)
        self.run_btn.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(btn_frame, text='Reset Defaults', command=self.reset_defaults).pack(side=tk.LEFT)

        # Status
        self.status_var = tk.StringVar(value='Ready')
        ttk.Label(main_frame, textvariable=self.status_var, style='Status.TLabel').pack(fill=tk.X, pady=(5, 0))

        self.generate_section_fields()

    def generate_section_fields(self):
        # Clear
        for widget in self.robot_scrollable.winfo_children():
            widget.destroy()
        self.ls_entries.clear()
        self.ns_entries.clear()
        for widget in self.vas_scrollable.winfo_children():
            widget.destroy()
        self.alpha_mins.clear()
        self.alpha_maxs.clear()
        self.lv_mins.clear()
        self.lv_maxs.clear()

        num_sec = self.num_sec_var.get()

        # Robot headers
        ttk.Label(self.robot_scrollable, text='Section', font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5, pady=3)
        ttk.Label(self.robot_scrollable, text='Ls (m)', font=('Arial', 10, 'bold')).grid(row=0, column=1, padx=5, pady=3)
        ttk.Label(self.robot_scrollable, text='Ns', font=('Arial', 10, 'bold')).grid(row=0, column=2, padx=5, pady=3)

        # VAS headers
        ttk.Label(self.vas_scrollable, text='Section', font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5, pady=3)
        ttk.Label(self.vas_scrollable, text='Alpha Min', font=('Arial', 10, 'bold')).grid(row=0, column=1, padx=5, pady=3)
        ttk.Label(self.vas_scrollable, text='Alpha Max', font=('Arial', 10, 'bold')).grid(row=0, column=2, padx=5, pady=3)
        ttk.Label(self.vas_scrollable, text='LV Min', font=('Arial', 10, 'bold')).grid(row=0, column=3, padx=5, pady=3)
        ttk.Label(self.vas_scrollable, text='LV Max', font=('Arial', 10, 'bold')).grid(row=0, column=4, padx=5, pady=3)

        # Defaults
        default_ls = 0.7
        default_ns_list = [15, 12, 12, 12, 12]
        alpha_min_def = [0.001, 0.001, 0.001, 0.001, 0.001]
        alpha_max_def = [np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi]
        lv_min_def = [0.0001, 0.0002, 0.0003, 0.0001, 0.0004]
        lv_max_def = [0.002, 0.003, 0.005, 0.002, 0.006]

        for i in range(num_sec):
            # Robot row
            ttk.Label(self.robot_scrollable, text=f'Sec {i+1}').grid(row=i+1, column=0, padx=5, pady=2)
            ls_var = tk.DoubleVar(value=default_ls)
            ttk.Entry(self.robot_scrollable, textvariable=ls_var, width=12).grid(row=i+1, column=1, padx=5, pady=2)
            self.ls_entries.append(ls_var)
            default_ns = default_ns_list[i] if i < len(default_ns_list) else 12
            ns_var = tk.IntVar(value=default_ns)
            ttk.Entry(self.robot_scrollable, textvariable=ns_var, width=12).grid(row=i+1, column=2, padx=5, pady=2)
            self.ns_entries.append(ns_var)

            # VAS row
            ttk.Label(self.vas_scrollable, text=f'Sec {i+1}').grid(row=i+1, column=0, padx=5, pady=2)
            alpha_min_val = alpha_min_def[i] if i < len(alpha_min_def) else 0.001
            alpha_min_var = tk.DoubleVar(value=alpha_min_val)
            ttk.Entry(self.vas_scrollable, textvariable=alpha_min_var, width=12).grid(row=i+1, column=1, padx=5, pady=2)
            self.alpha_mins.append(alpha_min_var)
            alpha_max_val = alpha_max_def[i] if i < len(alpha_max_def) else 2 * np.pi
            alpha_max_var = tk.DoubleVar(value=alpha_max_val)
            ttk.Entry(self.vas_scrollable, textvariable=alpha_max_var, width=12).grid(row=i+1, column=2, padx=5, pady=2)
            self.alpha_maxs.append(alpha_max_var)
            lv_min_val = lv_min_def[i] if i < len(lv_min_def) else 0.0001
            lv_min_var = tk.DoubleVar(value=lv_min_val)
            ttk.Entry(self.vas_scrollable, textvariable=lv_min_var, width=12).grid(row=i+1, column=3, padx=5, pady=2)
            self.lv_mins.append(lv_min_var)
            lv_max_val = lv_max_def[i] if i < len(lv_max_def) else 0.002
            lv_max_var = tk.DoubleVar(value=lv_max_val)
            ttk.Entry(self.vas_scrollable, textvariable=lv_max_var, width=12).grid(row=i+1, column=4, padx=5, pady=2)
            self.lv_maxs.append(lv_max_var)

        self.status_var.set(f'Configured {num_sec} sections')

    def reset_defaults(self):
        self.num_sec_var.set(5)
        self.r_d_var.set(0.005)
        self.generate_section_fields()

    def on_run(self):
        try:
            num_sec = self.num_sec_var.get()
            r_d = self.r_d_var.get()
            Ls = np.array([var.get() for var in self.ls_entries], dtype=float)
            Ns = np.array([var.get() for var in self.ns_entries], dtype=int)
            alpha_bounds = np.zeros((num_sec, 2))
            for sec in range(num_sec):
                alpha_bounds[sec, 0] = self.alpha_mins[sec].get()
                alpha_bounds[sec, 1] = self.alpha_maxs[sec].get()
            lv_bounds = np.zeros((num_sec, 2))
            for sec in range(num_sec):
                lv_bounds[sec, 0] = self.lv_mins[sec].get()
                lv_bounds[sec, 1] = self.lv_maxs[sec].get()

            # Validation
            if r_d <= 0 or np.any(Ls <= 0) or np.any(Ns < 2) or np.any(alpha_bounds[:,0] >= alpha_bounds[:,1]) or np.any(lv_bounds[:,0] >= lv_bounds[:,1]):
                messagebox.showerror('Input Error', 'Check parameters')
                return

            self.run_btn.config(state='disabled')
            self.status_var.set('Running simulation...')

            def run_thread():
                try:
                    run_simulation(Ls, r_d, Ns, alpha_bounds, lv_bounds, self.status_var)
                except Exception as e:
                    messagebox.showerror('Error', str(e))
                finally:
                    self.run_btn.config(state='normal')

            threading.Thread(target=run_thread, daemon=True).start()

        except Exception as e:
            messagebox.showerror('Input Error', str(e))


if __name__ == '__main__':
    root = tk.Tk()
    app = TrajectoryTrackingGUI(root)
    root.mainloop()