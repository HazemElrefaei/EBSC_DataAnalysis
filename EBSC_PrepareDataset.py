import os
import subprocess
import logging
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import pandas as pd
import cv2
import h5py
from tqdm import tqdm
import plotly.io as pio
import plotly.graph_objects as go

if hasattr(pio.kaleido, 'scope') and hasattr(pio.kaleido.scope, '_context'):
    original_exec = pio.kaleido.scope._context.subprocess.Popen

    def silent_popen(*args, **kwargs):
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
        return original_exec(*args, **kwargs)

    pio.kaleido.scope._context.subprocess.Popen = silent_popen
    
def resample_data_nan(current_time_sample, data, target_time_sample, method="linear"):
    current_time_sample = np.asarray(current_time_sample, dtype=float)
    data = np.asarray(data, dtype=float)
    target_time_sample = np.asarray(target_time_sample, dtype=float)

    if data.ndim == 1:
        data = data[:, None]   # convert to (N,1)

    M = len(target_time_sample)
    D = data.shape[1]
    resampled_data = np.zeros((M, D))
    # resampled_data = np.full((len(target_time_sample), data.shape[1]), np.nan)

    for j in range(data.shape[1]):
        f_interp = interp1d(
            current_time_sample,
            data[:, j],
            kind=method,
            bounds_error=False,
            fill_value=np.nan,
            assume_sorted=True
        )
        resampled_data[:, j] = f_interp(target_time_sample)

    return resampled_data
    
def resample_data(current_time_sample, data, target_time_sample, method="linear"):
    """
    Resample multi-dimensional time series data to a new set of timestamps.

    Parameters
    ----------
    current_time_sample : (N,) array_like
        Original timestamps (must be increasing).
    data : (N, D) array_like
        Data values corresponding to current_time_sample.
    target_time_sample : (M,) array_like
        New timestamps to resample the data to.
    method : str, optional
        Interpolation method. Options:
        - "linear"   : linear interpolation
        - "nearest"  : nearest neighbor
        - "previous" : zero-order hold (sample-and-hold)

    Returns
    -------
    resampled_data : (M, D) ndarray
        Data resampled at target_time_sample.
    """
    current_time_sample = np.asarray(current_time_sample, dtype=float)
    data = np.asarray(data, dtype=float)
    target_time_sample = np.asarray(target_time_sample, dtype=float)

    if data.ndim == 1:
        data = data[:, None]   # convert to (N,1)

    M = len(target_time_sample)
    D = data.shape[1]
    resampled_data = np.zeros((M, D))

    for j in range(D):
        f_interp = interp1d(
            current_time_sample,
            data[:, j],
            kind=method,
            fill_value="extrapolate",
            assume_sorted=True
        )
        resampled_data[:, j] = f_interp(target_time_sample)

    return resampled_data

def lowpass_filter(data, cutoff=2.0, fs=20.0, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def kalman_filter_position_velocity(pos, t, R_val=1e-4, Q_val=1e-6):
    """
    pos: (N,3) array of measured positions
    t:   (N,) timestamps (seconds)
    R_val: measurement noise covariance (position)
    Q_val: process noise covariance (velocity smoothness)
    """
    N = len(t)
    dt = np.diff(t, prepend=t[0])
    dt[0] = np.median(dt[1:])  # avoid zero at first step

    # State: [pos(3), vel(3)]
    x = np.zeros((6,))
    x[:3] = pos[0]

    # Covariance
    P = np.eye(6) * 1e-3

    # Matrices
    H = np.hstack([np.eye(3), np.zeros((3,3))])  # measure position only
    R = np.eye(3) * R_val

    xs = []
    for k in range(N):
        # Predict
        dt_k = dt[k]
        F = np.eye(6)
        F[0:3,3:6] = np.eye(3) * dt_k   # position update
        Q = np.eye(6) * Q_val

        x = F @ x
        P = F @ P @ F.T + Q

        # Update
        z = pos[k]
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        x = x + K @ y
        P = (np.eye(6) - K @ H) @ P

        xs.append(x.copy())

    xs = np.array(xs)   # (N,6)
    pos_filt = xs[:,0:3]
    vel_filt = xs[:,3:6]
    return pos_filt, vel_filt

def kalman_filter_position_velocity_valid_only(pos, t, R_val=1e-4, Q_val=1e-6):
    """
    Runs the position-velocity Kalman filter only on samples where pos is finite.
    Invalid samples remain NaN in the returned arrays.
    """
    pos = np.asarray(pos, dtype=float)
    t = np.asarray(t, dtype=float)

    pos_filt_full = np.full_like(pos, np.nan, dtype=float)
    vel_filt_full = np.full_like(pos, np.nan, dtype=float)

    valid = np.all(np.isfinite(pos), axis=1) & np.isfinite(t)

    if valid.sum() < 3:
        raise ValueError(f"Not enough valid samples for Kalman filtering: {valid.sum()}")

    pos_valid = pos[valid]
    t_valid = t[valid]

    pos_filt_valid, vel_filt_valid = kalman_filter_position_velocity(
        pos_valid,
        t_valid,
        R_val=R_val,
        Q_val=Q_val
    )

    pos_filt_full[valid] = pos_filt_valid
    vel_filt_full[valid] = vel_filt_valid

    return pos_filt_full, vel_filt_full, valid

def normalize_quaternion(q):
    return q / np.linalg.norm(q)

def omega_matrix(omega):
    wx, wy, wz = omega
    return np.array([
        [0,   -wx, -wy, -wz],
        [wx,   0,   wz, -wy],
        [wy,  -wz,  0,   wx],
        [wz,   wy, -wx,  0]
    ])


def quaternion_kalman_filter(q_meas, t, R_val=1e-4, Q_val=1e-6):
    """
    Quaternion Kalman filter for smoothing and angular velocity estimation.
    
    q_meas : (N,4) array of measured quaternions [x,y,z,w]
    t      : (N,) timestamps
    R_val  : measurement noise covariance
    Q_val  : process noise covariance
    """
    N = len(t)
    dt = np.diff(t, prepend=t[0])
    dt[0] = np.median(dt[1:])

    # State: [q(4), omega(3)]
    x = np.zeros(7)
    x[0:4] = normalize_quaternion(q_meas[0])
    P = np.eye(7) * 1e-3

    H = np.hstack([np.eye(4), np.zeros((4,3))])   # measure quaternion only
    R = np.eye(4) * R_val

    xs = []
    for k in range(N):
        dt_k = dt[k]

        # --- Prediction step ---
        q = x[0:4]
        omega = x[4:7]

        # Quaternion propagation
        q_dot = 0.5 * omega_matrix(omega) @ q
        q_pred = normalize_quaternion(q + q_dot * dt_k)

        # Assume constant angular velocity
        omega_pred = omega

        x_pred = np.hstack([q_pred, omega_pred])

        # Linearize (approximate F)
        F = np.eye(7)
        # Small process noise
        Q = np.eye(7) * Q_val

        P = F @ P @ F.T + Q
        x = x_pred

        # --- Update step ---
        z = normalize_quaternion(q_meas[k])
        y = z - H @ x   # innovation
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        x = x + K @ y
        P = (np.eye(7) - K @ H) @ P

        # Normalize quaternion
        x[0:4] = normalize_quaternion(x[0:4])

        xs.append(x.copy())

    xs = np.array(xs)  # (N,7)
    q_filt = xs[:,0:4]
    omega_filt = xs[:,4:7]
    return q_filt, omega_filt

def quaternion_kalman_filter_valid_only(q_meas, t, R_val=1e-4, Q_val=1e-6):
    """
    Runs quaternion Kalman filtering only on valid quaternion samples.
    Invalid samples remain NaN in the returned arrays.

    q_meas : (N,4) measured quaternions [x,y,z,w]
    t      : (N,) timestamps in seconds
    """
    q_meas = np.asarray(q_meas, dtype=float)
    t = np.asarray(t, dtype=float)

    q_filt_full = np.full_like(q_meas, np.nan, dtype=float)
    omega_filt_full = np.full((len(t), 3), np.nan, dtype=float)

    q_norm = np.linalg.norm(q_meas, axis=1)

    valid = (
        np.all(np.isfinite(q_meas), axis=1)
        & np.isfinite(t)
        & np.isfinite(q_norm)
        & (q_norm > 1e-12)
    )

    if valid.sum() < 3:
        raise ValueError(f"Not enough valid quaternion samples for filtering: {valid.sum()}")

    q_valid = q_meas[valid]
    t_valid = t[valid]

    q_filt_valid, omega_filt_valid = quaternion_kalman_filter(
        q_valid,
        t_valid,
        R_val=R_val,
        Q_val=Q_val
    )

    q_filt_full[valid] = q_filt_valid
    omega_filt_full[valid] = omega_filt_valid

    return q_filt_full, omega_filt_full, valid

def compute_ref_velocity(ref_velocity, motor_rpm_resampled, wheel_radius, radius_curvature, half_track_width,
                           use_wheels=(0, 2)):

    wheel_signs = np.array([+1, +1, -1, -1])

    # Apply sign correction
    corrected_rpm = motor_rpm_resampled[:, use_wheels] * wheel_signs[list(use_wheels)]

    # Convert RPM -> linear velocity (m/s)
    wheel_speeds = corrected_rpm * (2*np.pi/60) * wheel_radius  # (N,2)

    v_left  = wheel_speeds[:,0]
    v_right = wheel_speeds[:,1]

    # Forward velocity (average of left/right)
    ref_velocity[:,0] = ((v_left + v_right) / 2.0)

    # Yaw velocity (difference over track width)
    ref_velocity[:,1] = ((v_left - v_right) / (2*half_track_width))*radius_curvature

    return ref_velocity

def save_plotly_plot(x, y, label=['line1'],x2=None, y2=None,label2='line2',xlim = [], ylim = [], xlabel='', ylabel='', output_path='', show_markers=False, interactive = False):
    fig = go.Figure()
    shape_y = 1
    if len(y.shape) > shape_y:
        shape_y = y.shape[1]
        
    if shape_y > 1:
        for i in range(shape_y):
            fig.add_trace(go.Scatter(x=x,
                                    y=y[:,i],
                                    mode='lines+markers' if show_markers else 'lines',
                                    name= label[i],
                                    marker=dict(size=4) if show_markers else None,
                                    line=dict(width=2)
                                    ))
    else:
        for i in range(shape_y):
            fig.add_trace(go.Scatter(x=x,
                                    y=y,
                                    mode='lines+markers' if show_markers else 'lines',
                                    name= label[0],
                                    marker=dict(size=4) if show_markers else None,
                                    line=dict(width=2)
                                    ))
            
    
     # Second plot (Orange), if provided
    if x2 is not None and y2 is not None:
        fig.add_trace(go.Scatter(
            x=x2,
            y=y2,
            mode='lines+markers' if show_markers else 'lines',
            name= label2,
            marker=dict(size=4) if show_markers else None,
            line=dict(width=2)
        ))

    fig.update_layout(
        # title=title,
        # font=dict(
        #     family="Times New Roman",
        #     size=18,
        # ),
        xaxis=dict(
            title=dict(text=xlabel, font=dict(family='Times New Roman', size=18)),
            tickfont=dict(size=18, family="Times New Roman"),
            range=xlim
        ),
        yaxis=dict(
            title=dict(text=ylabel, font=dict(family='Times New Roman', size=18)),
            tickfont=dict(size=18, family="Times New Roman"),
            range=ylim
        ),
        legend=dict(font=dict(family="Times New Roman", size=14)),
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor='white'
    )
    fig.update_xaxes(
    mirror=True,
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    # Save as high-quality PNG
    if not interactive:
        fig.write_image(output_path, format='png', width=1200, height=400, scale=3)
    else:
        pio.write_html(fig, file=output_path)
        
def read_dataset_in_chunks(dataset, chunk_size=80000):
    """Read a HDF5 dataset in chunks with a progress bar."""
    total_size = dataset.shape[0]
    dtype = dataset.dtype
    data = np.empty((total_size,) + dataset.shape[1:], dtype=dtype)
    
    with tqdm(total=total_size, desc=f"Reading {dataset.name}", unit="rows") as pbar:
        for i in range(0, total_size, chunk_size):
            end = min(i + chunk_size, total_size)
            data[i:end] = dataset[i:end]
            pbar.update(end - i)
    
    return data

#####################
## Constants
#####################
window_size = 0.01
wheel_radius = 0.1              # meters
half_rover_length = 0.29
half_track_width = 0.22348
radius_curvature = np.sqrt(half_rover_length**2+half_track_width**2)
save_images = True

#####################
## Paths
#####################
if os.name == 'nt':
    _ = os.system('cls')
else:
    _ = os.system('clear')
    
output_dir = r"Path\To\Output\\"

path_to_h5 = r'Path\To\H5\\'

h5_file_list= [
#         '2025_07_17_17_19_17_Vel1_Lev3_ON',
# '2025_07_21_10_22_29_Vel1_Lev3_ON',
# '2025_07_21_10_43_41_Vel1_Lev3_ON',
# '2025_07_21_10_52_21_Vel1_Lev3_OFF',
# '2025_07_21_11_02_18_Vel1_Lev3_OFF',
# '2025_07_21_11_07_54_Vel1_Lev3_OFF',
# '2025_07_21_11_21_45_Vel2_Lev3_ON',
# '2025_07_21_11_28_46_Vel2_Lev3_ON',
# '2025_07_21_11_34_31_Vel2_Lev3_ON',
# '2025_07_21_11_40_08_Vel2_Lev3_OFF',
# '2025_07_21_11_46_03_Vel2_Lev3_OFF',
# '2025_07_21_11_51_48_Vel2_Lev3_OFF',
# '2025_07_21_12_17_42_Vel3_Lev3_ON',
# '2025_07_21_12_47_09_Vel3_Lev3_ON',
# '2025_07_21_12_51_01_Vel3_Lev3_ON',
# '2025_07_21_12_55_38_Vel3_Lev3_OFF',
# '2025_07_21_13_01_17_Vel3_Lev3_OFF',
# '2025_07_21_13_05_31_Vel3_Lev3_OFF',
# '2025_07_21_13_14_37_Vel4_Lev3_ON',
# '2025_07_21_13_25_38_Vel4_Lev3_ON',
# '2025_07_21_13_30_08_Vel4_Lev3_ON',
# '2025_07_21_13_34_10_Vel4_Lev3_OFF',
# '2025_07_21_13_39_22_Vel4_Lev3_OFF',
# '2025_07_21_13_43_31_Vel4_Lev3_OFF',
# '2025_07_21_14_47_02_Vel5_Lev3_ON',
# '2025_07_21_14_51_12_Vel5_Lev3_ON',
# '2025_07_21_14_54_15_Vel5_Lev3_ON',
# '2025_07_21_14_57_32_Vel5_Lev3_OFF',
# '2025_07_21_15_02_45_Vel5_Lev3_OFF',
# '2025_07_21_15_07_56_Vel5_Lev3_OFF',
# '2025_07_21_15_16_15_Vel1_50_Lev3_ON',
# '2025_07_21_15_26_32_Vel1_50_Lev3_ON',
# '2025_07_21_15_33_42_Vel1_50_Lev3_ON',
# '2025_07_21_15_41_37_Vel1_50_Lev3_OFF',
# '2025_07_21_15_47_39_Vel1_50_Lev3_OFF',
# '2025_07_21_15_54_53_Vel1_50_Lev3_OFF',
# '2025_07_21_16_03_00_Vel2_50_Lev3_ON',
# '2025_07_21_16_05_51_Vel2_50_Lev3_ON',
# '2025_07_21_16_09_50_Vel2_50_Lev3_ON',
# '2025_07_21_16_12_30_Vel2_50_Lev3_OFF',
# '2025_07_21_16_15_46_Vel2_50_Lev3_OFF',
# '2025_07_21_16_18_00_Vel2_50_Lev3_OFF',
# '2025_07_21_16_24_20_Vel3_50_Lev3_ON',
# '2025_07_21_16_27_38_Vel3_50_Lev3_ON',
# '2025_07_21_16_30_19_Vel3_50_Lev3_ON',
# '2025_07_21_16_33_37_Vel3_50_Lev3_OFF',
# '2025_07_21_16_36_17_Vel3_50_Lev3_OFF',
# '2025_07_21_16_39_20_Vel3_50_Lev3_OFF',
# '2025_07_21_17_17_25_Vel4_50_Lev3_ON',
# '2025_07_21_17_21_38_Vel4_50_Lev3_ON',
# '2025_07_21_17_24_02_Vel4_50_Lev3_ON',
# '2025_07_21_17_26_53_Vel4_50_Lev3_OFF',
# '2025_07_21_17_28_52_Vel4_50_Lev3_OFF',
# '2025_07_21_17_31_28_Vel4_50_Lev3_OFF',
# '2025_07_21_17_34_12_Vel5_50_Lev3_ON',
# '2025_07_21_17_36_05_Vel5_50_Lev3_ON',
# '2025_07_21_17_38_46_Vel5_50_Lev3_ON',
# '2025_07_21_17_42_55_Vel5_50_Lev3_OFF',
# '2025_07_21_17_45_12_Vel5_50_Lev3_OFF',
# '2025_07_21_17_47_13_Vel5_50_Lev3_OFF',
# '2025_07_21_19_05_31_Vel1_25_Lev3_ON',
# '2025_07_21_19_14_50_Vel1_25_Lev3_ON',
# '2025_07_21_19_22_23_Vel1_25_Lev3_ON',
# '2025_07_21_19_30_19_Vel1_25_Lev3_OFF',
# '2025_07_21_19_37_03_Vel1_25_Lev3_OFF',
# '2025_07_21_19_45_22_Vel1_25_Lev3_OFF',
# '2025_07_21_19_53_11_Vel2_25_Lev3_ON',
# '2025_07_21_19_57_08_Vel2_25_Lev3_ON',
# '2025_07_21_20_00_32_Vel2_25_Lev3_ON',
# '2025_07_21_20_06_24_Vel2_25_Lev3_OFF',
# '2025_07_21_20_11_12_Vel2_25_Lev3_OFF',
# '2025_07_21_20_15_38_Vel2_25_Lev3_OFF',
# '2025_07_21_20_23_00_Vel3_25_Lev3_ON',
# '2025_07_21_20_27_18_Vel3_25_Lev3_ON',
# '2025_07_21_20_29_46_Vel3_25_Lev3_ON',
# '2025_07_21_20_32_46_Vel3_25_Lev3_OFF',
# '2025_07_21_20_35_47_Vel3_25_Lev3_OFF',
# '2025_07_21_20_39_15_Vel3_25_Lev3_OFF',
# '2025_07_21_20_50_00_Vel4_25_Lev3_ON',
# '2025_07_21_20_52_45_Vel4_25_Lev3_ON',
# '2025_07_21_21_09_04_Vel4_25_Lev3_ON',
# '2025_07_21_21_14_17_Vel4_25_Lev3_OFF',
# '2025_07_21_22_19_41_Vel4_25_Lev3_OFF',
# '2025_07_21_22_28_35_Vel4_25_Lev3_OFF',
# '2025_07_21_22_38_20_Vel5_25_Lev3_ON',
# '2025_07_21_22_41_17_Vel5_25_Lev3_ON',
# '2025_07_21_22_44_37_Vel5_25_Lev3_ON',
# '2025_07_21_22_48_26_Vel5_25_Lev3_OFF',
# '2025_07_21_22_51_33_Vel5_25_Lev3_OFF',
# '2025_07_21_22_54_31_Vel5_25_Lev3_OFF',
# '2025_07_29_15_35_33_Vel1_Lev2_ON',
# '2025_07_29_15_43_14_Vel1_Lev2_ON',
# '2025_07_29_15_49_41_Vel1_Lev2_ON',
# '2025_07_29_15_53_06_Vel1_Lev2_OFF',
# '2025_07_29_15_57_13_Vel1_Lev2_OFF',
# '2025_07_29_16_03_07_Vel1_Lev2_OFF',
# '2025_07_29_16_06_27_Vel2_Lev2_ON',
# '2025_07_29_16_13_32_Vel2_Lev2_ON',
# '2025_07_29_16_15_51_Vel2_Lev2_ON',
# '2025_07_29_16_18_13_Vel2_Lev2_OFF',
# '2025_07_29_16_20_46_Vel2_Lev2_OFF',
# '2025_07_29_16_23_31_Vel2_Lev2_OFF',
# '2025_07_29_16_26_46_Vel3_Lev2_ON',
# '2025_07_29_16_30_04_Vel3_Lev2_ON',
# '2025_07_29_16_32_38_Vel3_Lev2_ON',
# '2025_07_29_16_35_04_Vel3_Lev2_OFF',
# '2025_07_29_16_36_55_Vel3_Lev2_OFF',
# '2025_07_29_16_39_26_Vel3_Lev2_OFF',
# '2025_07_29_16_42_19_Vel4_Lev2_ON',
# '2025_07_29_16_46_33_Vel4_Lev2_ON',
# '2025_07_29_16_48_58_Vel4_Lev2_ON',
# '2025_07_29_16_51_20_Vel4_Lev2_OFF',
# '2025_07_29_16_53_50_Vel4_Lev2_OFF',
# '2025_07_29_16_56_38_Vel4_Lev2_OFF',
# '2025_07_29_17_01_28_Vel5_Lev2_ON',
# '2025_07_29_17_03_32_Vel5_Lev2_ON',
# '2025_07_29_17_05_06_Vel5_Lev2_ON',
# '2025_07_29_17_07_39_Vel5_Lev2_OFF',
# '2025_07_29_17_09_38_Vel5_Lev2_OFF',
# '2025_07_29_17_11_40_Vel5_Lev2_OFF',
# '2025_07_30_10_50_30_Vel1_50_Lev2_ON',
# '2025_07_30_10_57_47_Vel1_50_Lev2_ON',
# '2025_07_30_11_00_51_Vel1_50_Lev2_ON',
# '2025_07_30_11_04_17_Vel1_50_Lev2_OFF',
# '2025_07_30_11_08_27_Vel1_50_Lev2_OFF',
# '2025_07_30_11_13_22_Vel1_50_Lev2_OFF',
# '2025_07_30_11_17_25_Vel2_50_Lev2_ON',
# '2025_07_30_11_19_49_Vel2_50_Lev2_ON',
# '2025_07_30_11_22_24_Vel2_50_Lev2_ON',
# '2025_07_30_11_26_22_Vel2_50_Lev2_OFF',
# '2025_07_30_11_28_26_Vel2_50_Lev2_OFF',
# '2025_07_30_11_31_11_Vel2_50_Lev2_OFF',
# '2025_07_30_11_37_42_Vel3_50_Lev2_ON',
# '2025_07_30_11_41_19_Vel3_50_Lev2_ON',
# '2025_07_30_11_45_21_Vel3_50_Lev2_ON',
# '2025_07_30_11_47_46_Vel3_50_Lev2_OFF',
# '2025_07_30_11_53_18_Vel3_50_Lev2_OFF',
# '2025_07_30_11_56_05_Vel3_50_Lev2_OFF',
# '2025_07_30_12_00_13_Vel4_50_Lev2_ON',
# '2025_07_30_12_02_24_Vel4_50_Lev2_ON',
# '2025_07_30_12_07_27_Vel4_50_Lev2_ON',
# '2025_07_30_12_10_02_Vel4_50_Lev2_OFF',
# '2025_07_30_12_12_03_Vel4_50_Lev2_OFF',
# '2025_07_30_12_14_59_Vel4_50_Lev2_OFF',
# '2025_07_30_12_30_52_Vel5_50_Lev2_ON',
# '2025_07_30_12_33_36_Vel5_50_Lev2_ON',
# '2025_07_30_12_35_52_Vel5_50_Lev2_ON',
# '2025_07_30_12_39_37_Vel5_50_Lev2_OFF',
# '2025_07_30_12_42_22_Vel5_50_Lev2_OFF',
# '2025_07_30_12_45_15_Vel5_50_Lev2_OFF',
# '2025_07_30_14_04_23_Vel1_25_Lev2_ON',
# '2025_07_30_14_08_46_Vel1_25_Lev2_ON',
# '2025_07_30_14_13_57_Vel1_25_Lev2_ON',
# '2025_07_30_14_16_46_Vel1_25_Lev2_OFF',
# '2025_07_30_14_20_46_Vel1_25_Lev2_OFF',
# '2025_07_30_14_23_23_Vel1_25_Lev2_OFF',
# '2025_07_30_14_27_43_Vel2_25_Lev2_ON',
# '2025_07_30_14_30_47_Vel2_25_Lev2_ON',
# '2025_07_30_14_34_48_Vel2_25_Lev2_ON',
# '2025_07_30_14_37_30_Vel2_25_Lev2_OFF',
# '2025_07_30_14_39_16_Vel2_25_Lev2_OFF',
# '2025_07_30_14_43_36_Vel2_25_Lev2_OFF',
# '2025_07_30_14_48_22_Vel3_25_Lev2_ON',
# '2025_07_30_14_50_37_Vel3_25_Lev2_ON',
# '2025_07_30_14_53_28_Vel3_25_Lev2_ON',
# '2025_07_30_14_58_34_Vel3_25_Lev2_OFF',
# '2025_07_30_15_00_44_Vel3_25_Lev2_OFF',
# '2025_07_30_15_03_19_Vel3_25_Lev2_OFF',
# '2025_07_30_15_09_29_Vel4_25_Lev2_ON',
# '2025_07_30_15_12_54_Vel4_25_Lev2_ON',
# '2025_07_30_15_16_26_Vel4_25_Lev2_ON',
# '2025_07_30_15_19_03_Vel4_25_Lev2_OFF',
# '2025_07_30_15_21_20_Vel4_25_Lev2_OFF',
# '2025_07_30_15_23_31_Vel4_25_Lev2_OFF',
# '2025_07_30_15_28_17_Vel5_25_Lev2_ON',
# '2025_07_30_15_29_24_Vel5_25_Lev2_ON',
# '2025_07_30_15_31_20_Vel5_25_Lev2_ON',
# '2025_07_30_15_33_21_Vel5_25_Lev2_OFF',
# '2025_07_30_15_35_13_Vel5_25_Lev2_OFF',
# '2025_07_30_15_37_09_Vel5_25_Lev2_OFF',
# '2025_07_30_17_54_22_Vel1_Lev1_ON',
# '2025_07_30_17_57_04_Vel1_Lev1_ON',
# '2025_07_30_17_59_27_Vel1_Lev1_ON',
# '2025_07_30_18_04_08_Vel1_Lev1_OFF',
# '2025_07_30_18_08_18_Vel1_Lev1_OFF',
# '2025_07_30_18_11_56_Vel1_Lev1_OFF',
# '2025_07_30_18_15_21_Vel2_Lev1_ON',
# '2025_07_30_18_16_53_Vel2_Lev1_ON',
# '2025_07_30_18_18_16_Vel2_Lev1_ON',
# '2025_07_30_18_20_12_Vel2_Lev1_OFF',
# '2025_07_30_18_22_54_Vel2_Lev1_OFF',
# '2025_07_30_18_25_44_Vel2_Lev1_OFF',
# '2025_07_30_18_28_49_Vel3_Lev1_ON',
# '2025_07_30_18_37_13_Vel3_Lev1_ON',
# '2025_07_30_18_40_06_Vel3_Lev1_ON',
# '2025_07_30_18_42_59_Vel3_Lev1_OFF',
# '2025_07_30_18_44_56_Vel3_Lev1_OFF',
# '2025_07_30_18_46_31_Vel3_Lev1_OFF',
# '2025_07_30_18_49_04_Vel4_Lev1_ON',
# '2025_07_30_18_50_26_Vel4_Lev1_ON',
# '2025_07_30_18_52_12_Vel4_Lev1_ON',
# '2025_07_30_18_55_15_Vel4_Lev1_OFF',
# '2025_07_30_18_57_21_Vel4_Lev1_OFF',
# '2025_07_30_19_01_31_Vel4_Lev1_OFF',
# '2025_07_30_19_04_15_Vel5_Lev1_ON',
# '2025_07_30_19_05_52_Vel5_Lev1_ON',
# '2025_07_30_19_06_58_Vel5_Lev1_ON',
# '2025_07_30_19_08_37_Vel5_Lev1_OFF',
# '2025_07_30_19_09_59_Vel5_Lev1_OFF',
# '2025_07_30_19_11_47_Vel5_Lev1_OFF',
# '2025_07_30_20_10_50_Vel1_50_Lev1_ON',
# '2025_07_30_20_16_10_Vel1_50_Lev1_ON',
# '2025_07_30_20_20_20_Vel1_50_Lev1_ON',
# '2025_07_30_20_26_00_Vel1_50_Lev1_OFF',
# '2025_07_30_20_30_20_Vel1_50_Lev1_OFF',
# '2025_07_30_20_35_02_Vel1_50_Lev1_OFF',
# '2025_07_30_20_52_14_Vel2_50_Lev1_ON',
# '2025_07_30_20_58_10_Vel2_50_Lev1_ON',
# '2025_07_30_21_01_20_Vel2_50_Lev1_ON',
# '2025_07_30_21_04_27_Vel2_50_Lev1_OFF',
# '2025_07_30_21_07_02_Vel2_50_Lev1_OFF',
# '2025_07_30_21_09_27_Vel2_50_Lev1_OFF',
# '2025_07_30_21_14_45_Vel3_50_Lev1_ON',
# '2025_07_30_21_17_11_Vel3_50_Lev1_ON',
# '2025_07_30_21_20_47_Vel3_50_Lev1_ON',
# '2025_07_30_21_29_18_Vel3_50_Lev1_OFF',
# '2025_07_30_21_30_57_Vel3_50_Lev1_OFF',
# '2025_07_30_21_32_33_Vel3_50_Lev1_OFF',
# '2025_07_30_21_34_41_Vel4_50_Lev1_ON',
# '2025_07_30_21_36_05_Vel4_50_Lev1_ON',
# '2025_07_30_21_37_51_Vel4_50_Lev1_ON',
# '2025_07_30_21_39_30_Vel4_50_Lev1_OFF',
# '2025_07_30_21_41_04_Vel4_50_Lev1_OFF',
# '2025_07_30_21_42_20_Vel4_50_Lev1_OFF',
# '2025_07_30_21_45_55_Vel5_50_Lev1_ON',
# '2025_07_30_21_48_40_Vel5_50_Lev1_ON',
# '2025_07_30_21_50_28_Vel5_50_Lev1_ON',
# '2025_07_30_21_52_56_Vel5_50_Lev1_OFF',
# '2025_07_30_21_55_08_Vel5_50_Lev1_OFF',
# '2025_07_30_21_57_14_Vel5_50_Lev1_OFF',
# '2025_07_30_22_03_57_Vel1_25_Lev1_ON',
# '2025_07_30_22_07_53_Vel1_25_Lev1_ON',
# '2025_07_30_22_11_18_Vel1_25_Lev1_ON',
# '2025_07_30_22_14_35_Vel1_25_Lev1_OFF',
# '2025_07_30_22_17_32_Vel1_25_Lev1_OFF',
# '2025_07_30_22_21_30_Vel1_25_Lev1_OFF',
# '2025_07_30_22_25_17_Vel2_25_Lev1_ON',
# '2025_07_30_22_28_33_Vel2_25_Lev1_ON',
# '2025_07_30_22_32_20_Vel2_25_Lev1_ON',
# '2025_07_30_22_37_49_Vel2_25_Lev1_OFF',
# '2025_07_30_22_42_43_Vel2_25_Lev1_OFF',
# '2025_07_30_22_46_54_Vel2_25_Lev1_OFF',
# '2025_07_30_22_52_03_Vel3_25_Lev1_ON',
# '2025_07_30_22_57_09_Vel3_25_Lev1_ON',
# '2025_07_30_23_00_29_Vel3_25_Lev1_ON',
# '2025_07_30_23_03_07_Vel3_25_Lev1_OFF',
# '2025_07_30_23_06_43_Vel3_25_Lev1_OFF',
# '2025_07_30_23_09_03_Vel3_25_Lev1_OFF',
# '2025_07_30_23_11_21_Vel4_25_Lev1_ON',
# '2025_07_30_23_13_43_Vel4_25_Lev1_ON',
# '2025_07_30_23_16_12_Vel4_25_Lev1_ON',
# '2025_07_30_23_19_18_Vel4_25_Lev1_OFF',
# '2025_07_30_23_22_06_Vel4_25_Lev1_OFF',
# '2025_07_30_23_24_19_Vel4_25_Lev1_OFF',
# '2025_07_30_23_27_36_Vel5_25_Lev1_ON',
# '2025_07_30_23_32_49_Vel5_25_Lev1_ON',
# '2025_07_30_23_34_43_Vel5_25_Lev1_ON',
# '2025_07_30_23_38_08_Vel5_25_Lev1_OFF',
# '2025_07_30_23_40_43_Vel5_25_Lev1_OFF',
# '2025_07_30_23_42_53_Vel5_25_Lev1_OFF',
]

file_index = 0

for h5_file_name in h5_file_list:
    os.makedirs(output_dir+h5_file_name, exist_ok=True)
    print(f"Processing file: {h5_file_name}")
    
    Opti = {}
    Motors_RPM = {}
    events = None
    with h5py.File(path_to_h5+h5_file_name + '.h5', 'r') as f:
        events = read_dataset_in_chunks(f['events']['data'])
        Opti['data'] = read_dataset_in_chunks(f['opti_pos']['data'])
        Opti['ros_timestamp'] = read_dataset_in_chunks(f['opti_pos']['time'])
        Motors_RPM['data'] = read_dataset_in_chunks(f['motor_speed']['data'])
        Motors_RPM['ros_timestamp'] = read_dataset_in_chunks(f['motor_speed']['time'])


        
    print(f"Messages have been read.")
    
    events = events.astype(np.float64)
    ts_first_ev = events[0,3]
    events[:,3] = (events[:,3] - ts_first_ev)*1e-6
    ts0 = events[0, 3]
    ts1 = events[-1, 3]
    print(f"Time span: {(ts1 - ts0)} seconds")


    # Breaks for 10ms intervals
    t_events = np.arange(ts0, ts1, window_size)
    
    # =========================
    #   Position (m)
    # =========================
    
    Opti_pos = np.asarray(Opti['data'][:, :3], dtype=float)   # (N,3)
    t_opti   = np.asarray(Opti['ros_timestamp'], dtype=float) # (N,)
    
    Opti_pos = resample_data_nan(t_opti, Opti_pos, t_events, method="linear")
    
    valid = (
        np.all(np.isfinite(Opti_pos), axis=1)
    )
    invalid = (
        ~np.all(np.isfinite(Opti_pos), axis=1) 
    )
    Opti_pos[invalid] = Opti_pos[valid][0]  # Replace invalid positions with the first valid position
    
    Opti_pos_filt, Opti_vel_filt, _ = kalman_filter_position_velocity_valid_only(Opti_pos, t_events)
    Opti_pos = Opti_pos_filt
    Opti_vel = Opti_vel_filt
    
    if save_images:
        save_plotly_plot(
            x=t_events,
            y=Opti_pos*100,
            label = ['X', 'Y', 'Z'],
            xlim = [0, np.max(t_events)],
            ylim = [np.nanmin(Opti_pos*100), np.nanmax(Opti_pos*100)],
            xlabel='Time (s)',
            ylabel='Position (cm)',
            output_path= output_dir+h5_file_name + '\\' + h5_file_name + "_Position.png"
        )
    print(f"Position is analyzed.")
    
    # ================================
    #   Inertial Linear velocity (m/s)
    # ================================
    # time_diffs = np.diff(t_opti, prepend=t_opti[0])  
    # time_diffs[0] = np.nan 
    
    if save_images:
        save_plotly_plot(
            x=t_events,
            y=Opti_vel*100,
            label = ['X', 'Y', 'Z'],
            xlim = [0, np.max(t_events)],
            ylim = [min(-20, np.nanmin(Opti_vel*100)), max(20, np.nanmax(Opti_vel*100))],
            xlabel='Time (s)',
            ylabel='Velocity (cm/s)',
            output_path= output_dir+h5_file_name + '\\' + h5_file_name + "_Velocity.png"
        )
    print(f"Velocity is analyzed.")
    # =========================
    #   Orientation (rad)
    # =========================
    Opti_ori = np.asarray(Opti['data'][:, 3:7], dtype=float)   # (N,4)
    
    Opti_ori = resample_data_nan(t_opti, Opti_ori, t_events, method="linear")
    
    
    quat_x = Opti_ori[:,0]
    quat_y = Opti_ori[:,1]
    quat_z = Opti_ori[:,2]
    quat_w = Opti_ori[:,3] 
    quat = np.column_stack((quat_x, quat_y, quat_z, quat_w))           # (N,4)
    
    quat, omega, _ = quaternion_kalman_filter_valid_only(quat, t_events)
    
    invalid = (
        ~np.all(np.isfinite(quat), axis=1) |
        (np.linalg.norm(quat, axis=1) < 1e-8)
    )
    
    quat[invalid] = [0, 0, 0, 1]
    r = R.from_quat(quat)
    eulers = r.as_euler('xyz', degrees=False)          # (N,3), radians

    Opti_ori = eulers
    Opti_ori_rate = omega
    
    if save_images:
        save_plotly_plot(
            x=t_events,
            y=Opti_ori,
            label = ['roll', 'pitch', 'yaw'],
            xlim = [0, np.max(t_events)],
            ylim = [np.nanmin(Opti_ori), np.nanmax(Opti_ori)],
            xlabel='Time (s)',
            ylabel='Orientation (rad)',
            output_path= output_dir+h5_file_name + '\\' + h5_file_name + "_Ori.png"
        )
    
    print(f"Orientation is analyzed.")
    # =========================
    #   Angular velocity (rad/s)
    # =========================
    # deul = np.diff(eulers, axis=0, prepend=eulers[0:1, :])  # (N,3)
    # Opti_angular_vel = deul / time_diffs[:, None]     
    # Opti_angular_vel[0, :] = 0.0
    
    # print(f"Orientation rate is analyzed.")
    # ================================
    #   Body Linear velocity (m/s)
    # ================================
    yaw = Opti_ori[:, 2]   #(N,1)
    
    vel_inertial_xy = Opti_vel[:, :2]   # take x,y only (N,2)
    vel_body_xy = np.zeros_like(vel_inertial_xy)
    vel_body_xy[:, 0] =  np.cos(yaw) * vel_inertial_xy[:, 0] + np.sin(yaw) * vel_inertial_xy[:, 1]  # body-x (forward)
    vel_body_xy[:, 1] = -np.sin(yaw) * vel_inertial_xy[:, 0] + np.cos(yaw) * vel_inertial_xy[:, 1]  # body-y (left)
    
    if save_images:
        save_plotly_plot(
            x=t_events,
            y=vel_body_xy*100,
            label = ['x', 'y'],
            xlim = [0, np.max(t_events)],
            ylim = [min(-20, np.nanmin(vel_body_xy*100)), max(20, np.nanmax(vel_body_xy*100))],
            xlabel='Time (s)',
            ylabel='Body Velocity (cm/s)',
            output_path= output_dir+h5_file_name + '\\' + h5_file_name + "_BodyVelocity.png"
        )
        
    print(f"Body Velocity is analyzed.")
    
    # =========================
    #   Motor RPM (rev/min)
    # ========================= 
    motor_rpm = np.array([rpm[:4] for rpm in Motors_RPM['data']], dtype=float)   # (RPM_N,4)
    t_motor   = np.asarray(Motors_RPM['ros_timestamp'], dtype=float)  # (RPM_N,1)
    
    
    
    motor_rpm = resample_data_nan(t_motor, motor_rpm, t_events, method="linear") # (N,4)
    
    valid = (
        np.all(np.isfinite(motor_rpm), axis=1)
    )
    invalid = (
        ~np.all(np.isfinite(motor_rpm), axis=1) 
    )
    motor_rpm[invalid] = motor_rpm[valid][0]  # Replace invalid RPMs with the first valid RPM
    

    if save_images:
        save_plotly_plot(
            x=t_events,
            y=motor_rpm,
            label = ['M1', 'M2', 'M3', 'M4'],
            xlim = [0, np.max(t_events)],
            ylim = [np.nanmin(motor_rpm), np.nanmax(motor_rpm)],
            xlabel='Time (s)',
            ylabel='Motor RPM (rev/min)',
            output_path= output_dir+h5_file_name + '\\' + h5_file_name + "_MotorRPM.png"
        )
    
    print(f"Motors RPM is analyzed.")
    # =========================
    #   Slip
    # =========================
    slip = np.zeros_like(vel_inertial_xy)
    ref_velocity = np.zeros_like(vel_inertial_xy)
    
    ref_velocity = compute_ref_velocity(ref_velocity, motor_rpm, wheel_radius, radius_curvature, half_track_width)  # (N,2)
    fs_events = 1.0 / window_size 
    ref_velocity[:,0] = lowpass_filter(ref_velocity[:,0], cutoff=2.0, fs=fs_events)
    ref_velocity[:,1] = lowpass_filter(ref_velocity[:,1], cutoff=2.0, fs=fs_events)
    
    print(f"Reference Velocity is analyzed.")
    
    nonzero_idx = ref_velocity[:,0] >= 1e-2 #Slip ratio is only defined when the reference velocity is non-zero
    
    slip[nonzero_idx,0] = ((ref_velocity[nonzero_idx,0] - vel_body_xy[nonzero_idx,0])/ ref_velocity[nonzero_idx,0])  # (N,1)
    slip[:,0] = np.clip(slip[:,0],-1,1)
    # slip[nonzero_idx,0] = (((np.pi/100) - vel_body_xy[nonzero_idx,0])/ (np.pi/100)) * 100   # (N,1)
    slip[:, 1] = np.arctan2(vel_body_xy[:,1], np.abs(vel_body_xy[:,0]))   # (N,1)
    
    
    if save_images:
        save_plotly_plot(
            x=t_events,
            y=slip[:,0],
            label = ['Longitudnal Slip Ratio'],
            xlim = [0, np.max(t_events)],
            ylim = [-1, 1],
            xlabel='Time (s)',
            ylabel='Slip Ratio',
            output_path= output_dir+h5_file_name + '\\' + h5_file_name + "_Slip.png"
        )
        
        save_plotly_plot(
            x=t_events,
            y=slip[:,1]*180/np.pi,
            label = ['Lateral Slip'],
            xlim = [0, np.max(t_events)],
            ylim = [-180, 180],
            xlabel='Time (s)',
            ylabel='Lateral Slip (deg)',
            output_path= output_dir+h5_file_name + '\\' + h5_file_name + "_LateralSlip.png"
        )
        
        save_plotly_plot(
            x=t_events,
            y=ref_velocity*100,
            label = ['X_ref', 'Y_ref'],
            xlim = [0, np.max(t_events)],
            ylim = [min(-20, np.nanmin(ref_velocity*100)), max(20, np.nanmax(ref_velocity*100))],
            xlabel='Time (s)',
            ylabel='Reference Velocity (cm/s)',
            output_path= output_dir+h5_file_name + '\\' + h5_file_name + "_ReferenceVelocity.png"
        )
        
    print(f"Slip is analyzed.")
    
    # =========================
    #   Save new HDF5 
    # =========================
    print(f"Saving new H5 file...")
    with h5py.File(output_dir+h5_file_name + '\\' + h5_file_name + '.h5', 'w') as f_out:
        f_out.create_dataset('events', data=events, compression='gzip', compression_opts=4)
        f_out.create_dataset('Opti_pos', data=Opti_pos, compression='gzip', compression_opts=4)
        f_out.create_dataset('Opti_vel', data=Opti_vel, compression='gzip', compression_opts=4)
        f_out.create_dataset('Opti_ori', data=Opti_ori, compression='gzip', compression_opts=4)
        f_out.create_dataset('Opti_ori_rate', data=Opti_ori_rate, compression='gzip', compression_opts=4)
        f_out.create_dataset('vel_body_xy', data=vel_body_xy, compression='gzip', compression_opts=4)
        f_out.create_dataset('motor_rpm', data=motor_rpm, compression='gzip', compression_opts=4)
        f_out.create_dataset('ref_velocity', data=ref_velocity, compression='gzip', compression_opts=4)
        f_out.create_dataset('slip', data=slip, compression='gzip', compression_opts=4)

    file_index += 1
    print(f"Files finished: {file_index}/{len(h5_file_list)}")