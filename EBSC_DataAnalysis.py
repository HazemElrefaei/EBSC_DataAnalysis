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
    
output_dir = r'output_path_to_store_images'

path_to_h5 = r'Path_to_h5_file'

h5_file_list= [
        '2025_07_17_17_19_17_Vel1_Lev3_ON',
]

file_index = 0

for h5_file_name in h5_file_list:
    os.makedirs(output_dir+h5_file_name, exist_ok=True)
    print(f"Processing file: {h5_file_name}")
    
    Opti = {}
    Motors = {}
    events = None
    Body_vel = None
    Slip = None
    with h5py.File(path_to_h5+h5_file_name + '.h5', 'r') as f:
        events = read_dataset_in_chunks(f['events'])
        Opti['Position'] = read_dataset_in_chunks(f['Opti_pos'])
        Opti['Velocity'] = read_dataset_in_chunks(f['Opti_vel'])
        Opti['Orientation'] = read_dataset_in_chunks(f['Opti_ori'])
        Opti['Ori_rate'] = read_dataset_in_chunks(f['Opti_ori_rate'])
        Body_vel = read_dataset_in_chunks(f['vel_body_xy'])
        Motors['RPM'] = read_dataset_in_chunks(f['motor_speed'])
        Motors['Ref_Vel'] = read_dataset_in_chunks(f['ref_velocity'])
        Slip = read_dataset_in_chunks(f['slip'])

        
    print(f"Messages have been read.")

    sensor_width = max(events[:,0]) + 1
    sensor_height = max(events[:,1]) + 1

    ts0 = events[0, 3]
    ts1 = events[-1, 3]
    print(f"Time span: {(ts1 - ts0)} seconds")
    

    # Breaks for 10ms intervals
    t_events = np.arange(ts0, ts1, window_size)
    
    # =========================
    #   Position (m)
    # =========================
    
    if save_images:
        save_plotly_plot(
            x=t_events,
            y=Opti['Position']*100,
            label = ['X', 'Y', 'Z'],
            xlim = [0, np.max(t_events)],
            ylim = [np.min(Opti_pos*100), np.max(Opti_pos*100)],
            xlabel='Time (s)',
            ylabel='Position (cm)',
            output_path= output_dir+h5_file_name + '\\' + h5_file_name + "_Position.png"
        )
    print(f"Position is analyzed.")
    
    # ================================
    #   Inertial Linear velocity (m/s)
    # ================================
    
    if save_images:
        save_plotly_plot(
            x=t_events,
            y=Opti['Velocity']*100,
            label = ['X', 'Y', 'Z'],
            xlim = [0, np.max(t_events)],
            ylim = [min(-20, np.min(Opti_vel*100)), max(20, np.max(Opti_vel*100))],
            xlabel='Time (s)',
            ylabel='Velocity (cm/s)',
            output_path= output_dir+h5_file_name + '\\' + h5_file_name + "_Velocity.png"
        )
    print(f"Velocity is analyzed.")
    # =========================
    #   Orientation (rad)
    # =========================
    
    if save_images:
        save_plotly_plot(
            x=t_events,
            y=Opti['Orientation'],
            label = ['roll', 'pitch', 'yaw'],
            xlim = [0, np.max(t_events)],
            ylim = [np.min(Opti_ori), np.max(Opti_ori)],
            xlabel='Time (s)',
            ylabel='Orientation (rad)',
            output_path= output_dir+h5_file_name + '\\' + h5_file_name + "_Ori.png"
        )
    
    print(f"Orientation is analyzed.")

    # ================================
    #   Body Linear velocity (m/s)
    # ================================
    if save_images:
        save_plotly_plot(
            x=t_events,
            y=Body_vel*100,
            label = ['x', 'y'],
            xlim = [0, np.max(t_events)],
            ylim = [min(-20, np.min(vel_body_xy*100)), max(20, np.max(vel_body_xy*100))],
            xlabel='Time (s)',
            ylabel='Body Velocity (cm/s)',
            output_path= output_dir+h5_file_name + '\\' + h5_file_name + "_BodyVelocity.png"
        )
        
    print(f"Body Velocity is analyzed.")
    
    # =========================
    #   Motor RPM (rev/min)
    # ========================= 
    if save_images:
        save_plotly_plot(
            x=t_events,
            y=Motors['RPM'],
            label = ['M1', 'M2', 'M3', 'M4'],
            xlim = [0, np.max(t_events)],
            ylim = [np.min(motor_rpm), np.max(motor_rpm)],
            xlabel='Time (s)',
            ylabel='Motor RPM (rev/min)',
            output_path= output_dir+h5_file_name + '\\' + h5_file_name + "_MotorRPM.png"
        )
    
    print(f"Motors RPM is analyzed.")
    # =========================
    #   Slip
    # =========================
    
    if save_images:
        save_plotly_plot(
            x=t_events,
            y=Slip[:,0],
            label = ['Longitudnal Slip Ratio'],
            xlim = [0, np.max(t_events)],
            ylim = [-1, 1],
            xlabel='Time (s)',
            ylabel='Slip Ratio',
            output_path= output_dir+h5_file_name + '\\' + h5_file_name + "_Slip.png"
        )
        
        save_plotly_plot(
            x=t_events,
            y=Slip[:,1]*180/np.pi,
            label = ['Lateral Slip'],
            xlim = [0, np.max(t_events)],
            ylim = [-180, 180],
            xlabel='Time (s)',
            ylabel='Lateral Slip (deg)',
            output_path= output_dir+h5_file_name + '\\' + h5_file_name + "_LateralSlip.png"
        )
        
        save_plotly_plot(
            x=t_events,
            y=Motors['Ref_Vel']*100,
            label = ['X_ref', 'Y_ref'],
            xlim = [0, np.max(t_events)],
            ylim = [min(-20, np.min(ref_velocity*100)), max(20, np.max(ref_velocity*100))],
            xlabel='Time (s)',
            ylabel='Reference Velocity (cm/s)',
            output_path= output_dir+h5_file_name + '\\' + h5_file_name + "_ReferenceVelocity.png"
        )
        
    print(f"Slip is analyzed.")
    

    file_index += 1
    print(f"Files finished: {file_index}/{len(h5_file_list)}")
