import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import multiprocessing
import logging
import sys
import cv2

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(processName)s] %(asctime)s [%(levelname)s]: %(message)s',
    stream=sys.stdout
)
log = logging.getLogger()

def read_dataset_in_chunks(dataset, chunk_size=40000):
    """Read a HDF5 dataset in chunks with a progress bar."""
    data = []
    total_size = dataset.shape[0]
    
    with tqdm(total=total_size, desc=f"Reading {dataset.name}", unit="rows") as pbar:
        for i in range(0, total_size, chunk_size):
            chunk = dataset[i:min(i + chunk_size, total_size)]
            data.append(chunk)
            pbar.update(len(chunk))
    
    return np.concatenate(data)


# === SETUP ===
path_to_h5 = r'path_to_h5_file'
output_dir = r'output_path'
os.makedirs(output_dir, exist_ok=True)

Representation = 'GRAY' # 'RAW' or 'JET' or 'GRAY'

file_list = [
    '2025_07_17_17_19_17_Vel1_Lev3_ON',
]

def set_RAW(S,x,y,polarity):
    if polarity == 1:
        S[y, x] = 1.0
    else:
        S[y, x] = -1.0
def set_EDTS(T, S, x, y, timedata):
    T[y, x] = timedata
    S[y, x] = 1.0

def update_EDTS(T, S, highest_timedata, tau):
    delta_T = T - highest_timedata
    S_temp = np.exp(delta_T / tau)
    np.multiply(S, S_temp, out=S) 
    
def process_events(events):
    """
    Process events to generate time surface frames
    Args:
        events: NumPy array of events with shape (N, 4) containing [x, y, polarity, timestamp]
        window_size: Time window size for frame generation
        tau: Decay constant for the time surface
    """
    # Initialize time surfaces
    tau = 0.4  # Define decay constant
    window_size = 0.01
    sensor_width = int(events[:,0].max() + 1)
    sensor_height = int(events[:,1].max() + 1)
    T1_ = np.zeros((sensor_height, sensor_width), dtype=np.float32)
    S1_ = np.zeros((sensor_height, sensor_width), dtype=np.float32)
    T0_ = np.zeros((sensor_height, sensor_width), dtype=np.float32)
    S0_ = np.zeros((sensor_height, sensor_width), dtype=np.float32)
    
    window_idx = 0
    frames = []
    frames_1 = []
    frames_0 = []
    # Process each event
    for ev in tqdm(events, desc="Processing events to S1_raw"):
        x = int(ev[0])
        y = int(ev[1])
        polarity = 1 if ev[2] else 0
        timedata = float(ev[3])
        
        
        if Representation == 'RAW':
            set_RAW(S1_,x,y,polarity)   
        else:
            if polarity == 1:
                set_EDTS(T1_, S1_, x, y, timedata)
            else:
                set_EDTS(T0_, S0_, x, y, timedata)

        if timedata > window_idx * window_size:
            if Representation == 'RAW':
                set_RAW(S1_,x,y,polarity) 
                frames.append(S1_.copy())
                S1_ = np.zeros((sensor_height, sensor_width), dtype=np.float32)
            else:
                # Save current frame
                if polarity == 1:
                    set_EDTS(T1_, S1_, x, y, timedata) 
                else:
                    set_EDTS(T0_, S0_, x, y, timedata)          

                frames_1.append(S1_.copy())
                frames_0.append(S0_.copy())
                # Update surfaces for next window
                update_EDTS(T1_, S1_, timedata, tau)
                update_EDTS(T0_, S0_, timedata, tau)
            window_idx += 1
    
    if Representation == 'RAW':
        return np.array(frames)
    else:
        return np.array(frames_1), np.array(frames_0) 

def save_frame(args):
    if Representation == 'RAW':
        i, S1_i, output_path, file_name = args
        frame_filename = os.path.join(output_path, f"{file_name}_image{i+1}.png")
        plt.imsave(frame_filename, S1_i, cmap='gray', format='png')
    else:
        i, S1_i, S0_i, output_path, file_name = args
        frame_filename_1 = os.path.join(output_path, f"{file_name}_S1_image{i+1}.png")
        frame_filename_0 = os.path.join(output_path, f"{file_name}_S0_image{i+1}.png")
        if Representation == 'JET':
            plt.imsave(frame_filename_1, S1_i, cmap='jet', format='png')
            plt.imsave(frame_filename_0, S0_i, cmap='jet', format='png')
        elif Representation == 'GRAY':
            plt.imsave(frame_filename_1, S1_i, cmap='gray', format='png')
            plt.imsave(frame_filename_0, S0_i, cmap='gray', format='png')

if __name__ == "__main__":
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')

    for file_name in file_list:
        print(f"Processing file: {file_name}")
        output_path = os.path.join(output_dir, f"{file_name}")
        os.makedirs(output_path, exist_ok=True)

        S1 = []
        S0 = []
        events = []
        frame_args = []
        with h5py.File(os.path.join(path_to_h5, file_name + '.h5'), 'r') as f:
            log.info(f"Reading events data ...")
            events = read_dataset_in_chunks(f['events'])
            if Representation == 'RAW':
                S1 = process_events(events)
            else:
                S1, S0 = process_events(events)
            events = []

        num_frames = S1.shape[0]
        print(f"{file_name}: {num_frames} frames")

        # Create list of (i, S1[i], file_name) for each frame
        if Representation == 'RAW':
            frame_args = [(i, S1[i], output_path,file_name) for i in range(num_frames)]
        else:
            frame_args = [(i, S1[i], S0[i], output_path,file_name) for i in range(num_frames)]

        with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 2) as pool:
            list(tqdm(pool.imap(save_frame, frame_args),
                      total=num_frames,
                      desc=f"Saving frames for {file_name}",
                      unit="frame"))
