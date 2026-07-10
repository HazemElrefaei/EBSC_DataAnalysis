#!/usr/bin/env python3
import os
import sys
import logging
import multiprocessing

import h5py
import numpy as np
import cv2
from tqdm import tqdm

# ========================
# CONFIG
# ========================

# Root where your H5 experiments live (each subfolder has <exp_id>.h5)
PUBLISHED_ROOT = r"FILE_PATH_TO_PUBLISHED_EXPERIMENTS"

# Root where gray frames will be written
# Final layout:
#   FRAMES_ROOT/<exp_id>/S1_undistorted/frame_0001.png, frame_0002.png, ...
FRAMES_ROOT = r"FILE_PATH_TO_FRAMES_DIRECTORY"

# EDTS settings
TAU = 0.4         # decay constant
WINDOW_SIZE = 0.01  # time window for generating new frame

# Use S1 (polarity == 1) EDTS, grayscale
REPRESENTATION = "GRAY"  # we focus on GRAY EDTS


# ========================
# LOGGING
# ========================

logging.basicConfig(
    level=logging.INFO,
    format='[%(processName)s] %(asctime)s [%(levelname)s]: %(message)s',
    stream=sys.stdout
)
log = logging.getLogger(__name__)


# ========================
# HDF5 HELPERS
# ========================

def read_dataset_in_chunks(dataset, chunk_size=40000):
    """
    Read a HDF5 dataset in chunks with a progress bar.
    dataset: h5py dataset, e.g. f['events'] of shape (N,4)
    Returns: np.ndarray of shape (N,4)
    """
    data = []
    total_size = dataset.shape[0]

    with tqdm(total=total_size, desc=f"Reading {dataset.name}", unit="rows") as pbar:
        for i in range(0, total_size, chunk_size):
            chunk = dataset[i:min(i + chunk_size, total_size)]
            data.append(chunk)
            pbar.update(len(chunk))

    return np.concatenate(data, axis=0)


# ========================
# EDTS HELPERS
# ========================

def set_RAW(S, x, y, polarity):
    if polarity == 1:
        S[y, x] = 1.0
    else:
        S[y, x] = -1.0


def set_EDTS(T, S, x, y, timedata):
    T[y, x] = timedata
    S[y, x] = 1.0


def update_EDTS(T, S, highest_timedata, tau):
    """
    Update EDTS surface:
        delta_T = T - highest_timedata
        S *= exp(delta_T / tau)
    delta_T <= 0, so exp(delta_T / tau) in (0,1].
    """
    delta_T = T - highest_timedata
    S_temp = np.exp(delta_T / tau)
    np.multiply(S, S_temp, out=S)


def process_events(events):
    """
    Process events to generate time-surface frames (EDTS) for S1 and S0.

    Args:
        events: NumPy array of events with shape (N, 4)
                columns: [x, y, polarity, timestamp]

    Returns:
        For REPRESENTATION == 'GRAY':
            S1_frames: np.ndarray of shape (num_frames, H, W) float32 in [0,1]
            S0_frames: np.ndarray of shape (num_frames, H, W) float32 in [0,1]
        For REPRESENTATION == 'RAW':
            S1_frames: np.ndarray of shape (num_frames, H, W)
    """
    tau = TAU
    window_size = WINDOW_SIZE

    # Infer sensor size from events
    sensor_width = int(events[:, 0].max() + 1)
    sensor_height = int(events[:, 1].max() + 1)

    # Initialize time surfaces
    T1_ = np.zeros((sensor_height, sensor_width), dtype=np.float32)
    S1_ = np.zeros((sensor_height, sensor_width), dtype=np.float32)
    T0_ = np.zeros((sensor_height, sensor_width), dtype=np.float32)
    S0_ = np.zeros((sensor_height, sensor_width), dtype=np.float32)

    window_idx = 0
    frames = []
    frames_1 = []
    frames_0 = []

    # Ensure events are sorted by time
    events = events[np.argsort(events[:, 3])]

    for ev in tqdm(events, desc="Processing events -> EDTS S1/S0"):
        x = int(ev[0])
        y = int(ev[1])
        polarity = 1 if ev[2] else 0
        timedata = float(ev[3])

        if REPRESENTATION == 'RAW':
            set_RAW(S1_, x, y, polarity)
        else:
            if polarity == 1:
                set_EDTS(T1_, S1_, x, y, timedata)
            # else:
            #     set_EDTS(T0_, S0_, x, y, timedata)

        # Advance window
        if timedata > window_idx * window_size:
            if REPRESENTATION == 'RAW':
                # Save current raw frame and reset
                frames.append(S1_.copy())
                S1_ = np.zeros((sensor_height, sensor_width), dtype=np.float32)
            else:
                # Save current EDTS snapshot
                if polarity == 1:
                    set_EDTS(T1_, S1_, x, y, timedata)
                # else:
                #     set_EDTS(T0_, S0_, x, y, timedata)

                frames_1.append(S1_.copy())
                # frames_0.append(S0_.copy())

                # Update surfaces for next window
                update_EDTS(T1_, S1_, timedata, tau)
                # update_EDTS(T0_, S0_, timedata, tau)

            window_idx += 1

    if REPRESENTATION == 'RAW':
        return np.array(frames, dtype=np.float32)
    else:
        S1_frames = np.array(frames_1, dtype=np.float32)
        # S0_frames = np.array(frames_0, dtype=np.float32)
        return S1_frames


# ========================
# FRAME SAVING
# ========================

def save_single_frame(args):
    """
    Save a single S1 frame to PNG.

    args: (i, S1_i, output_path, exp_id)
    """
    i, S1_i, output_path, exp_id = args
    os.makedirs(output_path, exist_ok=True)

    # File name: frame_0001.png, frame_0002.png, ...
    frame_filename = os.path.join(output_path, f"frame_{i+1:04d}.png")

    # S1_i is float32 EDTS (ideally in [0,1]); clip just in case
    img = np.clip(S1_i, 0.0, 1.0)

    # Scale to [0,255] and convert to uint8 for PNG
    img_u8 = (img * 255.0).astype(np.uint8)

    # Write as single-channel grayscale PNG
    cv2.imwrite(frame_filename, img_u8)


# ========================
# MAIN LOOP
# ========================

def main():
    # List experiments in PUBLISHED_ROOT
    exp_ids = []
    for name in sorted(os.listdir(PUBLISHED_ROOT)):
        exp_dir = os.path.join(PUBLISHED_ROOT, name)
        if not os.path.isdir(exp_dir):
            continue
        h5_path = os.path.join(exp_dir, name + ".h5")
        if os.path.isfile(h5_path):
            exp_ids.append(name)

    log.info(f"Found {len(exp_ids)} experiments in {PUBLISHED_ROOT}")

    for exp_id in exp_ids:
        log.info(f"Processing experiment: {exp_id}")

        h5_path = os.path.join(PUBLISHED_ROOT, exp_id, exp_id + ".h5")
        if not os.path.isfile(h5_path):
            log.warning(f"H5 file missing for {exp_id}: {h5_path}")
            continue

        # Output path: FRAMES_ROOT/<exp_id>/S1_undistorted
        output_path = os.path.join(FRAMES_ROOT, exp_id, "S1_undistorted")
        os.makedirs(output_path, exist_ok=True)

        # Read events
        with h5py.File(h5_path, "r") as f:
            if "events" not in f:
                log.warning(f"'events' dataset missing in {h5_path}")
                continue

            log.info(f"Reading events for {exp_id} ...")
            events = read_dataset_in_chunks(f["events"])

        # Process events -> EDTS frames
        if REPRESENTATION == "RAW":
            S1_frames = process_events(events)
            S0_frames = None
        else:
            S1_frames = process_events(events)

        events = []

        num_frames = S1_frames.shape[0]
        log.info(f"{exp_id}: generated {num_frames} S1 frames")

        # Save S1 frames only (these are what your training uses)
        frame_args = [
            (i, S1_frames[i], output_path, exp_id)
            for i in range(num_frames)
        ]

        # Use multiprocessing to speed up saving
        num_procs = max(1, multiprocessing.cpu_count() - 2)
        with multiprocessing.Pool(processes=num_procs) as pool:
            list(tqdm(
                pool.imap(save_single_frame, frame_args),
                total=num_frames,
                desc=f"Saving S1 frames for {exp_id}",
                unit="frame"
            ))

        log.info(f"Done with experiment: {exp_id}")


if __name__ == "__main__":
    if os.name == "nt":
        _ = os.system("cls")
    else:
        _ = os.system("clear")

    main()
