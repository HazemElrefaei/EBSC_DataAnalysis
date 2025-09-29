# EBSC_DataAnalysis

This repository provides tools for analyzing EBSC rover data and correcting camera image distortion using calibration parameters.

## Repository Structure

- `EBSC_DataAnalysis.py`: Main script for processing rover HDF5 data, generating plots, and saving analysis results.
- `undistort_images.py`: Script for undistorting images using camera calibration parameters from a `.mat` file.
- `CalibParameters_struct.mat`: Example MATLAB calibration file containing camera intrinsics and distortion coefficients.
- `README.md`: Documentation and usage instructions.

---

## Requirements

Install dependencies using pip:

```sh
pip install numpy scipy pandas opencv-python h5py tqdm plotly kaleido
```

---

## Usage

### 1. Data Analysis (`EBSC_DataAnalysis.py`)

This script reads rover data from HDF5 files, analyzes position, velocity, orientation, slip, and motor RPM, and saves high-quality plots.

#### Steps:

1. **Edit Paths**  
   - Set `output_dir` to your desired output folder.
   - Set `path_to_h5` to the folder containing your `.h5` data files.
   - Update `h5_file_list` with the filenames (without `.h5` extension) you want to process.

2. **Run the Script**  
   ```sh
   python EBSC_DataAnalysis.py
   ```
   - Plots will be saved in subfolders under `output_dir`.

#### Output

- Position, velocity, orientation, slip, and motor RPM plots as PNG images.

---

### 2. Image Undistortion (`undistort_images.py`)

This script uses camera calibration parameters to undistort images.

#### Steps:

1. **Edit Paths**  
   - Set `mat_path` to the path of your calibration `.mat` file (e.g., `CalibParameters_struct.mat`).
   - Set `img_path` to the path of your distorted image.

2. **Run the Script**  
   ```sh
   python undistort_images.py
   ```
   - The undistorted image will be saved with `_undistorted.png` appended to the original filename.

#### Notes

- The calibration `.mat` file must contain either `K` or `IntrinsicMatrix` for camera intrinsics, and `RD`/`RadialDistortion` and `TD`/`TangentialDistortion` for distortion coefficients.

---

## Example

Suppose you have:
- Calibration file: `CalibParameters_struct.mat`
- Distorted image: `distorted_image.png`
- Data file: `2025_07_17_17_19_17_Vel1_Lev3_ON.h5`

Set the paths in both scripts accordingly and run them as described above.

---

## Troubleshooting

- If you encounter missing fields in the `.mat` file, ensure it contains the required keys.
- For HDF5 data, verify the dataset names match those expected in the script.

---

## License

This repository is for academic and research use. Please contact the authors for other uses.

---

## Contact

For questions or support, please contact Hazem Elrefaei.