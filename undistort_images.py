import numpy as np
import cv2
from pathlib import Path
from scipy.io import loadmat

# ==== paths ====
mat_path = Path(r"Path_to_calibration_file_(added to this repository)")
img_path = Path(r"Path_to_distorted_image")
out_path = img_path.with_name(img_path.stem + "_undistorted.png")

md = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

def fetch(*names):
    for n in names:
        if n in md and md[n] is not None:
            return md[n]
    return None

# --- intrinsics (K) ---
K = fetch("K", "IntrinsicMatrix")
K = np.array(K, dtype=float)

# If it came from IntrinsicMatrix (MATLAB convention), transpose to OpenCV
if K.shape == (3, 3) and "IntrinsicMatrix" in md:
    K = K.T

# --- distortion (radial + tangential) ---
rad = fetch("RD", "RadialDistortion")
tan = fetch("TD", "TangentialDistortion")

if rad is None or tan is None:
    raise RuntimeError("Could not find distortion fields. Expected RD/RadialDistortion and TD/TangentialDistortion.")

rad = np.atleast_1d(np.array(rad, dtype=float).ravel())
tan = np.atleast_1d(np.array(tan, dtype=float).ravel())

# OpenCV order: [k1, k2, p1, p2, k3]
k1 = rad[0] if rad.size > 0 else 0.0
k2 = rad[1] if rad.size > 1 else 0.0
k3 = rad[2] if rad.size > 2 else 0.0
p1 = tan[0] if tan.size > 0 else 0.0
p2 = tan[1] if tan.size > 1 else 0.0
dist = np.array([k1, k2, p1, p2, k3], dtype=float)

# --- undistort ---
img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(f"Could not read {img_path}")

h, w = img.shape[:2]
newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)  # set alpha=1 to keep FOV with black borders
undist = cv2.undistort(img, K, dist, None, newCameraMatrix=newK)
cv2.imwrite(str(out_path), undist)

print("K:\n", K)
print("dist [k1 k2 p1 p2 k3]:", dist.tolist())
print("Saved:", out_path)
