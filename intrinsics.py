#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path
import time
import yaml

# ---------- CONFIG (edit if needed) ----------
PATTERN_SIZE   = (8, 6)    # inner corners (cols, rows). e.g. 9x6 means 10x7 squares printed
SQUARE_SIZE_M  = 0.0125    # side length of one checker square in meters
MIN_MOTION_PX  = 15.0      # require avg corner motion before saving another sample
COOLDOWN_S     = 0.4       # also wait this long between saved samples
WINDOW_NAME    = "Intrinsics Calibration (press Q/ESC to finish)"
# --------------------------------------------

def find_checker(gray, pattern_size=PATTERN_SIZE):
    """Return (found, corners[N,1,2] float32). Uses SB if available, else classic+subpix."""
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags)
        if ok:
            if corners.ndim == 2: corners = corners.reshape(-1,1,2)
            return True, corners.astype(np.float32)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ok:
        return False, None
    # subpixel refine
    corners = cv2.cornerSubPix(
        gray, corners, (11,11), (-1,-1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-6)
    )
    return True, corners

def make_object_points(pattern_size=PATTERN_SIZE, square_size=SQUARE_SIZE_M):
    w, h = pattern_size
    objp = np.zeros((w*h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2) * square_size
    return objp

def avg_motion(prev, curr):
    if prev is None or curr is None: return np.inf
    a = prev.reshape(-1,2); b = curr.reshape(-1,2)
    if a.shape != b.shape: return np.inf
    return float(np.linalg.norm(a - b, axis=1).mean())

def collect_samples(cap, pattern_size=PATTERN_SIZE, min_motion_px=MIN_MOTION_PX, cooldown_s=COOLDOWN_S):
    """Continuously collect (objpoints, imgpoints) until user quits."""
    objp = make_object_points(pattern_size)
    objpoints, imgpoints = [], []
    last_corners = None
    last_save_t  = 0.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print("Collecting… move/tilt the board; keep it sharp. Press Q or ESC to finish.")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Camera read failed; stopping.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = find_checker(gray, pattern_size)

        vis = frame.copy()
        if found:
            cv2.drawChessboardCorners(vis, pattern_size, corners, True)
            motion = avg_motion(last_corners, corners)
            can_save = (motion >= min_motion_px) and (time.time() - last_save_t >= cooldown_s)

            status = f"FOUND | samples: {len(objpoints)} | motion: {motion:.1f}px"
            color  = (0,255,0)
            if can_save:
                # store a sample
                objpoints.append(objp.copy())
                imgpoints.append(corners.copy())
                last_corners = corners.copy()
                last_save_t  = time.time()
                status += "  [saved]"
        else:
            status = f"NOT FOUND | samples: {len(objpoints)}"
            color  = (0,0,255)

        cv2.putText(vis, status, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(vis, "Press Q/ESC to finish", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow(WINDOW_NAME, vis)

        k = cv2.waitKey(10) & 0xFF
        if k in (27, ord('q'), ord('Q')):  # ESC or Q
            break

    cv2.destroyWindow(WINDOW_NAME)
    if not objpoints:
        raise RuntimeError("No samples collected. Make sure the checkerboard is visible and large in the frame.")
    image_size = (frame.shape[1], frame.shape[0])  # (w,h)
    return objpoints, imgpoints, image_size

def calibrate_intrinsics(objpoints, imgpoints, image_size):
    """Run cv2.calibrateCamera and return K, dist, rms, per_view_rms."""
    flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=objpoints,
        imagePoints=imgpoints,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=flags
    )
    # per-view RMS
    per_view = []
    O = objpoints[0]
    for rv, tv, imgp in zip(rvecs, tvecs, imgpoints):
        proj, _ = cv2.projectPoints(O, rv, tv, K, dist)
        err = float(np.sqrt(np.mean(np.sum((proj - imgp)**2, axis=2))))
        per_view.append(err)
    return K, dist, float(rms), per_view

def save_yaml(path, K, dist, image_size, pattern_size=PATTERN_SIZE, square_size=SQUARE_SIZE_M, rms=None):
    data = {
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "pattern_cols": int(pattern_size[0]),
        "pattern_rows": int(pattern_size[1]),
        "square_size_m": float(square_size),
        "K": K.tolist(),
        "dist": dist.ravel().tolist(),
    }
    if rms is not None:
        data["rms_px"] = float(rms)

    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, )

    print(f"[calib] wrote clean YAML to {path}")

if __name__ == "__main__":
    # Ask which camera to use
    cam_str = input("Enter camera index (e.g., 0): ").strip()
    try:
        cam_index = int(cam_str)
    except ValueError:
        print("Please enter an integer camera index (e.g., 0).")
        raise SystemExit(1)

    # Open camera
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Failed to open camera {cam_index}")
        raise SystemExit(1)

    try:
        objpoints, imgpoints, image_size = collect_samples(cap)
    finally:
        cap.release()

    print(f"Collected {len(objpoints)} samples. Calibrating…")
    K, dist, rms, per_view = calibrate_intrinsics(objpoints, imgpoints, image_size)

    median_rms = np.median(per_view)
    print(f"\nRMS reprojection error (overall): {rms:.3f} px")
    print(f"Median per-view RMS: {median_rms:.3f} px")

    # Feedback
    if rms <= 0.3:
        print("✅ Excellent calibration (sub-pixel accuracy).")
    elif rms <= 0.5:
        print("👍 Good calibration.")
    elif rms <= 1.0:
        print("⚠️  Marginal calibration. Check undistortion; consider recollecting sharper/more diverse views.")
    else:
        print("❌ Poor calibration. Likely wrong pattern size, square size, or bad detections.")

    print("\nCamera matrix (K):\n", K)
    print("\nDistortion coefficients:\n", dist.ravel())

    out_path = Path(f"intrinsics_cam{cam_index}.yml")
    save_yaml(out_path, K, dist, image_size, rms=rms)
    print(f"Saved intrinsics to {out_path.resolve()}\n"
          f"- image size: {image_size}\n- pattern: {PATTERN_SIZE} (inner corners)\n- square size: {SQUARE_SIZE_M} m")
