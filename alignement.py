import cv2, numpy as np, time
PATTERN_SIZE = (8, 6)      # inner corners; swap to (6,9) if needed
SQUARE_SIZE  = 0.0125       # meters
MIN_CAMS_PER_SAMPLE = 2

def find_checker(gray):
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, c = cv2.findChessboardCornersSB(gray, PATTERN_SIZE,
                                            cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if ok:
            c = c.astype(np.float32)
            if c.ndim == 2: c = c.reshape(-1,1,2)
            return True, c
    ok, c = cv2.findChessboardCorners(gray, PATTERN_SIZE,
                                      cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok: return False, None
    c = cv2.cornerSubPix(gray, c, (11,11), (-1,-1),
                         (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,40,1e-6))
    return True, c

def make_objp():
    w,h = PATTERN_SIZE
    objp = np.zeros((w*h,3), np.float32)
    objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)*SQUARE_SIZE
    return objp

def collect_sessions(caps, n_samples=30, autosave=False):
    objp = make_objp()
    sessions = []
    cv2.namedWindow("collect", cv2.WINDOW_NORMAL)
    while len(sessions) < n_samples:
        for c in caps: c.grab()
        frames, det, vis = [], {}, []
        for i,c in enumerate(caps):
            ok,f = c.retrieve(); frames.append(f if ok else None)
            if f is None: 
                vis.append(np.zeros((480,640,3),np.uint8)); continue
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            ok,corners = find_checker(g)
            if ok:
                det[i]=corners
                d=f.copy(); cv2.drawChessboardCorners(d,PATTERN_SIZE,corners,True); vis.append(d)
            else:
                vis.append(f)
        stacked = np.hstack(vis)
        msg = f"SPACE=save  A=autosave({autosave})  Q=quit | samples {len(sessions)}/{n_samples} | cams found {len(det)}"
        cv2.putText(stacked, msg, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0) if len(det)>=MIN_CAMS_PER_SAMPLE else (0,0,255), 2)
        cv2.imshow("collect", stacked)
        k = cv2.waitKey(20) & 0xFF
        if k==ord('q'): break
        if k==ord('a'): autosave = not autosave
        if ((k==32) or autosave) and len(det)>=MIN_CAMS_PER_SAMPLE:
            sessions.append({"objp":objp.copy(), "imgpoints":det})
            print(f"Saved sample {len(sessions)} with {len(det)} cams")
    cv2.destroyWindow("collect")
    first = next(f for f in frames if f is not None)
    return sessions, (first.shape[1], first.shape[0])

def calibrate_intrinsics(caps, sessions, image_size):
    from collections import defaultdict
    per_obj, per_img = defaultdict(list), defaultdict(list)
    for s in sessions:
        for ci,corners in s["imgpoints"].items():
            per_obj[ci].append(s["objp"])
            per_img[ci].append(corners)
    intr = {}
    for ci in range(len(caps)):
        if len(per_obj[ci]) < 8:
            raise RuntimeError(f"Camera {ci}: need >=8 detections, have {len(per_obj[ci])}")
        rms,K,dist,_,_ = cv2.calibrateCamera(per_obj[ci], per_img[ci], image_size, None, None,
                                             flags=cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3)
        intr[ci] = {"K":K, "dist":dist, "rms":rms}
        print(f"cam {ci}: RMS {rms:.3f}px")
    return intr

def extrinsics_to_ref(caps, sessions, intr, image_size, ref=0):
    def pair_samples(a,b):
        obj,ia,ib = [],[],[]
        for s in sessions:
            if a in s["imgpoints"] and b in s["imgpoints"]:
                obj.append(s["objp"]); ia.append(s["imgpoints"][a]); ib.append(s["imgpoints"][b])
        return obj,ia,ib
    extr = {ref:{"R":np.eye(3), "T":np.zeros((3,1)), "rms":0.0}}
    for ci in range(len(caps)):
        if ci==ref: continue
        obj,ia,ib = pair_samples(ref, ci)
        if len(obj) < 8:
            print(f"Skip cam {ci}: only {len(obj)} overlaps with ref {ref}")
            continue
        K0,d0 = intr[ref]["K"], intr[ref]["dist"]
        K1,d1 = intr[ci]["K"], intr[ci]["dist"]
        rms, *_ , R,T, E,F = cv2.stereoCalibrate(obj, ia, ib, K0,d0, K1,d1, image_size,
                                                 flags=cv2.CALIB_FIX_INTRINSIC,
                                                 criteria=(cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS,200,1e-9))
        extr[ci] = {"R":R, "T":T, "E":E, "F":F, "rms":rms}
        print(f"ref {ref} -> cam {ci}: RMS {rms:.3f}px, baseline {np.linalg.norm(T):.4f} m")
    return extr

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def _camera_center_world(R_wi, T_wi):
    # World->cam: X_cam = R_wi X_w + T_wi  =>  camera center in world: C = -R^T T
    return -R_wi.T @ T_wi

def _optical_axis_world(R_wi):
    # In OpenCV, camera looks along +Z in camera coords. Direction in world: R^T * [0,0,1]
    return R_wi.T @ np.array([0.0, 0.0, 1.0])

def _frustum_points_world(K, R_wi, T_wi, image_size, depth):
    """Return 5x3 array: [C, p_ul, p_ur, p_lr, p_ll] in world coords (ul=upper-left)."""
    w, h = image_size
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # Rays for the 4 image corners in camera coords at distance 'depth'
    # Using pinhole: x = (u-cx)/fx * z, y = (v-cy)/fy * z, z = depth
    corners_px = np.array([[0,   0],
                           [w-1, 0],
                           [w-1, h-1],
                           [0,   h-1]], dtype=np.float64)
    rays_cam = []
    for (u,v) in corners_px:
        x = (u - cx) / fx * depth
        y = (v - cy) / fy * depth
        z = depth
        rays_cam.append(np.array([x,y,z]))
    rays_cam = np.stack(rays_cam, axis=0)  # (4,3)

    # Transform to world: X_w = R^T (X_cam - T)
    P_world = (R_wi.T @ (rays_cam.T - T_wi)).T  # (4,3)
    C_world = _camera_center_world(R_wi, T_wi).reshape(1,3)
    return np.vstack([C_world, P_world])

def plot_cameras_3d(intr, extr, image_size, ref_cam=0, ax=None,
                    frustum_depth=None, dir_len=None, annotate=True):
    """
    intr: dict cam -> {'K', 'dist', 'rms'}
    extr: dict cam -> {'R','T','rms',...}, with ref_cam present and R=I, T=0
    image_size: (w,h)
    """
    cams = sorted(intr.keys())
    # Figure out a sensible scale from baselines
    baselines = []
    for k,v in extr.items():
        if k == ref_cam: continue
        if "T" in v and v["T"] is not None:
            baselines.append(np.linalg.norm(v["T"]))
    scale = np.median(baselines) if baselines else 0.2
    if frustum_depth is None:
        frustum_depth = max(0.2*scale, 0.05)  # meters
    if dir_len is None:
        dir_len = max(0.4*scale, 0.1)

    if ax is None:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')

    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4'])
    w,h = image_size

    for idx, ci in enumerate(cams):
        col = colors[idx % len(colors)]
        if ci == ref_cam:
            R_wi = np.eye(3)
            T_wi = np.zeros((3,1))
        else:
            if ci not in extr: 
                continue  # no solution for this cam
            R_wi = extr[ci]["R"]
            T_wi = extr[ci]["T"]
        K = intr[ci]["K"]

        # Camera center & axis
        C = _camera_center_world(R_wi, T_wi).ravel()
        dir_vec = _optical_axis_world(R_wi)
        dir_vec = dir_vec/np.linalg.norm(dir_vec+1e-12) * dir_len

        # Frustum
        P = _frustum_points_world(K, R_wi, T_wi, image_size, depth=frustum_depth)
        Cw, UL, UR, LR, LL = P[0], P[1], P[2], P[3], P[4]

        # Draw: center, axis, frustum edges
        ax.scatter([C[0]], [C[1]], [C[2]], marker='o', s=40, color=col, depthshade=True)
        ax.plot([C[0], C[0]+dir_vec[0]],
                [C[1], C[1]+dir_vec[1]],
                [C[2], C[2]+dir_vec[2]],
                lw=2, color=col, label=(f"cam {ci}" if annotate else None))

        # Pyramid sides
        for Q in [UL, UR, LR, LL]:
            ax.plot([Cw[0], Q[0]], [Cw[1], Q[1]], [Cw[2], Q[2]], color=col, lw=1, alpha=0.8)
        # Rectangle at depth
        ax.plot([UL[0], UR[0], LR[0], LL[0], UL[0]],
                [UL[1], UR[1], LR[1], LL[1], UL[1]],
                [UL[2], UR[2], LR[2], LL[2], UL[2]],
                color=col, lw=1)

        if annotate:
            fx, fy = K[0,0], K[1,1]
            ax.text(C[0], C[1], C[2], f"cam {ci}\nfx={fx:.0f}, fy={fy:.0f}",
                    color=col, fontsize=9)

    # Make axes equal-ish
    all_pts = []
    for v in extr.values():
        R_wi = v["R"] if "R" in v else np.eye(3)
        T_wi = v["T"] if "T" in v else np.zeros((3,1))
        all_pts.append(_camera_center_world(R_wi, T_wi).ravel())
    if all_pts:
        P = np.vstack(all_pts)
        mins, maxs = P.min(axis=0), P.max(axis=0)
        ctr = (mins+maxs)/2.0
        extent = (maxs-mins).max()
        extent = max(extent, frustum_depth*3 + dir_len)
        for i,(l,u) in enumerate(zip(ctr-extent/2, ctr+extent/2)):
            if i==0: ax.set_xlim(l,u)
            if i==1: ax.set_ylim(l,u)
            if i==2: ax.set_zlim(l,u)

    ax.set_xlabel('X (world)')
    ax.set_ylabel('Y (world)')
    ax.set_zlabel('Z (world)')
    ax.set_title('Multi-camera rig (world = reference camera frame)')
    if annotate:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    return ax

# --- run it ---
caps = [cv2.VideoCapture(0), cv2.VideoCapture(2)]
sessions, image_size = collect_sessions(caps, n_samples=30)
intr = calibrate_intrinsics(caps, sessions, image_size)
extr = extrinsics_to_ref(caps, sessions, intr, image_size, ref=0)

print(extr)

ax = plot_cameras_3d(intr, extr, image_size, ref_cam=0)
plt.show()