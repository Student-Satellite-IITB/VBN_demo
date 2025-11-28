import numpy as np
import cv2

# =======================
#  Pattern: ID & ordering
# =======================

def identify_and_order(pts2d, distinct_ratio=0.70):
    """
    Identify the 5-LED cross without circle/ellipse fitting.
    Returns ordered 2D points [Right, Up, Left, Down, Offset] plus helpers.
    """
    pts2d = np.asarray(pts2d, dtype=np.float32)
    if pts2d.shape[0] < 5:
        return {"ok": False, "reason": "need 5 points"}

    # # 1) inner vs outer by distance to centroid (of all 5)
    # C = np.mean(pts2d, axis=0)
    # dists = np.linalg.norm(pts2d - C, axis=1)
    # idx_sorted = np.argsort(dists)
    # idx_offset = int(idx_sorted[0])
    # idx_outer  = idx_sorted[1:5]
    offset_pt  = pts2d[4]
    outer_pts  = pts2d[:4]

    # # distinctness check
    # offset_d = float(dists[idx_offset])
    # med_outer = float(np.median(dists[idx_outer]))
    # if med_outer <= 1e-6 or (offset_d / med_outer) > distinct_ratio:
    #     return {"ok": False, "reason": "ambiguous inner/outer"}

    # # 2) local axes from C->offset (Up) and its +90° CW (Right)
    # v_up = offset_pt - C
    # n = float(np.linalg.norm(v_up))
    # if n < 1e-6:
    #     return {"ok": False, "reason": "degenerate up vector"}
    # e_up = v_up / n
    # e_right = np.array([e_up[1], -e_up[0]], dtype=np.float32)

    # # 3) project outers and pick extrema for labels
    # rel = outer_pts - C
    # x = rel @ e_right
    # y = rel @ e_up

    # labels = ["Right","Up","Left","Down"]
    # metrics = {"Right": x, "Up": y, "Left": -x, "Down": -y}
    # assigned, used = {}, set()
    # for lab in labels:
    #     vals = metrics[lab]
    #     order = np.argsort(-vals)
    #     pick = next((i for i in order if i not in used), None)
    #     if pick is None:
    #         return {"ok": False, "reason": "label assignment failed"}
    #     assigned[lab] = pick
    #     used.add(pick)

    # circle_ordered = np.vstack([
    #     outer_pts[assigned["Down"]],
    #     outer_pts[assigned["Right"]],
    #     outer_pts[assigned["Up"]],
    #     outer_pts[assigned["Left"]],
    # ]).astype(np.float32)
    circle_ordered = np.vstack([
        pts2d[0],
        pts2d[1],
        pts2d[2],
        pts2d[3],
    ]).astype(np.float32)

    ordered_pts2d = np.vstack([circle_ordered, pts2d[4]]).astype(np.float32)
    print ("ordered points",ordered_pts2d)
    return {
        "ok": True,
        "center_uv": np.mean(circle_ordered, axis=0).astype(np.float32),  # 4-LED centre
        "offset_uv": offset_pt,
        "outer_uv": outer_pts,
        "ordered_pts2d": ordered_pts2d
    }

# ==========================
#  Rotations & Euler (3-2-1)
# ==========================

def Rx(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, ca, -sa],
                     [0, sa,  ca]], dtype=np.float64)

def Ry(b):
    cb, sb = np.cos(b), np.sin(b)
    return np.array([[ cb, 0, sb],
                     [  0, 1,  0],
                     [-sb, 0, cb]], dtype=np.float64)

def Rz(c):
    cc, sc = np.cos(c), np.sin(c)
    return np.array([[cc, -sc, 0],
                     [sc,  cc, 0],
                     [ 0,   0, 1]], dtype=np.float64)

def euler321_from_R(R):
    """Aerospace Z-Y-X (yaw-pitch-roll) → (roll φ, pitch θ, yaw ψ) radians."""
    R = np.asarray(R, dtype=np.float64)
    pitch = np.arcsin(-np.clip(-R[2,0], -1.0, 1.0))
    cp = np.cos(pitch)
    if abs(cp) < 1e-8:
        roll = 0.0
        yaw  = np.arctan2(R[0,1], R[1,1])
    else:
        roll = np.arctan2(-R[2,1], R[2,2])
        yaw  = np.arctan2(-R[1,0], R[0,0])
    return roll, pitch, yaw

# Camera (x right, y down, z forward) -> Aerospace (x forward, y right, z down)
M_cam_to_aero = np.array([[0, 0, 1],
                          [1, 0, 0],
                          [0, 1, 0]], dtype=np.float64)

# =======================================
#  Full ANALYTICAL (α,β,γ, Az, El, Range)
# =======================================

def _safe_asin(x):
    return np.arcsin(np.clip(x, -1.0, 1.0))

def analytic_rpy_range_full(ordered_pts_px, K, R_phys):
    """
    Full closed-form (pixel-space, uses fx/fy), following the C draft:
      - Centre from 4 outer LEDs (pixels)
      - Az = atan((u_c - cx)/fx),  El = atan(-(v_c - cy)/fy)
      - Define relative coords x'_iy = u_i - u_c, x'_iz = v_i - v_c
      - α = atan2(-x'_1z, x'_2z)
      - γ = asin( (-x'_5y / x'_4z) * cos α ) - Az
      - β = asin( (cos(γ+Az) cos α) / (sin(γ+Az) sin α + (x'_3y / x'_5z)) ) - El
      - R = ( (R_phys * fx) / x'_1y ) * (cos α cos(γ+Az) - sin α sin(γ+Az) sin(β+El))
    Returns: rpy321_deg (φ,θ,ψ), range_m, AzEl_deg
    """
    pts = np.asarray(ordered_pts_px, dtype=np.float64).reshape(5,2)
    fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])

    # 4-LED centre (pixels)
    uc = float(np.mean(pts[:4, 0]))
    vc = float(np.mean(pts[:4, 1]))

    # LoS from centre
    Az = np.arctan((uc - cx) / fx)
    El = np.arctan(-(vc - cy) / fy)

    # Relative pixel coords wrt centre
    x_r = np.zeros((5,3), dtype=np.float64)
    x_r[:,1] = pts[:,0] - uc  # y'
    x_r[:,2] = pts[:,1] - vc  # z'

    y1, z1 = x_r[0,1], x_r[0,2]   # Up
    y2, z2 = x_r[1,1], x_r[1,2]   # Left
    y3, z3 = x_r[2,1], x_r[2,2]   # Down
    y4, z4 = x_r[3,1], x_r[3,2]   # Right
    y5, z5 = x_r[4,1], x_r[4,2]   # Offset (near centre)

    # Guard degeneracies
    if abs(z4) < 1e-9 or abs(z5) < 1e-9 or abs(y1) < 1e-9 or (abs(z2) < 1e-12 and abs(z1) < 1e-12):
        raise FloatingPointError("Analytic ill-conditioned (denominator ~ 0).")

    # α
    print ("z1,z2 - ",z1,z2)
    alpha = np.arctan2(-z1, z2)
    ca, sa = np.cos(alpha), np.sin(alpha)

    # γ
    gamma_plus_Az = _safe_asin((-y5 / z4) * ca)
    gamma = gamma_plus_Az - Az

    # β
    num_b = (np.cos(gamma_plus_Az) * ca)
    den_b = (np.sin(gamma_plus_Az) * sa + (y3 / z5))
    beta_plus_El = _safe_asin(num_b / den_b)
    beta = beta_plus_El - El
    print("alpha,beta,gamma - ",np.degrees(alpha) , np.degrees(beta) , np.degrees(gamma))
    # Range
    Df = float(R_phys) * fx  # D * f_x (because y' is in pixels)
    range_R = (Df / y1) * (np.cos(alpha) * np.cos(gamma_plus_Az)
                           - np.sin(alpha) * np.sin(gamma_plus_Az) * np.sin(beta_plus_El))

    # Object->camera rotation (1-2-3) then to aerospace for display
    R_cam = Rz(gamma) @ Ry(beta) @ Rx(alpha)
    # R_aero = M_cam_to_aero @ R_cam @ M_cam_to_aero.T
    # roll, pitch, yaw = euler321_from_R(R_aero)
    rpy_deg = tuple(np.degrees([alpha,beta,gamma]))
    AzEl_deg = (float(np.degrees(Az)), float(np.degrees(El)))

    return {"ok": True, "rpy321_deg": rpy_deg, "range_m": float(range_R), "AzEl_deg": AzEl_deg}

# ===========================
#  Full PnP (relative vectors)
# ===========================

def _model_points_centered(R_phys, d_phys):
    """
    3D model with origin at the centroid of the 4 outer LEDs (target frame).
    All LEDs lie in the Z=0 plane; the 5th LED is offset along +Y (in-plane).
    Ordering: [Right, Up, Left, Down, Offset]
    """
    R = float(R_phys); d = float(d_phys)
    model = np.array([
        [ +R,  0.0, 0.0],  # Right
        [  0.0, +R,  0.0], # Up
        [ -R,  0.0, 0.0],  # Left
        [  0.0, -R,  0.0], # Down
        [  0.0,  0,  -d], # Offset (near centre)
    ], dtype=np.float64)
    # Centroid of the 4 outers is already (0,0,0), so model is centred as-is.
    return model

def pnp_rpy_range_centered(ordered_pts_px, K, distCoeffs, R_phys, d_phys=0.010):
    """
    Full PnP using a model centred at the 4-LED centroid (relative position vectors).
    Returns RPY (aerospace 3-2-1) and metric range = ||tvec||, plus reprojection and Z_cam.
    """
    pts = np.asarray(ordered_pts_px, dtype=np.float64).reshape(5,2)
    model = _model_points_centered(R_phys, d_phys)

    ok, rvec, tvec = cv2.solvePnP(model, pts, K, distCoeffs, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return {"ok": False, "reason": "solvePnP failed"}

    R_cam, _ = cv2.Rodrigues(rvec)
    R_aero = M_cam_to_aero @ R_cam @ M_cam_to_aero.T
    roll, pitch, yaw = euler321_from_R(R_aero)
    rpy_deg = tuple(np.degrees([roll, pitch, yaw]))

    # Metric range is distance from camera centre to the MODEL origin (pattern centre)
    rng = float(np.linalg.norm(tvec))

    # Optional: reprojection + per-point camera Z (for safe drawing)
    pts_cam = (R_cam @ model.T + tvec.reshape(3,1)).T
    z_cam = pts_cam[:, 2].astype(np.float64)
    reproj = np.full((model.shape[0], 2), np.nan, dtype=np.float64)
    valid = np.isfinite(z_cam) & (z_cam > 1e-6)
    if np.any(valid):
        proj_valid, _ = cv2.projectPoints(model[valid], rvec, tvec, K, distCoeffs)
        reproj[valid] = proj_valid.reshape(-1,2)

    # LOS from solved translation (more geometric than centre-pixel LOS)
    # In camera axes: x_right, y_down, z_forward
    tx, ty, tz = tvec.reshape(3)
    Az = float(np.degrees(np.arctan2(tx, tz)))         # yaw
    El = float(np.degrees(np.arctan2(-ty, np.hypot(tx, tz))))  # pitch

    return {"ok": True, "rpy321_deg": rpy_deg, "range_m": rng,
            "AzEl_deg": (Az, El), "reproj": reproj, "z_cam": z_cam}

# ==============
#  Public  API
# ==============

def estimate_pose(ordered_pts_px, K, distCoeffs, R_phys, d_phys=0.010):
    """https://github.com/dangercomix07/VBN_DEMO.git
    Returns:
      {
        "analytic": {"ok": bool, "rpy321_deg": (φ,θ,ψ), "range_m": R, "AzEl_deg": (Az,El)},
        "pnp":      {"ok": bool, "rpy321_deg": (φ,θ,ψ), "range_m": R, "AzEl_deg": (Az,El),
                     "reproj": Nx2, "z_cam": N}
      }
    """
    out = {"analytic": {"ok": False}, "pnp": {"ok": False}}

    # Full analytical (paper-style)
    try:
        out["analytic"] = analytic_rpy_range_full(ordered_pts_px, K, R_phys)
    except Exception as e:
        out["analytic"] = {"ok": False, "reason": str(e)}

    # Full PnP with relative (centred) model
    try:
        out["pnp"] = pnp_rpy_range_centered(ordered_pts_px, K, distCoeffs, R_phys, d_phys=d_phys)
    except Exception as e:
        out["pnp"] = {"ok": False, "reason": str(e)}

    return out
def convert_analytic_to_matrix(out):
    analytic = out["analytic"]

    if not analytic["ok"]:
        return None  # Don't process if failed

    R = analytic["range_m"]
    az_deg, el_deg = analytic["AzEl_deg"]
    roll, pitch, yaw = analytic["rpy321_deg"]  # alpha, beta, gamma

    # Convert degrees to radians
    az = math.radians(az_deg)
    el = math.radians(el_deg)

    # Position
    x = R * math.cos(el) * math.cos(az)
    y = R * math.cos(el) * math.sin(az)
    z = R * math.sin(el)

    # Form the 6×1 matrix
    matrix_6x1 = np.array([[x], [y], [z], [roll], [pitch], [yaw]])

    return readings
