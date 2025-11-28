# main.py — VBN measurement producer (no HUD, no streaming)
import os
import time
import cv2
import numpy as np
import socket, json, tempfile

from capture_frame import FrameSource
from led_detection_man import detect_leds_python
from pose_estimation import identify_and_order, estimate_pose
from display import draw_raw_points, draw_pattern
import platform

# IPC paths
SOCKET_PATH = "/tmp/kf_socket"
FRAME_PATH = "/tmp/vbn_last.jpg"

# ---------- Pattern geometry (meters) ----------
R = 0.010  # radius from center to each of the 4 outer LEDs

# ---------- Camera intrinsics & distortion ----------
camMatrix = np.array([[908.62425565  , 0, 643.93436085],
                    [0, 908.92570486, 393.45889572],
                    [0, 0, 1.0]], dtype=np.float64)

distCoeff = np.array([ 2.18984921e-01, -5.80493965e-01, 1.15200278e-04,
                      -2.04177566e-03, 4.48611005e-01], dtype=np.float64)

def _los_from_center(center_uv, K):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    u, v = float(center_uv[0]), float(center_uv[1])
    x_n = (u - cx) / fx
    y_n = (v - cy) / fy
    az  = np.degrees(np.arctan2(x_n, 1.0))
    el  = np.degrees(np.arctan2(-y_n, 1.0))
    return az, el

def main():
    cap = FrameSource()

    # UNIX DGRAM socket -> KF
    _meas_client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)

    t0 = time.time(); frames = 0
    priors = { "lock": False, "range": 0, "centre": [], "area_const": 3236.41, "length_const": 8600 }

    try:
        while True:
            ok, frame, frame_colour = cap.read()
            if not ok or frame is None:
                print("\n[capture] no frame.")
                time.sleep(0.02)
                continue

            frames += 1
            now = time.time()
            fps = frames / max(1e-6, (now - t0))

            # detect bright LEDs (max 5)
            pts = detect_leds_python(frame, priors, 20, 5, max_pts=5)
            pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2) if pts is not None else np.empty((0,2))

            vis = frame_colour.copy()
            draw_raw_points(vis, pts)  # optional debugging dots

            console_msg = ""

            if len(pts) == 5:
                res = identify_and_order(pts)
                if res["ok"]:
                    C        = res["center_uv"]
                    off      = res["offset_uv"]
                    outs     = res["outer_uv"]
                    ordered  = res["ordered_pts2d"]
                    priors["centre"] = C
                    priors["lock"] = True

                    draw_pattern(vis, C, off, outs)

                    # estimate pose
                    try:
                        pose = estimate_pose(ordered, camMatrix, distCoeff, R)
                    except Exception as e:
                        pose = {"analytic": {"ok": False, "reason": str(e)}, "pnp": {"ok": False}}

                    analytic = pose.get("analytic", {})
                    if analytic.get("ok", False):

                        Rm = analytic["range_m"]
                        az_deg, el_deg = analytic["AzEl_deg"]
                        roll_deg, pitch_deg, yaw_deg = analytic["rpy321_deg"]

                        az = np.deg2rad(az_deg)
                        el = np.deg2rad(el_deg)

                        x = Rm * np.cos(el) * np.cos(az)
                        y = Rm * np.cos(el) * np.sin(az)
                        z = Rm * np.sin(el)

                        z6 = [float(x), float(y), float(z),
                              float(roll_deg), float(pitch_deg), float(yaw_deg)]

                        pkt = json.dumps({"meas": z6, "ts": time.time()})
                        try:
                            _meas_client.sendto(pkt.encode(), SOCKET_PATH)
                        except FileNotFoundError:
                            pass

                    console_msg = "Pattern OK; LEDs: " + str(len(pts))
                else:
                    console_msg = res.get("reason", "pattern id failed")
                    priors["lock"] = False

            else:
                console_msg = f"Need 5 LEDs (got {len(pts)})"
                priors["lock"] = False

            # encode frame for KF
            ok_jpg, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok_jpg:
                try:
                    tmpf = FRAME_PATH + ".tmp"
                    with open(tmpf, "wb") as f:
                        f.write(jpg.tobytes())
                    os.replace(tmpf, FRAME_PATH)
                except Exception:
                    pass

            # console output
            print(f"\rFPS: {fps:5.1f} | {console_msg:>s}   ", end="", flush=True)

    except KeyboardInterrupt:
        print("\n[run] interrupted")
    finally:
        cap.release()
        try: cv2.destroyAllWindows()
        except: pass
        print("\n[run] shutdown complete.")

if __name__ == "__main__":
    main()
