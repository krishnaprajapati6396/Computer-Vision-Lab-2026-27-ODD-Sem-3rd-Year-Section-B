import os
import urllib.request
import cv2
import numpy as np

# -------------------------------------------------------------------------
# Step 1: Load video sequence (downloading car-detection benchmark)
# -------------------------------------------------------------------------
VIDEO_PATH = "input_video.mp4"


if not os.path.exists(VIDEO_PATH):
    print("Downloading car-detection.mp4...")
    urllib.request.urlretrieve(VIDEO_URL, VIDEO_PATH)
    print("Download finished.")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

ret, first_frame = cap.read()
if not ret:
    cap.release()
    raise RuntimeError("Failed to read video.")

# -------------------------------------------------------------------------
# Step 2: Grayscale conversion & Shi-Tomasi Feature Corner Detection
# -------------------------------------------------------------------------
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
feature_params = dict(maxCorners=150, qualityLevel=0.3, minDistance=7, blockSize=7)
p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# Parameters for Step 3: Lucas-Kanade Sparse Flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

# Canvases for Step 4 & Step 6
trajectory_canvas = np.zeros_like(first_frame)
hsv_canvas = np.zeros_like(first_frame)
hsv_canvas[..., 1] = 255  # Fixed saturation

# Tracking metrics accumulators for Step 5 & Step 8
cumulative_distance = 0.0
frame_count = 0
PANEL_W, PANEL_H = 480, 270

print("Running full optical flow pipeline. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lk_vis = frame.copy()

    # Step 2 refresh: Re-detect feature corners if points drop
    if p0 is None or len(p0) < 15:
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    # ---------------------------------------------------------------------
    # Step 3, 4, 5, 8: Lucas-Kanade Flow, Trajectory, Direction & Speed
    # ---------------------------------------------------------------------
    current_speeds = []
    current_angles = []
    active_points_count = 0

    if p0 is not None:
        p1, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, frame_gray, p0, None, **lk_params
        )

        if p1 is not None and status is not None:
            good_new = p1[status == 1]
            good_old = p0[status == 1]
            active_points_count = len(good_new)

            for new_pt, old_pt in zip(good_new, good_old):
                x_new, y_new = new_pt.ravel()
                x_old, y_old = old_pt.ravel()

                dx = float(x_new - x_old)
                dy = float(y_new - y_old)

                # Step 5: Magnitude (displacement) and direction (angle)
                displacement = float(np.hypot(dx, dy))
                angle_deg = float(np.degrees(np.arctan2(dy, dx))) % 360.0

                current_speeds.append(displacement)
                current_angles.append(angle_deg)

                # Step 4: Draw persistent motion trajectory paths & vectors
                trajectory_canvas = cv2.line(
                    trajectory_canvas,
                    (int(x_new), int(y_new)),
                    (int(x_old), int(y_old)),
                    (0, 255, 0),
                    2,
                )
                lk_vis = cv2.circle(
                    lk_vis, (int(x_new), int(y_new)), 4, (0, 0, 255), -1
                )

            p0 = good_new.reshape(-1, 1, 2)

    # Overlay trajectory lines on LK frame
    lk_vis = cv2.add(lk_vis, trajectory_canvas)

    # Step 8: Calculate instantaneous and cumulative motion metrics
    frame_avg_speed = float(np.mean(current_speeds)) if current_speeds else 0.0
    frame_avg_angle = float(np.mean(current_angles)) if current_angles else 0.0
    cumulative_distance += frame_avg_speed
    real_time_sec = frame_count / fps

    # ---------------------------------------------------------------------
    # Step 5 & 6: Gunnar Farnebäck Dense Optical Flow (HSV Visualization)
    # ---------------------------------------------------------------------
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        frame_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    flow_mag, flow_ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv_canvas[..., 0] = flow_ang * 180 / np.pi / 2
    hsv_canvas[..., 2] = cv2.normalize(flow_mag, None, 0, 255, cv2.NORM_MINMAX)
    dense_vis = cv2.cvtColor(hsv_canvas, cv2.COLOR_HSV2BGR)

    prev_gray = frame_gray.copy()

    # ---------------------------------------------------------------------
    # Visualization & Live Telemetry Dashboard
    # ---------------------------------------------------------------------
    p1_view = cv2.resize(frame, (PANEL_W, PANEL_H))
    p2_view = cv2.resize(lk_vis, (PANEL_W, PANEL_H))
    p3_view = cv2.resize(dense_vis, (PANEL_W, PANEL_H))

    # Panel Titles
    cv2.putText(p1_view, "1. Input Stream", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(p2_view, "2. Lucas-Kanade (Sparse)", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(p3_view, "3. Farneback (Dense HSV)", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # Step 5 & Step 8 HUD Telemetry (Points, Speed, Direction, Trajectory Displacement)
    hud_bg = p2_view.copy()
    cv2.rectangle(hud_bg, (10, PANEL_H - 105), (280, PANEL_H - 10), (0, 0, 0), -1)
    p2_view = cv2.addWeighted(hud_bg, 0.6, p2_view, 0.4, 0)

    cv2.putText(p2_view, f"Tracked Points: {active_points_count}", (15, PANEL_H - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(p2_view, f"Speed: {frame_avg_speed:.2f} px/frame", (15, PANEL_H - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(p2_view, f"Direction: {frame_avg_angle:.1f} deg", (15, PANEL_H - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(p2_view, f"Total Displ: {cumulative_distance:.1f} px", (15, PANEL_H - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)

    # Combine into a horizontal dashboard
    dashboard = np.hstack([p1_view, p2_view, p3_view])
    cv2.imshow("Optical Flow Tracking & Motion Analysis", dashboard)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()