import cv2
import numpy as np

# --------------------------------------------------
# 1. Open Video
# --------------------------------------------------

video_path = "input_video.mp4"   # Change this to your video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read first frame
ret, old_frame = cap.read()

if not ret:
    print("Error: Could not read video.")
    cap.release()
    exit()

# Convert first frame to grayscale
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# --------------------------------------------------
# 2. Parameters for Lucas-Kanade Optical Flow
# --------------------------------------------------

feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)

# Detect feature points
p0 = cv2.goodFeaturesToTrack(
    old_gray,
    mask=None,
    **feature_params
)

# Mask for drawing trajectories
mask = np.zeros_like(old_frame)

# --------------------------------------------------
# 3. Process Video
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------
    # Lucas-Kanade Sparse Optical Flow
    # --------------------------------------------------

    if p0 is not None and len(p0) > 0:

        p1, status, error = cv2.calcOpticalFlowPyrLK(
            old_gray,
            frame_gray,
            p0,
            None,
            **lk_params
        )

        if p1 is not None and status is not None:

            # Select successfully tracked points
            good_new = p1[status == 1]
            good_old = p0[status == 1]

            # Draw trajectories
            for new, old in zip(good_new, good_old):

                x_new, y_new = new.ravel()
                x_old, y_old = old.ravel()

                x_new, y_new = int(x_new), int(y_new)
                x_old, y_old = int(x_old), int(y_old)

                # Draw trajectory
                mask = cv2.line(
                    mask,
                    (x_new, y_new),
                    (x_old, y_old),
                    (0, 255, 0),
                    2
                )

                # Draw current point
                frame = cv2.circle(
                    frame,
                    (x_new, y_new),
                    4,
                    (0, 0, 255),
                    -1
                )

            # Combine frame and trajectories
            sparse_output = cv2.add(frame, mask)

            # Update points
            p0 = good_new.reshape(-1, 1, 2)

        else:
            p0 = None

    else:
        sparse_output = frame.copy()

        # Detect new features if no points remain
        p0 = cv2.goodFeaturesToTrack(
            frame_gray,
            mask=None,
            **feature_params
        )

    # --------------------------------------------------
    # Farneback Dense Optical Flow
    # --------------------------------------------------

    flow = cv2.calcOpticalFlowFarneback(
        old_gray,
        frame_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # Convert flow from Cartesian coordinates
    # to magnitude and direction
    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    # HSV image for visualization
    hsv = np.zeros_like(frame)

    # Saturation
    hsv[..., 1] = 255

    # Direction -> Hue
    hsv[..., 0] = angle * 180 / np.pi / 2

    # Magnitude -> Brightness
    hsv[..., 2] = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # Convert HSV to BGR
    dense_output = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )

    # --------------------------------------------------
    # 4. Display Results
    # --------------------------------------------------

    cv2.imshow(
        "Lucas-Kanade Sparse Optical Flow",
        sparse_output
    )

    cv2.imshow(
        "Farneback Dense Optical Flow",
        dense_output
    )

    # Update previous frame
    old_gray = frame_gray.copy()

    # Press ESC to exit
    key = cv2.waitKey(30) & 0xFF

    if key == 27:
        break

# --------------------------------------------------
# 5. Release Resources
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()