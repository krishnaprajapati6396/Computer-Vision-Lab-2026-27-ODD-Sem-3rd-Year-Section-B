import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# ==============================================================================
# STEP 1 & 2: Load real-world image, convert to grayscale, and preprocess
# ==============================================================================
image_path = "colourful.jpeg"

if not os.path.exists(image_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "colourful.jpeg")

img_bgr = cv2.imread(image_path)

if img_bgr is None:
    # Synthetic fallback if image is missing
    img_bgr = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img_bgr, (60, 60), (240, 240), (200, 150, 50), -1)
    cv2.circle(img_bgr, (150, 150), 50, (50, 50, 200), -1)

# Convert to grayscale
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# ==============================================================================
# STEP 3 & 4: SIFT Feature Extraction & Keypoint Visualization
# ==============================================================================
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img_gray, None)

# Draw rich keypoints (indicating scale and orientation)
img_sift_keypoints = cv2.drawKeypoints(
    img_gray, 
    keypoints, 
    None, 
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# ==============================================================================
# STEP 5 & 6: HOG Feature Extraction & Gradient Visualization
# ==============================================================================
# Extract HOG features and gradient visualization using scikit-image
hog_features, hog_image = hog(
    img_gray,
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    visualize=True,
    channel_axis=None
)

# Rescale intensity for clearer gradient visualization
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# ==============================================================================
# STEP 7 & 8: Image Matching using SIFT Features
# ==============================================================================
# Create a transformed secondary image (rotated and scaled) to test invariance
rows, cols = img_gray.shape
M_transform = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle=35, scale=0.85)
img_transformed = cv2.warpAffine(img_gray, M_transform, (cols, rows))

# Detect SIFT keypoints on the second image
kp2, des2 = sift.detectAndCompute(img_transformed, None)

# FLANN-based descriptor matcher with Lowe's ratio test
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(descriptors, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw top 30 matching keypoint pairs
img_matches = cv2.drawMatches(
    img_gray, keypoints,
    img_transformed, kp2,
    good_matches[:30], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# ==============================================================================
# STEP 9: Visualize and Save Outputs
# ==============================================================================
plt.figure(figsize=(15, 10))

# Display Original Image
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
plt.title("Step 1: Original Image", fontsize=11)
plt.axis("off")

# Display SIFT Keypoints
plt.subplot(2, 2, 2)
plt.imshow(img_sift_keypoints)
plt.title(f"Step 4: SIFT Keypoints (Count: {len(keypoints)})", fontsize=11)
plt.axis("off")

# Display HOG Orientations
plt.subplot(2, 2, 3)
plt.imshow(hog_image_rescaled, cmap="gray")
plt.title(f"Step 6: HOG Gradient Features ({len(hog_features)} dims)", fontsize=11)
plt.axis("off")

# Display SIFT Matching
plt.subplot(2, 2, 4)
plt.imshow(img_matches)
plt.title(f"Step 8: SIFT Keypoint Matches ({len(good_matches)} valid matches)", fontsize=11)
plt.axis("off")

plt.tight_layout()

output_file = "exp5_sift_hog_output.png"
plt.savefig(output_file, dpi=300)
print(f"Output successfully saved as '{output_file}'.")

plt.show(block=True)

# ==============================================================================
# STEP 10: Analytical Documentation & Observations
# ==============================================================================
print("\n--- Step 10: Analytical Observations ---")
print("1. SIFT: Sparse local feature detector. Extremely robust to geometric rotation, scale, and affine skew.")
print("2. HOG: Dense global structure descriptor. Captures edge directions and silhouettes, ideal for pedestrian/object shapes.")
print("3. Feature Matching: SIFT reliably correlates corresponding points across scaled and rotated variations.")