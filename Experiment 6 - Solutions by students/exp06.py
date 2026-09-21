import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# STEP 1: Import libraries and load a real-world image
# ==============================================================================
image_path = "colourful.jpeg"

if not os.path.exists(image_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "colourful.jpeg")

img_bgr = cv2.imread(image_path)

if img_bgr is None:
    # Synthetic fallback with overlapping circular regions for watershed demonstration
    img_bgr = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img_bgr, (120, 150), 60, (220, 220, 220), -1)
    cv2.circle(img_bgr, (180, 150), 60, (220, 220, 220), -1)

# Convert to RGB for proper Matplotlib rendering
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ==============================================================================
# STEP 2: Grayscale conversion and Gaussian Blur preprocessing
# ==============================================================================
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

# ==============================================================================
# STEP 3: Global Thresholding
# ==============================================================================
_, thresh_global = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

# ==============================================================================
# STEP 4: Otsu's Thresholding
# ==============================================================================
otsu_val, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# ==============================================================================
# STEP 5: Adaptive Thresholding (Mean-C)
# ==============================================================================
thresh_adaptive = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
)

# ==============================================================================
# STEP 6: Watershed Segmentation for overlapping/touching objects
# ==============================================================================
# 6a. Binary thresholding for morphological operations
_, thresh_w = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 6b. Morphological opening to remove stray background noise
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh_w, cv2.MORPH_OPEN, kernel, iterations=2)

# 6c. Definite background identification using dilation
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# 6d. Definite foreground identification via Euclidean Distance Transform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# 6e. Unknown/boundary region calculation
unknown = cv2.subtract(sure_bg, sure_fg)

# 6f. Marker labeling
num_labels, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1          # Background gets 1 instead of 0
markers[unknown == 255] = 0    # Unknown boundary markers marked with 0

# 6g. Apply watershed algorithm and mark boundary lines in red
watershed_img = img_rgb.copy()
cv2.watershed(watershed_img, markers)
watershed_img[markers == -1] = [255, 0, 0]  # Red boundary lines

# ==============================================================================
# STEP 7: K-Means Clustering for Color-Based Segmentation
# ==============================================================================
pixel_values = img_rgb.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Termination criteria: Stop after 100 iterations or when epsilon reaches 0.2
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 4  # Number of color clusters
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Reconstruct quantized segmented image
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
kmeans_segmented = segmented_data.reshape(img_rgb.shape)

# ==============================================================================
# STEP 8 & 9: Visual Comparison & Output Plotting
# ==============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

titles = [
    "Step 1: Original Image",
    "Step 3: Global Thresholding (T=127)",
    f"Step 4: Otsu's Thresholding (T={int(otsu_val)})",
    "Step 5: Adaptive Thresholding",
    "Step 6: Watershed Segmentation",
    f"Step 7: K-Means Clustering (K={k})"
]

displays = [
    img_rgb,
    thresh_global,
    thresh_otsu,
    thresh_adaptive,
    watershed_img,
    kmeans_segmented
]

cmaps = [None, "gray", "gray", "gray", None, None]

for ax, title, data, cmap in zip(axes.ravel(), titles, displays, cmaps):
    if cmap:
        ax.imshow(data, cmap=cmap)
    else:
        ax.imshow(data)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.tight_layout()

output_file = "exp6_segmentation_output.png"
plt.savefig(output_file, dpi=300)
print(f"Segmentation comparison saved successfully as '{output_file}'.")

plt.show(block=True)

# ==============================================================================
# STEP 10: Observations & Analytical Documentation
# ==============================================================================
print("\n--- Step 10: Analytical Observations ---")
print("1. Global vs Otsu: Global relies on hardcoded thresholds; Otsu computes the bimodal variance optimum automatically.")
print("2. Adaptive Thresholding: Calculates thresholds per local window, handling uneven shadows and gradients.")
print("3. Watershed: Uses topographic distance gradients to identify boundaries of touching/overlapping objects.")
print("4. K-Means: Groups color distributions in multi-dimensional space, preserving multi-class regions without binary constraints.")