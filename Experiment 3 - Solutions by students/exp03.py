import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load your image file
image_path = "colourful.jpeg"

# Fallback check: look in the current script directory if not found in root
if not os.path.exists(image_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "colourful.jpeg")

# Read as grayscale for clear filter and edge analysis
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(
        f"Could not find '{image_path}'. Make sure 'colourful.jpeg' is placed inside 'd:\\ss\\EXPIREMENT-03\\'"
    )

# 2. Generate Salt-and-Pepper Noise on the image
noisy_img = img.copy()
num_noise_pixels = int(0.05 * img.size * 0.5)

# Add Salt (white pixels)
coords = [np.random.randint(0, i, num_noise_pixels) for i in img.shape]
noisy_img[tuple(coords)] = 255

# Add Pepper (black pixels)
coords = [np.random.randint(0, i, num_noise_pixels) for i in img.shape]
noisy_img[tuple(coords)] = 0

# 3. Apply Low-Pass Filters (Smoothing & Denoising)
mean_filtered = cv2.blur(noisy_img, (5, 5))
gaussian_filtered = cv2.GaussianBlur(noisy_img, (5, 5), sigmaX=1.5)
median_filtered = cv2.medianBlur(noisy_img, 5)

# 4. Apply High-Pass Filters (Edge Detection)
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y))
laplacian = cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_64F, ksize=3))

# 5. Plot the 8 steps into a single grid
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

titles = [
    "Original Grayscale", "Noisy (Salt & Pepper)", "Mean Filter (5x5)", "Gaussian Filter (5x5)",
    "Median Filter (5x5)", "Sobel X (Vertical Edges)", "Sobel Magnitude (All Edges)", "Laplacian Filter"
]
images = [
    img, noisy_img, mean_filtered, gaussian_filtered,
    median_filtered, cv2.convertScaleAbs(sobel_x), sobel_combined, laplacian
]

for ax, title, image in zip(axes.ravel(), titles, images):
    ax.imshow(image, cmap="gray")
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.tight_layout()

# Save the final result to disk so you can open it immediately
output_path = "output_result.png"
plt.savefig(output_path, dpi=300)
print(f"Success! Output image saved as: {output_path}")

# Display the window if running from an interactive terminal
plt.show(block=True)




