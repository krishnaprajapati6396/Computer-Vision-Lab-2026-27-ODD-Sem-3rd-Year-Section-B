import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# STEP 1: Import required libraries and load a grayscale image
# ==============================================================================
image_path = "colourful.jpeg"

if not os.path.exists(image_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "colourful.jpeg")

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    # Synthetic fallback if image is missing
    img = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(img, (60, 60), (240, 240), 200, -1)
    cv2.circle(img, (150, 150), 45, 50, -1)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2  # Center coordinates

# ==============================================================================
# STEP 2: Compute Discrete Fourier Transform (DFT) using NumPy
# ==============================================================================
dft = np.fft.fft2(img.astype(np.float32))

# ==============================================================================
# STEP 3: Shift zero-frequency component to center of frequency spectrum
# ==============================================================================
dft_shifted = np.fft.fftshift(dft)

# ==============================================================================
# STEP 4: Display and analyze magnitude spectrum
# ==============================================================================
magnitude_spectrum = 20 * np.log(np.abs(dft_shifted) + 1e-5)

# ==============================================================================
# STEP 5: Design and apply an Ideal/Gaussian Low-Pass Filter (LPF)
# ==============================================================================
# Create coordinate grid
u = np.arange(rows)
v = np.arange(cols)
u, v = np.meshgrid(u, v, indexing='ij')
D = np.sqrt((u - crow)**2 + (v - ccol)**2)

# Ideal Low-Pass Filter Mask (cutoff radius D0)
D0_lpf = 40
lpf_mask = np.zeros((rows, cols), dtype=np.float32)
lpf_mask[D <= D0_lpf] = 1.0

# Apply LPF mask in frequency domain
fshift_lpf = dft_shifted * lpf_mask

# ==============================================================================
# STEP 6: Design and apply an Ideal High-Pass Filter (HPF)
# ==============================================================================
D0_hpf = 20
hpf_mask = np.ones((rows, cols), dtype=np.float32)
hpf_mask[D <= D0_hpf] = 0.0

# Apply HPF mask in frequency domain
fshift_hpf = dft_shifted * hpf_mask

# ==============================================================================
# STEP 7: Perform Inverse Fourier Transform (IDFT) to reconstruct spatial images
# ==============================================================================
# Inverse shift and inverse 2D FFT for LPF
f_ishift_lpf = np.fft.ifftshift(fshift_lpf)
img_lpf = np.fft.ifft2(f_ishift_lpf)
img_lpf = np.abs(img_lpf)

# Inverse shift and inverse 2D FFT for HPF
f_ishift_hpf = np.fft.ifftshift(fshift_hpf)
img_hpf = np.fft.ifft2(f_ishift_hpf)
img_hpf = np.abs(img_hpf)

# ==============================================================================
# STEP 8 & 9: Plot and compare outputs
# ==============================================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

titles = [
    "Step 1: Original Image",
    "Step 4: Magnitude Spectrum",
    "Step 5: LPF Mask (Radius=40)",
    "Step 5 & 7: LPF Reconstructed",
    "Step 6: HPF Mask (Radius=20)",
    "Step 6 & 7: HPF Reconstructed (Edges)"
]

displays = [
    img,
    magnitude_spectrum,
    lpf_mask,
    img_lpf,
    hpf_mask,
    img_hpf
]

for ax, title, data in zip(axes.ravel(), titles, displays):
    ax.imshow(data, cmap="gray")
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.tight_layout()

# Save image file to allow viewing in Code Runner / non-GUI environments
output_file = "exp4_frequency_filtering_output.png"
plt.savefig(output_file, dpi=300)
print(f"Result successfully saved as '{output_file}'.")

plt.show(block=True)

# ==============================================================================
# STEP 10: Observations & Analytical Documentation
# ==============================================================================
print("\n--- Step 10: Analytical Observations ---")
print("1. Low-Pass Filtering: Attenuates outer high frequencies, resulting in image smoothing and noise reduction.")
print("2. High-Pass Filtering: Blocks central DC/low-frequency energy, isolating sharp gradients, edges, and fine details.")
print("3. Convolution Theorem: Multiplication in frequency domain corresponds to 2D spatial convolution, reducing complexity.")