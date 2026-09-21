import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def compute_metrics(original, enhanced):
    """Calculates contrast, dynamic range, PSNR, and SSIM metrics."""
    c_rms = enhanced.std()
    p_val = psnr(original, enhanced, data_range=255)
    s_val = ssim(original, enhanced, data_range=255)
    return c_rms, p_val, s_val

# ---------------------------------------------------------
# Step 1 & 2: Load Image and Convert to Grayscale
# ---------------------------------------------------------
image_path = 'image.jpg'
original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Synthetic fallback if no image file is found
if original_img is None:
    np.random.seed(42)
    base = np.linspace(90, 140, 300, dtype=np.uint8)
    original_img = np.tile(base, (300, 1))
    noise = np.random.normal(0, 5, (300, 300)).astype(np.int16)
    original_img = np.clip(original_img.astype(np.int16) + noise, 80, 150).astype(np.uint8)

# ---------------------------------------------------------
# Step 4: Robust Contrast Stretching (2nd - 98th Percentile)
# ---------------------------------------------------------
# Percentile clipping prevents single-pixel noise from degrading the stretch
p2, p98 = np.percentile(original_img, (2, 98))
stretched_img = np.clip((original_img - p2) / (p98 - p2 + 1e-5) * 255.0, 0, 255).astype(np.uint8)

# ---------------------------------------------------------
# Step 5: Global Histogram Equalization (GHE)
# ---------------------------------------------------------
ghe_img = cv2.equalizeHist(original_img)

# ---------------------------------------------------------
# Step 6: CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ---------------------------------------------------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(original_img)

# ---------------------------------------------------------
# Step 7 & 8: Compute Quantitative Metrics & CDFs
# ---------------------------------------------------------
images = {
    'Original (Low Contrast)': original_img,
    'Contrast Stretched (2-98%)': stretched_img,
    'Global HE': ghe_img,
    'CLAHE (clip=2.0, grid=8x8)': clahe_img
}

results = {}
for name, img in images.items():
    rms, p, s = compute_metrics(original_img, img)
    results[name] = {'RMS Contrast': rms, 'PSNR (dB)': p, 'SSIM': s}

# ---------------------------------------------------------
# Step 7 & 8: High-Resolution Multi-Panel Visualizations
# ---------------------------------------------------------
fig, axes = plt.subplots(len(images), 2, figsize=(13, 14), constrained_layout=True)

for idx, (title, img) in enumerate(images.items()):
    # Image Display
    axes[idx, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[idx, 0].set_title(title, fontsize=11, fontweight='bold')
    axes[idx, 0].axis('off')
    
    # Histogram Calculation
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / (cdf.max() + 1e-5)
    
    # Histogram & CDF Dual Plot
    ax_hist = axes[idx, 1]
    ax_hist.bar(range(256), hist, color='#2b5c8f', width=1.0, alpha=0.7, label='Histogram')
    ax_hist.plot(cdf_normalized, color='#d95f02', linewidth=1.5, label='Normalized CDF')
    ax_hist.set_xlim([0, 256])
    ax_hist.set_ylabel('Pixel Count', fontsize=9)
    ax_hist.grid(True, linestyle=':', alpha=0.6)
    
    if idx == 0:
        ax_hist.legend(loc='upper left', fontsize=8)
    if idx == len(images) - 1:
        ax_hist.set_xlabel('Intensity Value (0–255)', fontsize=10)

plt.show()

# ---------------------------------------------------------
# Step 9 & 10: Print Quantitative Comparison Table
# ---------------------------------------------------------
print(f"\n{'Technique':<30} | {'RMS Contrast':<12} | {'PSNR (dB)':<10} | {'SSIM':<6}")
print("-" * 65)
for name, m in results.items():
    print(f"{name:<30} | {m['RMS Contrast']:<12.2f} | {m['PSNR (dB)']:<10.2f} | {m['SSIM']:<6.4f}")