# ============================================================
# IMAGE PROCESSING USING OPENCV, NUMPY AND MATPLOTLIB
# ============================================================

# 1. IMPORT REQUIRED LIBRARIES
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# 2. LOAD AND DISPLAY COLOR IMAGE
# ============================================================

image_path = "image.jpg"

# Check whether image exists
if not os.path.exists(image_path):
    raise FileNotFoundError(
        f"Image not found: {image_path}\n"
        "Please put the image in the same folder as this Python file."
    )

# OpenCV reads image in BGR format
image = cv2.imread(image_path)

if image is None:
    raise ValueError("Could not read the image. Check the file path.")


# Display using OpenCV
cv2.imshow("Original Image - OpenCV", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Display using Matplotlib
# Convert BGR to RGB because Matplotlib uses RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 5))
plt.imshow(image_rgb)
plt.title("Original Image - Matplotlib")
plt.axis("off")
plt.show()


# ============================================================
# 3. EXAMINE IMAGE PROPERTIES
# ============================================================

height, width = image.shape[:2]
channels = image.shape[2]
data_type = image.dtype

print("\n========== IMAGE PROPERTIES ==========")
print("Width       :", width, "pixels")
print("Height      :", height, "pixels")
print("Resolution  :", width, "x", height)
print("Channels    :", channels)
print("Data Type   :", data_type)
print("Shape       :", image.shape)
print("Total Pixels:", width * height)


# ============================================================
# 4. SAVE IMAGE IN JPEG AND PNG FORMATS
# ============================================================

jpeg_path = "output_image.jpg"
png_path = "output_image.png"

cv2.imwrite(jpeg_path, image)
cv2.imwrite(png_path, image)

print("\n========== FILE INFORMATION ==========")
print("JPEG saved as:", jpeg_path)
print("PNG saved as :", png_path)

if os.path.exists(jpeg_path):
    print("JPEG size    :", os.path.getsize(jpeg_path), "bytes")

if os.path.exists(png_path):
    print("PNG size     :", os.path.getsize(png_path), "bytes")


# ============================================================
# 5. COLOR SPACE CONVERSIONS
# ============================================================

# BGR to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# BGR to LAB
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


# ============================================================
# 6. GEOMETRIC TRANSFORMATIONS
# ============================================================

# ---------- Resizing ----------
resized = cv2.resize(image, (600, 400))


# ---------- Rotation ----------
height, width = image.shape[:2]

center = (width // 2, height // 2)

# Rotate image by 45 degrees
rotation_matrix = cv2.getRotationMatrix2D(
    center,
    45,
    1.0
)

rotated = cv2.warpAffine(
    image,
    rotation_matrix,
    (width, height)
)


# ---------- Horizontal Flip ----------
horizontal_flip = cv2.flip(image, 1)


# ---------- Vertical Flip ----------
vertical_flip = cv2.flip(image, 0)


# ============================================================
# 7. NEGATIVE / COMPLEMENT IMAGE
# ============================================================

negative = 255 - image


# ============================================================
# 8. CROP REGION OF INTEREST (ROI)
# ============================================================

# Make sure coordinates are within image dimensions
x1 = 100
y1 = 100
x2 = min(500, width)
y2 = min(400, height)

# Check that ROI is valid
if x1 >= x2 or y1 >= y2:
    raise ValueError(
        "ROI coordinates are outside the image dimensions."
    )

roi = image[y1:y2, x1:x2]

print("\n========== ROI PROPERTIES ==========")
print("ROI Shape       :", roi.shape)
print("ROI Height      :", roi.shape[0])
print("ROI Width       :", roi.shape[1])
print("ROI Channels    :", roi.shape[2])
print("ROI Data Type   :", roi.dtype)
print("ROI Mean Value  :", np.mean(roi))
print("ROI Min Value   :", np.min(roi))
print("ROI Max Value   :", np.max(roi))


# ============================================================
# 9. DISPLAY ORIGINAL AND PROCESSED IMAGES
# ============================================================

plt.figure(figsize=(16, 12))


# Original
plt.subplot(3, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")


# Grayscale
plt.subplot(3, 4, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")


# HSV
plt.subplot(3, 4, 3)
plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
plt.title("HSV")
plt.axis("off")


# LAB
plt.subplot(3, 4, 4)
plt.imshow(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
plt.title("LAB")
plt.axis("off")


# Resized
plt.subplot(3, 4, 5)
plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
plt.title("Resized")
plt.axis("off")


# Rotated
plt.subplot(3, 4, 6)
plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
plt.title("Rotated 45°")
plt.axis("off")


# Horizontal Flip
plt.subplot(3, 4, 7)
plt.imshow(cv2.cvtColor(horizontal_flip, cv2.COLOR_BGR2RGB))
plt.title("Horizontal Flip")
plt.axis("off")


# Vertical Flip
plt.subplot(3, 4, 8)
plt.imshow(cv2.cvtColor(vertical_flip, cv2.COLOR_BGR2RGB))
plt.title("Vertical Flip")
plt.axis("off")


# Negative
plt.subplot(3, 4, 9)
plt.imshow(cv2.cvtColor(negative, cv2.COLOR_BGR2RGB))
plt.title("Negative")
plt.axis("off")


# ROI
plt.subplot(3, 4, 10)
plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
plt.title("ROI")
plt.axis("off")


plt.tight_layout()
plt.show()


# ============================================================
# 10. SAVE PROCESSED IMAGES
# ============================================================

cv2.imwrite("grayscale.jpg", gray)
cv2.imwrite("hsv.jpg", hsv)
cv2.imwrite("lab.jpg", lab)
cv2.imwrite("resized.jpg", resized)
cv2.imwrite("rotated.jpg", rotated)
cv2.imwrite("horizontal_flip.jpg", horizontal_flip)
cv2.imwrite("vertical_flip.jpg", vertical_flip)
cv2.imwrite("negative.jpg", negative)
cv2.imwrite("roi.jpg", roi)

print("\n========== PROCESSING COMPLETE ==========")
print("All processed images have been saved successfully.")
