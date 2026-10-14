import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

# ==========================================================
# 1. SETUP & MODEL LOADING
# ==========================================================
def load_model(model_name="yolov8n.pt"):
    """
    Loads a pre-trained YOLO model from Ultralytics.
    Available options: yolov8n.pt (Nano - fast), yolov8s.pt, yolov8m.pt
    """
    print(f"[INFO] Loading pre-trained {model_name} model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running inference on device: {device.upper()}")
    
    # Load model pre-trained on MS COCO dataset (80 classes)
    model = YOLO(model_name)
    return model

# ==========================================================
# 2. IMAGE PREPARATION & INFERENCE
# ==========================================================
def run_object_detection(model, image_source, conf_threshold=0.25):
    """
    Runs object detection inference on an input image.
    """
    print(f"[INFO] Running object detection on: {image_source}")
    
    # Measure inference execution time
    start_time = time.time()
    results = model.predict(source=image_source, conf=conf_threshold, save=False)
    end_time = time.time()
    
    inference_speed = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"[INFO] Inference completed in {inference_speed:.2f} ms")
    
    return results[0], inference_speed

# ==========================================================
# 3. PROCESSING & VISUALIZATION
# ==========================================================
def analyze_and_display_results(result, image_source):
    """
    Extracts bounding boxes, labels, confidence scores, and renders the result.
    """
    # Extract original image and detection data
    img = cv2.imread(image_source) if isinstance(image_source, str) else image_source
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    boxes = result.boxes
    print("\n" + "="*50)
    print(f"DETLECTION RESULTS SUMMARY (Total Objects Detected: {len(boxes)})")
    print("="*50)
    
    for idx, box in enumerate(boxes):
        # Extract coordinates, confidence, and class
        coords = box.xyxy[0].cpu().numpy().astype(int)  # [xmin, ymin, xmax, ymax]
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        class_name = result.names[cls_id]
        
        print(f"Object {idx+1}: Class='{class_name}', Confidence={conf:.2f}, Box={coords.tolist()}")
    
    print("="*50 + "\n")
    
    # Render annotated image with bounding boxes
    annotated_frame = result.plot()  # Draws bounding boxes, labels, and confidence scores
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    # Display results using Matplotlib
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Input Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(annotated_frame_rgb)
    plt.title(f"YOLOv8 Detection ({len(boxes)} Objects Detected)")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

# ==========================================================
# 4. MAIN EXECUTION PIPELINE
# ==========================================================
if __name__ == "__main__":
    # Standard public sample image from MS COCO repository
    sample_image_url = "https://ultralytics.com/images/bus.jpg"
    
    # Step 1: Load pre-trained object detection model
    yolo_model = load_model(model_name="yolov8n.pt")
    
    # Step 2: Run inference
    detection_result, speed = run_object_detection(yolo_model, sample_image_url, conf_threshold=0.3)
    
    # Step 3: Parse metrics and display output visualization
    analyze_and_display_results(detection_result, sample_image_url)