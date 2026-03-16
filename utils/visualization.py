import cv2
import numpy as np

def draw_masks_by_class(image: np.ndarray, masks: np.ndarray, classes, alpha=0.5):
    overlay = image.copy()
    color_map = {
        0: (0, 0, 255),      # синий
        1: (255, 0, 0),      # красный
        2: (255, 255, 0)     # жёлтый
    }
    for i, mask in enumerate(masks):
        c = classes[i]
        color = color_map[c]
        mask_bin = (mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
        color_layer = np.zeros_like(image)
        color_layer[mask_bin > 0] = color
        overlay = cv2.addWeighted(overlay, 1, color_layer, alpha, 0)
    return overlay

def masks_to_28x28(masks, target_size=(28,28), dilate_kernel_size=3, blur_ksize=(3,3), blur_sigma=0.8):
    if len(masks) == 0:
        return np.zeros(target_size, dtype=np.float32)

    h, w = masks[0].shape
    combined = np.zeros((h, w), dtype=np.uint8)
    for mask in masks:
        mask_bin = (mask > 0.5).astype(np.uint8)
        combined = cv2.bitwise_or(combined, mask_bin * 255)

    resized = cv2.resize(combined, target_size, interpolation=cv2.INTER_NEAREST)

    kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    dilated = cv2.dilate(resized, kernel, iterations=1)

    blurred = cv2.GaussianBlur(dilated, blur_ksize, blur_sigma)

    normalized = blurred.astype(np.float32) / 255.0
    return normalized