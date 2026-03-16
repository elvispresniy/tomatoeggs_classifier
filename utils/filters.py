import cv2
import numpy as np

def filter_masks(gray: np.ndarray, mask: np.ndarray) -> bool:
    imgsize = np.prod(mask.shape)
    masksize = mask.sum()
    if masksize >= imgsize * 0.025 or masksize <= imgsize * 0.003:
        return False

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False
    contour = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    area = cv2.contourArea(hull)
    if perimeter == 0:
        return False

    tau = ((4 * np.pi * area) / (perimeter * perimeter) + (area / (perimeter * perimeter)) * (2 + np.pi) ** 2 / np.pi) / 2
    tau = (tau - 0.4) / 0.4
    lambda_ = (gray * mask).sum() / mask.sum() / 400
    lambda_ *= 2

    asd = (1 - tau) * 0.75 + 0.2
    if lambda_ < asd or tau < 0.76:
        return False
    return True

def get_internal_gradient_variance(gray_img, mask):
    gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    internal_grad_values = magnitude[eroded_mask > 0]
    grad_variance = np.std(internal_grad_values)
    return grad_variance

def is_egg(original: np.ndarray, mask: np.ndarray) -> bool:
    lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    pixels_lab = lab[mask > 0]
    pixels_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)[mask > 0]

    raw_grad_std = get_internal_gradient_variance(original[:, :, 2], mask)
    mean_bright = np.mean(pixels_gray) + 1e-5
    tau = raw_grad_std / mean_bright / 20

    mean_a = np.mean(pixels_lab[:, 1])
    lambda_ = mean_a / 255.0

    if tau > 0.1:
        return True
    if tau + lambda_ < 0.625:
        return False
    return True