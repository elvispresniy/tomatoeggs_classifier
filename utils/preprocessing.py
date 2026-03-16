import cv2
import numpy as np

def increase_contrast_sharpness(img_bgr, clip_limit=2.0, tile_size=(8,8), sharpness_strength=1.0):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_contrast = clahe.apply(l)
    if sharpness_strength > 0:
        blurred = cv2.GaussianBlur(l_contrast, (0, 0), sigmaX=1.0)
        l_sharp = cv2.addWeighted(l_contrast, 1.0 + sharpness_strength, blurred, -sharpness_strength, 0)
        l_final = np.clip(l_sharp, 0, 255).astype(np.uint8)
    else:
        l_final = l_contrast
    lab_result = cv2.merge((l_final, a, b))
    return cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)

def clahe_and_blur(img_bgr):
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5,5))
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    l_blur = cv2.medianBlur(l, 3)
    l = cv2.addWeighted(l, 2.5, l_blur, -1.5, 0)
    l = clahe.apply(l)
    l = cv2.GaussianBlur(l, (5, 5), 1)
    l = clahe.apply(l)
    l = cv2.medianBlur(l, 3)
    l = clahe.apply(l)
    img_res = cv2.merge((l, a, b))
    return cv2.cvtColor(img_res, cv2.COLOR_LAB2BGR)

def remove_shadows(img_bgr, strength=1.0, shadow_threshold=0.1, mask_blur=21):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_float = l.astype(np.float32)
    scales = [15, 51, 151]
    retinex = np.zeros_like(l_float)
    for scale in scales:
        kernel_size = scale if scale % 2 == 1 else scale + 1
        blur = cv2.GaussianBlur(l_float, (kernel_size, kernel_size), 0)
        retinex += np.log(l_float + 1) - np.log(blur + 1)
    retinex /= len(scales)
    retinex_norm = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    l_norm = l.astype(np.float32) / 255.0
    shadow_cond = (l_norm < shadow_threshold) & (s < 0.5)
    shadow_mask = shadow_cond.astype(np.float32)
    if mask_blur % 2 == 0:
        mask_blur += 1
    shadow_mask = cv2.GaussianBlur(shadow_mask, (mask_blur, mask_blur), 0)

    l_corrected = (1 - strength * shadow_mask) * l_float + (strength * shadow_mask) * retinex_norm
    l_corrected = np.clip(l_corrected, 0, 255).astype(np.uint8)
    mask_inv = 1 - shadow_mask
    a_bg = np.sum(a * mask_inv) / (np.sum(mask_inv) + 1e-6)
    b_bg = np.sum(b * mask_inv) / (np.sum(mask_inv) + 1e-6)
    a_corrected = (1 - strength * shadow_mask) * a + (strength * shadow_mask) * a_bg
    b_corrected = (1 - strength * shadow_mask) * b + (strength * shadow_mask) * b_bg
    a_corrected = np.clip(a_corrected, 0, 255).astype(np.uint8)
    b_corrected = np.clip(b_corrected, 0, 255).astype(np.uint8)
    lab_corrected = cv2.merge([l_corrected, a_corrected, b_corrected])
    img_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
    return img_corrected, (shadow_mask * 255).astype(np.uint8)