import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from .filters import filter_masks, is_egg
from .preprocessing import clahe_and_blur, increase_contrast_sharpness, remove_shadows
from models.fastsam_loader import get_fastsam_model

def merge_masks(base_masks, new_masks, overlap_threshold=0.1):
    if len(base_masks) == 0:
        return new_masks.copy()
    if len(new_masks) == 0:
        return base_masks.copy()
    h, w = base_masks[0].shape
    base_overlay = np.zeros((h, w), dtype=np.uint8)
    for mask in base_masks:
        base_overlay[(mask > 0.5)] = 1
    merged = list(base_masks)
    for mask in new_masks:
        mask_area = np.sum(mask > 0.5)
        if mask_area == 0:
            continue
        intersection = np.sum((mask > 0.5) & (base_overlay > 0))
        if intersection / mask_area <= overlap_threshold:
            merged.append(mask)
            base_overlay[(mask > 0.5)] = 1
    return np.array(merged)

def remove_near_duplicates(masks, overlap_threshold=0.8):
    if len(masks) <= 1:
        return masks
    areas = np.sum(masks > 0.5, axis=(1, 2))
    sorted_indices = np.argsort(areas)[::-1]
    selected_indices = []
    for i in sorted_indices:
        mask_i = masks[i] > 0.5
        area_i = areas[i]
        if area_i == 0:
            continue
        duplicate = False
        for j in selected_indices:
            mask_j = masks[j] > 0.5
            area_j = areas[j]
            intersection = np.sum(mask_i & mask_j)
            if intersection / area_i > overlap_threshold or intersection / area_j > overlap_threshold:
                duplicate = True
                break
        if not duplicate:
            selected_indices.append(i)
    return masks[selected_indices]

def get_masks_and_classes(
    img_bgr,
    invert_egg=True,
    dbscan_eps=5,
    dbscan_min_samples=2,
    fallback_a_thresh=20,
    overlap_threshold=0.1,
    iou_threshold=0.8
):
    model = get_fastsam_model()

    def get_masks_single(img):
        gray = np.max(img, axis=2).astype(np.uint8)
        results = model(img, retina_masks=True, conf=0.3, verbose=False, imgsz=1024, max_det=1000)
        if results[0].masks is None:
            return np.array([])
        masks = results[0].masks.data.cpu().numpy()
        valid = []
        for m in masks:
            mask_bin = (m > 0.5).astype(np.uint8)
            if filter_masks(gray, mask_bin):
                valid.append(m)
        return np.array(valid)

    masks_orig = get_masks_single(img_bgr)
    masks_test2 = get_masks_single(clahe_and_blur(img_bgr.copy()))
    masks_contrast = get_masks_single(increase_contrast_sharpness(img_bgr.copy()))
    masks_shadow = get_masks_single(remove_shadows(img_bgr.copy(), strength=1.0, shadow_threshold=0.1)[0])

    base_masks = masks_orig.copy()
    base_masks = merge_masks(base_masks, masks_test2, overlap_threshold=overlap_threshold)
    base_masks = merge_masks(base_masks, masks_contrast, overlap_threshold=overlap_threshold)
    base_masks = merge_masks(base_masks, masks_shadow, overlap_threshold=overlap_threshold)

    base_masks = remove_near_duplicates(base_masks, iou_threshold)

    if len(base_masks) == 0:
        return base_masks, [], []

    classes = []
    tomato_indices = []
    tomato_features = []

    for idx, mask in enumerate(base_masks):
        egg = is_egg(img_bgr, mask)
        if invert_egg:
            egg = not egg
        if egg:
            classes.append(0)
        else:
            classes.append(-1)
            tomato_indices.append(idx)
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            pixels_lab = lab[mask > 0]
            mean_a = np.mean(pixels_lab[:, 1])
            mean_b = np.mean(pixels_lab[:, 2])
            tomato_features.append([mean_a, mean_b])

    if len(tomato_features) >= 2:
        X = np.array(tomato_features)
        clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(X)
        labels = clustering.labels_
        unique_labels = set(labels)
        valid_labels = [l for l in unique_labels if l != -1]

        if len(valid_labels) == 0:
            for i, idx in enumerate(tomato_indices):
                if tomato_features[i][0] > fallback_a_thresh:
                    classes[idx] = 1
                else:
                    classes[idx] = 2
        elif len(valid_labels) == 1:
            cluster = valid_labels[0]
            cluster_points = [tomato_features[i] for i in range(len(tomato_features)) if labels[i] == cluster]
            mean_a_cluster = np.mean([p[0] for p in cluster_points])
            for i, idx in enumerate(tomato_indices):
                if labels[i] == cluster:
                    if mean_a_cluster > fallback_a_thresh:
                        classes[idx] = 1
                    else:
                        classes[idx] = 2
                else:
                    if tomato_features[i][0] > fallback_a_thresh:
                        classes[idx] = 1
                    else:
                        classes[idx] = 2
        else:
            cluster_mean_b = {}
            cluster_points_dict = {}
            for l in valid_labels:
                points_idx = [i for i in range(len(tomato_features)) if labels[i] == l]
                cluster_points_dict[l] = points_idx
                mean_b = np.mean([tomato_features[i][1] for i in points_idx])
                cluster_mean_b[l] = mean_b

            sorted_clusters = sorted(cluster_mean_b.items(), key=lambda x: x[1], reverse=True)
            yellow_cluster = sorted_clusters[0][0]
            red_cluster = sorted_clusters[-1][0]

            red_points = [tomato_features[i] for i in cluster_points_dict[red_cluster]]
            yellow_points = [tomato_features[i] for i in cluster_points_dict[yellow_cluster]]

            for i, idx in enumerate(tomato_indices):
                if labels[i] == red_cluster:
                    classes[idx] = 1
                elif labels[i] == yellow_cluster:
                    classes[idx] = 2
                else:
                    dist_red = min([np.linalg.norm(np.array(tomato_features[i]) - np.array(p)) for p in red_points]) if red_points else float('inf')
                    dist_yellow = min([np.linalg.norm(np.array(tomato_features[i]) - np.array(p)) for p in yellow_points]) if yellow_points else float('inf')
                    if dist_red < dist_yellow:
                        classes[idx] = 1
                    else:
                        classes[idx] = 2
    else:
        for i, idx in enumerate(tomato_indices):
            if tomato_features[i][0] > fallback_a_thresh:
                classes[idx] = 1
            else:
                classes[idx] = 2

    return base_masks, classes, tomato_features