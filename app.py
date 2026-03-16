import os
import cv2
import numpy as np
import gradio as gr
import torch

from config import (
    FIXED_INVERT_EGG, FIXED_DBSCAN_EPS, FIXED_DBSCAN_MIN_SAMPLES,
    FIXED_FALLBACK_A_THRESH, FIXED_OVERLAP_THRESHOLD, FIXED_IOU_THRESHOLD,
    TARGET_SIZE, DILATE_KERNEL_SIZE, BLUR_KSIZE, BLUR_SIGMA,
    CNN_WEIGHTS_PATH
)
from models.cnn_classifier import load_cnn_model, get_device, MAPPING
from utils.mask_ops import get_masks_and_classes
from utils.visualization import draw_masks_by_class, masks_to_28x28
from utils.preprocessing import clahe_and_blur, increase_contrast_sharpness, remove_shadows

device = get_device()
cnn = load_cnn_model(CNN_WEIGHTS_PATH)

def process_ui(input_img, classify_letter):
    if input_img is None:
        dummy = np.zeros((100,100,3), dtype=np.uint8)
        return [dummy]*4, dummy, "Загрузите изображение", "", dummy

    img_bgr = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

    orig_rgb = input_img
    test2_rgb = cv2.cvtColor(clahe_and_blur(img_bgr.copy()), cv2.COLOR_BGR2RGB)
    contrast_rgb = cv2.cvtColor(increase_contrast_sharpness(img_bgr.copy()), cv2.COLOR_BGR2RGB)
    shadow_rgb = cv2.cvtColor(remove_shadows(img_bgr.copy(), strength=1.0, shadow_threshold=0.1)[0], cv2.COLOR_BGR2RGB)
    gallery_images = [orig_rgb, test2_rgb, contrast_rgb, shadow_rgb]

    masks, classes, _ = get_masks_and_classes(
        img_bgr,
        invert_egg=FIXED_INVERT_EGG,
        dbscan_eps=FIXED_DBSCAN_EPS,
        dbscan_min_samples=FIXED_DBSCAN_MIN_SAMPLES,
        fallback_a_thresh=FIXED_FALLBACK_A_THRESH,
        overlap_threshold=FIXED_OVERLAP_THRESHOLD,
        iou_threshold=FIXED_IOU_THRESHOLD
    )

    if len(masks) == 0:
        mask_28 = np.zeros(TARGET_SIZE, dtype=np.float32)
        mask_28_disp = (mask_28 * 255).astype(np.uint8)
        return gallery_images, orig_rgb, "Объектов не обнаружено.", "", mask_28_disp

    result_rgb = draw_masks_by_class(orig_rgb, masks, classes, alpha=0.4)

    num_eggs = classes.count(0)
    num_red = classes.count(1)
    num_yellow = classes.count(2)
    stats = f"Яиц: {num_eggs}, Красных томатов: {num_red}, Жёлтых томатов: {num_yellow}"

    mask_28 = masks_to_28x28(masks, target_size=TARGET_SIZE,
                              dilate_kernel_size=DILATE_KERNEL_SIZE,
                              blur_ksize=BLUR_KSIZE, blur_sigma=BLUR_SIGMA)
    mask_28_disp = (mask_28 * 255).astype(np.uint8)

    letter_result = ""
    if classify_letter:
        input_tensor = torch.from_numpy(mask_28).unsqueeze(0).unsqueeze(0).float()
        input_tensor = input_tensor.transpose(-2, -1).to(device)
        with torch.no_grad():
            output = cnn(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()
            raw_logit = output[0, pred_idx].item()

        if pred_idx <= 9:
            letter_result = f"Цифра {MAPPING[pred_idx]} (буква не обнаружена)"
        else:
            letter_result = f"Предполагаемая буква: {MAPPING[pred_idx]}"

        letter_result += f"\nУверенность: {confidence:.4f} | raw logit: {raw_logit:.4f}"
        if raw_logit < 5:
            letter_result += "\n⚠️ НИЗКАЯ УВЕРЕННОСТЬ!"

    return gallery_images, result_rgb, stats, letter_result, mask_28_disp

with gr.Blocks(title="Детекция помидоров и яиц + классификация букв") as demo:
    gr.Markdown("# Сегментация и классификация помидоров и яиц + классификация букв из масок")
    gr.Markdown("Загрузите изображение. Параметры обработки зафиксированы.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Входное изображение", type="numpy")
            classify_letter = gr.Checkbox(label="Классифицировать букву", value=False)
            submit_btn = gr.Button("Обработать")
        with gr.Column():
            gallery = gr.Gallery(label="Промежуточные преобразования", columns=2, rows=2, object_fit="contain", height="auto")
            output_image = gr.Image(label="Результат")

    with gr.Row():
        with gr.Column():
            output_stats = gr.Textbox(label="Статистика объектов", lines=2)
        with gr.Column():
            output_letter = gr.Textbox(label="Результат классификации буквы", lines=4)

    with gr.Row():
        output_28 = gr.Image(label="Маска 28x28 для CNN", height=200, width=200)

    submit_btn.click(
        fn=process_ui,
        inputs=[input_image, classify_letter],
        outputs=[gallery, output_image, output_stats, output_letter, output_28]
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch()