import io
import base64

import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torchvision.transforms.functional as F

from ultralytics import YOLO
from PIL import Image
import numpy as np

# -------------------------
# DEVICE & CONSTANTS
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FASTFOOD_CLASSES = {
    "pizza",
    "hot dog",
    "sandwich",   # burger
    "donut",
    "cake",
    "apple",
    "banana",
    "orange",
    "broccoli",
    "carrot",
}

CALORIES = {
    "pizza": 250,
    "sandwich": 400,
    "hot dog": 300,
    "donut": 200,
    "cake": 350,
    "apple": 95,
    "banana": 105,
    "orange": 62,
    "broccoli": 55,
    "carrot": 25,
}

YOLO_CONF_THRES = 0.35
MASK_CONF_THRES = 0.5

# -------------------------
# LOAD MODELS (at import)
# -------------------------
print("[diet_analyzer] Loading models on", device)

# YOLO – object detection
yolo_model = YOLO("yolov8n.pt")  # or "yolo11n.pt" depending on your weights file

# Mask R-CNN – instance segmentation
mask_weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
mask_model = maskrcnn_resnet50_fpn(weights=mask_weights).to(device)
mask_model.eval()
mask_categories = mask_weights.meta.get("categories", None)

# ResNet18 – CNN classifier
cnn_weights = ResNet18_Weights.DEFAULT
cnn_model = resnet18(weights=cnn_weights).to(device)
cnn_model.eval()
cnn_preprocess = cnn_weights.transforms()
cnn_categories = cnn_weights.meta.get("categories", None)

print("[diet_analyzer] Models loaded.")


# -------------------------
# CORE HELPERS
# -------------------------
def run_yolo(image_pil):
    """Run YOLO on a PIL image and return filtered detections."""
    results = yolo_model(image_pil)[0]
    if len(results.boxes) == 0:
        return []

    boxes_xyxy = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    scores = results.boxes.conf.cpu().numpy()

    detections = []
    for box, cls_id, score in zip(boxes_xyxy, classes, scores):
        label = results.names[int(cls_id)]
        if score < YOLO_CONF_THRES:
            continue
        if label not in FASTFOOD_CLASSES:
            continue
        detections.append({
            "label": label,
            "score": float(score),
            "box": box,  # [x1, y1, x2, y2]
        })
    return detections


def run_mask_rcnn(image_pil):
    """Run Mask R-CNN and return segmentation results for our food classes."""
    img_t = mask_weights.transforms()(image_pil).to(device)
    with torch.no_grad():
        output = mask_model([img_t])[0]

    boxes = output["boxes"].cpu()
    labels = output["labels"].cpu()
    scores = output["scores"].cpu()
    masks = output["masks"].cpu()

    seg = []
    for i in range(len(boxes)):
        if scores[i] < MASK_CONF_THRES:
            continue

        label_idx = int(labels[i].item())
        if mask_categories is not None and 0 < label_idx <= len(mask_categories):
            label_str = mask_categories[label_idx - 1]
        else:
            label_str = str(label_idx)

        if label_str not in FASTFOOD_CLASSES:
            continue

        seg.append({
            "label": label_str,
            "score": float(scores[i].item()),
            "box": boxes[i].numpy(),
            "mask": (masks[i, 0] > 0.5),
        })
    return seg


def classify_crop_with_resnet(crop_pil):
    """Classify a cropped food patch using ResNet18 (ImageNet labels)."""
    t = cnn_preprocess(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = cnn_model(t)
        probs = torch.softmax(logits, dim=1)[0]
        top_prob, top_idx = torch.topk(probs, 1)

    label = cnn_categories[top_idx.item()] if cnn_categories else str(int(top_idx.item()))
    return label, float(top_prob.item())


def estimate_calories(detections, img_width, img_height):
    """
    Estimate calories using:
      - PIZZA: relative area + aspect ratio
        * only 'roundish' boxes (aspect ~1) can be full/half pizza
        * long/triangular boxes are always treated as 1 slice
      - OTHER FOODS: small/medium/large factor based on relative area
    """
    total_cal = 0.0
    per_item = []

    img_area = img_width * img_height

    for det in detections:
        label = det["label"]
        if label not in CALORIES:
            continue

        base_cal = CALORIES[label]
        x1, y1, x2, y2 = det["box"]

        w_box = max((x2 - x1), 1)
        h_box = max((y2 - y1), 1)
        box_area = w_box * h_box
        rel_area = box_area / img_area
        aspect = w_box / (h_box + 1e-6)  # >1 = wide, <1 = tall

        # ---------------- PIZZA ----------------
        if label == "pizza":
            # Only almost-square boxes (round pizza) can be treated
            # as full / half pizzas. Long triangular boxes = slices.
            is_roundish = 0.9 <= aspect <= 1.1

            if is_roundish:
                # thresholds for a round pizza in the frame
                if rel_area > 0.45:
                    slices = 8      # full pizza
                elif rel_area > 0.25:
                    slices = 4      # ~half pizza
                elif rel_area > 0.12:
                    slices = 2      # 2–3 slices
                else:
                    slices = 1      # small piece / far away
            else:
                # Non-round (triangular / long) area => force ONE SLICE
                slices = 1

            cal = base_cal * slices
            size_desc = f"~{slices} slice(s)"
            factor = slices

        # ---------------- OTHER FOODS ----------------
        else:
            if rel_area < 0.03:
                size_desc = "small"
                factor = 0.5
            elif rel_area < 0.07:
                size_desc = "medium"
                factor = 1.0
            else:
                size_desc = "large"
                factor = 1.5

            cal = base_cal * factor

        total_cal += cal

        per_item.append({
            "label": label,
            "size": size_desc,
            "base_cal": base_cal,
            "factor": factor,
            "calories": cal,
            "score": det["score"],
            "rel_area": rel_area,
            "aspect": aspect,
        })

        # Uncomment while tuning:
        # print(f"[DEBUG] {label}: rel_area={rel_area:.3f}, aspect={aspect:.2f}, size={size_desc}, cal={cal:.1f}")

    return per_item, total_cal


def pil_to_base64(pil_img):
    """Convert PIL image to base64 string (for HTML embedding)."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# -------------------------
# MAIN PUBLIC API
# -------------------------
def analyze_image(pil_img, return_images=True):
    """
    Main function to call, like predict_tumor() in NeuroScan.
    Input:  PIL.Image (RGB)
    Output: dict with
        - yolo_dets
        - mask_dets
        - per_item
        - total_cal
        - boxed_b64  (if return_images)
        - seg_b64    (if return_images)
    """
    img_w, img_h = pil_img.size
    img_tensor_uint8 = (F.to_tensor(pil_img) * 255.0).to(torch.uint8)

    # YOLO detections
    yolo_dets = run_yolo(pil_img)

    # CNN classification on each detected crop
    for det in yolo_dets:
        x1, y1, x2, y2 = det["box"]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        crop = pil_img.crop((x1, y1, x2, y2))
        cls_label, cls_prob = classify_crop_with_resnet(crop)
        det["cnn_label"] = cls_label
        det["cnn_conf"] = cls_prob

    # Calorie estimation
    per_item, total_cal = estimate_calories(yolo_dets, img_w, img_h)

    # Optional visualizations
    boxed_b64 = None
    seg_b64 = None

    if return_images:
        # YOLO bounding boxes
        if yolo_dets:
            boxes = []
            labels = []
            for det in yolo_dets:
                x1, y1, x2, y2 = det["box"]
                boxes.append([x1, y1, x2, y2])
                labels.append(f"{det['label']} {det['score']:.2f}")
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            boxed_img = draw_bounding_boxes(
                img_tensor_uint8,
                boxes=boxes_tensor,
                labels=labels,
                width=2
            )
            boxed_pil = F.to_pil_image(boxed_img)
            boxed_b64 = pil_to_base64(boxed_pil)

        # Mask R-CNN segmentations
        mask_dets = run_mask_rcnn(pil_img)
        if mask_dets:
            masks = [det["mask"] for det in mask_dets]
            masks_stack = torch.stack(masks, dim=0)
            seg_img = draw_segmentation_masks(
                img_tensor_uint8.clone(),
                masks=masks_stack,
                alpha=0.5
            )
            seg_pil = F.to_pil_image(seg_img)
            seg_b64 = pil_to_base64(seg_pil)
        else:
            mask_dets = []
    else:
        # still compute mask_dets if caller cares
        mask_dets = run_mask_rcnn(pil_img)

    return {
        "yolo_dets": yolo_dets,
        "mask_dets": mask_dets,
        "per_item": per_item,
        "total_cal": total_cal,
        "boxed_b64": boxed_b64,
        "seg_b64": seg_b64,
    }


# -------------------------
# CLI TEST (run from terminal / VS Code)
# -------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python diet_analyzer.py path/to/image.jpg")
        sys.exit(1)

    path = sys.argv[1]
    img = Image.open(path).convert("RGB")
    res = analyze_image(img, return_images=False)

    print("Detected items:")
    for item in res["per_item"]:
        print(f"- {item['label']} ({item['size']}): {item['calories']:.1f} kcal")

    print("Total calories:", res["total_cal"])
