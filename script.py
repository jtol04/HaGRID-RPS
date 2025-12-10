import json
from pathlib import Path
import cv2
import numpy as np

HAGRID_ROOT = "/home/jt3577/hagrid_data"
OUTPUT_DIR  = "/home/jt3577/hagrid_data/output"
SPLIT = "train"


def load_annotations(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp(v):
    return max(0.0, min(1.0, v))


def resize_with_padding_cv(img, size=224, pad_color=(0, 0, 0)):
    h, w = img.shape[:2]
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), pad_color, dtype=np.uint8)

    left = (size - new_w) // 2
    top = (size - new_h) // 2

    out[top:top + new_h, left:left + new_w] = resized
    return out


def crop_from_annotations(images_root: Path, annotations_root: Path, output_root: Path, split: str):
    target_gestures = {"fist", "no_gesture", "peace", "stop", "stop_inverted", "peace_inverted"}

    ann_dir = annotations_root / split
    json_files = [f for f in sorted(ann_dir.glob("*.json")) if f.stem in target_gestures]

    for ann_file in json_files:
        gesture = ann_file.stem
        gesture_img_dir = images_root / gesture
        out_dir = output_root / split / gesture
        out_dir.mkdir(parents=True, exist_ok=True)

        ann = load_annotations(ann_file)

        for image_id, rec in ann.items():
            img_path = gesture_img_dir / f"{image_id}.jpg"
            if not img_path.exists():
                img_path = gesture_img_dir / f"{image_id}.png"
                if not img_path.exists():
                    print(f"img not found {img_path}")
                    continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            H, W = img.shape[:2]
            bboxes = rec.get("bboxes", [])
            labels = rec.get("labels", [])

            for idx, bbox in enumerate(bboxes):
                if len(bbox) != 4:
                    continue

                x, y, w, h = (clamp(v) for v in bbox)
                left = int(x * W)
                top = int(y * H)
                right = int((x + w) * W)
                bottom = int((y + h) * H)

                crop = img[top:bottom, left:right]
                if crop.size == 0:
                    continue

                crop = resize_with_padding_cv(crop, size=224)

                label = labels[idx] if idx < len(labels) else gesture
                out_name = f"{image_id}_{idx}_{label}.jpg"
                cv2.imwrite(str(out_dir / out_name), crop)


def main():
    root = Path(HAGRID_ROOT)
    images_root = root
    annotations_root = root / "annotations"
    output_root = Path(OUTPUT_DIR)

    print("Train")
    crop_from_annotations(images_root, annotations_root, output_root, "train")
    print("Test")
    crop_from_annotations(images_root, annotations_root, output_root, "test")
    print("Val")
    crop_from_annotations(images_root, annotations_root, output_root, "val")


if __name__ == "__main__":
    main()
