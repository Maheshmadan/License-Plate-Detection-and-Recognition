import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_resize import create_yolo_labels
from ultralytics import YOLO


project_root = Path(__file__).resolve().parents[1]
image_folder = project_root / "model" / "data" / "license_plates_detection_train"
csv_path = project_root / "data" / "Licplatesdetection_train.csv"

train_images_dir = project_root / "model" / "data" / "images" / "train"
val_images_dir = project_root / "model" / "data" / "images" / "val"
train_labels_dir = project_root / "model" / "data" / "labels" / "train"
val_labels_dir = project_root / "model" / "data" / "labels" / "val"


def prepare_yolo_dataset():
    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    create_yolo_labels(
        csv_path=str(csv_path),
        image_dir=str(image_folder),
        output_dir=project_root / "model" / "data" / "labels",
    )

    images = sorted(image_folder.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images found in {image_folder}")

    random.seed(42)
    random.shuffle(images)
    val_size = max(1, int(len(images) * 0.2))
    val_images = images[:val_size]
    train_images = images[val_size:]

    for image_path in train_images:
        shutil.copy2(image_path, train_images_dir / image_path.name)
        label_path = project_root / "model" / "data" / "labels" / f"{image_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, train_labels_dir / label_path.name)

    for image_path in val_images:
        shutil.copy2(image_path, val_images_dir / image_path.name)
        label_path = project_root / "model" / "data" / "labels" / f"{image_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, val_labels_dir / label_path.name)

    data_yaml_path = project_root / "data.yaml"
    data_yaml_path.write_text(
        "path: model/data\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: license_plate\n",
        encoding="utf-8",
    )

    print(f"Prepared {len(train_images)} training images and {len(val_images)} validation images")
    print(f"YOLO config written to {data_yaml_path}")


def _metric_value(metric):
    if metric is None:
        return 0.0
    if hasattr(metric, "__len__") and not isinstance(metric, (str, bytes)):
        return float(metric[0]) if len(metric) else 0.0
    return float(metric)


def evaluate_yolo_model(model, data_yaml: Path, imgsz: int = 640):
    print("Evaluating model on the validation split...")
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, split="val", stream=False)

    evaluation = {
        "precision": round(_metric_value(metrics.box.p), 4),
        "recall": round(_metric_value(metrics.box.r), 4),
        "mAP50": round(_metric_value(metrics.box.map50), 4),
        "mAP50_95": round(_metric_value(metrics.box.map), 4),
    }

    print("Evaluation metrics:")
    for name, value in evaluation.items():
        print(f"  {name}: {value}")

    metrics_file = project_root / "runs" / "detect" / "train" / "evaluation_metrics.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"Saved evaluation metrics to {metrics_file}")
    return evaluation


def train_yolo_model(epochs: int = 5, imgsz: int = 640):
    data_yaml = project_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"YOLO config not found at {data_yaml}")

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        project=str(project_root / "runs"),
        name="detect/train",
        exist_ok=True,
    )
    print("Training complete")
    evaluation = evaluate_yolo_model(model, data_yaml, imgsz=imgsz)
    return {"training_results": results, "evaluation": evaluation}


if __name__ == "__main__":
    prepare_yolo_dataset()
    train_yolo_model()