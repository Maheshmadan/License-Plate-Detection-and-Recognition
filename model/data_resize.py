from pathlib import Path
import pandas as pd
from PIL import Image


def yolo_data(path: str):
    data = pd.read_csv(path)
    data = data.copy()
    data["center_x"] = (data["xmin"] + data["xmax"]) / 2
    data["center_y"] = (data["ymin"] + data["ymax"]) / 2
    data["width"] = data["xmax"] - data["xmin"]
    data["height"] = data["ymax"] - data["ymin"]
    return data


def create_yolo_labels(csv_path: str, image_dir: str, output_dir: str, class_id: int = 0):
    csv_path = Path(csv_path)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = yolo_data(str(csv_path))

    for image_path in sorted(image_dir.glob("*.jpg")):
        image_name = image_path.stem
        image_annotations = data[data["img_id"] == f"{image_name}.jpg"]

        if image_annotations.empty:
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        label_lines = []
        for _, row in image_annotations.iterrows():
            x_min = float(row["xmin"])
            y_min = float(row["ymin"])
            x_max = float(row["xmax"])
            y_max = float(row["ymax"])

            center_x = (x_min + x_max) / 2 / width
            center_y = (y_min + y_max) / 2 / height
            bbox_width = (x_max - x_min) / width
            bbox_height = (y_max - y_min) / height

            label_lines.append(
                f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}"
            )

        label_path = output_dir / f"{image_name}.txt"
        label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    return output_dir