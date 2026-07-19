from pathlib import Path

from ultralytics import YOLO


project_root = Path(__file__).resolve().parents[1]
model_path = project_root / "runs" / "detect" / "train" / "weights" / "best.pt"
test_images_dir = project_root / "data" / "test_images"
output_dir = project_root / "runs" / "detect" / "predict"

def run_prediction(model_path: Path = model_path, test_images_dir: Path = test_images_dir):
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Train the model first.")

    test_images_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    results = model(test_images_dir, save=True, project=str(output_dir.parent), name=output_dir.name, exist_ok=True)
    print(f"Predictions saved to {output_dir}")
    return results


if __name__ == "__main__":
    run_prediction()
