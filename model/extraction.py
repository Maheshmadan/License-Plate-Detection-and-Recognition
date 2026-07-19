import zipfile
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data"
extract_folder = Path(__file__).parent / "data"

extract_folder.mkdir(exist_ok=True)

archive_paths = [
    data_dir / "Licplatesrecognition_train.zip",
    data_dir / "Licplatesdetection_train.zip",
]

for archive_path in archive_paths:
    if archive_path.exists() and archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(extract_folder)

