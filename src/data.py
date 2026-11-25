from pathlib import Path
from typing import List, Tuple
from PIL import Image

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_split_dir(split_dir: Path, label_to_idx: dict) -> Tuple[List[str], List[int]]:
    paths: List[str] = []
    labels: List[int] = []

    for class_name, label_idx in label_to_idx.items():
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.rglob("*")):
            if img_path.suffix.lower() in VALID_EXTS:
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                    paths.append(str(img_path))
                    labels.append(label_idx)
                except Exception:
                    continue

    return paths, labels


def load_image_dataset(
    data_dir: str,
) -> Tuple[List[str], List[str], List[int], List[int], List[str]]:
    
    root = Path(data_dir)
    train_dir = root / "train"
    test_dir = root / "test"

    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Expected 'train' and 'test' subdirectories inside {data_dir}"
        )

    class_names = sorted(
        {
            d.name
            for d in list(train_dir.iterdir()) + list(test_dir.iterdir())
            if d.is_dir()
        }
    )
    if len(class_names) < 2:
        raise ValueError(f"Expected at least 2 classes, found {class_names}")

    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    train_paths, train_labels = _load_split_dir(train_dir, label_to_idx)
    test_paths, test_labels = _load_split_dir(test_dir, label_to_idx)

    return train_paths, test_paths, train_labels, test_labels, class_names