"""
Create a tiny HAM10000 subset ZIP for the Vercel web app.

Colab usage:
    pip install kagglehub pandas pillow tqdm
    python tools/create_subset.py --data-dir auto --per-class 8 --out ham10000_small_subset.zip

Kaggle usage after adding dataset skin-cancer-mnist-ham10000:
    python tools/create_subset.py --data-dir /kaggle/input/skin-cancer-mnist-ham10000 --per-class 8 --out ham10000_small_subset.zip
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def find_dataset(data_dir: str) -> Path:
    if data_dir != "auto":
        return Path(data_dir)
    try:
        import kagglehub  # type: ignore
        path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
        return Path(path)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not download HAM10000 with kagglehub. In Colab, upload kaggle.json or set KAGGLE_USERNAME/KAGGLE_KEY."
        ) from exc


def find_metadata(root: Path) -> Path:
    candidates = list(root.rglob("HAM10000_metadata.csv"))
    if not candidates:
        candidates = list(root.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No metadata CSV found under {root}")
    return candidates[0]


def build_image_index(root: Path) -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for path in root.rglob(ext):
            image_index[path.stem] = path
    return image_index


def create_subset(data_dir: str, per_class: int, out_path: str, seed: int) -> None:
    root = find_dataset(data_dir)
    metadata_path = find_metadata(root)
    df = pd.read_csv(metadata_path)
    required = {"lesion_id", "image_id", "dx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")

    image_index = build_image_index(root)
    rng = random.Random(seed)
    selected_frames = []
    for cls in CLASSES:
        cls_df = df[df["dx"] == cls].copy()
        cls_df = cls_df[cls_df["image_id"].map(lambda image_id: str(image_id) in image_index)]
        rows = cls_df.to_dict("records")
        rng.shuffle(rows)
        selected_frames.append(pd.DataFrame(rows[:per_class]))

    subset = pd.concat(selected_frames, ignore_index=True)
    subset = subset.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    temp = Path("_ham10000_small_subset")
    if temp.exists():
        shutil.rmtree(temp)
    (temp / "images").mkdir(parents=True)
    subset.to_csv(temp / "HAM10000_metadata_subset.csv", index=False)

    for row in tqdm(subset.to_dict("records"), desc="Copying/resizing"):
        image_id = str(row["image_id"])
        src = image_index[image_id]
        dst = temp / "images" / f"{image_id}.jpg"
        with Image.open(src).convert("RGB") as img:
            img = img.resize((64, 64))
            img.save(dst, quality=92)

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in temp.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(temp))
    shutil.rmtree(temp)
    print(f"Saved {out_path} with {len(subset)} images from {root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="auto", help="auto uses kagglehub; or pass Kaggle dataset folder")
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("--out", default="ham10000_small_subset.zip")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_subset(args.data_dir, args.per_class, args.out, args.seed)


if __name__ == "__main__":
    main()
