"""
Download the Amazon ML Challenge 2026 dataset from Kaggle and place it
into the project's dataset/ folder structure.

WHY: kagglehub's Pandas adapter loads one file at a time and requires
knowing its exact internal path upfront. We need multiple files (3
train sources + ground truth + 3 test sources), so it's more reliable
to download the full dataset once, inspect it, then read everything
locally with pandas 
"""

import shutil
from pathlib import Path

import kagglehub

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"


def download_dataset() -> Path:
    # WHY: dataset_download caches locally and returns the extracted path —
    # avoids re-downloading on every run
    path = kagglehub.dataset_download("gowthamdd/amazon-ml-challenge-2026")
    return Path(path)


def inspect_dataset(path: Path) -> None:
    # WHY: confirm actual folder/file names before writing any copy logic —
    # Kaggle dataset layouts vary and shouldn't be assumed
    print(f"Downloaded to: {path}\n")
    for item in sorted(path.rglob("*")):
        print(item.relative_to(path))


def copy_to_project(cache_path: Path) -> None:
    # WHY: kagglehub's cache lives outside the project (~/.cache/), which is
    # inconvenient to work with and invisible to PyCharm's project tree —
    # copy into dataset/ so every later script can use one fixed local path
    for split in ("train", "test"):
        src_dir = cache_path / split
        dst_dir = DATASET_DIR / split
        dst_dir.mkdir(parents=True, exist_ok=True)

        for file in src_dir.glob("*.tsv"):
            shutil.copy2(file, dst_dir / file.name)
            print(f"Copied {file.name} -> {dst_dir}")


def main():
    raw_path = download_dataset()
    inspect_dataset(raw_path)
    copy_to_project(raw_path)


if __name__ == "__main__":
    main()