"""Download version 2 of the Kaggle chest X-ray dataset used in this homework."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import kagglehub


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=".kaggle-cache")
    parser.add_argument("--link", default="data/chest_xray")
    parser.add_argument("--version", type=int, default=2)
    args = parser.parse_args()
    if args.version < 1:
        parser.error("--version must be positive")

    cache_dir = Path(args.cache_dir).resolve()
    os.environ["KAGGLEHUB_CACHE"] = str(cache_dir)
    version_root = (cache_dir / "datasets" / "paultimothymooney" / "chest-xray-pneumonia" / "versions")
    cached_version = version_root / str(args.version)
    downloaded = (cached_version if (cached_version / "chest_xray/train").is_dir() else Path( kagglehub.dataset_download( f"paultimothymooney/chest-xray-pneumonia/versions/{args.version}" ) ))
    source = downloaded / "chest_xray"
    if not (source / "train").is_dir():
        raise FileNotFoundError(f"Unexpected Kaggle dataset layout under {downloaded}")

    link = Path(args.link)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if link.resolve() != source.resolve():
            raise FileExistsError(f"Refusing to replace existing path: {link}")
    else:
        link.symlink_to(source)
    print(f"Dataset: {source}")
    print(f"Linked:  {link} -> {link.resolve()}")


if __name__ == "__main__":
    main()
