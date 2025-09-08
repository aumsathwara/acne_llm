import subprocess
from pathlib import Path


def main() -> None:
    """Clone the public acne segmentation dataset with masks.

    The dataset is sourced from Merligus/acne-segmentation, which provides
    JPEG images and corresponding binary masks under the ``data`` directory.
    If the dataset already exists locally this function is a no-op.  The
    repository uses Git LFS for the image assets, so ``git lfs`` must be
    installed before cloning in order to fetch the mask and image files.
    """
    target = Path("Datasets") / "acne_segmentation"
    if target.exists():
        print(f"Dataset already present at {target}")
        return

    url = "https://github.com/Merligus/acne-segmentation.git"
    subprocess.run(["git", "lfs", "install"], check=True)
    subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)


if __name__ == "__main__":
    main()
