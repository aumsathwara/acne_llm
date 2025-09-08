import argparse
from pathlib import Path

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image


class AcneSegmentationDataset(Dataset):
    """Dataset reading JPEG images and corresponding masks.

    Assumes the following structure inside ``root``::

        data/
          JPEGImages/  # RGB images
          mask/        # PNG binary masks
          with_mask_train.txt
          with_mask_val.txt

    Each line in the split files begins with an image filename. The mask file
    is expected to share the same stem with ``.png`` extension.
    """

    def __init__(self, root: Path, split_file: str, transforms: A.Compose | None = None):
        self.root = Path(root)
        self.img_dir = self.root / "data" / "JPEGImages"
        self.mask_dir = self.root / "data" / "mask"
        with open(self.root / "data" / split_file) as f:
            self.ids = [line.strip().split()[0] for line in f]
        self.transforms = transforms

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_path = self.img_dir / self.ids[idx]
        mask_name = Path(self.ids[idx]).stem + ".png"
        mask_path = self.mask_dir / mask_name
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        if self.transforms:
            aug = self.transforms(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]
        return image, mask.long()


def train(args: argparse.Namespace) -> None:
    root = Path(args.data_root)
    transform = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ])
    train_ds = AcneSegmentationDataset(root, "with_mask_train.txt", transform)
    val_ds = AcneSegmentationDataset(root, "with_mask_test.txt", transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name="vit_b_16",
        encoder_weights="imagenet",
        classes=1,
        activation=None,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = smp.losses.DiceLoss(mode="binary")

    pred_dir = root / "pred_masks"
    pred_dir.mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()

        model.eval()
        ious, accs, scores = [], [], []
        with torch.no_grad():
            for idx, (images, masks) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                probs = logits.sigmoid()
                preds = (probs > 0.5).float()
                intersection = (preds * masks).sum((1, 2, 3))
                union = preds.sum((1, 2, 3)) + masks.sum((1, 2, 3)) - intersection
                ious.extend((intersection / union).cpu().numpy())
                accs.extend(((preds == masks).float().mean((1, 2, 3))).cpu().numpy())
                scores.extend(probs.mean((1, 2, 3)).cpu().numpy())
                for b in range(images.size(0)):
                    save_image(
                        preds[b].cpu(),
                        pred_dir / f"epoch{epoch+1:02d}_{idx*val_loader.batch_size + b:04d}.png",
                    )
        print(
            f"Epoch {epoch + 1}: IoU {np.mean(ious):.3f} "
            f"Acc {np.mean(accs):.3f} Score {np.mean(scores):.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT-based segmentation for acne")
    parser.add_argument("--data-root", type=str, default="Datasets/acne_segmentation", help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=256)
    train(parser.parse_args())
