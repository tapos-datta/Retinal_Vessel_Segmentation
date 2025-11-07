from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import transform
from torch.utils.data import DataLoader, Dataset


@dataclass
class AugmentationConfig:
    flip: bool = False
    rotate: bool = False
    brightness: bool = False
    blur: bool = False
    noise: bool = False

    @classmethod
    def from_dict(cls, config: Optional[Dict[str, bool]]) -> "AugmentationConfig":
        if not config:
            return cls()
        return cls(
            flip=config.get("flip", False),
            rotate=config.get("rotate", False),
            brightness=config.get("brightness", False),
            blur=config.get("blur", False),
            noise=config.get("noise", False),
        )


class PatchRetinalDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        patch_size: Tuple[int, int] = (496, 496),
        stride: Optional[int] = None,
        augment_config: Optional[Dict[str, bool]] = None,
        use_full_random: bool = True,
        downsample_size: Tuple[int, int] = (384, 384),
        use_patch_prob: float = 0.7,
        load_into_ram: bool = True,
    ) -> None:
        """Dataset that provides random vessel patches or downsampled full images."""

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_paths = sorted(p for p in self.image_dir.iterdir() if p.is_file())
        self.mask_paths = sorted(p for p in self.mask_dir.iterdir() if p.is_file())

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("Image and mask counts do not match.")

        self.patch_h, self.patch_w = patch_size
        self.stride = stride or self.patch_h
        self.use_full_random = use_full_random
        self.downsample_size = downsample_size
        self.use_patch_prob = use_patch_prob
        self.load_into_ram = load_into_ram
        self.aug_cfg = AugmentationConfig.from_dict(augment_config)

        if self.load_into_ram:
            self.images = [self._load_image(path) for path in self.image_paths]
            self.masks = [self._load_mask(path) for path in self.mask_paths]
        else:
            self.images = None
            self.masks = None

    def __len__(self) -> int:
        return len(self.image_paths)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        img, mask = self._fetch_pair(idx)
        img_patch, mask_patch = self._sample_patch(img, mask)
        img_patch, mask_patch = self._apply_augmentations(img_patch, mask_patch)
        img_patch, mask_patch = self._normalize_pair(img_patch, mask_patch)

        img_tensor = torch.from_numpy(img_patch.transpose(2, 0, 1).astype(np.float32))
        mask_tensor = torch.from_numpy(mask_patch.astype(np.float32)).unsqueeze(0)

        return img_tensor, mask_tensor

    def get_full_image(self, idx: int):
        img = self._load_image(self.image_paths[idx]) if not self.load_into_ram else self.images[idx].copy()
        mask = self._load_mask(self.mask_paths[idx]) if not self.load_into_ram else self.masks[idx].copy()

        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        return img, mask

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fetch_pair(self, idx: int):
        if self.load_into_ram:
            img = self.images[idx].copy()
            mask = self.masks[idx].copy()
        else:
            img = self._load_image(self.image_paths[idx])
            mask = self._load_mask(self.mask_paths[idx])

        return img, mask

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32)

    @staticmethod
    def _load_mask(path: Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to load mask: {path}")
        return mask.astype(np.float32)

    def _sample_patch(self, img: np.ndarray, mask: np.ndarray):
        h, w, _ = img.shape
        if self.use_full_random and random.random() < self.use_patch_prob:
            top = np.random.randint(0, max(1, h - self.patch_h + 1))
            left = np.random.randint(0, max(1, w - self.patch_w + 1))
            img_patch = img[top : top + self.patch_h, left : left + self.patch_w]
            mask_patch = mask[top : top + self.patch_h, left : left + self.patch_w]
        else:
            target_size = self.downsample_size or (h, w)
            img_patch = transform.resize(
                img,
                target_size,
                order=1,
                mode="reflect",
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float32)
            mask_patch = transform.resize(
                mask,
                target_size,
                order=0,
                mode="reflect",
                preserve_range=True,
            ).astype(np.float32)

        return img_patch, mask_patch

    def _apply_augmentations(self, img: np.ndarray, mask: np.ndarray):
        if self.aug_cfg.flip and random.random() < 0.5:
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            if random.random() < 0.5:
                img = cv2.flip(img, 0)
                mask = cv2.flip(mask, 0)

        if self.aug_cfg.rotate:
            if random.random() < 0.5:
                angle = random.choice([0, 90, 180, 270])
            else:
                angle = random.uniform(-15.0, 15.0)
            img, mask = self.rotate_pair(img, mask, angle)

        if self.aug_cfg.brightness and random.random() < 0.3:
            factor = 1.0 + 0.1 * (random.random() - 0.5)
            img = np.clip(img * factor, 0, 255)

        if self.aug_cfg.blur and random.random() < 0.3:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        if self.aug_cfg.noise and random.random() < 0.3:
            noise = np.random.normal(0, 2, img.shape)
            img = np.clip(img + noise, 0, 255)

        return img, mask

    @staticmethod
    def _normalize_pair(img: np.ndarray, mask: np.ndarray):
        img_norm = img.astype(np.float32) / 255.0
        mask_bin = (mask > 127).astype(np.float32)
        return img_norm, mask_bin

    # ------------------------------------------------------------------
    # Optional preprocessing utilities
    # ------------------------------------------------------------------
    def preprocess_chase_image(self, img: np.ndarray) -> np.ndarray:
        """Optional preprocessing for CHASE_DB1 images."""
        green = img[..., 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        green = clahe.apply(green)
        sharpened = cv2.addWeighted(green, 1.5, cv2.GaussianBlur(green, (5, 5), 3), -1.5, 0)
        _ = self.preprocess_stage3(sharpened)
        return np.stack([sharpened] * 3, axis=-1)

    @staticmethod
    def preprocess_stage3(sharp: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(sharp, d=5, sigmaColor=30, sigmaSpace=15)

    @staticmethod
    def rotate_pair(img: np.ndarray, mask: np.ndarray, angle: float):
        h, w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img_rot = cv2.warpAffine(
            img,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        mask_rot = cv2.warpAffine(
            mask,
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return img_rot, mask_rot


def visualize_batch(imgs: torch.Tensor, masks: torch.Tensor, num_samples: int = 4) -> None:
    """Visualize a few samples from your dataloader."""

    imgs = imgs[:num_samples]
    masks = masks[:num_samples]

    plt.figure(figsize=(10, num_samples * 3))

    for i in range(num_samples):
        img = imgs[i].permute(1, 2, 0).numpy()
        mask = masks[i][0].numpy()

        img = np.clip(img, 0, 1)

        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(img)
        plt.title("Augmented Image")
        plt.axis("off")

        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.imshow(img)
        plt.imshow(mask, cmap="Reds", alpha=0.4)
        plt.title("Overlayed Mask")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cfg = {
        "flip": True,
        "rotate": True,
        "brightness": True,
        "blur": True,
        "noise": True,
    }

    dataset = PatchRetinalDataset(
        image_dir="data/HRF/Train/images",
        mask_dir="data/HRF/Train/masks",
        patch_size=(512, 512),
        augment_config=cfg,
        use_full_random=True,
    )

    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
    imgs, masks = next(iter(loader))
    print(imgs.shape, masks.shape)
    visualize_batch(imgs, masks, num_samples=2)
