import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import PatchRetinalDataset  
from model.U2NetE import U2NetE  
from model.U2NetArc import U2NET_lite
import os
import logging
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

class TverskyLoss(nn.Module):
    """
    Tversky Loss for segmentation, handling severe class imbalance.
    A generalized version of Dice Loss.
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        """
        Args:
            alpha (float): Controls the penalty for False Positives (FP). 
                           Lower alpha reduces FP penalty.
            beta (float): Controls the penalty for False Negatives (FN). 
                          Higher beta increases FN penalty.
            smooth (float): Small constant to prevent division by zero.
        
        Note: alpha + beta should typically sum to 1.0.
              For vessel segmentation, set beta > alpha (e.g., 0.3/0.7)
              to prioritize finding all vessels (reduce FNs).
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
        
        # True Positives: Where both are 1 (Prediction is correct vessel pixel)
        TP = (inputs * targets).sum()
        
        # False Positives: Where input is 1 (vessel) but target is 0 (background)
        FP = ((1 - targets) * inputs).sum()
        
        # False Negatives: Where input is 0 (background) but target is 1 (vessel)
        FN = (targets * (1 - inputs)).sum()
        
        # Tversky Index (TI)
        TI = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        # Tversky Loss
        return 1 - TI


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainerConfig:
    train_image_dir: str = "data/HRF/Train/images"
    train_mask_dir: str = "data/HRF/Train/masks"
    val_image_dir: str = "data/HRF/Val_augmented/images"
    val_mask_dir: str = "data/HRF/Val_augmented/masks"
    patch_size: Tuple[int, int] = (384, 384)
    augment_config: Dict[str, bool] = field(default_factory=lambda: {
        "flip": True,
        "rotate": True,
        "brightness": True,
        "blur": True,
        "noise": True,
    })
    train_use_full_random: bool = True
    train_patch_prob: float = 0.70
    train_load_into_ram: bool = False
    val_use_full_random: bool = False
    val_patch_prob: float = 0.0
    val_load_into_ram: bool = False
    batch_size: int = 6
    num_workers: int = 2
    lr: float = 3e-4
    weight_decay: float = 1e-5
    num_epochs: int = 10000
    alpha_bce: float = 0.5
    alpha_dice: float = 0.5
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    save_dir: str = "trained_models"
    checkpoint_dir: str = "checkpoints_HYBRID_4SET"
    log_dir: str = "logs"
    log_filename: str = "HYBRID_4set_training_log.txt"
    val_interval: int = 3
    val_subset_size: int = 35
    random_seed: int = 42
    log_base: int = 2
    log_tolerance: float = 0.05
    train_visual_dir: str = "./results"
    val_visual_dir: str = "./val_results"
    visualize_index: int = 6
    infer_patch_size: int = 384
    infer_overlap: int = 64
    infer_batch_size: int = 6
    device: torch.device = field(default_factory=get_default_device)

    def log_path(self) -> str:
        return os.path.join(self.log_dir, self.log_filename)


class HybridTrainer:
    def __init__(self, config: TrainerConfig):
        self.cfg = config
        self.device = config.device

        self._ensure_directories()
        self._setup_logging()

        self.bce_criterion = nn.BCELoss(reduction='mean')
        self.tversky_loss = TverskyLoss(alpha=self.cfg.tversky_alpha, beta=self.cfg.tversky_beta)

        self.train_loader, self.val_dataset = self._build_datasets()
        self.model = self._build_model()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

        self.best_val_loss = float("inf")
        self.fixed_val_indices = self._select_fixed_indices(self.val_dataset)

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _ensure_directories(self) -> None:
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        os.makedirs(self.cfg.train_visual_dir, exist_ok=True)
        os.makedirs(self.cfg.val_visual_dir, exist_ok=True)

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            handlers=[
                logging.FileHandler(self.cfg.log_path(), mode='w'),
                logging.StreamHandler(),
            ],
            force=True,
        )

    def _build_datasets(self) -> Tuple[DataLoader, PatchRetinalDataset]:
        train_dataset = PatchRetinalDataset(
            image_dir=self.cfg.train_image_dir,
            mask_dir=self.cfg.train_mask_dir,
            patch_size=self.cfg.patch_size,
            augment_config=self.cfg.augment_config,
            use_full_random=self.cfg.train_use_full_random,
            use_patch_prob=self.cfg.train_patch_prob,
            load_into_ram=self.cfg.train_load_into_ram,
        )

        val_dataset = PatchRetinalDataset(
            image_dir=self.cfg.val_image_dir,
            mask_dir=self.cfg.val_mask_dir,
            patch_size=self.cfg.patch_size,
            use_full_random=self.cfg.val_use_full_random,
            use_patch_prob=self.cfg.val_patch_prob,
            load_into_ram=self.cfg.val_load_into_ram,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
        )

        return train_loader, val_dataset

    def _build_model(self) -> nn.Module:
        base_model = U2NET_lite()
        model = U2NetE(base_model=base_model, enhancement_mode="hybrid", normalize=True)
        return model.to(self.device)

    def _select_fixed_indices(self, dataset: PatchRetinalDataset) -> List[int]:
        total_val_images = len(dataset)
        if total_val_images == 0:
            return []

        subset_size = min(self.cfg.val_subset_size, total_val_images)
        rng = np.random.default_rng(self.cfg.random_seed)
        return rng.choice(total_val_images, size=subset_size, replace=False).tolist()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self) -> None:
        print(f"Training started on device {self.device}")
        logging.info("Training started...")

        for epoch in range(self.cfg.num_epochs):
            train_metrics = self._train_one_epoch(epoch)
            self._log_epoch(epoch, train_metrics)

            if (epoch + 1) % self.cfg.val_interval == 0:
                self._run_validation(epoch)

            self._maybe_visualize_training_snapshot(epoch, train_metrics)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()

        running_total_loss = 0.0
        running_bce_loss = 0.0
        running_dice_loss = 0.0
        running_d0_bce = 0.0
        running_d0_dice = 0.0

        dataset_size = len(self.train_loader.dataset)

        for imgs, masks in self.train_loader:
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(imgs)

            bce_loss_sum = 0.0
            dice_loss_sum = 0.0

            for out in outputs:
                bce_val = self.bce_criterion(out, masks)
                dice_val = self.tversky_loss(out, masks)
                bce_loss_sum += bce_val
                dice_loss_sum += dice_val

            bce_loss_avg = bce_loss_sum / len(outputs)
            dice_loss_avg = dice_loss_sum / len(outputs)
            total_loss = self.cfg.alpha_bce * bce_loss_avg + self.cfg.alpha_dice * dice_loss_avg

            primary_output = outputs[1] if len(outputs) > 1 else outputs[0]
            d0_bce = self.bce_criterion(primary_output, masks)
            d0_dice = self.tversky_loss(primary_output, masks)

            total_loss.backward()
            self.optimizer.step()

            batch_size_actual = imgs.size(0)
            running_total_loss += total_loss.item() * batch_size_actual
            running_bce_loss += bce_loss_avg.item() * batch_size_actual
            running_dice_loss += dice_loss_avg.item() * batch_size_actual
            running_d0_bce += d0_bce.item() * batch_size_actual
            running_d0_dice += d0_dice.item() * batch_size_actual

        epoch_total = running_total_loss / dataset_size
        epoch_bce = running_bce_loss / dataset_size
        epoch_dice = running_dice_loss / dataset_size
        epoch_d0_bce = running_d0_bce / dataset_size
        epoch_d0_dice = running_d0_dice / dataset_size

        return {
            "total": epoch_total,
            "bce": epoch_bce,
            "dice": epoch_dice,
            "d0_bce": epoch_d0_bce,
            "d0_dice": epoch_d0_dice,
        }

    def _log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        logging.info(
            "Epoch [%d/%d] | Total Loss: %.6f | BCE: %.6f | Dice: %.6f || d0_BCE: %.6f | d0_Dice: %.6f",
            epoch + 1,
            self.cfg.num_epochs,
            metrics["total"],
            metrics["bce"],
            metrics["dice"],
            metrics["d0_bce"],
            metrics["d0_dice"],
        )

    def _maybe_visualize_training_snapshot(self, epoch: int, metrics: Dict[str, float]) -> None:
        epoch_num = epoch + 1
        if should_save(epoch_num, log_base=self.cfg.log_base, tolerance=self.cfg.log_tolerance):
            visualize_fixed_image(
                self.model,
                self.val_dataset,
                self.device,
                epoch,
                metrics["total"],
                metrics["bce"],
                metrics["dice"],
                index=self.cfg.visualize_index,
                save_dir=self.cfg.train_visual_dir,
                patch_size=self.cfg.infer_patch_size,
                overlap=self.cfg.infer_overlap,
                infer_batch_size=self.cfg.infer_batch_size,
            )

    # ------------------------------------------------------------------
    # Validation and checkpointing
    # ------------------------------------------------------------------
    def _run_validation(self, epoch: int) -> None:
        val_total, val_bce, val_dice = self._validate_one_epoch()

        if val_total is None:
            return

        logging.info(
            "   → Validation | Total: %.4f | BCE: %.4f | Dice: %.4f",
            val_total,
            val_bce,
            val_dice,
        )

        if val_total < self.best_val_loss:
            self.best_val_loss = val_total
            best_model_path = f"best_model_epoch_{epoch+1}_val_{val_total:.4f}.pth"
            self._save_checkpoint(epoch, best_model_path, val_total)

            visualize_fixed_image(
                self.model,
                self.val_dataset,
                self.device,
                epoch,
                val_total,
                val_bce,
                val_dice,
                index=self.cfg.visualize_index,
                save_dir=self.cfg.val_visual_dir,
                patch_size=self.cfg.infer_patch_size,
                overlap=self.cfg.infer_overlap,
                infer_batch_size=self.cfg.infer_batch_size,
            )

    def _validate_one_epoch(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if not self.fixed_val_indices:
            return None, None, None

        self.model.eval()
        total_bce = 0.0
        total_dice = 0.0
        count = 0

        with torch.no_grad():
            for idx in self.fixed_val_indices:
                img, gt_mask = self.val_dataset.get_full_image(idx=idx)
                pred_mask, _ = infer_full_image_with_enhance(
                    self.model,
                    img=img,
                    patch_size=self.cfg.infer_patch_size,
                    overlap=self.cfg.infer_overlap,
                    device=self.device,
                    batch_size=self.cfg.infer_batch_size,
                )

                pred_t = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0).to(self.device)
                gt_t = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0).float().to(self.device)

                bce_val = self.bce_criterion(pred_t, gt_t)
                dice_val = self.tversky_loss(pred_t, gt_t)

                total_bce += bce_val.item()
                total_dice += dice_val.item()
                count += 1

        if count == 0:
            return None, None, None

        avg_bce = total_bce / count
        avg_dice = total_dice / count
        avg_total = self.cfg.alpha_bce * avg_bce + self.cfg.alpha_dice * avg_dice

        return avg_total, avg_bce, avg_dice

    def _save_checkpoint(self, epoch: int, filename: str, total_loss: float) -> None:
        checkpoint_path = os.path.join(self.cfg.checkpoint_dir, filename)
        torch.save(
            {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'total_loss': total_loss,
            },
            checkpoint_path,
        )

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def infer_full_image_with_enhance(model, img, patch_size=512, overlap=64, device='cpu', batch_size=6):
    """
    Patch-based inference for full image with enhancement visualization,
    processing patches in batches for GPU efficiency.

    Args:
        model: trained U2NetE model.
        img: np.ndarray, H x W x C (RGB).
        patch_size: int, size of square patches.
        overlap: int, overlapping pixels between patches.
        device: 'cpu' | 'cuda' | 'mps'.
        batch_size: int, number of patches to process concurrently.
    
    Returns:
        full_pred: merged d0 prediction (H x W).
        full_enhanced: merged enhanced image (H x W x 3).
    """
    
    H, W, C = img.shape
    stride = patch_size - overlap

    # Pad image to fit patch grid
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    img_pad = np.pad(img, ((0, pad_h), (0, pad_w), (0,0)), mode='reflect')
    H_pad, W_pad, _ = img_pad.shape

    # Convert to tensor (B=1)
    img_pad_tensor = torch.from_numpy(img_pad.transpose(2,0,1)).unsqueeze(0).float()

    # Initialize accumulation tensors
    full_pred = torch.zeros((1, 1, H_pad, W_pad), device=device)
    full_enhanced = torch.zeros((1, 3, H_pad, W_pad), device=device)
    weight_map = torch.zeros_like(full_pred)

    model.eval()
    with torch.no_grad():
        ys = list(range(0, H_pad - patch_size + 1, stride))
        xs = list(range(0, W_pad - patch_size + 1, stride))

        # Ensure last patch covers the image edge
        if ys[-1] != H_pad - patch_size:
            ys.append(H_pad - patch_size)
        if xs[-1] != W_pad - patch_size:
            xs.append(W_pad - patch_size)
            
        # --- BATCH PROCESSING START ---
        
        all_patches = []
        patch_coords = []
        
        # 1. Collect all patches and coordinates
        for y in ys:
            for x in xs:
                # Patch is still 4D (1, C, P, P)
                patch = img_pad_tensor[:, :, y:y+patch_size, x:x+patch_size] 
                all_patches.append(patch)
                patch_coords.append((y, x))
                
        num_patches = len(all_patches)
        
        # 2. Process Patches in Batches
        for i in range(0, num_patches, batch_size):
            batch_end = min(i + batch_size, num_patches)
            
            # Concatenate patches to form the batch tensor (B, C, P, P)
            patch_batch = torch.cat(all_patches[i:batch_end], dim=0).to(device)
            
            # Forward pass is now highly efficient!
            outputs, enhanced_batch = model(patch_batch, return_enhanced=True)
            # Assuming outputs[1] is the main prediction d0
            d0_batch = outputs[1]
            
            # 3. Deconstruct and Accumulate Results
            for j in range(patch_batch.size(0)):
                # Get the result and coordinates for the j-th patch in the current batch
                y, x = patch_coords[i + j]
                d0 = d0_batch[j:j+1] # Keep 4D [1, 1, P, P] for accumulation
                enhanced = enhanced_batch[j:j+1] # Keep 4D [1, 3, P, P]

                full_pred[:, :, y:y+patch_size, x:x+patch_size] += d0
                full_enhanced[:, :, y:y+patch_size, x:x+patch_size] += enhanced
                weight_map[:, :, y:y+patch_size, x:x+patch_size] += 1.0

        # --- BATCH PROCESSING END ---

        # Average overlapping regions
        full_pred /= torch.clamp(weight_map, min=1e-6)
        full_enhanced /= torch.clamp(weight_map.expand_as(full_enhanced), min=1e-6)

        # Apply normalization (normPRED) once after merging
        full_pred = normPRED(full_pred)
        full_enhanced = normPRED(full_enhanced)

        # Convert to numpy for visualization
        full_pred = full_pred.squeeze().cpu().numpy()
        full_enhanced = full_enhanced.squeeze().permute(1,2,0).cpu().numpy()

    # Crop to original size
    full_pred = full_pred[:H, :W]
    full_enhanced = full_enhanced[:H, :W, :]

    return full_pred, full_enhanced


def visualize_full_with_enhance(img, enhanced, pred_mask, gt_mask, save_path=None):
    fig, axs = plt.subplots(1, 4, figsize=(20,5))
    axs[0].imshow(img)
    axs[0].set_title("Input Image")
    axs[0].axis('off')

    axs[1].imshow(enhanced)
    axs[1].set_title("Enhanced Image")
    axs[1].axis('off')

    axs[2].imshow(pred_mask, cmap='gray')
    axs[2].set_title("Predicted Mask")
    axs[2].axis('off')

    axs[3].imshow(gt_mask, cmap='gray')
    axs[3].set_title("Ground Truth Mask")
    axs[3].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_fixed_image(
    model,
    dataset,
    device,
    epoch,
    epoch_total,
    epoch_bce,
    epoch_dice,
    index: int = 15,
    save_dir: str = "./results",
    patch_size: int = 384,
    overlap: int = 64,
    infer_batch_size: int = 6,
):
    """
    Visualize a fixed image from the dataset without augmentation and save with epoch losses in filename.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    img, mask = dataset.get_full_image(idx=index)
    pred_mask, enhanced_img = infer_full_image_with_enhance(
        model,
        img=img,
        patch_size=patch_size,
        overlap=overlap,
        device=device,
        batch_size=infer_batch_size,
    )


    # Save with epoch and losses in filename
    filename = f"epoch_{epoch+1}_total_{epoch_total:.4f}_bce_{epoch_bce:.4f}_dice_{epoch_dice:.4f}.png"
    save_path = os.path.join(save_dir, filename)
    visualize_full_with_enhance(img=img,enhanced=enhanced_img,pred_mask=pred_mask,gt_mask=mask, save_path=save_path)
    model.train()

def should_save(epoch: int, log_base: int = 2, tolerance: float = 0.05) -> bool:
    """
    Checks if the current epoch is close to a power of the given log base.
    
    Args:
        epoch (int): The current epoch number (should be >= 1).
        log_base (int): The base of the logarithm (e.g., 2 for doubling frequency).
        tolerance (float): The tolerance for checking if the log is close to an integer.
        
    Returns:
        bool: True if the epoch should be saved/logged.
    """
    if epoch < 1:
        return False
        
    # Calculate log_base of the current epoch
    log_value = math.log(epoch, log_base)
    
    # Check if the log value is very close to an integer (a power of the base)
    # E.g., log2(8) = 3.0, log2(9) != integer
    is_power_of_base = abs(log_value - round(log_value)) < tolerance
    
    # Always save the first few epochs (e.g., first 5) for initial sanity check
    if epoch <= 5:
        return True
        
    # After the initial period, only save if it's near a power of the base
    return is_power_of_base


def main() -> None:
    config = TrainerConfig()
    trainer = HybridTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()