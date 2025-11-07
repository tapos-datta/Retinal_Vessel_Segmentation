import torch
import numpy as np
import os
import glob
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from dataloader import PatchRetinalDataset  # your dataset module
from model.U2NetE import U2NetE  # your model module
from model.U2NetArc import U2NET_lite
from sklearn.metrics import accuracy_score, jaccard_score, recall_score
from operator import itemgetter # Used for easy sorting


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# --- I. METRIC HELPER FUNCTIONS ---

def dice_coefficient(pred, target, smooth=1e-6):
    """Calculates the Dice Coefficient (F1 Score)."""
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def calculate_metrics(y_true, y_pred):
    """
    Calculates key segmentation metrics using flattened binary arrays.
    
    Args:
        y_true (np.ndarray): Flattened ground truth mask (0 or 1).
        y_pred (np.ndarray): Flattened predicted mask (0 or 1).
    """
    # Overall Accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # Jaccard Index (IoU)
    iou = jaccard_score(y_true, y_pred)
    
    # Sensitivity (Recall) - How well we found all vessel pixels (TP / (TP + FN))
    sensitivity = recall_score(y_true, y_pred)
    
    # Specificity - How well we found all background pixels (TN / (TN + FP))
    # We calculate it manually because sklearn's recall_score defaults to positive class (1).
    # Specificity is Recall of the negative class (0).
    specificity = recall_score(y_true, y_pred, pos_label=0)
    
    # Dice Coefficient (F1) - Calculated using the formula
    dice = dice_coefficient(y_pred, y_true)

    return {
        'Dice': dice.item(),
        'IoU': iou,
        'Accuracy': acc,
        'Sensitivity': sensitivity,
        'Specificity': specificity
    }


@dataclass
class TestConfig:
    test_image_dir: str = "data/HRF/Test/images"
    test_mask_dir: str = "data/HRF/Test/masks"
    patch_size: int = 384
    overlap: int = 64
    infer_batch_size: int = 6
    tta_folds: int = 8
    threshold: float = 0.5
    checkpoint_dir: str = "checkpoints_HYBRID_4SET"
    num_checkpoints: int = 15
    log_dir: str = "logs"
    log_filename: str = "HYBRID_4set_test_log.txt"
    device: torch.device = field(default_factory=get_default_device)
    enhancement_mode: str = "hybrid"
    normalize: bool = True

    def log_path(self) -> str:
        return os.path.join(self.log_dir, self.log_filename)


class HybridEvaluator:
    def __init__(self, config: TestConfig):
        self.cfg = config
        self.device = config.device

        self._ensure_directories()
        self._setup_logging()

        self.dataset = self._build_dataset()
        self.model = self._build_model()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _ensure_directories(self) -> None:
        os.makedirs(self.cfg.log_dir, exist_ok=True)

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

    def _build_dataset(self) -> PatchRetinalDataset:
        return PatchRetinalDataset(
            image_dir=self.cfg.test_image_dir,
            mask_dir=self.cfg.test_mask_dir,
            patch_size=(self.cfg.patch_size, self.cfg.patch_size),
            use_full_random=False,
            use_patch_prob=0.0,
            load_into_ram=False,
        )

    def _build_model(self) -> U2NetE:
        base_model = U2NET_lite()
        model = U2NetE(
            base_model=base_model,
            enhancement_mode=self.cfg.enhancement_mode,
            normalize=self.cfg.normalize,
        )
        return model.to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, float]]:
        if not os.path.exists(checkpoint_path):
            logging.warning("Checkpoint %s does not exist", checkpoint_path)
            return None

        logging.info("Loading checkpoint from %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        metrics = self._run_evaluation()
        if metrics is None:
            logging.warning("No metrics computed for checkpoint %s", checkpoint_path)
            return None

        logging.info("\n--- FINAL TEST PERFORMANCE REPORT ---")
        logging.info("Average Dice Coefficient: %.4f", metrics['Dice'])
        logging.info("Average IoU (Jaccard): %.4f", metrics['IoU'])
        logging.info("Average Accuracy: %.4f", metrics['Accuracy'])
        logging.info("Average Sensitivity (Recall): %.4f", metrics['Sensitivity'])
        logging.info("Average Specificity: %.4f", metrics['Specificity'])
        logging.info("-------------------------------------")

        return metrics

    def evaluate_latest_checkpoints(self) -> Dict[str, Dict[str, float]]:
        checkpoints = self._find_recent_checkpoints(self.cfg.checkpoint_dir, self.cfg.num_checkpoints)
        if not checkpoints:
            return {}

        results: Dict[str, Dict[str, float]] = {}
        for path in checkpoints:
            name = os.path.basename(path)
            logging.info("\n--- Evaluating Checkpoint: %s ---", name)
            metrics = self.evaluate_checkpoint(path)
            if metrics is not None:
                results[name] = metrics

        if results:
            logging.info("\n--- SUMMARY OF ALL TEST RESULTS ---")
            for filename, metrics in results.items():
                logging.info(
                    "%s: Dice=%.4f, IoU=%.4f, Sensitivity=%.4f",
                    filename,
                    metrics['Dice'],
                    metrics['IoU'],
                    metrics['Sensitivity'],
                )

        return results

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def _run_evaluation(self) -> Optional[Dict[str, float]]:
        if len(self.dataset) == 0:
            logging.warning("Test dataset is empty. Skipping evaluation.")
            return None

        all_metrics: List[Dict[str, float]] = []

        with torch.no_grad():
            for idx in range(len(self.dataset)):
                img, gt_mask = self.dataset.get_full_image(idx=idx)
                pred_mask = self._infer_with_tta(img)

                binary_pred = (pred_mask > self.cfg.threshold).astype(np.int32)
                gt_flat = gt_mask.flatten().astype(np.int32)
                pred_flat = binary_pred.flatten().astype(np.int32)

                metrics = calculate_metrics(gt_flat, pred_flat)
                all_metrics.append(metrics)

                logging.info(
                    "Image %d/%d - Dice: %.4f, Sensitivity: %.4f",
                    idx + 1,
                    len(self.dataset),
                    metrics['Dice'],
                    metrics['Sensitivity'],
                )

        aggregated = {
            key: float(np.mean([m[key] for m in all_metrics]))
            for key in all_metrics[0].keys()
        }

        return aggregated

    def _infer_with_tta(self, img: np.ndarray) -> np.ndarray:
        transforms = [
            lambda x: x,
            lambda x: np.flip(x, axis=1),
            lambda x: np.flip(x, axis=0),
            lambda x: np.rot90(x, k=1, axes=(0, 1)),
            lambda x: np.rot90(x, k=2, axes=(0, 1)),
            lambda x: np.rot90(x, k=3, axes=(0, 1)),
            lambda x: np.rot90(np.flip(x, axis=1), k=1, axes=(0, 1)),
            lambda x: np.rot90(np.flip(x, axis=1), k=3, axes=(0, 1)),
        ]

        inverse_transforms = [
            lambda x: x,
            lambda x: np.flip(x, axis=1),
            lambda x: np.flip(x, axis=0),
            lambda x: np.rot90(x, k=-1, axes=(0, 1)),
            lambda x: np.rot90(x, k=-2, axes=(0, 1)),
            lambda x: np.rot90(x, k=-3, axes=(0, 1)),
            lambda x: np.flip(np.rot90(x, k=-1, axes=(0, 1)), axis=1),
            lambda x: np.flip(np.rot90(x, k=-3, axes=(0, 1)), axis=1),
        ]

        preds = []
        for idx, transform in enumerate(transforms):
            img_aug = transform(img.copy())
            pred_aug, _ = self._infer_full_image_with_enhance(img_aug)
            pred_original = inverse_transforms[idx](pred_aug)
            preds.append(pred_original)

        return np.mean(preds, axis=0)

    def _infer_full_image_with_enhance(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        patch_size = self.cfg.patch_size
        overlap = self.cfg.overlap
        batch_size = self.cfg.infer_batch_size

        H, W, _ = img.shape
        stride = patch_size - overlap

        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        img_pad = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        H_pad, W_pad, _ = img_pad.shape

        img_pad_tensor = torch.from_numpy(img_pad.transpose(2, 0, 1)).unsqueeze(0).float()

        full_pred = torch.zeros((1, 1, H_pad, W_pad), device=self.device)
        full_enhanced = torch.zeros((1, 3, H_pad, W_pad), device=self.device)
        weight_map = torch.zeros_like(full_pred)

        self.model.eval()
        with torch.no_grad():
            ys = list(range(0, H_pad - patch_size + 1, stride))
            xs = list(range(0, W_pad - patch_size + 1, stride))

            if ys[-1] != H_pad - patch_size:
                ys.append(H_pad - patch_size)
            if xs[-1] != W_pad - patch_size:
                xs.append(W_pad - patch_size)

            all_patches = []
            patch_coords = []
            for y in ys:
                for x in xs:
                    patch = img_pad_tensor[:, :, y:y + patch_size, x:x + patch_size]
                    all_patches.append(patch)
                    patch_coords.append((y, x))

            num_patches = len(all_patches)
            for i in range(0, num_patches, batch_size):
                batch_end = min(i + batch_size, num_patches)
                patch_batch = torch.cat(all_patches[i:batch_end], dim=0).to(self.device)

                outputs, enhanced_batch = self.model(patch_batch, return_enhanced=True)
                d0_batch = outputs[1]

                for j in range(patch_batch.size(0)):
                    y, x = patch_coords[i + j]
                    d0 = d0_batch[j:j + 1]
                    enhanced = enhanced_batch[j:j + 1]

                    full_pred[:, :, y:y + patch_size, x:x + patch_size] += d0
                    full_enhanced[:, :, y:y + patch_size, x:x + patch_size] += enhanced
                    weight_map[:, :, y:y + patch_size, x:x + patch_size] += 1.0

            full_pred /= torch.clamp(weight_map, min=1e-6)
            full_enhanced /= torch.clamp(weight_map.expand_as(full_enhanced), min=1e-6)

            full_pred = normPRED(full_pred)
            full_enhanced = normPRED(full_enhanced)

            full_pred = full_pred.squeeze().cpu().numpy()
            full_enhanced = full_enhanced.squeeze().permute(1, 2, 0).cpu().numpy()

        full_pred = full_pred[:H, :W]
        full_enhanced = full_enhanced[:H, :W, :]

        return full_pred, full_enhanced

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _find_recent_checkpoints(directory: str, limit: int) -> List[str]:
        search_path = os.path.join(directory, "*.pth")
        all_files = glob.glob(search_path)

        if not all_files:
            logging.warning("No checkpoint files found in %s.", directory)
            return []

        files_with_time = [(path, os.path.getmtime(path)) for path in all_files]
        files_with_time.sort(key=itemgetter(1), reverse=True)

        selected = [path for path, _ in files_with_time[:limit]]
        logging.info(
            "Found %d checkpoints. Testing the %d most recent ones.",
            len(all_files),
            len(selected),
        )
        return selected


def main() -> None:
    config = TestConfig()
    evaluator = HybridEvaluator(config)
    evaluator.evaluate_latest_checkpoints()


if __name__ == "__main__":
    main()