import argparse
import os
import torch

from model.U2NetArc import U2NET_lite
from model.U2NetE import U2NetE


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def export_checkpoint_to_pth(
    checkpoint_path: str,
    output_path: str,
    enhancement_mode: str = "hybrid",
    normalize: bool = True,
) -> None:
    """
    Extract model state dict from checkpoint and save as standalone .pth file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        output_path: Destination .pth file path
        enhancement_mode: Model enhancement mode (for validation)
        normalize: Whether model uses normalization (for validation)
    """
    device = get_default_device()
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("Found 'model_state_dict' in checkpoint")
    else:
        # Assume the checkpoint itself is the state dict
        state_dict = checkpoint
        print("Checkpoint appears to be a state dict")
    
    # Optional: Validate by loading into model
    try:
        base_model = U2NET_lite()
        model = U2NetE(base_model=base_model, enhancement_mode=enhancement_mode, normalize=normalize)
        model.load_state_dict(state_dict)
        model.eval()
        print("✓ State dict validation successful")
    except Exception as e:
        print(f"⚠ Warning: State dict validation failed: {e}")
        print("  Proceeding anyway - the state dict may be for a different model configuration")
    
    # Save as standalone .pth file
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(state_dict, output_path)
    
    # Get file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"✓ Successfully exported model state dict to {output_path}")
    print(f"  File size: {file_size:.2f} MB")
    
    # Optionally save with metadata
    if 'epoch' in checkpoint or 'total_loss' in checkpoint:
        metadata_path = output_path.replace('.pth', '_metadata.txt')
        with open(metadata_path, 'w') as f:
            f.write("Checkpoint Metadata:\n")
            f.write("=" * 50 + "\n")
            if 'epoch' in checkpoint:
                f.write(f"Epoch: {checkpoint['epoch']}\n")
            if 'total_loss' in checkpoint:
                f.write(f"Total Loss: {checkpoint['total_loss']:.6f}\n")
            if 'optimizer_state_dict' in checkpoint:
                f.write("Optimizer state: Included in original checkpoint\n")
        print(f"  Metadata saved to {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract model state dict from checkpoint and save as .pth file"
    )
    parser.add_argument('--checkpoint', required=True, help='Path to the checkpoint file (.pth)')
    parser.add_argument('--output', required=True, help='Destination .pth file path')
    parser.add_argument('--enhancement-mode', default='hybrid', 
                        choices=['rgb', 'green', 'hybrid'],
                        help='Enhancement mode for validation (default: hybrid)')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Model uses normalization (for validation)')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                        help='Model does not use normalization')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    export_checkpoint_to_pth(
        args.checkpoint,
        args.output,
        args.enhancement_mode,
        args.normalize,
    )


if __name__ == '__main__':
    main()

