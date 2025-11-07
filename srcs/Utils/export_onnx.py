import argparse
import os
import torch

from model.U2NetArc import U2NET_lite
from model.U2NetE import U2NetE


def load_model(checkpoint_path: str, enhancement_mode: str, normalize: bool, device: torch.device) -> U2NetE:
    base_model = U2NET_lite()
    model = U2NetE(base_model=base_model, enhancement_mode=enhancement_mode, normalize=normalize)
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_to_onnx(
    model: U2NetE,
    output_path: str,
    height: int,
    width: int,
    device: torch.device,
    opset: int = 18,
    use_dynamic_batch: bool = False,
) -> None:
    dummy_input = torch.randn(1, 3, height, width, device=device)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Test forward pass first
    with torch.no_grad():
        outputs = model(dummy_input)
        num_outputs = len(outputs) if isinstance(outputs, (list, tuple)) else 1

    export_kwargs = {
        'model': model,
        'args': dummy_input,
        'f': output_path,
        'input_names': ['input'],
        'output_names': [f'output_{i}' for i in range(num_outputs)],
        'opset_version': opset,
        'do_constant_folding': True,
        'verbose': False,
    }

    if use_dynamic_batch:
        export_kwargs['dynamic_axes'] = {
            'input': {0: 'batch'},
            **{f'output_{i}': {0: 'batch'} for i in range(num_outputs)}
        }

    torch.onnx.export(**export_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export U²Net-E checkpoint to ONNX format")
    parser.add_argument('--checkpoint', required=True, help='Path to the PyTorch checkpoint (.pth)')
    parser.add_argument('--output', required=True, help='Destination ONNX file path')
    parser.add_argument('--height', type=int, default=384, help='Input tensor height')
    parser.add_argument('--width', type=int, default=384, help='Input tensor width')
    parser.add_argument('--enhancement-mode', default='hybrid', choices=['rgb', 'green', 'hybrid'], help='Enhancement module mode')
    parser.add_argument('--normalize', action='store_true', help='Apply ImageNet-style normalization before backbone')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false', help='Disable ImageNet-style normalization')
    parser.set_defaults(normalize=True)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='Device for export runtime')
    parser.add_argument('--opset', type=int, default=18, help='ONNX opset version (default: 11, recommended for stability)')
    parser.add_argument('--dynamic-batch', action='store_true', help='Enable dynamic batch size in ONNX model')
    return parser.parse_args()


def resolve_device(arg: str) -> torch.device:
    if arg == 'cpu':
        return torch.device('cpu')
    if arg == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device('cpu')


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    
    print(f"Loading checkpoint from {args.checkpoint}")
    model = load_model(args.checkpoint, args.enhancement_mode, args.normalize, device)
    
    print(f"Exporting to ONNX (opset {args.opset})...")
    try:
        export_to_onnx(
            model, 
            args.output, 
            args.height, 
            args.width, 
            device, 
            opset=args.opset,
            use_dynamic_batch=args.dynamic_batch
        )
        print(f"✓ Successfully exported ONNX model to {args.output}")
    except Exception as e:
        print(f"✗ Export failed: {e}")
        print("\nTry:")
        print("  - Lower opset version (e.g., --opset 11)")
        print("  - Remove --dynamic-batch flag")
        print("  - Use CPU device (--device cpu)")
        raise


if __name__ == '__main__':
    main()

