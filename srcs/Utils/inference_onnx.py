import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def load_onnx_model(onnx_path: str, providers: list = None) -> ort.InferenceSession:
    """Load ONNX model and create inference session."""
    if providers is None:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(
        onnx_path,
        sess_options=session_options,
        providers=providers
    )
    
    print(f"Loaded ONNX model: {onnx_path}")
    print(f"Input shape: {session.get_inputs()[0].shape}")
    print(f"Output count: {len(session.get_outputs())}")
    for i, output in enumerate(session.get_outputs()):
        print(f"  Output {i}: {output.name}, shape: {output.shape}")
    
    return session


def preprocess_image(image_path: str, target_size: tuple = (384, 384)) -> np.ndarray:
    """Load and preprocess image for inference."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Resize to target size
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0, 1] and convert to float32
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Convert HWC to CHW and add batch dimension
    img_tensor = np.transpose(img_normalized, (2, 0, 1))  # [3, H, W]
    img_tensor = np.expand_dims(img_tensor, axis=0)  # [1, 3, H, W]
    
    return img_tensor, (h, w)


def postprocess_output(output: np.ndarray, original_size: tuple) -> np.ndarray:
    """Postprocess model output to binary mask."""
    # Remove batch dimension if present
    if len(output.shape) == 4:
        output = output[0]
    
    # If multiple outputs, take the first one (or concatenate if needed)
    if len(output.shape) == 4 and output.shape[0] > 1:
        output = output[0]  # Take first channel
    
    # Remove channel dimension if present
    if len(output.shape) == 3:
        output = output[0]  # [H, W]
    
    # Binarize
    probability_map = (output * 255).astype(np.uint8)
    
    # Resize to original image size
    if original_size:
        probability_map = cv2.resize(
            probability_map, 
            (original_size[1], original_size[0]), 
            interpolation=cv2.INTER_LINEAR
        )
    
    return probability_map


def infer_patch_wise(
    session: ort.InferenceSession,
    image: np.ndarray,
    patch_size: int = 384,
    overlap: int = 64,
    batch_size: int = 4
) -> np.ndarray:
    """
    Run patch-wise inference on a full-size image.
    
    Args:
        session: ONNX inference session
        image: Full image as numpy array (H, W, 3) in RGB, float32 [0, 1]
        patch_size: Size of each patch
        overlap: Overlapping pixels between patches
        batch_size: Number of patches to process in parallel
    
    Returns:
        Full-size probability map (H, W) as uint8 [0, 255]
    """
    H, W, C = image.shape
    stride = patch_size - overlap
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape
    
    # Check if model expects fixed batch size
    # Handle both fixed and dynamic batch sizes
    if len(input_shape) > 0:
        first_dim = input_shape[0]
        if first_dim is not None and isinstance(first_dim, (int, str)):
            if isinstance(first_dim, int) and first_dim == 1:
                # Model expects fixed batch size of 1
                batch_size = 1
                print(f"Model expects batch size 1, processing patches individually")
            elif isinstance(first_dim, str) and first_dim.lower() in ['batch', 'batch_size']:
                # Dynamic batch size - try to use provided batch_size
                pass
            elif isinstance(first_dim, int) and first_dim > 1:
                # Model expects specific batch size
                batch_size = min(batch_size, first_dim)
                print(f"Model expects batch size {first_dim}, using batch_size={batch_size}")
    
    # Pad image to fit patch grid
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    img_pad = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    H_pad, W_pad, _ = img_pad.shape
    
    # Initialize accumulation arrays
    full_pred = np.zeros((H_pad, W_pad), dtype=np.float32)
    weight_map = np.zeros((H_pad, W_pad), dtype=np.float32)
    
    # Generate patch coordinates
    ys = list(range(0, H_pad - patch_size + 1, stride))
    xs = list(range(0, W_pad - patch_size + 1, stride))
    
    # Ensure last patch covers the edge
    if ys[-1] != H_pad - patch_size:
        ys.append(H_pad - patch_size)
    if xs[-1] != W_pad - patch_size:
        xs.append(W_pad - patch_size)
    
    # Collect all patches
    all_patches = []
    patch_coords = []
    for y in ys:
        for x in xs:
            patch = img_pad[y:y+patch_size, x:x+patch_size]
            # Convert to CHW and add batch dimension
            patch_tensor = np.transpose(patch, (2, 0, 1))  # [3, H, W]
            patch_tensor = np.expand_dims(patch_tensor, axis=0)  # [1, 3, H, W]
            all_patches.append(patch_tensor)
            patch_coords.append((y, x))
    
    num_patches = len(all_patches)
    print(f"Processing {num_patches} patches in batches of {batch_size}")
    
    # Process patches in batches
    for i in range(0, num_patches, batch_size):
        batch_end = min(i + batch_size, num_patches)
        
        if batch_size == 1:
            # Process one patch at a time
            for j in range(i, batch_end):
                patch_tensor = all_patches[j]
                y, x = patch_coords[j]
                
                # Run inference on single patch
                outputs = session.run(None, {input_name: patch_tensor})
                
                # Handle multiple outputs
                if len(outputs) > 1:
                    output = outputs[1] if len(outputs) > 1 else outputs[0]
                else:
                    output = outputs[0]
                
                # Remove batch/channel dimensions if present
                pred_patch = output[0] if len(output.shape) >= 3 else output
                if len(pred_patch.shape) == 3:
                    pred_patch = pred_patch[0]  # [H, W]
                elif len(pred_patch.shape) == 2:
                    pred_patch = pred_patch  # Already [H, W]
                
                # Accumulate predictions
                full_pred[y:y+patch_size, x:x+patch_size] += pred_patch
                weight_map[y:y+patch_size, x:x+patch_size] += 1.0
        else:
            # Process batch of patches
            batch_patches = all_patches[i:batch_end]
            batch_tensor = np.concatenate(batch_patches, axis=0)  # [B, 3, H, W]
            
            # Run inference
            outputs = session.run(None, {input_name: batch_tensor})
            
            # Handle multiple outputs
            if len(outputs) > 1:
                output_batch = outputs[1] if len(outputs) > 1 else outputs[0]
            else:
                output_batch = outputs[0]
            
            # Process each patch in the batch
            for j in range(batch_tensor.shape[0]):
                y, x = patch_coords[i + j]
                
                # Extract prediction for this patch
                pred_patch = output_batch[j]
                
                # Remove batch/channel dimensions if present
                if len(pred_patch.shape) == 4:
                    pred_patch = pred_patch[0]
                if len(pred_patch.shape) == 3:
                    pred_patch = pred_patch[0]  # [H, W]
                
                # Accumulate predictions
                full_pred[y:y+patch_size, x:x+patch_size] += pred_patch
                weight_map[y:y+patch_size, x:x+patch_size] += 1.0
    
    # Average overlapping regions
    full_pred = np.divide(full_pred, np.maximum(weight_map, 1e-6))
    
    # Crop to original size
    full_pred = full_pred[:H, :W]
    
    # Convert to uint8 probability map
    probability_map = (full_pred * 255).astype(np.uint8)
    
    return probability_map


def infer_single_image(
    session: ort.InferenceSession,
    image_path: str,
    output_path: str = None,
    target_size: tuple = (384, 384),
    use_patch_wise: bool = False,
    patch_size: int = 384,
    overlap: int = 64,
    batch_size: int = 4
) -> np.ndarray:
    """Run inference on a single image."""
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = img.shape[:2]
    img_normalized = img.astype(np.float32) / 255.0
    
    if use_patch_wise:
        # Patch-wise inference
        mask = infer_patch_wise(
            session, 
            img_normalized, 
            patch_size=patch_size,
            overlap=overlap,
            batch_size=batch_size
        )
    else:
        # Standard inference (resize to target size)
        img_tensor, _ = preprocess_image(image_path, target_size)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img_tensor})
        
        # Handle multiple outputs
        if len(outputs) > 1:
            output = outputs[1] if len(outputs) > 1 else outputs[0]
        else:
            output = outputs[0]
        
        # Postprocess
        mask = postprocess_output(output, original_size)
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, mask)
        print(f"Saved prediction to {output_path}")
    
    return mask


def infer_batch_images(
    session: ort.InferenceSession,
    image_dir: str,
    output_dir: str,
    target_size: tuple = (384, 384),
    use_patch_wise: bool = False,
    patch_size: int = 384,
    overlap: int = 64,
    batch_size: int = 4
) -> None:
    """Run inference on all images in a directory."""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_paths = [p for p in image_dir.iterdir() 
                   if p.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_paths)} images")
    
    for i, img_path in enumerate(image_paths):
        print(f"Processing {i+1}/{len(image_paths)}: {img_path.name}")
        output_path = output_dir / f"{img_path.stem}_prediction.png"
        
        try:
            infer_single_image(
                session, 
                str(img_path), 
                str(output_path), 
                target_size,
                use_patch_wise=use_patch_wise,
                patch_size=patch_size,
                overlap=overlap,
                batch_size=batch_size
            )
        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with ONNX model")
    parser.add_argument('--onnx', required=True, help='Path to ONNX model file')
    parser.add_argument('--input', required=True, help='Input image file or directory')
    parser.add_argument('--output', help='Output image file or directory')
    parser.add_argument('--height', type=int, default=384, help='Input height (for standard inference)')
    parser.add_argument('--width', type=int, default=384, help='Input width (for standard inference)')
    parser.add_argument('--cpu-only', action='store_true', help='Use CPU only (disable GPU)')
    parser.add_argument('--patch-wise', action='store_true', help='Use patch-wise inference for full-size images')
    parser.add_argument('--patch-size', type=int, default=384, help='Patch size for patch-wise inference')
    parser.add_argument('--overlap', type=int, default=64, help='Overlap between patches')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for processing patches')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Setup providers
    if args.cpu_only:
        providers = ['CPUExecutionProvider']
    else:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    # Load model
    session = load_onnx_model(args.onnx, providers)
    
    # Get target size from model input or args
    input_shape = session.get_inputs()[0].shape
    if len(input_shape) >= 3:
        target_size = (input_shape[2], input_shape[3]) if input_shape[2] and input_shape[3] else (args.height, args.width)
    else:
        target_size = (args.height, args.width)
    
    if args.patch_wise:
        print(f"Using patch-wise inference (patch_size={args.patch_size}, overlap={args.overlap}, batch_size={args.batch_size})")
    else:
        print(f"Using standard inference (input size: {target_size})")
    
    # Run inference
    input_path = Path(args.input)
    if input_path.is_file():
        # Single image
        output_path = args.output or str(input_path.parent / f"{input_path.stem}_prediction.png")
        infer_single_image(
            session, 
            str(input_path), 
            output_path, 
            target_size,
            use_patch_wise=args.patch_wise,
            patch_size=args.patch_size,
            overlap=args.overlap,
            batch_size=args.batch_size
        )
    elif input_path.is_dir():
        # Directory of images
        output_dir = args.output or str(input_path.parent / "predictions")
        infer_batch_images(
            session, 
            str(input_path), 
            output_dir, 
            target_size,
            use_patch_wise=args.patch_wise,
            patch_size=args.patch_size,
            overlap=args.overlap,
            batch_size=args.batch_size
        )
    else:
        raise ValueError(f"Input path does not exist: {args.input}")


if __name__ == '__main__':
    main()

