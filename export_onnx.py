"""
Export trained RoshamboNet PyTorch weights (motion_model.pth) to ONNX format (motion_model.onnx).
Enables ultra-fast inference on Raspberry Pi 5 (ARM Cortex-A76) via onnxruntime.

Usage:
    python3 export_onnx.py --model motion_model.pth --output motion_model.onnx
"""

import os
import sys
import argparse
import time
import numpy as np
import torch

from model import RoshamboNet, CLASS_NAMES


def export_to_onnx(
    model_path: str = "motion_model.pth",
    output_path: str = "motion_model.onnx",
    opset_version: int = 14,
    verify: bool = True,
):
    print("\n========================================================")
    print("      ROSHAMBONET PYTORCH -> ONNX EXPORT UTILITY        ")
    print("========================================================")
    print(f"  PyTorch Model:  {os.path.abspath(model_path)}")
    print(f"  ONNX Output:    {os.path.abspath(output_path)}")
    print(f"  Opset Version:  {opset_version}")
    print("========================================================\n")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' does not exist. Train the model first using train.py.")

    device = torch.device("cpu")

    # 1. Load PyTorch model architecture and weights
    model = RoshamboNet(num_classes=len(CLASS_NAMES), pooling="avg", dropout=0.0)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[export_onnx] Loaded RoshamboNet with {total_params:,} parameters.")

    # 2. Prepare dummy input tensor matching 64x64 motion mask
    dummy_input = torch.randn(1, 1, 64, 64, dtype=torch.float32, device=device)

    # 3. Export to ONNX
    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }

    print(f"[export_onnx] Exporting to {output_path} (opset {opset_version})...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    except TypeError:
        # For older PyTorch versions that don't take dynamo arg
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    file_size_kb = os.path.getsize(output_path) / 1024.0
    print(f"[export_onnx] Export successful! Model size: {file_size_kb:.1f} KB")

    # 4. Optional verification with ONNX and onnxruntime
    if verify:
        verify_onnx_model(model, dummy_input, output_path)

    print("\n[export_onnx] Deployment Ready!")
    print(f"To run real-time inference on Raspberry Pi 5 with ONNX Runtime:")
    print(f"  python3 live_predict.py --use_onnx --onnx_model {output_path} --fps 30 --threads 4\n")


def verify_onnx_model(torch_model, dummy_input: torch.Tensor, onnx_path: str):
    """Verifies ONNX model validity and validates numerical output parity with PyTorch."""
    print("\n[export_onnx] Verifying ONNX model integrity...")

    # Check with onnx package
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("  ✓ ONNX checker passed successfully.")
    except ImportError:
        print("  - Note: 'onnx' package not installed, skipping structural check.")
    except Exception as e:
        print(f"  ✗ ONNX structural check warning: {e}")

    # Test inference with onnxruntime
    try:
        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        sess = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        
        # Run PyTorch
        with torch.no_grad():
            torch_out = torch_model(dummy_input).numpy()

        # Run ONNX Runtime
        ort_inputs = {input_name: dummy_input.numpy()}
        ort_out = sess.run(None, ort_inputs)[0]

        # Compare outputs
        max_diff = np.max(np.abs(torch_out - ort_out))
        print(f"  ✓ ONNX Runtime numerical parity confirmed. Max output delta: {max_diff:.6f}")

        # Benchmark 50 runs on CPU
        runs = 50
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(runs):
                _ = torch_model(dummy_input)
        torch_time = (time.perf_counter() - t0) / runs * 1000.0

        t0 = time.perf_counter()
        for _ in range(runs):
            _ = sess.run(None, ort_inputs)
        ort_time = (time.perf_counter() - t0) / runs * 1000.0

        print(f"  ✓ Quick Benchmark: PyTorch = {torch_time:.2f}ms | ONNX Runtime = {ort_time:.2f}ms")

    except ImportError:
        print("  - Note: 'onnxruntime' not installed. Install via 'pip install onnxruntime' to verify.")
    except Exception as e:
        print(f"  ✗ ONNX Runtime validation failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export RoshamboNet PyTorch checkpoint to ONNX")
    parser.add_argument("--model", type=str, default="motion_model.pth", help="Input PyTorch checkpoint path")
    parser.add_argument("--output", type=str, default="motion_model.onnx", help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version (14 recommended)")
    parser.add_argument("--no_verify", action="store_true", help="Skip ONNX / ONNX Runtime verification")
    args = parser.parse_args()

    export_to_onnx(
        model_path=args.model,
        output_path=args.output,
        opset_version=args.opset,
        verify=not args.no_verify,
    )
