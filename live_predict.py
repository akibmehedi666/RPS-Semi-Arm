"""
Real-Time Ultra-Low Latency Invincible Rock-Paper-Scissors System.
Optimized for Laptop & Raspberry Pi 5 (ARM64, Debian Bookworm).

Pipeline:
  Webcam Frame (USB V4L2) -> Temporal Differencing |F_t - F_{t-1}| ->
  Binary Threshold @ 18 -> 64x64 Mask ->
  Inference Engine (PyTorch CPU / ONNX Runtime) ->
  5-Frame Temporal Majority Filter ->
  Invincible Counter-Move Decision Engine
"""

import os
import sys
import time
import argparse
import platform
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import RoshamboNet, MajorityVote, CLASS_NAMES, LABEL_TO_SYMBOL, COUNTER_MOVES
from pseudo_event_producer import PseudoEventProducer


class InvincibleRoshamboEngine:
    def __init__(
        self,
        model_path: str = "motion_model.pth",
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        threshold_value: int = 18,
        window_length: int = 5,
        threads: int = 4,
        latency_budget_ms: float = 16.0,
        use_onnx: bool = False,
        onnx_model_path: str = "motion_model.onnx",
        mock_mode: bool = False,
    ):
        self.device = torch.device("cpu")
        self.threads = threads
        self.latency_budget_ms = latency_budget_ms
        self.use_onnx = use_onnx
        self.onnx_model_path = onnx_model_path
        self.last_latency_warning_time = 0.0

        # 1. Configure CPU threading (crucial for Raspberry Pi 5's 4 Cortex-A76 cores)
        torch.set_num_threads(self.threads)
        print(f"[InvincibleRoshamboEngine] System: {platform.system()} ({platform.machine()})")
        print(f"[InvincibleRoshamboEngine] CPU thread allocation: {self.threads}")

        # 2. Initialize Inference Engine (ONNX Runtime vs PyTorch)
        if self.use_onnx:
            self._init_onnx_runtime()
        else:
            self._init_pytorch_engine(model_path)

        # 3. Temporal Majority Voter (ported from dextra-roshambo-python consumer.py)
        self.voter = MajorityVote(window_length=window_length, num_classes=len(CLASS_NAMES))

        # 4. Pseudo-Event Motion Producer (with V4L2 USB camera backend)
        self.producer = PseudoEventProducer(
            camera_index=camera_index,
            width=width,
            height=height,
            fps=fps,
            threshold_value=threshold_value,
            mock_mode=mock_mode,
        )

        # Performance & Game State Tracking
        self.stats = {
            "inference_times_ms": [],
            "loop_times_ms": [],
            "user_wins": 0,
            "ai_wins": 0,
            "ties": 0,
        }

    def _init_pytorch_engine(self, model_path: str):
        """Initializes native PyTorch model and warms up CPU execution."""
        print(f"[InvincibleRoshamboEngine] Engine: PyTorch CPU (Eager)")
        self.model = RoshamboNet(num_classes=len(CLASS_NAMES), pooling="avg")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"[InvincibleRoshamboEngine] Loaded PyTorch weights: {os.path.abspath(model_path)}")
        else:
            print(f"[InvincibleRoshamboEngine] WARNING: Weights '{model_path}' not found! Running uninitialized.")

        self.model.to(self.device)
        self.model.eval()

        # Warm up JIT/CPU caches
        dummy = torch.randn(1, 1, 64, 64, device=self.device)
        with torch.inference_mode():
            for _ in range(10):
                _ = self.model(dummy)
        print("[InvincibleRoshamboEngine] PyTorch warm-up complete.")

    def _init_onnx_runtime(self):
        """Initializes ONNX Runtime session with ARM NEON / intra-op thread tuning."""
        print(f"[InvincibleRoshamboEngine] Engine: ONNX Runtime (Optimized for ARM Cortex-A76 / Pi 5)")
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is not installed. To run with --use_onnx, run:\n"
                "  pip install onnxruntime\n"
                "or run with native PyTorch (omit --use_onnx)."
            )

        if not os.path.exists(self.onnx_model_path):
            raise FileNotFoundError(
                f"ONNX model file '{self.onnx_model_path}' not found.\n"
                f"Export it from your trained weights using:\n"
                f"  python3 export_onnx.py --model motion_model.pth --output {self.onnx_model_path}"
            )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.ort_session = ort.InferenceSession(
            self.onnx_model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.ort_input_name = self.ort_session.get_inputs()[0].name
        print(f"[InvincibleRoshamboEngine] Loaded ONNX model: {os.path.abspath(self.onnx_model_path)}")

        # Warm up ONNX Runtime
        dummy_np = np.zeros((1, 1, 64, 64), dtype=np.float32)
        for _ in range(10):
            _ = self.ort_session.run(None, {self.ort_input_name: dummy_np})
        print("[InvincibleRoshamboEngine] ONNX Runtime warm-up complete.")

    def predict_mask(self, mask_64: np.ndarray):
        """
        Runs model inference on a single 64x64 uint8 motion mask.
        Supports both PyTorch and ONNX Runtime backends.
        Returns: (pred_label, confidence, inference_ms)
        """
        t_start = time.perf_counter()

        if self.use_onnx:
            # Preprocess: uint8 [0, 255] -> float32 [0.0, 1.0], shape (1, 1, 64, 64)
            inp = (mask_64.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]
            ort_outs = self.ort_session.run(None, {self.ort_input_name: inp})
            logits = ort_outs[0][0]
            # Numerically stable softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            pred_label = int(np.argmax(probs))
            confidence = float(probs[pred_label])
        else:
            # PyTorch Eager inference
            tensor_in = torch.from_numpy(mask_64).float().div_(255.0).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                logits = self.model(tensor_in)
                probs = F.softmax(logits, dim=1)[0]
                pred_label = int(torch.argmax(probs).item())
                confidence = float(probs[pred_label].item())

        inference_ms = (time.perf_counter() - t_start) * 1000.0
        return pred_label, confidence, inference_ms

    def run(self, benchmark_frames: int = 0):
        """
        Main real-time loop.
        If benchmark_frames > 0 or running in headless SSH environment, skips GUI window.
        """
        is_headless = (
            benchmark_frames > 0 or
            (platform.system() == "Linux" and os.environ.get("DISPLAY") is None and os.environ.get("WAYLAND_DISPLAY") is None)
        )

        window_name = "Invincible Rock-Paper-Scissors (Webcam Pseudo-Events)"
        if not is_headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        else:
            print("[InvincibleRoshamboEngine] Running in HEADLESS mode (no GUI display).")

        print("\n========================================================")
        print("  INVINCIBLE ROCK-PAPER-SCISSORS SYSTEM ACTIVE          ")
        print("========================================================")
        print(f"  Backend: {'ONNX Runtime' if self.use_onnx else 'PyTorch CPU'}")
        print(f"  Target FPS: {self.producer.fps} | Latency Budget: {self.latency_budget_ms:.1f}ms")
        print("  Controls:")
        print("    [Q] / ESC -> Exit")
        print("    [R]       -> Reset Win/Streak Counters")
        print("========================================================\n")

        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0.0
        frame_idx = 0

        last_stable_label = 3
        last_announced_move = None
        consecutive_wins = 0

        try:
            while True:
                loop_start = time.perf_counter()
                ret, raw_frame, mask_64 = self.producer.get_motion_mask()
                if not ret or raw_frame is None:
                    time.sleep(0.005)
                    continue

                frame_idx += 1

                # 1. Real-time Inference (<1ms target)
                raw_label, confidence, inf_ms = self.predict_mask(mask_64)
                self.stats["inference_times_ms"].append(inf_ms)
                if len(self.stats["inference_times_ms"]) > 100:
                    self.stats["inference_times_ms"].pop(0)

                # 2. 5-Frame Majority Vote Filter
                voted_label = self.voter.new_prediction_and_vote(raw_label)
                if voted_label is None:
                    active_label = last_stable_label
                else:
                    active_label = voted_label
                    last_stable_label = voted_label

                # 3. Invincible Counter Decision
                counter_info = COUNTER_MOVES[active_label]
                user_symbol = LABEL_TO_SYMBOL[active_label]
                ai_symbol = counter_info["ai_symbol"]
                match_desc = counter_info["desc"]

                if active_label != 3 and active_label != last_announced_move:
                    consecutive_wins += 1
                    self.stats["ai_wins"] += 1
                    last_announced_move = active_label
                    if is_headless:
                        print(f"[MOVE] User: {user_symbol.upper()} -> AI plays: {ai_symbol.upper()} ({match_desc}) | Streak: {consecutive_wins}")

                # FPS calculation
                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    current_fps = fps_counter / (time.time() - fps_timer)
                    fps_counter = 0
                    fps_timer = time.time()

                # Check frame processing latency against budget
                frame_loop_ms = (time.perf_counter() - loop_start) * 1000.0
                self.stats["loop_times_ms"].append(frame_loop_ms)
                if len(self.stats["loop_times_ms"]) > 100:
                    self.stats["loop_times_ms"].pop(0)

                if frame_loop_ms > self.latency_budget_ms:
                    now = time.time()
                    if now - self.last_latency_warning_time > 2.0:
                        print(
                            f"[PERF WARNING] Frame latency: {frame_loop_ms:.1f}ms exceeds budget "
                            f"({self.latency_budget_ms:.1f}ms). High processing overhead detected on Pi/CPU.",
                            file=sys.stderr
                        )
                        self.last_latency_warning_time = now

                # Benchmarking termination check
                if benchmark_frames > 0 and frame_idx >= benchmark_frames:
                    print(f"[Benchmark] Completed {frame_idx} test frames.")
                    break

                if is_headless:
                    continue

                # 4. Render Rich HUD & Visualizer (Desktop only)
                display = raw_frame.copy()
                dh, dw, _ = display.shape

                # Center ROI outline
                min_dim = min(dh, dw)
                sy = (dh - min_dim) // 2
                sx = (dw - min_dim) // 2
                cv2.rectangle(display, (sx, sy), (sx + min_dim, sy + min_dim), (80, 80, 80), 1)

                # PiP Preview: Scaled 64x64 Motion Mask
                mask_display = cv2.resize(mask_64, (150, 150), interpolation=cv2.INTER_NEAREST)
                mask_display_bgr = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(mask_display_bgr, (0, 0), (149, 149), (255, 255, 0), 2)
                display[15:165, dw - 165 : dw - 15] = mask_display_bgr
                cv2.putText(display, "64x64 Motion Mask", (dw - 160, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

                # Top Banner: System Title & Latency
                avg_inf_ms = np.mean(self.stats["inference_times_ms"]) if self.stats["inference_times_ms"] else 0.0
                backend_str = "ONNX" if self.use_onnx else "PyTorch"
                cv2.rectangle(display, (0, 0), (dw, 55), (20, 20, 20), -1)
                cv2.putText(display, f"INVINCIBLE ROSHAMBO ({backend_str})", (15, 25),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
                latency_color = (0, 255, 0) if avg_inf_ms < 3.0 else (0, 165, 255)
                cv2.putText(display, f"Inference: {avg_inf_ms:.2f}ms | Loop: {frame_loop_ms:.1f}ms | FPS: {current_fps:.1f}",
                            (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, latency_color, 1)

                # Player vs AI Decision Card
                card_y = dh - 120
                cv2.rectangle(display, (15, card_y), (dw - 15, dh - 15), (15, 15, 15), -1)
                cv2.rectangle(display, (15, card_y), (dw - 15, dh - 15), (60, 60, 60), 1)

                # User Move
                user_text = f"YOU: {user_symbol.upper()} (conf: {confidence*100:.1f}%)"
                cv2.putText(display, user_text, (30, card_y + 35),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (200, 200, 200), 1)

                # AI Invincible Counter Move
                if active_label != 3:
                    ai_text = f"AI COUNTER: {ai_symbol.upper()} -> {match_desc}"
                    cv2.putText(display, ai_text, (30, card_y + 70),
                                cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)
                    cv2.putText(display, f"INVINCIBLE STREAK: {consecutive_wins}", (30, card_y + 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                else:
                    cv2.putText(display, "AI: Waiting for hand motion...", (30, card_y + 70),
                                cv2.FONT_HERSHEY_DUPLEX, 0.65, (120, 120, 120), 1)

                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF

                if key in [ord('q'), 27]:
                    break
                elif key == ord('r'):
                    consecutive_wins = 0
                    self.stats["ai_wins"] = 0
                    print("[InvincibleRoshamboEngine] Streak reset to 0.")

        finally:
            self.producer.release()
            cv2.destroyAllWindows()
            self._print_summary()

    def _print_summary(self):
        times = self.stats["inference_times_ms"]
        loops = self.stats["loop_times_ms"]
        if times:
            print("\n================ LATENCY & PERFORMANCE SUMMARY ================")
            print(f"  Backend:                  {'ONNX Runtime' if self.use_onnx else 'PyTorch CPU'}")
            print(f"  CPU Threads Configured:   {self.threads}")
            print(f"  Mean Inference Latency:   {np.mean(times):.2f} ms")
            print(f"  Median Inference Latency: {np.median(times):.2f} ms")
            print(f"  95th Percentile Latency:  {np.percentile(times, 95):.2f} ms")
            if loops:
                print(f"  Mean Total Frame Loop:    {np.mean(loops):.2f} ms (Budget: {self.latency_budget_ms:.1f} ms)")
            print(f"  Sub-3ms Target Met:       {'YES [PASS]' if np.mean(times) < 3.0 else 'NO'}")
            print(f"  Total Counter Moves:      {self.stats['ai_wins']}")
            print("===============================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Invincible Rock-Paper-Scissors (Laptop & Raspberry Pi 5)")
    parser.add_argument("--model", type=str, default="motion_model.pth", help="Path to PyTorch motion_model.pth")
    parser.add_argument("--camera", type=int, default=0, help="USB webcam device index (e.g. 0 or 2)")
    parser.add_argument("--width", type=int, default=640, help="Webcam width")
    parser.add_argument("--height", type=int, default=480, help="Webcam height")
    parser.add_argument("--fps", type=int, default=30, help="Webcam target FPS (30 recommended for Pi 5 USB cameras)")
    parser.add_argument("--threshold", type=int, default=18, help="Motion differencing threshold (15-20 recommended)")
    parser.add_argument("--window_length", type=int, default=5, help="Majority vote temporal window size")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 4), help="Number of CPU inference threads (default: 4 for Pi 5)")
    parser.add_argument("--latency_budget_ms", type=float, default=16.0, help="Warning threshold for per-frame processing latency in ms")
    parser.add_argument("--use_onnx", action="store_true", help="Run inference with ONNX Runtime instead of eager PyTorch")
    parser.add_argument("--onnx_model", type=str, default="motion_model.onnx", help="Path to exported ONNX model")
    parser.add_argument("--mock", action="store_true", help="Simulate camera frames for testing without hardware")
    parser.add_argument("--benchmark", type=int, default=0, help="Run headless benchmark for N frames and exit")
    args = parser.parse_args()

    engine = InvincibleRoshamboEngine(
        model_path=args.model,
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        threshold_value=args.threshold,
        window_length=args.window_length,
        threads=args.threads,
        latency_budget_ms=args.latency_budget_ms,
        use_onnx=args.use_onnx,
        onnx_model_path=args.onnx_model,
        mock_mode=args.mock,
    )
    engine.run(benchmark_frames=args.benchmark)
