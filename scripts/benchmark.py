"""Measure the processing speed of each stage on a video.

Usage:
    python scripts/benchmark.py [--video videos/demo_atelier.mp4] [--frames 150]
                                [--max-side 720] [--device cpu]

Writes reports/benchmark.md. Timings exclude video decoding and Streamlit
display, so the speed seen in the app is lower.
"""
import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402

import helper  # noqa: E402
import settings  # noqa: E402
from posture import PostureEstimator  # noqa: E402


def cpu_name():
    """Readable CPU model (platform.processor() is cryptic on Windows)."""
    try:
        if platform.system() == "Windows":
            import subprocess
            out = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True, text=True).stdout
            names = [l.strip() for l in out.splitlines()[1:] if l.strip()]
            if names:
                return names[0]
        elif Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or platform.machine()


def timed(fn, frames, warmup=5):
    for f in frames[:warmup]:
        fn(f)
    out = []
    for f in frames:
        t = time.perf_counter()
        fn(f)
        out.append((time.perf_counter() - t) * 1000)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(ROOT / "videos" / "demo_atelier.mp4"))
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--max-side", type=int, default=720)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    frames = []
    while len(frames) < args.frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(helper.fit_frame(f, args.max_side))
    cap.release()
    h, w = frames[0].shape[:2]

    ppe_model = YOLO(str(settings.DETECTION_MODEL))
    person_model = YOLO(str(settings.PERSON_MODEL))
    pose = PostureEstimator(static_image=False)
    proc = helper.FrameProcessor(ppe_model, person_model, settings.DEFAULT_DECISION_CONFIDENCE,
                                 True, True, max_side=args.max_side)
    stages = {
        "PPE detector (YOLOv8s)": lambda f: ppe_model.predict(f, conf=0.25, device=args.device, verbose=False),
        "Person counter (YOLOv8n)": lambda f: person_model.predict(f, classes=[0], device=args.device, verbose=False),
        "Pose (MediaPipe)": lambda f: pose.estimate(f),
        "Full pipeline (PPE + people + posture + drawing)": lambda f: proc(f),
    }
    results = {name: timed(fn, frames) for name, fn in stages.items()}
    pose.close()
    proc.close()

    import torch
    import ultralytics
    lines = ["# Processing speed benchmark\n",
             f"Command: `python scripts/benchmark.py {' '.join(sys.argv[1:])}`\n",
             f"- Hardware: {cpu_name()}, "
             f"{torch.get_num_threads()} torch threads, CUDA: {torch.cuda.is_available()}",
             f"- Software: Python {platform.python_version()}, ultralytics {ultralytics.__version__}, "
             f"torch {torch.__version__}, OS {platform.platform()}",
             f"- Input: `{Path(args.video).name}`, {len(frames)} frames resized to {w}×{h}, "
             "YOLO inference at imgsz 640, device " + args.device,
             "- Excludes video decoding and Streamlit display.\n",
             "| Stage | median ms/frame | p90 ms/frame | frames/s (median) |",
             "|---|--:|--:|--:|"]
    for name, ms in results.items():
        med = statistics.median(ms)
        p90 = sorted(ms)[int(0.9 * (len(ms) - 1))]
        lines.append(f"| {name} | {med:.0f} | {p90:.0f} | {1000 / med:.1f} |")
    out = ROOT / "reports" / "benchmark.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
