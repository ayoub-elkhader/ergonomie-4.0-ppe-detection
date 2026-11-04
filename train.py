"""Train or evaluate the PPE detector on the soudage_ia dataset.

Examples
--------
Evaluate the provided weights on the test split:
    python train.py --evaluate

Re-train from scratch (GPU strongly recommended, e.g. Google Colab):
    python train.py --epochs 200 --model yolov8s.yaml
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "data.yaml"
WEIGHTS = ROOT / "weights" / "best.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluate", action="store_true",
                        help="only evaluate weights/best.pt on the val and test splits")
    parser.add_argument("--model", default="yolov8s.yaml",
                        help="model config (from scratch) or .pt checkpoint (fine-tuning)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="'cpu', '0', ... (auto by default)")
    args = parser.parse_args()

    if args.evaluate:
        model = YOLO(str(WEIGHTS))
        for split in ("val", "test"):
            m = model.val(data=str(DATA), split=split, imgsz=args.imgsz,
                          device=args.device, plots=False).box
            print(f"[{split}] mAP50={m.map50:.3f}  mAP50-95={m.map:.3f}  "
                  f"precision={m.mp:.3f}  recall={m.mr:.3f}")
        return

    model = YOLO(args.model)
    model.train(data=str(DATA), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device,
                project=str(ROOT / "runs"), name="ppe")
    print("Done. Copy runs/ppe*/weights/best.pt to weights/best.pt to use it in the app.")


if __name__ == "__main__":
    main()
