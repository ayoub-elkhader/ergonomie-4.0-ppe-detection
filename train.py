"""Train the PPE detector (model selection uses the VALIDATION split only).

The provided weights/best.pt was trained from scratch:
    python train.py --model yolov8s.yaml --epochs 200 --name scratch

Comparison with COCO-pretrained weights, same data, seed and budget:
    python train.py --model yolov8s.pt --epochs 100 --name pretrained
    python train.py --model yolov8s.yaml --epochs 100 --name scratch100

Compare the runs on the validation split (runs/<name>/results.csv, or
`python scripts/evaluate.py --splits val --model runs/<name>/weights/best.pt`),
choose ONE model, and only then evaluate it once on the test split:
    python scripts/evaluate.py --splits test --model runs/<name>/weights/best.pt

A GPU is strongly recommended (e.g. Google Colab). On a laptop CPU one epoch
takes several minutes.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "data.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="yolov8s.yaml",
                        help="yolov8s.yaml = from scratch, yolov8s.pt = COCO-pretrained")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="'cpu', '0', ... (auto by default)")
    parser.add_argument("--name", default="ppe")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(data=str(DATA), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                seed=args.seed, deterministic=True, device=args.device,
                project=str(ROOT / "runs"), name=args.name, exist_ok=False)
    # Ultralytics validates on the 'val' split during training; the test split is never used here.
    print(f"Done. Compare runs on the validation split before touching the test split.")


if __name__ == "__main__":
    main()
