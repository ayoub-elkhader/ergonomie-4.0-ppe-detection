"""Reproducible evaluation of the PPE detector (and of the app's decision).

Usage:
    python scripts/evaluate.py [--splits val test] [--model weights/best.pt]
                               [--conf 0.5] [--device cpu] [--out reports/evaluation]

Outputs (in --out):
    evaluation.md / evaluation.json   environment, model and data fingerprints,
                                      COCO-style metrics, per-class results with
                                      supports, results at the app threshold,
                                      decision-level agreement
    <split>/                          Ultralytics plots: confusion matrices,
                                      PR / P / R / F1 curves
    <split>/examples/                 images of false positives / negatives
                                      (contain people: NOT for publication)

mAP is a ranking metric over all score thresholds; it is not the percentage
of correct compliance decisions. The decision-level table compares the app's
decision computed from predictions with the same rule applied to the ground
truth labels, which may mix several people in one image.
"""
import argparse
import hashlib
import json
import platform
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

import ppe  # noqa: E402
import settings  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_dirs(data_yaml, split):
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    img_dir = (Path(data_yaml).resolve().parent / cfg[split]).resolve()
    return img_dir, img_dir.parent / "labels"


def fingerprint(img_dir, lbl_dir):
    """Hash of file names + label contents: identifies the exact split used."""
    h = hashlib.md5()
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    for p in images:
        h.update(p.name.encode())
        lbl = lbl_dir / (p.stem + ".txt")
        h.update(lbl.read_bytes() if lbl.exists() else b"<missing>")
    return h.hexdigest(), len(images)


def read_gt(lbl_path, w, h):
    boxes = []
    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                c, x, y, bw, bh = int(float(parts[0])), *map(float, parts[1:])
                boxes.append((c, np.array([(x - bw / 2) * w, (y - bh / 2) * h,
                                           (x + bw / 2) * w, (y + bh / 2) * h])))
    return boxes


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts, iou_thr=0.5):
    """Greedy per-class matching. preds: [(cls, box, score)], gts: [(cls, box)].
    Returns lists of TP preds, FP preds, FN gts."""
    tp, fp, used = [], [], set()
    for p in sorted(preds, key=lambda p: -p[2]):
        best, best_j = 0.0, None
        for j, g in enumerate(gts):
            if j in used or g[0] != p[0]:
                continue
            v = iou(p[1], g[1])
            if v > best:
                best, best_j = v, j
        if best_j is not None and best >= iou_thr:
            used.add(best_j)
            tp.append(p)
        else:
            fp.append(p)
    fn = [g for j, g in enumerate(gts) if j not in used]
    return tp, fp, fn


def environment():
    import torch
    import ultralytics
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "ultralytics": ultralytics.__version__,
            "torch": torch.__version__, "opencv": cv2.__version__, "numpy": np.__version__,
            "cuda": torch.cuda.is_available()}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
        info["cpu_count"] = psutil.cpu_count(logical=True)
    except ImportError:
        pass
    return info


def evaluate_split(model, person_model, names, data_yaml, split, conf, device, out_dir,
                   max_examples=12):
    img_dir, lbl_dir = split_dirs(data_yaml, split)
    split_out = out_dir / split
    if split_out.exists():
        shutil.rmtree(split_out)
    split_out.mkdir(parents=True)

    # 1) Ultralytics COCO-style evaluation (all thresholds), with plots
    m = model.val(data=str(data_yaml), split=split, imgsz=640, batch=8, device=device,
                  plots=True, project=str(out_dir / "_ultralytics"), name=split,
                  exist_ok=True, verbose=False)
    for f in (out_dir / "_ultralytics" / split).glob("*.png"):
        if "batch" not in f.name:  # batch previews show people
            shutil.copy(f, split_out / f.name)
    shutil.rmtree(out_dir / "_ultralytics" / split, ignore_errors=True)

    box = m.box
    per_class = {}
    for i, c in enumerate(box.ap_class_index):
        per_class[names[int(c)]] = {"precision": float(box.p[i]), "recall": float(box.r[i]),
                                    "mAP50": float(box.ap50[i]), "mAP50_95": float(box.ap[i])}

    # 2) Own pass at the app threshold: supports, TP/FP/FN, decisions
    support_inst, support_img = Counter(), Counter()
    counts = defaultdict(Counter)
    fp_examples, fn_examples = [], []
    decisions = Counter()
    persons_hist = Counter()
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    for path in images:
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        gts = read_gt(lbl_dir / (path.stem + ".txt"), w, h)
        for c in {g[0] for g in gts}:
            support_img[names[c]] += 1
        for g in gts:
            support_inst[names[g[0]]] += 1

        res = model.predict(img, conf=settings.UNCERTAIN_CONFIDENCE, device=device, verbose=False)[0]
        preds_all = [(int(c), b.numpy(), float(s)) for c, b, s in
                     zip(res.boxes.cls, res.boxes.xyxy.cpu(), res.boxes.conf.cpu())]
        preds = [p for p in preds_all if p[2] >= conf]
        tp, fp, fn = match(preds, gts)
        for p in tp:
            counts[names[p[0]]]["tp"] += 1
        for p in fp:
            counts[names[p[0]]]["fp"] += 1
            fp_examples.append((p[2], path.name, names[p[0]], p[1]))
        for g in fn:
            counts[names[g[0]]]["fn"] += 1
            fn_examples.append((path.name, names[g[0]], g[1]))

        n_persons = ppe.count_persons(person_model.predict(
            img, classes=[settings.PERSON_CLASS_ID], conf=settings.PERSON_CONFIDENCE,
            device=device, verbose=False)[0])
        persons_hist[n_persons] += 1
        pred_dec = ppe.check_ppe([ppe.Detection(names[c], s) for c, _, s in preds_all],
                                 n_persons, conf)["overall"]
        gt_dec = ppe.check_ppe([ppe.Detection(names[g[0]], 1.0) for g in gts],
                               n_persons, conf)["overall"]
        decisions[(gt_dec, pred_dec)] += 1

    at_thr = {}
    for n in names:
        c = counts[n]
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None
        at_thr[n] = {"tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "precision": p, "recall": r}

    # 3) Example images (gitignored)
    ex_dir = split_out / "examples"
    ex_dir.mkdir()
    fp_examples.sort(key=lambda e: -e[0])
    for k, (score, name, cls, b) in enumerate(fp_examples[:max_examples]):
        _save_example(img_dir / name, b, f"FP {cls} {score:.2f}", (0, 0, 255), ex_dir / f"fp_{k:02d}_{cls}.jpg")
    for k, (name, cls, b) in enumerate(fn_examples[:max_examples]):
        _save_example(img_dir / name, b, f"FN {cls}", (0, 200, 255), ex_dir / f"fn_{k:02d}_{cls}.jpg")

    fp_list = [{"image": n, "class": c, "score": round(s, 3)} for s, n, c, _ in fp_examples]
    fn_list = [{"image": n, "class": c} for n, c, _ in fn_examples]
    return {
        "images": len(images),
        "metrics": {"precision": float(box.mp), "recall": float(box.mr),
                    "mAP50": float(box.map50), "mAP50_95": float(box.map)},
        "per_class": per_class,
        "support_instances": dict(support_inst), "support_images": dict(support_img),
        "at_app_threshold": at_thr,
        "false_positives": fp_list, "false_negatives": fn_list,
        "decisions": {f"{g} -> {p}": n for (g, p), n in sorted(decisions.items())},
        "persons_per_image": {str(k): v for k, v in sorted(persons_hist.items(), key=lambda x: str(x[0]))},
    }


def _save_example(img_path, box, text, color, out):
    img = cv2.imread(str(img_path))
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(str(out), img)


def fmt(v):
    return "–" if v is None else f"{v:.3f}"


def to_markdown(r):
    lines = [f"# Evaluation of the PPE detector\n\nGenerated {r['generated']} with "
             f"`python scripts/evaluate.py {' '.join(r['args'])}`.\n"]
    e, m = r["environment"], r["model"]
    lines.append("## Setup\n")
    lines.append(f"- Environment: Python {e['python']}, ultralytics {e['ultralytics']}, torch {e['torch']}, "
                 f"OpenCV {e['opencv']}, CUDA: {e['cuda']}; {e['platform']}; {e.get('processor','')}")
    lines.append(f"- Model: `{m['path']}` (MD5 `{m['md5']}`), saved {m['saved']} by ultralytics "
                 f"{m['ultralytics_version']}; trained from `{m['train_args'].get('model')}`, "
                 f"pretrained={m['train_args'].get('pretrained')}, epochs={m['train_args'].get('epochs')}, "
                 f"imgsz={m['train_args'].get('imgsz')}, seed={m['train_args'].get('seed')}. "
                 "No training log is available.")
    lines.append(f"- Data: `{r['data']['yaml']}` (MD5 `{r['data']['yaml_md5']}`); split fingerprints: "
                 + ", ".join(f"{s} `{v['fingerprint'][:12]}` ({v['images']} images)"
                             for s, v in r["data"]["splits"].items()))
    lines.append(f"- Validation settings: imgsz 640, Ultralytics defaults (conf 0.001, IoU 0.7 NMS). "
                 f"App threshold for the second table: score ≥ {r['conf']}, IoU ≥ 0.5.\n")

    for split, s in r["splits"].items():
        mm = s["metrics"]
        lines.append(f"## Split `{split}` ({s['images']} images)\n")
        lines.append(f"| Precision | Recall | mAP50 | mAP50-95 |\n|--:|--:|--:|--:|\n"
                     f"| {mm['precision']:.3f} | {mm['recall']:.3f} | {mm['mAP50']:.3f} | {mm['mAP50_95']:.3f} |\n")
        lines.append("Precision/recall are means over classes at the F1-optimal threshold chosen by "
                     "Ultralytics. mAP is not a percentage of correct decisions.\n")
        lines.append("| Class | Instances (images) | P | R | AP50 | AP50-95 | TP / FP / FN @app | P @app | R @app |\n"
                     "|---|--:|--:|--:|--:|--:|--:|--:|--:|")
        for n in r["classes"]:
            pc = s["per_class"].get(n, {})
            a = s["at_app_threshold"][n]
            inst, imgs = s["support_instances"].get(n, 0), s["support_images"].get(n, 0)
            flag = " ⚠️" if inst < 10 else ""
            lines.append(f"| {n}{flag} | {inst} ({imgs}) | {fmt(pc.get('precision'))} | {fmt(pc.get('recall'))} "
                         f"| {fmt(pc.get('mAP50'))} | {fmt(pc.get('mAP50_95'))} | {a['tp']} / {a['fp']} / {a['fn']} "
                         f"| {fmt(a['precision'])} | {fmt(a['recall'])} |")
        lines.append("\n⚠️ fewer than 10 instances: the per-class figures are not reliable.\n")
        lines.append("**Decision agreement** (rule applied to ground-truth labels → to predictions, "
                     f"people counted by YOLOv8n; people per image: {s['persons_per_image']}):\n")
        for k, v in s["decisions"].items():
            lines.append(f"- {k}: {v}")
        lines.append(f"\nFalse positives at the app threshold: {len(s['false_positives'])}; "
                     f"false negatives: {len(s['false_negatives'])}. Files: `{split}/examples/` "
                     "(local only, images show people). Plots: "
                     + ", ".join(f"`{split}/{p}`" for p in s["plots"]) + "\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(settings.DETECTION_MODEL))
    ap.add_argument("--data", default=str(ROOT / "dataset" / "data.yaml"))
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--conf", type=float, default=settings.DEFAULT_DECISION_CONFIDENCE)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "reports" / "evaluation"))
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    person_model = YOLO(str(settings.PERSON_MODEL))
    names = [model.names[i] for i in sorted(model.names)]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ck = model.ckpt or {}
    rel = lambda p: Path(p).resolve().relative_to(ROOT).as_posix() if Path(p).resolve().is_relative_to(ROOT) else Path(p).name  # noqa: E731
    report = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "args": [a for a in sys.argv[1:]],
        "environment": environment(),
        "model": {"path": rel(args.model), "md5": md5(args.model), "saved": ck.get("date"),
                  "ultralytics_version": ck.get("version"),
                  "train_args": {k: ck.get("train_args", {}).get(k) for k in
                                 ("model", "pretrained", "data", "epochs", "imgsz", "batch", "seed",
                                  "optimizer", "patience")}},
        "data": {"yaml": rel(args.data), "yaml_md5": md5(args.data), "splits": {}},
        "conf": args.conf, "classes": names, "splits": {},
    }
    for split in args.splits:
        fp_hash, n = fingerprint(*split_dirs(args.data, split))
        report["data"]["splits"][split] = {"fingerprint": fp_hash, "images": n}
        res = evaluate_split(model, person_model, names, args.data, split, args.conf,
                             args.device, out_dir)
        res["plots"] = sorted(p.name for p in (out_dir / split).glob("*.png"))
        report["splits"][split] = res
    shutil.rmtree(out_dir / "_ultralytics", ignore_errors=True)

    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    (out_dir / "evaluation.md").write_text(to_markdown(report), encoding="utf-8")
    print(f"Report written to {rel(out_dir)}/evaluation.md")


if __name__ == "__main__":
    main()
