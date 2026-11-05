"""Audit the YOLO dataset: counts, label validity, class balance and leakage signals.

Usage:
    python scripts/audit_dataset.py [--data dataset/data.yaml] [--out reports/dataset_audit]

Writes <out>.md (human readable) and <out>.json (machine readable).
Nothing in the dataset is modified.

Leakage checks (none of them can PROVE independence between splits):
  * exact duplicates (MD5 of the image bytes) inside and across splits;
  * near duplicates (64-bit difference hash, Hamming distance <= --near);
  * same source photo across splits (Roboflow names: <source>_jpg.rf.<hash>.jpg);
  * neighbouring photo numbers across splits (IMG_3349 vs IMG_3350 are usually
    taken seconds apart, i.e. same person / same session).
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_RE = re.compile(r"^(?P<src>.+?)_(?:jpe?g|png|heic|JPG|JPEG|PNG|HEIC)\.rf\.[0-9a-f]+$")
NUMBER_RE = re.compile(r"(\d+)$")


def source_name(stem):
    m = SOURCE_RE.match(stem)
    return m.group("src") if m else stem


def dhash(path, size=8):
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def parse_label(path, nc):
    """Return (list of (cls, x, y, w, h), list of problems)."""
    boxes, problems = [], []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            problems.append(f"line {n}: {len(parts)} fields (expected 5)")
            continue
        try:
            c = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            problems.append(f"line {n}: not numeric")
            continue
        if not 0 <= c < nc:
            problems.append(f"line {n}: class {c} out of range")
        if not all(0 <= v <= 1 for v in (x, y, w, h)):
            problems.append(f"line {n}: coordinates outside [0, 1]")
        if w <= 0 or h <= 0:
            problems.append(f"line {n}: zero-area box")
        elif w * h < 1e-4:
            problems.append(f"line {n}: tiny box (area {w * h:.1e})")
        if x - w / 2 < -0.01 or x + w / 2 > 1.01 or y - h / 2 < -0.01 or y + h / 2 > 1.01:
            problems.append(f"line {n}: box exceeds image")
        boxes.append((c, x, y, w, h))
    dup = len(boxes) - len(set(boxes))
    if dup:
        problems.append(f"{dup} identical duplicated box(es)")
    return boxes, problems


def audit(data_yaml, near_threshold):
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    base = Path(data_yaml).resolve().parent
    names = cfg["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    nc = len(names)

    report = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "data_yaml": str(Path(data_yaml).as_posix()), "classes": names, "splits": {}}
    hashes, dhashes, sources = {}, {}, defaultdict(set)
    problems = []

    for split in SPLITS:
        img_dir = (base / cfg[split]).resolve()
        lbl_dir = img_dir.parent / "labels"
        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        labels = {p.stem: p for p in lbl_dir.glob("*.txt")}
        inst, img_per_class = Counter(), Counter()
        empty, srcs, sizes = 0, set(), Counter()
        exif_dates = 0

        for img in images:
            hashes.setdefault(hashlib.md5(img.read_bytes()).hexdigest(), []).append((split, img.name))
            dhashes[(split, img.name)] = dhash(img)
            with Image.open(img) as im:
                sizes[im.size] += 1
                exif = im.getexif()
                exif_dates += bool(exif.get(306) or exif.get(36867))
            src = source_name(img.stem)
            srcs.add(src)
            sources[src].add(split)

            lbl = labels.pop(img.stem, None)
            if lbl is None:
                problems.append({"split": split, "file": img.name, "problem": "missing label file"})
                continue
            boxes, probs = parse_label(lbl, nc)
            problems += [{"split": split, "file": lbl.name, "problem": p} for p in probs]
            if not boxes:
                empty += 1
            for c in {b[0] for b in boxes if 0 <= b[0] < nc}:
                img_per_class[names[c]] += 1
            for b in boxes:
                if 0 <= b[0] < nc:
                    inst[names[b[0]]] += 1
        for stem in labels:
            problems.append({"split": split, "file": stem + ".txt", "problem": "label without image"})

        report["splits"][split] = {
            "images": len(images), "source_photos": len(srcs),
            "variants_per_source": round(len(images) / max(1, len(srcs)), 2),
            "empty_label_files": empty, "image_sizes": {f"{w}x{h}": n for (w, h), n in sizes.items()},
            "images_with_exif_date": exif_dates,
            "instances": {n: inst[n] for n in names},
            "images_per_class": {n: img_per_class[n] for n in names},
        }

    # ---------------------------------------------------------- leakage
    exact = [v for v in hashes.values() if len(v) > 1]
    keys = list(dhashes)
    near = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a[0] == b[0] and source_name(Path(a[1]).stem) == source_name(Path(b[1]).stem):
                continue  # augmented variants of the same photo
            d = bin(dhashes[a] ^ dhashes[b]).count("1")
            if d <= near_threshold:
                near.append({"a": f"{a[0]}/{a[1]}", "b": f"{b[0]}/{b[1]}", "hamming": d})
    cross_near = [n for n in near if n["a"].split("/")[0] != n["b"].split("/")[0]]
    shared_sources = {s: sorted(sp) for s, sp in sources.items() if len(sp) > 1}

    numbered = {}
    for s, sp in sources.items():
        m = NUMBER_RE.search(s)
        if m and len(sp) == 1:
            numbered[int(m.group(1))] = next(iter(sp))
    neighbours = Counter()
    nums = sorted(numbered)
    for a, b in zip(nums, nums[1:]):
        if b - a == 1:
            pair = tuple(sorted((numbered[a], numbered[b])))
            neighbours[pair] += 1

    report["leakage"] = {
        "exact_duplicate_groups": exact,
        "exact_duplicates_across_splits": [g for g in exact if len({s for s, _ in g}) > 1],
        "near_duplicate_pairs": near,
        "near_duplicate_pairs_across_splits": cross_near,
        "near_threshold_bits": near_threshold,
        "sources_in_several_splits": shared_sources,
        "consecutive_photo_numbers": {f"{a}-{b}": n for (a, b), n in sorted(neighbours.items())},
    }
    report["problems"] = problems
    return report


def to_markdown(r):
    names, sp = r["classes"], r["splits"]
    out = [f"# Dataset audit\n\nGenerated {r['generated']} by `scripts/audit_dataset.py` "
           f"from `{r['data_yaml']}`.\n"]
    out.append("## Images\n\n| Split | Images | Source photos | Images / source | Empty label files |\n"
               "|---|--:|--:|--:|--:|")
    for s in SPLITS:
        d = sp[s]
        out.append(f"| {s} | {d['images']} | {d['source_photos']} | {d['variants_per_source']} "
                   f"| {d['empty_label_files']} |")
    out.append(f"\nImage sizes: " + "; ".join(f"{s}: {sp[s]['image_sizes']}" for s in SPLITS))
    out.append("\nImages with an EXIF capture date: "
               + ", ".join(f"{s} {sp[s]['images_with_exif_date']}" for s in SPLITS)
               + " (Roboflow strips EXIF, so capture time and camera are unknown).")

    out.append("\n## Instances (images) per class\n\n| id | Class | " +
               " | ".join(SPLITS) + " | Total instances | Share |\n|--:|---|" + "--:|" * (len(SPLITS) + 2))
    total = {n: sum(sp[s]["instances"][n] for s in SPLITS) for n in names}
    grand = sum(total.values()) or 1
    for i, n in enumerate(names):
        cells = " | ".join(f"{sp[s]['instances'][n]} ({sp[s]['images_per_class'][n]})" for s in SPLITS)
        out.append(f"| {i} | {n} | {cells} | {total[n]} | {100 * total[n] / grand:.1f} % |")
    nz = [v for v in total.values() if v]
    out.append(f"\nImbalance ratio (most / least frequent class): {max(nz) / min(nz):.1f}. "
               "Train counts include the ×3 augmentation.")
    small = [f"{n} ({sp['test']['instances'][n]})" for n in names if sp["test"]["instances"][n] < 10]
    if small:
        out.append(f"\n**Classes with fewer than 10 test instances** (per-class test metrics are "
                   f"fragile): {', '.join(small)}.")

    lk = r["leakage"]
    out.append("\n## Leakage signals\n")
    out.append(f"- Exact duplicate groups: {len(lk['exact_duplicate_groups'])} "
               f"(across splits: {len(lk['exact_duplicates_across_splits'])})")
    out.append(f"- Near-duplicate pairs (dHash distance ≤ {lk['near_threshold_bits']} bits, excluding "
               f"augmentations of the same photo): {len(lk['near_duplicate_pairs'])} "
               f"(across splits: {len(lk['near_duplicate_pairs_across_splits'])})")
    out.append(f"- Source photos present in several splits: {len(lk['sources_in_several_splits'])}")
    out.append("- Consecutive photo numbers falling in different splits (same session, "
               "probably same person): " + (", ".join(f"{k}: {v}" for k, v in
                                                     lk["consecutive_photo_numbers"].items() if k.split('-')[0] != k.split('-')[1]) or "none"))
    if lk["near_duplicate_pairs_across_splits"]:
        out.append("\nClosest cross-split pairs:\n")
        for p in sorted(lk["near_duplicate_pairs_across_splits"], key=lambda p: p["hamming"])[:15]:
            out.append(f"- `{p['a']}` ↔ `{p['b']}` (distance {p['hamming']})")

    out.append(f"\n## Label problems ({len(r['problems'])})\n")
    kinds = Counter(p["problem"].split(":")[-1].strip().split(" (")[0] for p in r["problems"])
    for k, v in kinds.most_common():
        out.append(f"- {k}: {v}")
    for p in r["problems"][:20]:
        out.append(f"  - `{p['split']}/{p['file']}`: {p['problem']}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "dataset" / "data.yaml"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "dataset_audit"))
    ap.add_argument("--near", type=int, default=6, help="max dHash Hamming distance for near duplicates")
    args = ap.parse_args()
    report = audit(args.data, args.near)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report["data_yaml"] = Path(args.data).resolve().relative_to(ROOT).as_posix() \
        if Path(args.data).resolve().is_relative_to(ROOT) else Path(args.data).name
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    out.with_suffix(".md").write_text(to_markdown(report), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(to_markdown(report))


if __name__ == "__main__":
    main()
