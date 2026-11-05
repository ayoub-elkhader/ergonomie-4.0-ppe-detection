# soudage_ia dataset (v2)

PPE annotations for welding operators, YOLOv8 format, exported from a private
Roboflow project (`ensam-wpmdf/soudage_ia`, version 2, exported 2023-12-15).
Photos were taken by the authors in the ENSAM Meknès welding workshop.

**The images are not distributed with this repository** (they show real people
and no publication consent is documented). Only `data.yaml` and this file are
versioned.

The figures below were produced by `python scripts/audit_dataset.py`; the full
report is in [`reports/dataset_audit.md`](../reports/dataset_audit.md).

## Splits

| Split | Images | Source photos | Images per photo |
|---|--:|--:|--:|
| train | 978 | 326 | 3 (Roboflow augmentation: crop 0–20 %, brightness and exposure ±25 %) |
| valid | 93 | 93 | 1 |
| test | 47 | 47 | 1 |

All images are 640×640 (stretched). EXIF data was stripped by Roboflow:
capture time, camera and the identity of the people are not recorded.

## Instances (images containing the class) per split

| id | class | meaning | train* | valid | test |
|--:|---|---|--:|--:|--:|
| 0 | avec_casque | welding helmet worn | 777 (747) | 67 (65) | 38 (37) |
| 1 | avec_gants | gloves worn | 1047 (756) | 102 (76) | 55 (39) |
| 2 | chaussure_adequat | suitable shoes | 623 (457) | 55 (40) | 35 (24) |
| 3 | chaussure_inadequat | unsuitable shoes | 90 (54) | 10 (5) | 8 (4) |
| 4 | presence_accessoires | accessories (jewellery, watch…) | 60 (51) | 2 (2) | 4 (4) |
| 5 | presence_cheveux | hair not covered | 89 (66) | 9 (5) | 5 (3) |
| 6 | sans_casque | no helmet | 259 (244) | 30 (26) | 11 (11) |
| 7 | sans_gants | no gloves | 267 (210) | 17 (11) | 10 (8) |
| 8 | sans_tablier | no apron | 48 (48) | 3 (3) | 1 (1) |
| 9 | tablier_ferme | apron closed | 600 (567) | 54 (49) | 30 (30) |
| 10 | tablier_ouvert | apron open | 132 (132) | 10 (10) | 7 (7) |

\* train counts include the ×3 augmentation (≈ one third are distinct photos).
Most / least frequent class ratio: 23. Five classes have fewer than 10 test
instances (`sans_tablier` has a single one), so their test metrics are not reliable.

The previous version of this file listed 76 `avec_casque` instances. That table
was wrong: it was computed by concatenating label files that have no trailing
newline, which merged lines and dropped most instances.

## Label quality

No missing, empty or malformed label file, no class id out of range, no box
outside the image. One photo (`IMG_3649`, 3 augmented copies) has two very
small boxes (area < 0.01 % of the image) that may be annotation slips.

## Split independence — important

The split was made **photo by photo**, not by person, session or sequence:

- no exact duplicate and no photo shared between splits, but
- 40 near-duplicate pairs across splits (difference hash ≤ 6 bits), e.g.
  `IMG_3463` (train) and `IMG_3464` (test) show the same person in the same
  place seconds apart;
- consecutive photo numbers fall into different splits 205 times
  (train–val 125, train–test 65, val–test 15).

The validation and test sets are therefore **not independent of training**,
and the reported scores are likely optimistic for new people, sessions or
workshops. Without capture metadata (person, session) a grouped re-split
can only be approximated, e.g. by grouping consecutive photo numbers; any
such re-split requires retraining, and the current scores would no longer
apply to it.

Images may contain several people: the YOLOv8n person counter finds 2–3 people
in 7 of the 47 test images (and none in 1). Labels are per image, not per person.

## Layout

```
dataset/
├── data.yaml
├── train/{images,labels}/
├── valid/{images,labels}/
└── test/{images,labels}/
```

Label line: `class x_center y_center width height` (normalised 0–1).

## Getting the data

The Roboflow project is private. Authorised members can export version 2 in
"YOLOv8" format and unzip it into this folder (keep this `data.yaml`).
