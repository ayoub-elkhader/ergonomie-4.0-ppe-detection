# Evaluation of the PPE detector

Generated 2026-11-04T23:34:46+00:00 with `python scripts/evaluate.py --splits val test --device cpu`.

## Setup

- Environment: Python 3.11.5, ultralytics 8.4.202, torch 2.15.0+cpu, OpenCV 4.11.0, CUDA: False; Windows-10-10.0.19045-SP0; Intel64 Family 6 Model 61 Stepping 4, GenuineIntel
- Model: `weights/best.pt` (MD5 `297aa6322c8b0ae5c109ecf71e5ce791`), saved 2024-01-31T18:11:08.355460 by ultralytics 8.0.20; trained from `yolov8s.yaml`, pretrained=False, epochs=200, imgsz=640, seed=0. No training log is available.
- Data: `dataset/data.yaml` (MD5 `abfdb2bbcebfcce41307db28cc81d982`); split fingerprints: val `2e8757cc9f86` (93 images), test `90ff5d49d1ff` (47 images)
- Validation settings: imgsz 640, Ultralytics defaults (conf 0.001, IoU 0.7 NMS). App threshold for the second table: score ≥ 0.5, IoU ≥ 0.5.

## Split `val` (93 images)

| Precision | Recall | mAP50 | mAP50-95 |
|--:|--:|--:|--:|
| 0.707 | 0.758 | 0.771 | 0.372 |

Precision/recall are means over classes at the F1-optimal threshold chosen by Ultralytics. mAP is not a percentage of correct decisions.

| Class | Instances (images) | P | R | AP50 | AP50-95 | TP / FP / FN @app | P @app | R @app |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| avec_casque | 67 (65) | 0.884 | 0.940 | 0.943 | 0.498 | 62 / 9 / 5 | 0.873 | 0.925 |
| avec_gants | 102 (76) | 0.713 | 0.794 | 0.747 | 0.306 | 82 / 20 / 20 | 0.804 | 0.804 |
| chaussure_adequat | 55 (40) | 0.582 | 0.710 | 0.634 | 0.345 | 31 / 17 / 24 | 0.646 | 0.564 |
| chaussure_inadequat | 10 (5) | 0.839 | 0.800 | 0.954 | 0.431 | 6 / 0 / 4 | 1.000 | 0.600 |
| presence_accessoires ⚠️ | 2 (2) | 0.305 | 0.500 | 0.495 | 0.297 | 1 / 1 / 1 | 0.500 | 0.500 |
| presence_cheveux ⚠️ | 9 (5) | 0.532 | 0.384 | 0.444 | 0.165 | 2 / 3 / 7 | 0.400 | 0.222 |
| sans_casque | 30 (26) | 0.931 | 0.896 | 0.909 | 0.486 | 26 / 2 / 4 | 0.929 | 0.867 |
| sans_gants | 17 (11) | 0.741 | 0.882 | 0.878 | 0.446 | 15 / 4 / 2 | 0.789 | 0.882 |
| sans_tablier ⚠️ | 3 (3) | 1.000 | 0.928 | 0.995 | 0.432 | 2 / 0 / 1 | 1.000 | 0.667 |
| tablier_ferme | 54 (49) | 0.569 | 0.708 | 0.682 | 0.298 | 33 / 19 / 21 | 0.635 | 0.611 |
| tablier_ouvert | 10 (10) | 0.678 | 0.800 | 0.795 | 0.383 | 6 / 2 / 4 | 0.750 | 0.600 |

⚠️ fewer than 10 instances: the per-class figures are not reliable.

**Decision agreement** (rule applied to ground-truth labels → to predictions, people counted by YOLOv8n; people per image: {'0': 3, '1': 75, '2': 13, '3': 1, '4': 1}):

- anomaly -> anomaly: 32
- anomaly -> incomplete: 7
- anomaly -> no_anomaly: 1
- incomplete -> anomaly: 2
- incomplete -> incomplete: 42
- incomplete -> no_anomaly: 2
- no_anomaly -> anomaly: 1
- no_anomaly -> incomplete: 3
- no_anomaly -> no_anomaly: 3

False positives at the app threshold: 77; false negatives: 93. Files: `val/examples/` (local only, images show people). Plots: `val/BoxF1_curve.png`, `val/BoxPR_curve.png`, `val/BoxP_curve.png`, `val/BoxR_curve.png`, `val/confusion_matrix.png`, `val/confusion_matrix_normalized.png`

## Split `test` (47 images)

| Precision | Recall | mAP50 | mAP50-95 |
|--:|--:|--:|--:|
| 0.857 | 0.707 | 0.809 | 0.396 |

Precision/recall are means over classes at the F1-optimal threshold chosen by Ultralytics. mAP is not a percentage of correct decisions.

| Class | Instances (images) | P | R | AP50 | AP50-95 | TP / FP / FN @app | P @app | R @app |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| avec_casque | 38 (37) | 0.938 | 0.974 | 0.954 | 0.558 | 35 / 3 / 3 | 0.921 | 0.921 |
| avec_gants | 55 (39) | 0.743 | 0.745 | 0.700 | 0.273 | 37 / 13 / 18 | 0.740 | 0.673 |
| chaussure_adequat | 35 (24) | 0.690 | 0.657 | 0.630 | 0.304 | 18 / 8 / 17 | 0.692 | 0.514 |
| chaussure_inadequat ⚠️ | 8 (4) | 0.751 | 0.625 | 0.695 | 0.332 | 4 / 1 / 4 | 0.800 | 0.500 |
| presence_accessoires ⚠️ | 4 (4) | 1.000 | 0.000 | 0.534 | 0.291 | 0 / 0 / 4 | – | 0.000 |
| presence_cheveux ⚠️ | 5 (3) | 1.000 | 0.796 | 0.962 | 0.493 | 3 / 0 / 2 | 1.000 | 0.600 |
| sans_casque | 11 (11) | 1.000 | 0.860 | 0.905 | 0.507 | 9 / 0 / 2 | 1.000 | 0.818 |
| sans_gants | 10 (8) | 0.952 | 0.800 | 0.898 | 0.379 | 8 / 0 / 2 | 1.000 | 0.800 |
| sans_tablier ⚠️ | 1 (1) | 0.680 | 1.000 | 0.995 | 0.497 | 1 / 0 / 0 | 1.000 | 1.000 |
| tablier_ferme | 30 (30) | 0.722 | 0.606 | 0.737 | 0.332 | 18 / 5 / 12 | 0.783 | 0.600 |
| tablier_ouvert ⚠️ | 7 (7) | 0.948 | 0.714 | 0.893 | 0.389 | 2 / 0 / 5 | 1.000 | 0.286 |

⚠️ fewer than 10 instances: the per-class figures are not reliable.

**Decision agreement** (rule applied to ground-truth labels → to predictions, people counted by YOLOv8n; people per image: {'0': 1, '1': 39, '2': 5, '3': 2}):

- anomaly -> anomaly: 13
- anomaly -> incomplete: 8
- incomplete -> incomplete: 15
- incomplete -> no_anomaly: 1
- no_anomaly -> incomplete: 9
- no_anomaly -> no_anomaly: 1

False positives at the app threshold: 30; false negatives: 69. Files: `test/examples/` (local only, images show people). Plots: `test/BoxF1_curve.png`, `test/BoxPR_curve.png`, `test/BoxP_curve.png`, `test/BoxR_curve.png`, `test/confusion_matrix.png`, `test/confusion_matrix_normalized.png`
