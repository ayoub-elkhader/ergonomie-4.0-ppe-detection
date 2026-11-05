# Dataset audit

Generated 2026-11-04T23:24:20+00:00 by `scripts/audit_dataset.py` from `dataset/data.yaml`.

## Images

| Split | Images | Source photos | Images / source | Empty label files |
|---|--:|--:|--:|--:|
| train | 978 | 326 | 3.0 | 0 |
| val | 93 | 93 | 1.0 | 0 |
| test | 47 | 47 | 1.0 | 0 |

Image sizes: train: {'640x640': 978}; val: {'640x640': 93}; test: {'640x640': 47}

Images with an EXIF capture date: train 0, val 0, test 0 (Roboflow strips EXIF, so capture time and camera are unknown).

## Instances (images) per class

| id | Class | train | val | test | Total instances | Share |
|--:|---|--:|--:|--:|--:|--:|
| 0 | avec_casque | 777 (747) | 67 (65) | 38 (37) | 882 | 19.4 % |
| 1 | avec_gants | 1047 (756) | 102 (76) | 55 (39) | 1204 | 26.4 % |
| 2 | chaussure_adequat | 623 (457) | 55 (40) | 35 (24) | 713 | 15.7 % |
| 3 | chaussure_inadequat | 90 (54) | 10 (5) | 8 (4) | 108 | 2.4 % |
| 4 | presence_accessoires | 60 (51) | 2 (2) | 4 (4) | 66 | 1.4 % |
| 5 | presence_cheveux | 89 (66) | 9 (5) | 5 (3) | 103 | 2.3 % |
| 6 | sans_casque | 259 (244) | 30 (26) | 11 (11) | 300 | 6.6 % |
| 7 | sans_gants | 267 (210) | 17 (11) | 10 (8) | 294 | 6.5 % |
| 8 | sans_tablier | 48 (48) | 3 (3) | 1 (1) | 52 | 1.1 % |
| 9 | tablier_ferme | 600 (567) | 54 (49) | 30 (30) | 684 | 15.0 % |
| 10 | tablier_ouvert | 132 (132) | 10 (10) | 7 (7) | 149 | 3.3 % |

Imbalance ratio (most / least frequent class): 23.2. Train counts include the ×3 augmentation.

**Classes with fewer than 10 test instances** (per-class test metrics are fragile): chaussure_inadequat (8), presence_accessoires (4), presence_cheveux (5), sans_tablier (1), tablier_ouvert (7).

## Leakage signals

- Exact duplicate groups: 0 (across splits: 0)
- Near-duplicate pairs (dHash distance ≤ 6 bits, excluding augmentations of the same photo): 155 (across splits: 40)
- Source photos present in several splits: 0
- Consecutive photo numbers falling in different splits (same session, probably same person): test-train: 65, test-val: 15, train-val: 125

Closest cross-split pairs:

- `train/IMG_3585_jpg.rf.9f9827491db1836d8b159594c6ee83e9.jpg` ↔ `val/IMG_3586_jpg.rf.4cf4ebf48360bd37958444699281d0e8.jpg` (distance 1)
- `train/IMG_3585_jpg.rf.bbdb249ab074d7725067917d14b56da6.jpg` ↔ `val/IMG_3586_jpg.rf.4cf4ebf48360bd37958444699281d0e8.jpg` (distance 1)
- `train/IMG_3622_jpg.rf.9b38a4b5b55caf91d64ad86a03556ed3.jpg` ↔ `val/IMG_3623_jpg.rf.098a54442820d7b4f5ac2dfcb38ba70e.jpg` (distance 2)
- `train/IMG_3622_jpg.rf.dc161d631f9aeb116c205144fc394476.jpg` ↔ `val/IMG_3623_jpg.rf.098a54442820d7b4f5ac2dfcb38ba70e.jpg` (distance 2)
- `train/IMG_3719_jpg.rf.4c4c991cb2abcc0bae5744ba9cd2d882.jpg` ↔ `val/IMG_3717_jpg.rf.09fdd38ed1fd248a9b73bd6fed70aa84.jpg` (distance 2)
- `train/IMG_3444_jpg.rf.e03c902b0b595da9dae6326646198735.jpg` ↔ `test/IMG_3443_jpg.rf.a465199ec7540077167d46535cd124dc.jpg` (distance 3)
- `train/IMG_3452_jpg.rf.59802acc6f6a2903663ff10e2b2b0780.jpg` ↔ `val/IMG_3451_jpg.rf.9871c59d977d54e16eb74b00d1c2b389.jpg` (distance 3)
- `train/IMG_3453_jpg.rf.f0d5489825098e2cb831fec7013ea633.jpg` ↔ `val/IMG_3451_jpg.rf.9871c59d977d54e16eb74b00d1c2b389.jpg` (distance 3)
- `train/IMG_3463_jpg.rf.425ef3b8a87666ce1e35904ea1669439.jpg` ↔ `test/IMG_3464_jpg.rf.ecaec9a590606d5694ca979f14e785d9.jpg` (distance 3)
- `train/IMG_3531_jpg.rf.6c206dc6066256b176599f0d0558e1aa.jpg` ↔ `val/IMG_3530_jpg.rf.0f629e7b1bd19c20bd804ba68d47681e.jpg` (distance 3)
- `train/IMG_3531_jpg.rf.d519d88bfc9b325b95346357415900a0.jpg` ↔ `val/IMG_3530_jpg.rf.0f629e7b1bd19c20bd804ba68d47681e.jpg` (distance 3)
- `train/IMG_3542_jpg.rf.2b13011a5b73fcff3664853d24922fcf.jpg` ↔ `val/IMG_3541_jpg.rf.fb93c46221b7cc1db5142953b23e02ca.jpg` (distance 3)
- `train/IMG_3585_jpg.rf.4de8e75e36fce7ec710f6d2b5dce2701.jpg` ↔ `val/IMG_3586_jpg.rf.4cf4ebf48360bd37958444699281d0e8.jpg` (distance 3)
- `train/IMG_3594_jpg.rf.ce0c6afb410b67218c8c82303918d620.jpg` ↔ `val/IMG_3593_jpg.rf.e5e8e4093155711affe99d91e2215a13.jpg` (distance 3)
- `train/IMG_3594_jpg.rf.09e1f15cd73924745acdce8dca0fa84f.jpg` ↔ `val/IMG_3593_jpg.rf.e5e8e4093155711affe99d91e2215a13.jpg` (distance 4)

## Label problems (6)

- tiny box: 6
  - `train/IMG_3649_jpg.rf.37d9f913d351ab9636f79e3384af0b61.txt`: line 3: tiny box (area 5.9e-05)
  - `train/IMG_3649_jpg.rf.37d9f913d351ab9636f79e3384af0b61.txt`: line 5: tiny box (area 6.4e-05)
  - `train/IMG_3649_jpg.rf.701cbf89ab2d786d0f3dce8ebda1f916.txt`: line 3: tiny box (area 6.3e-05)
  - `train/IMG_3649_jpg.rf.701cbf89ab2d786d0f3dce8ebda1f916.txt`: line 5: tiny box (area 6.4e-05)
  - `train/IMG_3649_jpg.rf.e18cb2e6decead72a9ffc6c7a976308c.txt`: line 3: tiny box (area 5.1e-05)
  - `train/IMG_3649_jpg.rf.e18cb2e6decead72a9ffc6c7a976308c.txt`: line 5: tiny box (area 6.0e-05)
