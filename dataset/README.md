# soudage_ia dataset (v2)

PPE annotations for welding operators, YOLOv8 format.

| Split | Images |
|-------|-------:|
| train | 978 (≈326 photos × 3 augmentations) |
| valid | 93 |
| test  | 47 |

Annotated instances per class (all splits):

| id | class | instances |
|---:|-------|----------:|
| 0 | avec_casque | 76 |
| 1 | avec_gants | 1183 |
| 2 | chaussure_adequat | 690 |
| 3 | chaussure_inadequat | 98 |
| 4 | presence_accessoires | 62 |
| 5 | presence_cheveux | 77 |
| 6 | sans_casque | 88 |
| 7 | sans_gants | 291 |
| 8 | sans_tablier | 51 |
| 9 | tablier_ferme | 678 |
| 10 | tablier_ouvert | 146 |

Layout:

```
dataset/
├── data.yaml
├── train/{images,labels}/
├── valid/{images,labels}/
└── test/{images,labels}/
```

Each label file has one line per object: `class x_center y_center width height`
(normalised 0–1).

> The image folders are excluded from the Git repository by `.gitignore`
> (the photos show real people). Only `data.yaml` and this file are versioned.
