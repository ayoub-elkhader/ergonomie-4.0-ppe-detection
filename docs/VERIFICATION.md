# Verification log

Checks actually run on 2026-11-04 on the author's laptop (Windows 10, Intel
Core i7-5600U, no GPU). Commands are run from the repository root.

## 1. Audit findings re-checked before fixing

| Finding | How it was checked | Result |
|---|---|---|
| "PPE compliant" with only a helmet | `check_ppe` called with a single `avec_casque` detection | Already fixed in commit `9fa4159` (returned "incomplete"); the rule was then extended with uncertain evidence, conflicts and the one-operator scope |
| "Trunk" angle is a thigh–torso angle | synthetic skeletons fed to the previous `Ergonomy` class | Reproduced: standing 0°, seated upright 90°, whole body tilted 45° → 0° |
| Wrong class counts in `dataset/README.md` | `scripts/audit_dataset.py` | Reproduced: 882 `avec_casque` instances, not 76. Cause: label files have no trailing newline (978/978), and the old count concatenated them |
| Origin of the README metrics | `YOLO('weights/best.pt').ckpt` | Saved 2024-01-31 by Ultralytics 8.0.20, `yolov8s.yaml`, `pretrained=False`, 200 epochs, seed 0, `patience=0`, data `/content/datasets/soudage_ia-2`. No training log. The 0.809 / 0.396 / 0.707 test figures were computed during this session, not taken from an original log |
| Test supports (1 `sans_tablier`, 4 accessories, 5 hair) | audit script | Confirmed |

## 2. Unit tests

```
pytest            # 31 passed
```

`tests/test_ppe.py` (17 tests): helmet only, nothing detected, all items,
confirmed / uncertain / conflicting evidence, hazards, 0 / 2 / 3 / unknown
people, two helmets with one person counted, threshold sensitivity.

`tests/test_posture.py` (14 tests): standing, seated upright, whole body tilted
45°, bent forward 70°, neck flexion 30°, asymmetry, hidden landmarks on one or
both sides, all landmarks hidden, coincident points, NaN coordinates,
configurable thresholds, exposure timing with gaps.

## 3. Dataset audit

`python scripts/audit_dataset.py` → `reports/dataset_audit.md` / `.json`.
Main results: 978 / 93 / 47 images; 326 / 93 / 47 source photos; no missing or
malformed labels; 0 exact duplicates; 40 cross-split near-duplicate pairs;
205 consecutive photo numbers split across sets. A train/test pair
(`IMG_3463` / `IMG_3464`) was inspected visually: same person, same place.

## 4. Evaluation

`python scripts/evaluate.py --splits val test --device cpu` →
`reports/evaluation/evaluation.md`, `evaluation.json`, plots in
`reports/evaluation/{val,test}/`. Example FP/FN images are written to
`reports/evaluation/*/examples/` and are git-ignored (they show people).

Reproduced: val mAP50 0.771, mAP50-95 0.372; test mAP50 0.809, mAP50-95 0.396,
mean recall 0.707, mean precision 0.857.

Not done: comparison with COCO-pretrained weights. Training on this CPU would
take hours per run; the protocol is in `train.py`. No result is claimed.

## 5. Speed

`python scripts/benchmark.py` → `reports/benchmark.md`: full pipeline 424 ms per
frame (2.4 frames/s) on 400×720 frames. Not real-time on this machine.

## 6. Application

- `streamlit.testing.v1.AppTest`: 3 analysis modes × 4 sources, including a
  full analysis of the demo video and an invalid RTSP URL: no exception.
- Manual check in a browser (Streamlit 1.30): landing page images load; image
  analysis shows the checklist ("Evaluation incomplete — Safety shoes not
  observed" on `epi1.jpg`) and the posture report.
- Pipeline on the 47 test images and on the 311-frame demo video with
  ByteTrack: no error; exposure durations accumulated from video timestamps.
- Not tested: a real webcam, a real RTSP camera, several simultaneous browser
  sessions.

## 7. Repository hygiene

- No secret, API key or personal path in versioned files (searched for
  `lenovo`, `C:/`, `api_key`, `token`, `password`). The Roboflow API key found
  in the original Colab notebook is not in this repository; it should be
  revoked on Roboflow because the notebook is shared.
- Licenses: Ultralytics AGPL-3.0 (project license kept); interface adapted from
  CodingMantras (Apache-2.0, checked through the GitHub API) — attribution in
  `NOTICE`.
- Media showing people (consent not documented): `images/epi1.jpg`,
  `images/epi1cor.jpg`, `docs/interface.png`, `videos/demo_atelier.mp4`.
- Commit metadata contains the author's e-mail address.

## 8. Clean environment install

A new virtual environment (Python 3.11.5, `python -m venv`, nothing else
installed) was created and `pip install -r requirements-dev.txt` was run:

- install succeeded, `pip check`: "No broken requirements found";
- resolved versions: ultralytics 8.4.202, streamlit 1.39.1, mediapipe 0.10.14,
  numpy 1.26.4, opencv-python 4.11.0.86 (+ opencv-contrib-python pulled by
  mediapipe), torch 2.15.0 (CPU), lap 0.5.14;
- `pytest`: 31 passed;
- AppTest run of the 12 mode × source combinations (video analysed end to end):
  no exception.

The interactive browser check (section 6) was done with Streamlit 1.30.0 in the
author's Anaconda environment, not with 1.39.1. Linux and macOS were not tested.
`pandas` was then removed from `requirements.txt` (not imported by the project;
Streamlit installs it anyway).
