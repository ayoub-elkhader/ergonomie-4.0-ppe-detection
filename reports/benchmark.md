# Processing speed benchmark

Command: `python scripts/benchmark.py `

- Hardware: Intel(R) Core(TM) i7-5600U CPU @ 2.60GHz, 3 torch threads, CUDA: False
- Software: Python 3.11.5, ultralytics 8.4.202, torch 2.15.0+cpu, OS Windows-10-10.0.19045-SP0
- Input: `demo_atelier.mp4`, 150 frames resized to 400×720, YOLO inference at imgsz 640, device cpu
- Excludes video decoding and Streamlit display.

| Stage | median ms/frame | p90 ms/frame | frames/s (median) |
|---|--:|--:|--:|
| PPE detector (YOLOv8s) | 288 | 569 | 3.5 |
| Person counter (YOLOv8n) | 109 | 142 | 9.2 |
| Pose (MediaPipe) | 34 | 43 | 29.2 |
| Full pipeline (PPE + people + posture + drawing) | 424 | 512 | 2.4 |
