# Human Action Recognition using YOLOv11

Real-time human action recognition (Run, Sit, Stand, Walk) built with YOLOv11 (Ultralytics) and OpenCV, trained on a custom Roboflow dataset.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.8-orange)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-purple)


---

## 📌 Overview

| | |
|---|---|
| **Model** | YOLOv11s (`yolo11s.pt` base) |
| **Classes** | Run, Sit, Stand, Walk |
| **Dataset source** | [Roboflow](https://roboflow.com) — `hadis-gh/action-recognition-y9uip` |
| **Dataset size** | ~9,600 images (70/20/10 train/valid/test split) |
| **Training hardware** | NVIDIA RTX 2050 (4GB VRAM) |
| **Framework** | Ultralytics YOLOv11, PyTorch (CUDA 12.8), OpenCV |
| **Overall mAP50** | ~0.629 |

---

## 📂 Project Structure

```
.
├── 1_ split_dataset.py        # Create train/valid/test split
├── 2_ check_gpu_setup.py      # Step 3: verify PyTorch + Ultralytics + GPU
├── 3_ train.py                # Train the model
├── 4_ test_model.py           # Evaluate the model
├── 5_ webcam_inference.py     # Real-time webcam inference
├── 6_ analyze_and_improve.py  # Chart the results + improvement suggestions
├── requirements.txt
└── README.md
```
---

## 🛠️ Setup

```bash
conda create -n obj python=3.10 -y
conda activate obj
pip install -r requirements.txt
```

For a CUDA-enabled PyTorch build specifically (recommended over the plain `pip install torch` in requirements.txt):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

> **Note:** On some Windows/PowerShell setups, `conda activate` inside scripts doesn't reliably persist. If a script can't find packages, call Python directly with the full interpreter path, e.g.:
> `C:\Users\ASUS\anaconda3\envs\obj\python.exe train.py`

---

## 🚀 Step-by-Step Pipeline

### 1️⃣ Get the dataset from Roboflow

Download your dataset from the Roboflow website in **YOLOv8/YOLOv11 format**. (Roboflow's "YOLO v26" label under labeling settings is just a format identifier, not an architecture — YOLOv11 is fully compatible with YOLOv8-format exports.)

If the downloaded `data.yaml` has broken relative paths (`../train/images`), fix them:
```powershell
(Get-Content data.yaml) -replace '\.\./', '' | Set-Content data.yaml
```

### 2️⃣ Create train / valid / test split

```bash
python split_dataset.py --source dataset/ --output dataset_split/ --seed 42
```

Splits the ~9,600 images **70/20/10** with a fixed random seed for reproducibility (only needed if your Roboflow download didn't already include a split).

### 3️⃣ Install PyTorch + Ultralytics for GPU

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics
```

Quick check that the GPU is visible to PyTorch:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```
**Don't move on to training until this prints `True`** — otherwise training silently falls back to CPU and is far slower.

### 4️⃣ Train the model

```bash
C:\Users\ASUS\anaconda3\envs\obj\python.exe train.py
```

Trains `yolo11s.pt` for 20 epochs with `batch=8` and `workers=2` — reduced values to fit a 4GB-VRAM GPU and avoid `MemoryError` on this system. Outputs (weights, `results.png`, `confusion_matrix.png`, `results.csv`) are saved to:runs/detect/action_recognition_v1/

### 5️⃣ Test the model

```bash
C:\Users\ASUS\anaconda3\envs\obj\python.exe test_model.py --weights runs/detect/action_recognition_v1/weights/best.pt --data data/data.yaml --split test
```

Prints overall mAP50, mAP50-95, and **per-class mAP** — this is what tells you which action (e.g. Stand) is underperforming.

### 6️⃣ Analyze the data/results with charts + get improvement suggestions

```bash
python analyze_and_improve.py --data data/data.yaml --results runs/detect/action_recognition_v1/results.csv
```

This generates two charts and prints suggestions:
- `assets/class_distribution.png` — instances per class, computed directly from your label files (no guessing).
- `assets/training_curves.png` — mAP, precision/recall, and loss curves over epochs, parsed from Ultralytics' own `results.csv`.
- Console output with concrete next steps tailored to your actual class balance (see below).

### 🎥 Bonus: real-time webcam inference

```bash
C:\Users\ASUS\anaconda3\envs\obj\python.exe webcam_inference.py
```

Ultralytics' built-in `show=True` didn't reliably render on this system, so this uses a custom OpenCV display loop instead. Press **q** to quit.

---

## 📊 Results Snapshot

| Metric | Value |
|---|---|
| **mAP50 (overall)** | ~0.629 |
| **Epochs** | 20 |
| **Base model** | YOLOv11s |
| **Batch size** | 8 |

**Class imbalance** is the main issue observed so far: **Walk** has ~581 annotated instances while **Stand** has only ~179, and Stand is the weakest-performing class as a direct result. Run `analyze_and_improve.py` on your dataset to get the exact current counts and per-class breakdown as charts.

---

## 📈 How to Improve the Model Further

1. **Fix class imbalance (highest priority)** — collect/annotate more samples for **Stand** (and any other under-represented class) until counts are closer to Walk's (~581). Use Roboflow's **Label Assist** to speed this up.
2. **Train longer / tune hyperparameters** — 20 epochs is likely under-trained for a 4-class problem; increase epochs (with early stopping) once the dataset is more balanced. Try `yolo11m` instead of `yolo11s` if VRAM allows.
3. **Augmentation** — add rotation, brightness/contrast jitter, and motion blur to simulate real webcam conditions; Ultralytics' default mosaic/mixup helps too, but new real samples for the weak class help more.
4. **Data quality checks** — manually review borderline "Stand vs Sit" and "Walk vs Run" annotations, which are visually ambiguous and a common source of label noise.
5. **Evaluation discipline** — track per-class mAP every run (via `test_model.py`), not just overall mAP50, so gains on one class don't hide regressions on another.
6. **Temporal information** — a future version could use a few consecutive frames (optical flow or a lightweight temporal model) instead of single-frame detection, since actions unfold over time.

---
