# Human Action Recognition using YOLO11

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![YOLO11](https://img.shields.io/badge/YOLO-v11-green)

A Human Action Recognition (HAR) project built with **YOLO11** using the Ultralytics framework. The project includes dataset analysis, model training, image inference, and real-time webcam inference using a custom YOLO-format dataset.

---
## ✨ Features

- Train a YOLO11 model for human action recognition
- Perform image inference using a trained model
- Real-time webcam inference
- Analyze dataset statistics before training
- Split datasets into training, validation, and test sets
- GPU acceleration with CUDA (if available)
## 📂 Project Structure

```
human-action-recognition/
│
├── train.py               # Train the YOLO11 model
├── test.py                # Run inference on an image
├── webcam.py              # Real-time webcam inference
├── split_dataset.py       # Split dataset into train/valid/test
├── analyze_dataset.py     # Analyze dataset statistics
├── check_gpu.py           # Check CUDA/GPU availability
├── requirements.txt       # Project dependencies
├── README.md
└── dataset/               # YOLO-format dataset
```
## 🖼️ Image Inference

Run inference on an image using the trained model.

```bash
python test.py
```

The prediction result will be saved automatically to:

```
runs/detect/predict/
```

Example output:

- Predicted action label
- Bounding box
- Confidence score

  
## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Hadisgh1260/human-action-recognition.git
cd human-action-recognition
```

### 2. Create a virtual environment (Recommended)

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

Python **3.10 or later** is recommended.

```bash
pip install -r requirements.txt
```

### 4. Verify GPU availability (Optional)

```bash
python check_gpu.py
```
## 🚀 Getting Started

### 1. Analyze the Dataset

Before training, you can inspect your dataset statistics.

```bash
python analyze_dataset.py
```

---

### 2. Split the Dataset

Split the dataset into training, validation, and testing sets.

```bash
python split_dataset.py
```

---

### 3. Train the Model

Train the YOLO11 model.

```bash
python train.py
```

The trained weights will be saved automatically in:

```
runs/train/
```

---

### 4. Run Image Inference

Run inference on an image using the trained model.

```bash
python test.py
```

Prediction results will be saved to:

```
runs/detect/predict/
```

---

### 5. Real-Time Webcam Inference

Run real-time human action recognition using your webcam.

```bash
python webcam.py
```

## 📂 Dataset

The model was trained using a custom Human Action Recognition dataset in YOLO format.

Classes:

- Run
- Sit
- Stand
- Walk

Dataset Source:

https://app.roboflow.com/hadis-gh/action-recognition-y9uip

---

## 📊 Model Performance

The trained YOLO11 model achieved the following performance on the evaluation dataset.

| Metric | Value |
|---------|------:|
| Precision | 0.618 |
| Recall | 0.590 |
| mAP@50 | 0.629 |
| mAP@50-95 | 0.424 |

> These results were obtained using the current version of the dataset and training configuration.


## 📈 Future Improvements

1. **Address class imbalance** — collect and annotate more samples for underrepresented classes (such as **Stand**) to improve class balance and overall model performance.

2. **Train longer and tune hyperparameters** — experiment with more training epochs, early stopping, learning rate scheduling, and larger YOLO11 variants (e.g., `yolo11m`) if hardware resources allow.

3. **Apply stronger data augmentation** — use techniques such as rotation, brightness/contrast adjustment, motion blur, and other augmentations to improve model robustness in real-world scenarios.

4. **Improve annotation quality** — review ambiguous samples (e.g., **Stand vs. Sit** or **Walk vs. Run**) to reduce label noise and improve classification accuracy.

5. **Perform comprehensive model evaluation** — monitor Precision, Recall, mAP, and per-class performance after each training session to better identify weak classes and measure improvements.

6. **Extend to temporal action recognition** — future versions could leverage multiple consecutive video frames (e.g., optical flow or temporal models) instead of single-frame inference for more accurate action recognition.
