# 🕳️ Pothole Detector

A real-time pothole detection system powered by **YOLOv5** and **Python**, designed to identify road potholes from images, videos, or live webcam feeds with high accuracy.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![YOLOv5](https://img.shields.io/badge/YOLOv5-00FFFF?style=flat&logo=yolo&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## 🔍 Overview

Pothole Detector uses the YOLOv5 object detection architecture to detect and localize potholes in road images or video feeds. The project includes both a **training pipeline** to train a custom model and a **detection script** to run inference on new images.

It was trained on a dataset of **665 images** with a 70/20/10 train/validation/test split.

---

## ✨ Features

- Real-time pothole detection using YOLOv5
- Full training pipeline with GPU support
- Displays confidence scores for each detected pothole
- Supports image, video, and webcam input
- Exports trained model to ONNX, TorchScript, or TensorRT
- Early stopping to prevent overfitting
- Clean command-line interface

---

## ⚙️ Requirements

- Python 3.8+
- NVIDIA GPU with CUDA (recommended) or CPU
- pip

---

## 🚀 Installation

### 1. Clone the repository

​```bash
git clone https://github.com/Bolice1/Pothole-detector.git
cd Pothole-detector
​```

### 2. Install PyTorch

**For CUDA 11.8:**
​```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
​```

**For CUDA 12.x:**
​```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
​```

**For CPU only:**
​```bash
pip install torch torchvision torchaudio
​```

### 3. Install dependencies

​```bash
pip install ultralytics opencv-python pillow matplotlib pyyaml
​```

---

## 🖥️ Usage

### Basic detection

​```bash
python pothole_detector.py path/to/image.jpg
​```

### Save the output image

​```bash
python pothole_detector.py path/to/image.jpg --save
​```

### Display the result in a window

​```bash
python pothole_detector.py path/to/image.jpg --show
​```

### Use a custom trained model

​```bash
python pothole_detector.py path/to/image.jpg --model path/to/best.pt
​```

### All CLI options

| Argument | Short | Description |
|----------|-------|-------------|
| `image` | | Path to the input image (required) |
| `--model` | `-m` | Path to model file (default: `best.pt`) |
| `--save` | `-s` | Save output image with detections |
| `--show` | | Display the result in a window |
| `--output` | `-o` | Custom output filename |

---

## 🏋️ Training

### 1. Set your dataset path

Open `pothole_yolo5_train.py` and update:

​```python
DATASET_PATH = "path/to/your/dataset"
​```

Your dataset should follow this structure:

​```
dataset/
├── images/
│   ├── train/
│   ├── valid/
│   └── test/
└── labels/
    ├── train/
    ├── valid/
    └── test/
​```

### 2. Run training

​```bash
python pothole_yolo5_train.py
​```

Training configuration (editable at top of file):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMG_SIZE` | 640 | Input image size |
| `BATCH_SIZE` | 16 | Batch size |
| `EPOCHS` | 100 | Max training epochs |
| `MODEL_TYPE` | `yolov5s` | YOLOv5 model variant |
| `DEVICE` | `0` | GPU device (use `'cpu'` for CPU) |

### 3. Use your trained model

After training, your best model is saved to:

​```
pothole_detection/train/weights/best.pt
​```

Run detection with it:

​```bash
python pothole_detector.py image.jpg --model pothole_detection/train/weights/best.pt
​```

---

## 📁 Project Structure

​```
Pothole-detector/
├── pothole_detector.py      # Detection script — run inference on images
├── pothole_yolo5_train.py   # Training pipeline — train YOLOv5 on custom data
├── LICENSE                  # MIT License
└── README.md                # Project documentation
​```

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and open a pull request.

---

## 📄 License

This project is licensed under the MIT License.