# 🦺 PPE Detection — Construction Safety Compliance

Detect Personal Protective Equipment (PPE) on construction sites using **YOLO** and the [Construction-PPE dataset](https://docs.ultralytics.com/datasets/detect/construction-ppe/).

The model identifies **11 classes** — both worn and missing PPE — enabling real-time safety compliance monitoring.

| Worn PPE (✅ Compliant) | Missing PPE (❌ Violation) |
|---|---|
| helmet | no_helmet |
| gloves | no_gloves |
| vest | no_boots |
| boots | no_goggle |
| goggles | none |
| Person | — |

---

## 📁 Project Structure

```
safety_compliance/
├── config/
│   ├── __init__.py
│   ├── dataset.yaml        # Dataset paths & class definitions
│   └── settings.py         # Central project settings
├── utils/
│   ├── __init__.py
│   └── visualization.py    # Annotation & compliance helpers
├── train.py                # Train YOLO on Construction-PPE
├── predict.py              # Run inference (image/video/webcam)
├── evaluate.py             # Evaluate model metrics
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

> The dataset (~178 MB) will auto-download on the first training run.

---

## 🏋️ Training

```bash
# Default: 100 epochs, 640px, yolo11n.pt
python train.py

# Quick test run
python train.py --epochs 5 --imgsz 320 --batch 8

# Use a larger model
python train.py --model yolo11s.pt --epochs 100

# Resume interrupted training
python train.py --resume
```

Trained weights are saved to `runs/detect/ppe_detection/weights/best.pt`.

---

## 🔍 Inference

```bash
# Single image
python predict.py --source path/to/image.jpg --show

# Folder of images (save annotated results)
python predict.py --source path/to/images/ --save

# Video
python predict.py --source video.mp4 --save

# Webcam (live)
python predict.py --source 0 --show

# Custom model + confidence
python predict.py --source image.jpg --model runs/detect/ppe_detection/weights/best.pt --conf 0.4
```

---

## 📊 Evaluation

```bash
# Evaluate on validation set
python evaluate.py

# Evaluate on test set with per-class breakdown
python evaluate.py --split test --verbose

# Custom model
python evaluate.py --model path/to/best.pt --split test --verbose
```

Reports **mAP@0.5**, **mAP@0.5:0.95**, **Precision**, and **Recall**.

---

## 📚 References

- [Construction-PPE Dataset — Ultralytics Docs](https://docs.ultralytics.com/datasets/detect/construction-ppe/)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [Dataset Download](https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip)

---

## 📄 License

This project uses the Construction-PPE dataset provided under the [AGPL-3.0 License](https://ultralytics.com/license) by Ultralytics.
