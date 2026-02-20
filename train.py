"""
train.py — Train a YOLO model on the Construction-PPE dataset.

Usage:
    python train.py                          # defaults (100 epochs, 640px)
    python train.py --epochs 50 --imgsz 320  # quick run
    python train.py --model yolo11s.pt       # use a larger model variant
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from config.settings import (
    DATASET_YAML,
    DEFAULT_MODEL,
    DEFAULT_EPOCHS,
    DEFAULT_IMGSZ,
    DEFAULT_BATCH,
    DEFAULT_DEVICE,
    DEFAULT_PROJECT,
    DEFAULT_NAME,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO on the Construction-PPE dataset",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Pretrained model checkpoint (default: {DEFAULT_MODEL})")
    parser.add_argument("--data", type=str, default=str(DATASET_YAML),
                        help="Path to dataset YAML config")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help=f"Input image size (default: {DEFAULT_IMGSZ})")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help=f"Batch size (default: {DEFAULT_BATCH})")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                        help="Device to train on (e.g. '0', 'cpu', '' for auto)")
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT,
                        help="Output project directory")
    parser.add_argument("--name", type=str, default=DEFAULT_NAME,
                        help="Experiment name")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  PPE Detection — Training")
    print("=" * 60)
    print(f"  Model      : {args.model}")
    print(f"  Dataset    : {args.data}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Image Size : {args.imgsz}")
    print(f"  Batch Size : {args.batch}")
    print(f"  Device     : {args.device or 'auto'}")
    print(f"  Output     : {args.project}/{args.name}")
    print("=" * 60)

    # Load model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        exist_ok=True,
        resume=args.resume,
    )

    print("\n✅ Training complete!")
    print(f"   Best weights saved to: {args.project}/{args.name}/weights/best.pt")

    return results


if __name__ == "__main__":
    main()
