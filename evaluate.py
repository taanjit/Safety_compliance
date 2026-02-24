"""
evaluate.py — Evaluate a trained YOLO model on the Construction-PPE dataset.

Usage:
    python evaluate.py                                   # evaluate on val split
    python evaluate.py --split test                      # evaluate on test split
    python evaluate.py --model path/to/best.pt --split test
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
    DEFAULT_IMGSZ,
    DEFAULT_DEVICE,
    DEFAULT_PROJECT,
    CLASS_NAMES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO model on the Construction-PPE dataset",
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model weights (.pt). "
                             "Defaults to runs/detect/ppe_detection/weights/best.pt")
    parser.add_argument("--data", type=str, default=str(DATASET_YAML),
                        help="Path to dataset YAML config")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help=f"Image size for validation (default: {DEFAULT_IMGSZ})")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                        help="Dataset split to evaluate on (default: val)")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                        help="Device for evaluation")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed per-class metrics")
    return parser.parse_args()


def _resolve_model_path(model_arg: str | None) -> str:
    """Resolve the model weights path."""
    if model_arg:
        return model_arg

    default_weights = Path(DEFAULT_PROJECT) / "ppe_detection" / "weights" / "best.pt"
    if default_weights.exists():
        return str(default_weights)

    print("❌ No trained model found. Train a model first with train.py")
    print(f"   Expected at: {default_weights}")
    sys.exit(1)


def main():
    args = parse_args()
    model_path = _resolve_model_path(args.model)

    print("=" * 60)
    print("  PPE Detection — Evaluation")
    print("=" * 60)
    print(f"  Model      : {model_path}")
    print(f"  Dataset    : {args.data}")
    print(f"  Split      : {args.split}")
    print(f"  Image Size : {args.imgsz}")
    print("=" * 60)

    # Load model
    model = YOLO(model_path)

    # Run validation
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        split=args.split,
        device=args.device if args.device else None,
        project=DEFAULT_PROJECT,
        name="ppe_evaluate",
        exist_ok=True,
        verbose=args.verbose,
    )

    # Print summary metrics
    print("\n" + "=" * 60)
    print("  Evaluation Results")
    print("=" * 60)
    print(f"  mAP@0.5       : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95  : {metrics.box.map:.4f}")
    print(f"  Precision      : {metrics.box.mp:.4f}")
    print(f"  Recall         : {metrics.box.mr:.4f}")

    # Per-class metrics
    if args.verbose and hasattr(metrics.box, "ap50"):
        print("\n  Per-Class AP@0.5:")
        print("  " + "-" * 40)
        for i, ap in enumerate(metrics.box.ap50):
            class_name = CLASS_NAMES.get(i, f"class_{i}")
            print(f"    {class_name:<15s} : {ap:.4f}")

    print("=" * 60)
    print("✅ Evaluation complete!")
    return metrics


if __name__ == "__main__":
    main()
