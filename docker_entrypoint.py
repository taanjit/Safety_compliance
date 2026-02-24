"""
docker_entrypoint.py — Docker entrypoint for PPE Detection scheduler.

Reads input/output/confidence from environment variables or CLI args,
overrides settings at runtime, and starts the scheduler.

Environment Variables:
    INPUT_DIR      — Path to input folder  (default: /data/input)
    OUTPUT_DIR     — Path to output folder (default: /data/output)
    CONFIDENCE     — Confidence threshold  (default: 0.5)
    FRAME_SKIP     — Frame skip rate       (default: 10)
    POLL_INTERVAL  — Polling interval in s (default: 30)
    IMGSZ          — Inference image size  (default: 640)
    DEVICE         — Device for inference  (default: "")
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config.settings as settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPE Detection Docker Entrypoint",
    )
    parser.add_argument("--input-dir", type=str,
                        default=os.environ.get("INPUT_DIR", "/data/input"),
                        help="Path to input video folder")
    parser.add_argument("--output-dir", type=str,
                        default=os.environ.get("OUTPUT_DIR", "/data/output"),
                        help="Path to output folder")
    parser.add_argument("--conf", type=float,
                        default=float(os.environ.get("CONFIDENCE", "0.5")),
                        help="Confidence threshold")
    parser.add_argument("--frame-skip", type=int,
                        default=int(os.environ.get("FRAME_SKIP", "10")),
                        help="Process every Nth frame")
    parser.add_argument("--poll", type=int,
                        default=int(os.environ.get("POLL_INTERVAL", "30")),
                        help="Polling interval in seconds")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model weights (.pt)")
    parser.add_argument("--imgsz", type=int,
                        default=int(os.environ.get("IMGSZ", "640")),
                        help="Inference image size")
    parser.add_argument("--device", type=str,
                        default=os.environ.get("DEVICE", ""),
                        help="Device for inference (e.g. 0 or cpu)")
    parser.add_argument("--once", action="store_true",
                        help="Run a single scan and exit")
    return parser.parse_args()


def main():
    args = parse_args()

    # Override global settings with runtime values
    settings.INPUT_DIR = Path(args.input_dir)
    settings.OUTPUT_DIR = Path(args.output_dir)
    settings.LOG_FILE = settings.OUTPUT_DIR / "processed.log"

    print("=" * 60)
    print("  PPE Detection — Docker Container")
    print("=" * 60)
    print(f"  Input Dir   : {settings.INPUT_DIR}")
    print(f"  Output Dir  : {settings.OUTPUT_DIR}")
    print(f"  Confidence  : {args.conf}")
    print(f"  Frame Skip  : {args.frame_skip}")
    print(f"  Poll Interval: {args.poll}s")
    print("=" * 60)

    # Build scheduler CLI args and delegate to scheduler.main()
    sys.argv = [
        "scheduler.py",
        "--poll", str(args.poll),
        "--frame-skip", str(args.frame_skip),
        "--conf", str(args.conf),
        "--imgsz", str(args.imgsz),
        "--device", args.device,
    ]
    if args.model:
        sys.argv += ["--model", args.model]
    if args.once:
        sys.argv.append("--once")

    # Import and run scheduler
    from scheduler import main as scheduler_main
    scheduler_main()


if __name__ == "__main__":
    main()
