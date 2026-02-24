#!/bin/bash
# reproduce_issue.sh

TEST_IMG="/home/tadmnit/AI_Team/Anjit/Fire_smoke_detection/datasets/construction-ppe/images/val/image1010.jpg"

echo "Running prediction with DEFAULT confidence (0.8)..."
python3 predict.py --source "$TEST_IMG" --save-txt

echo "---------------------------------------------------"

echo "Running prediction with LOWER confidence (0.25)..."
python3 predict.py --source "$TEST_IMG" --conf 0.25 --save-txt
