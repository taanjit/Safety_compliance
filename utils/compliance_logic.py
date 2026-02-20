"""
compliance_logic.py — Heuristic logic to infer missing PPE.

This module provides "Strict Mode" compliance checking.
Instead of relying solely on the model's "no_helmet" / "no_gloves" detections (which can be unreliable),
this logic checks every detected PERSON.

If a Person is detected, we check if they have overlapping bounding boxes with required PPE (Helmet, Gloves, etc.).
If the PPE is not found on the person, we EXPLICITLY report a violation (e.g., "no_helmet").
"""

import numpy as np
from config.settings import CLASS_NAMES, WORN_PPE_CLASSES, MISSING_PPE_CLASSES

# Mapping of required PPE for a person
# If a person is detected, they MUST have these (or else we infer "no_X")
REQUIRED_PPE = {
    0: "no_helmet",  # If class 0 (helmet) is missing -> report "no_helmet"
    # 1: "no_gloves", # If class 1 (gloves) is missing -> report "no_gloves" 
    # (Note: gloves are tricky because they might be obscured; enabling strict check for now per user request)
    # 2: "no_vest",
    # 3: "no_boots",
}

# Add gloves to required list if strict checking is desired
REQUIRED_PPE_NAMES = {
    "helmet": "no_helmet",
    "gloves": "no_gloves",
    "boots": "no_boots",
    "vest": "no_vest",
    "goggles": "no_goggle"
}

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    Boxes are [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0
    return intersection_area / union_area


def is_overlapping(person_box, ppe_box, threshold=0.01):
    """
    Check if PPE box overlaps significantly with Person box.
    Simple IoU or containment check.
    For PPE, often the PPE is small and inside the Person box.
    """
    # Check if PPE is mostly INSIDE the person box (containment)
    # or if there is simply an intersection.
    
    # Simple intersection check
    x1 = max(person_box[0], ppe_box[0])
    y1 = max(person_box[1], ppe_box[1])
    x2 = min(person_box[2], ppe_box[2])
    y2 = min(person_box[3], ppe_box[3])
    
    if x2 > x1 and y2 > y1:
        return True # Any overlap is considered "associated" for now
    
    return False


def check_compliance_strict(results):
    """
    Analyze YOLO results and enforce strict PPE compliance.
    
    Args:
        results: ultralytics Results object (single frame)
        
    Returns:
        summary: Dict with counts of worn/missing PPE.
        annotations: List of dicts {'box', 'label', 'color'} to draw on the frame.
    """
    if results.boxes is None or len(results.boxes) == 0:
        return {
            "total_persons": 0,
            "worn_ppe": {},
            "missing_ppe": {},
            "is_compliant": True,
            "inferred_violations": [] 
        }, []

    persons = []
    ppe_items = []

    # 1. Separate Persons and PPE
    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist() # [x1, y1, x2, y2]
        
        item = {
            "cls_id": cls_id,
            "conf": conf,
            "box": xyxy,
            "name": CLASS_NAMES.get(cls_id, "unknown")
        }

        if cls_id == 6: # Person
            persons.append(item)
        elif cls_id in WORN_PPE_CLASSES:
            ppe_items.append(item)
        elif cls_id in MISSING_PPE_CLASSES:
            # We still track existing "no_X" detections if the model produces them
            ppe_items.append(item)


    # 2. Results Data Structure
    summary = {
        "total_persons": len(persons),
        "worn_ppe": {},
        "missing_ppe": {},
        "is_compliant": True,
        "inferred_violations": [] 
    }
    
    annotations = []
    # Red color for violations
    COLOR_VIOLATION = (0, 0, 220) 

    # 3. For each Person, check which PPE they have
    for person in persons:
        # What PPE is this person wearing?
        person_worn = set()
        
        for ppe in ppe_items:
            # Skip if it's a "missing" class (no_helmet etc), we only care if they are WEARING it
            if ppe["cls_id"] in MISSING_PPE_CLASSES:
                continue
                
            if is_overlapping(person["box"], ppe["box"]):
                person_worn.add(ppe["name"])
        
        # Check against Requirements
        # We assume EVERY person needs: Helmet, Gloves, Boots. 
        required_names = ["helmet", "gloves", "boots"]
        
        person_violations = []
        for req in required_names:
            if req not in person_worn:
                # VIOLATION INFERRED
                violation_name = REQUIRED_PPE_NAMES[req]
                summary["missing_ppe"][violation_name] = summary["missing_ppe"].get(violation_name, 0) + 1
                summary["is_compliant"] = False
                person_violations.append(violation_name)
        
        # Create an annotation for this person if they have violations
        if person_violations:
            label = ", ".join(person_violations)
            annotations.append({
                "box": person["box"],
                "label": label,
                "color": COLOR_VIOLATION
            })
                
    
    # 4. Also add count of actual PPE items detected (worn)
    for ppe in ppe_items:
        if ppe["cls_id"] in WORN_PPE_CLASSES:
            name = ppe["name"]
            summary["worn_ppe"][name] = summary["worn_ppe"].get(name, 0) + 1
            
    return summary, annotations
