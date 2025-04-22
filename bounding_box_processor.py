"""Bounding box processing module for YOLOv8 model outputs.

This module provides functionality to process raw YOLOv8 model outputs and convert them
into properly formatted bounding boxes with class labels and confidence scores.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import json
import os

# --- Configuration Loading (Specific to BBP) ---
def load_config(config_path="config.json"):
    """Loads relevant configuration from a JSON file."""
    default_config = {
        "object_detection": {
            "classes_path": "coco.names",
            "confidence_threshold": 0.4,
            "nms_threshold": 0.45
        }
    }
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}. Using default BBP values.")
        return default_config["object_detection"]
    try:
        with open(config_path, 'r') as f:
            config = json.load()
        # Merge with defaults
        bbp_config = {**default_config["object_detection"], **config.get("object_detection", {})}
        return bbp_config
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}. Using default BBP values.")
        return default_config["object_detection"]

bbp_config = load_config()
DEFAULT_CONFIDENCE_THRESHOLD = float(bbp_config.get("confidence_threshold", 0.4))
DEFAULT_NMS_THRESHOLD = float(bbp_config.get("nms_threshold", 0.45))
DEFAULT_CLASSES_PATH = bbp_config.get("classes_path", "coco.names")

class BoundingBoxProcessor:
    """Processes YOLOv8 model outputs into bounding boxes with labels and confidence scores."""
    
    def __init__(
        self,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        classes: Optional[List[str]] = None,
        classes_path: str = DEFAULT_CLASSES_PATH
    ) -> None:
        """Initialize the bounding box processor.
        
        Args:
            conf_threshold (float): Confidence threshold for detections from config.
            nms_threshold (float): Non-maximum suppression threshold from config.
            classes (List[str], optional): List of class names. If provided, overrides classes_path.
            classes_path (str): Path to class names file (used if classes is None).
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.classes = classes

        # Load classes from file if not provided directly
        if self.classes is None:
            try:
                with open(classes_path, 'r') as f:
                    self.classes = [line.strip() for line in f.readlines()]
                if not self.classes:
                    raise ValueError(f"Class file is empty: {classes_path}")
                print(f"BBP loaded {len(self.classes)} classes from {classes_path}")
            except FileNotFoundError:
                print(f"Error: BBP could not find classes file: {classes_path}. Labels will be numeric IDs.")
                self.classes = None # Ensure it's None if loading fails
            except Exception as e:
                 print(f"Error loading class file {classes_path} in BBP: {e}")
                 self.classes = None

    def process_output(
        self,
        output: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> List[Dict[str, any]]:
        """Process YOLOv8 model output tensor into a list of detection dictionaries.
        
        Args:
            output (np.ndarray): Raw model output tensor of shape [1, 84, 8400]
            frame_shape (Tuple[int, int]): Shape of the input frame (height, width)
            
        Returns:
            List[Dict[str, any]]: List of filtered detections, each containing:
                - label (str): Class name or ID
                - confidence (float): Detection confidence
                - box (List[int]): Bounding box [x, y, w, h]
        """
        frame_height, frame_width = frame_shape
        class_ids = []
        confidences = []
        boxes = []
        
        # Process the output tensor: [1, 84, 8400] -> [8400, 84]
        try:
            output = output[0].transpose(1, 0)
        except IndexError:
             print("Error: Unexpected model output shape. Expected [1, 84, 8400]")
             return [] # Return empty list on shape error

        for detection in output:
            scores = detection[4:] # Class probabilities start at index 4
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > self.conf_threshold:
                # Box coordinates are normalized [center_x, center_y, width, height]
                center_x = detection[0] * frame_width
                center_y = detection[1] * frame_height
                w = detection[2] * frame_width
                h = detection[3] * frame_height
                
                # Calculate top-left corner (x, y)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, int(w), int(h)]) # Store as int
                confidences.append(float(confidence))
                class_ids.append(class_id)
        
        # Apply non-maximum suppression
        try:
            indices = cv2.dnn.NMSBoxes(
                boxes,
                confidences,
                self.conf_threshold,
                self.nms_threshold
            )
        except Exception as e:
            print(f"Error during NMS: {e}")
            indices = [] # Proceed without NMS if it fails
        
        # Format results
        detections_list = []
        if len(indices) > 0:
             # Ensure indices are handled correctly (can be 1D or 2D array)
            processed_indices = indices.flatten() if hasattr(indices, 'flatten') else indices
            for i in processed_indices:
                try:
                    box = boxes[i]
                    confidence = confidences[i]
                    class_id = class_ids[i]
                    
                    # Use class name if available, otherwise use ID
                    if self.classes and 0 <= class_id < len(self.classes):
                         label = self.classes[class_id]
                    else:
                         label = f"ID:{class_id}" # Use ID if classes missing or out of bounds

                    detections_list.append({
                        "label": label,
                        "confidence": confidence,
                        "box": box
                    })
                except IndexError:
                    print(f"Warning: Index {i} out of bounds during NMS processing. Max index: {len(boxes)-1}")
                    continue # Skip this problematic index

        return detections_list

    # Visualization is now separate, this class focuses only on processing.
    # def draw_boxes(...): # Removed
    #     pass 