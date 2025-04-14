"""Object detection and tracking module for the Smart Desk Assistant project.

This module provides YOLOv8-based object detection and tracking capabilities
using OpenCV's DNN module and the Supervision library.
It is optimized for CPU execution and provides efficient object detection and visualization.
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from bounding_box_processor import BoundingBoxProcessor
import json
import os
import supervision as sv # Import Supervision

# --- Configuration Loading ---
def load_config(config_path="config.json"):
    """Loads configuration from a JSON file."""
    default_config = {
        "object_detection": {
            "model_path": "models/yolov8m.onnx",
            "classes_path": "coco.names",
            "confidence_threshold": 0.4,
            "nms_threshold": 0.45,
            "input_width": 640,
            "input_height": 640
        },
        "tracking": {
            "enabled": False, # Default to disabled if config missing
            "track_thresh": 0.25,
            "track_buffer": 30,
            "match_thresh": 0.8,
            "frame_rate": 30
        }
    }
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}. Using default values.")
        return default_config
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        # Merge with defaults to ensure all keys exist
        config["object_detection"] = {**default_config["object_detection"], **config.get("object_detection", {})}
        config["tracking"] = {**default_config["tracking"], **config.get("tracking", {})}
        return config
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}. Using default values.")
        return default_config

config = load_config()
od_config = config.get("object_detection", {})
tracking_config = config.get("tracking", {})

# Default configuration values (fallbacks)
DEFAULT_MODEL_PATH = od_config.get("model_path", "models/yolov8m.onnx")
DEFAULT_CLASSES_PATH = od_config.get("classes_path", "coco.names")
DEFAULT_CONFIDENCE_THRESHOLD = float(od_config.get("confidence_threshold", 0.4))
DEFAULT_NMS_THRESHOLD = float(od_config.get("nms_threshold", 0.45))
DEFAULT_INPUT_WIDTH = int(od_config.get("input_width", 640))
DEFAULT_INPUT_HEIGHT = int(od_config.get("input_height", 640))
TRACKING_ENABLED = bool(tracking_config.get("enabled", False))


class ObjectDetector:
    """Handles object detection and tracking using YOLOv8 ONNX model via OpenCV DNN and Supervision.

    Attributes:
        net (cv2.dnn_Net): The loaded YOLOv8 ONNX model.
        classes (List[str]): List of class names for detected objects.
        conf_threshold (float): Confidence threshold for detections.
        nms_threshold (float): Non-maximum suppression threshold.
        input_width (int): Input image width for the model.
        input_height (int): Input image height for the model.
        output_layers (List[str]): Names of the network's output layers.
        box_processor (BoundingBoxProcessor): Handles post-processing and NMS.
        tracker (sv.ByteTrack, optional): Object tracker if enabled.
        tracking_enabled (bool): Flag indicating if tracking is enabled.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        classes_path: str = DEFAULT_CLASSES_PATH,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        input_width: int = DEFAULT_INPUT_WIDTH,
        input_height: int = DEFAULT_INPUT_HEIGHT,
        use_tracking: bool = TRACKING_ENABLED
    ) -> None:
        """Initialize the object detector with specified parameters from config or defaults.

        Args:
            model_path (str): Path to the YOLOv8 ONNX model file.
            classes_path (str): Path to the file containing class names.
            conf_threshold (float): Confidence threshold for detections.
            nms_threshold (float): Non-maximum suppression threshold.
            input_width (int): Input image width for the model.
            input_height (int): Input image height for the model.
            use_tracking (bool): Whether to enable object tracking.

        Raises:
            FileNotFoundError: If the model or classes file cannot be found/loaded.
            Exception: For other OpenCV or initialization errors.
        """
        print("Initializing Object Detector...")
        # Store configuration parameters
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_width = input_width
        self.input_height = input_height
        self.tracking_enabled = use_tracking
        self.tracker = None
        self.classes = []
        self.net = None

        # Load class names first
        try:
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            if not self.classes:
                 raise ValueError("Class names file is empty.")
            print(f"Successfully loaded {len(self.classes)} class names from {classes_path}")
        except FileNotFoundError:
            print(f"Fatal Error: Class names file not found at {classes_path}")
            raise
        except Exception as e:
            print(f"Fatal Error loading class names from {classes_path}: {e}")
            raise

        # Load the YOLOv8 model
        if not os.path.exists(model_path):
             print(f"Fatal Error: Model file not found at {model_path}")
             raise FileNotFoundError(f"Model file not found: {model_path}")
        try:
            self.net = cv2.dnn.readNet(model_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print(f"Successfully loaded YOLOv8 model from {model_path}")
        except cv2.error as e:
            print(f"Fatal Error loading YOLOv8 model: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred loading the model: {e}")
            raise

        # Get output layer names
        try:
            self.output_layers = self.net.getUnconnectedOutLayersNames()
            if not self.output_layers:
                raise RuntimeError("Could not get output layer names from the model.")
        except Exception as e:
            print(f"Error getting output layer names: {e}")
            raise

        # Initialize bounding box processor (depends on classes being loaded)
        self.box_processor = BoundingBoxProcessor(
            conf_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            classes=self.classes
        )

        # Initialize tracker if enabled
        if self.tracking_enabled:
            try:
                # Use default settings for ByteTrack initially
                # For customization, check supervision docs for ByteTrackArgs
                self.tracker = sv.ByteTrack(
                    frame_rate=int(tracking_config.get("frame_rate", 30)) # Frame rate is still relevant
                )
                print(f"Object tracker (ByteTrack) initialized with frame_rate={tracking_config.get('frame_rate', 30)}.")
            except Exception as e:
                print(f"Warning: Failed to initialize ByteTrack: {e}. Tracking disabled.")
                speak("Warning: Object tracker could not be initialized.") # Speak warning
                self.tracking_enabled = False # Disable tracking if init fails
                self.tracker = None

        print("Object Detector initialized successfully.")


    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess the input frame for YOLOv8 inference."""
        # Ensure frame is valid
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or invalid.")
        try:
            return cv2.dnn.blobFromImage(
                frame,
                1/255.0,
                (self.input_width, self.input_height),
                swapRB=True,
                crop=False
            )
        except Exception as e:
            print(f"Error during frame preprocessing: {e}")
            raise # Re-raise for caller handling

    def detect_objects(self, frame: np.ndarray) -> sv.Detections:
        """Perform object detection and optional tracking.

        Args:
            frame (np.ndarray): Input frame to process.

        Returns:
            sv.Detections: Supervision Detections object containing xyxy coordinates,
                           confidence, class_id, and potentially tracker_id.
                           Returns empty Detections if error occurs.
        """
        start_time = time.time()

        if self.net is None:
            print("Error: Model not loaded.")
            return sv.Detections.empty() # Return empty detections

        try:
            # Preprocess frame
            blob = self._preprocess(frame)

            # Set input and perform inference
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)

            # Process outputs using the bounding box processor
            # This gives us filtered boxes, confidences, and class IDs
            detections_list = self.box_processor.process_output(
                outputs[0],  # YOLOv8 has a single output
                frame.shape[:2]  # Pass frame height and width
            )

            # Convert to Supervision Detections object
            if not detections_list:
                # print("No raw detections found.")
                return sv.Detections.empty()

            boxes = np.array([d['box'] for d in detections_list]) # [x, y, w, h]
            confidences = np.array([d['confidence'] for d in detections_list])
            class_ids = np.array([self.classes.index(d['label']) for d in detections_list]) # Get class index

            # Convert [x, y, w, h] to [x1, y1, x2, y2] for Supervision
            xyxy = np.zeros((len(boxes), 4))
            xyxy[:, 0] = boxes[:, 0]                 # x1 = x
            xyxy[:, 1] = boxes[:, 1]                 # y1 = y
            xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]   # x2 = x + w
            xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]   # y2 = y + h

            detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidences,
                class_id=class_ids
            )

            # Apply tracking if enabled
            if self.tracking_enabled and self.tracker is not None:
                 # Make sure detections are not empty before tracking
                if len(detections.xyxy) > 0:
                    detections = self.tracker.update_with_detections(detections)
                    # print(f"Tracked Detections: {len(detections.tracker_id) if detections.tracker_id is not None else 0}")
                # else: # No need to update tracker if there are no detections
                    # Pass empty detections to update tracker state if needed by tracker logic (ByteTrack might handle this internally)
                    # self.tracker.update_with_detections(sv.Detections.empty())
                    # print("No detections to track.")
            # else:
                # print("Tracking disabled or tracker not initialized.")

            # print(f"Detection & Tracking Time: {(time.time() - start_time) * 1000:.2f} ms")
            return detections

        except Exception as e:
            print(f"Error during object detection/tracking: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback
            return sv.Detections.empty() # Return empty detections on error

    # Note: draw_boxes is now handled externally or by the BoundingBoxProcessor/GUI
    # to decouple detection logic from specific visualization implementations.
    # You might want a separate Visualizer class later.

    # def draw_boxes(...): # Removed from here
    #    pass

    def draw_boxes(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, any]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """Draw bounding boxes and labels on the frame.
        
        Args:
            frame (np.ndarray): Input frame to draw on.
            detections (List[Dict[str, any]]): List of detections to draw.
            color (Tuple[int, int, int]): Color of the bounding boxes.
            thickness (int): Thickness of the bounding boxes.
            
        Returns:
            np.ndarray: Frame with drawn boxes and labels.
        """
        return self.box_processor.draw_boxes(frame, detections, color, thickness) 