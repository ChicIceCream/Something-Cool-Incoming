"""Camera management module for the Smart Desk Assistant project.

This module provides a robust interface for managing webcam access using OpenCV.
It handles camera initialization, frame capture, and resource cleanup.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import json
import os

# --- Configuration Loading ---
def load_config(config_path="config.json"):
    """Loads configuration from a JSON file."""
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}. Using default values.")
        return {
            "camera": {"index": 0} # Default camera index
        }
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}. Using default values.")
        return {
            "camera": {"index": 0}
        }

config = load_config()
CAMERA_INDEX = config.get("camera", {}).get("index", 0)

class CameraManager:
    """Manages webcam access and frame capture for computer vision tasks.
    
    This class provides a high-level interface for interacting with the system's
    webcam using OpenCV. It handles camera initialization, frame capture, and
    proper resource cleanup.
    
    Attributes:
        cap (cv2.VideoCapture): OpenCV VideoCapture object for camera access.
        source (int): Camera source index used for initialization.
    """
    
    def __init__(self, source: int = CAMERA_INDEX) -> None:
        """Initialize the camera manager with the specified source.
        
        Args:
            source (int): Camera source index. Defaults to value from config or 0.
            
        Raises:
            IOError: If the camera cannot be opened at the specified source.
        """
        self.source = source
        self.cap = None # Initialize cap to None
        try:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                # Use specific error type
                raise ConnectionError(f"Failed to open camera at source index {source}")
            print(f"Successfully initialized camera at source index {source}")
        except Exception as e:
            print(f"Error initializing camera: {e}")
            # Ensure cap is released if partially opened and failed
            if self.cap is not None:
                self.cap.release()
            raise # Re-raise the exception to signal failure
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture a single frame from the camera.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: A tuple containing:
                - bool: Success status of frame capture
                - Optional[np.ndarray]: The captured frame if successful, None otherwise
        """
        if self.cap is None or not self.cap.isOpened():
            print("Warning: Camera is not initialized or opened.")
            return False, None

        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Warning: Failed to capture frame from camera")
                return False, None
            return ret, frame
        except Exception as e:
            print(f"Error reading frame from camera: {e}")
            return False, None
    
    def release(self) -> None:
        """Release the camera resource.
        
        This method safely closes the camera connection and frees system resources.
        """
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            print("Camera resources have been released")
            self.cap = None # Set cap to None after release
    
    def __del__(self) -> None:
        """Cleanup method called when the object is garbage collected.
        
        Ensures that camera resources are properly released even if the object
        is not explicitly released.
        """
        self.release()
