import cv2
import time
from camera_handler import CameraManager
from object_detector import ObjectDetector

def main():
    """Test the camera and object detection functionality."""
    print("Initializing camera...")
    camera = CameraManager()  # Camera is initialized in the constructor
    
    print("Initializing object detector...")
    detector = ObjectDetector(
        model_path="models/yolov8m.onnx",
        classes_path="coco.names"  # Explicitly specify the classes file
    )  # Detector is initialized in the constructor

    print("\nStarting object detection test...")
    print("Press 'q' to quit")

    try:
        while True:
            # Get frame from camera
            success, frame = camera.get_frame()
            if not success or frame is None:
                print("Failed to get frame from camera")
                break

            # Detect objects
            start_time = time.time()
            detections = detector.detect_objects(frame)
            end_time = time.time()
            detection_time = (end_time - start_time) * 1000  # Convert to milliseconds

            # Print detections
            if detections:
                print("\nDetected objects:")
                for detection in detections:
                    print(f"- {detection['label']}: {detection['confidence']:.2f}")

            # Draw bounding boxes
            detector.draw_boxes(frame, detections)

            # Print detection time
            print(f"Detection time: {detection_time:.2f}ms")

            # Display frame
            cv2.imshow("Object Detection", frame)

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera resources have been released")

if __name__ == "__main__":
    main() 