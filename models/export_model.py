import os
from ultralytics import YOLO

def export_model():
    """Export YOLOv8 model to ONNX format."""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    try:
        print("Loading YOLOv8m model...")
        # Load a pretrained YOLOv8m model
        model = YOLO('yolov8m.pt')
        
        print("Exporting model to ONNX format...")
        # Export the model to ONNX format
        model.export(format='onnx', imgsz=640)
        
        # Move the exported model to our models directory
        os.rename('yolov8m.onnx', 'models/yolov8m.onnx')
        print("Model exported successfully to models/yolov8m.onnx")
        
    except Exception as e:
        print(f"Error exporting model: {e}")
        print("\nTo fix this, please:")
        print("1. Install the required package: pip install ultralytics")
        print("2. Make sure you have an internet connection")
        print("3. Run this script again")
        return False
    
    return True

if __name__ == "__main__":
    export_model() 