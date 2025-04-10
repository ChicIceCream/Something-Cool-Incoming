import os
import sys
import subprocess

def download_model():
    """Download and export YOLOv8 model to ONNX format."""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Check if model already exists
    model_path = "models/yolov8m.onnx"
    if os.path.exists(model_path):
        print("Model file already exists.")
        return True
    
    try:
        # Install required packages
        print("Installing required packages...")
        
        # First, uninstall any existing ONNX installation
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "onnx", "-y"])
        
        # Install specific version of ONNX that works well on Windows
        subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx==1.14.0"])
        
        # Install ultralytics package if not already installed
        print("Checking for ultralytics package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        
        # Run the export script
        print("Exporting YOLOv8 model to ONNX format...")
        subprocess.check_call([sys.executable, "models/export_model.py"])
        
        print("Model downloaded and exported successfully!")
        return True
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTo fix this, please try the following steps manually:")
        print("1. Uninstall ONNX: pip uninstall onnx")
        print("2. Install specific ONNX version: pip install onnx==1.14.0")
        print("3. Install ultralytics: pip install ultralytics")
        print("4. Run the export script: python models/export_model.py")
        return False

if __name__ == "__main__":
    download_model()
