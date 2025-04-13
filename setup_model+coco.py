import os
import urllib.request

def setup_object_detection():
    """Set up the required files for object detection."""
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Download YOLOv5s ONNX model
    model_url = "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/yolov5/model/yolov5s.onnx"
    model_path = "models/yolov5s.onnx"
    
    if not os.path.exists(model_path):
        print("Downloading YOLOv5s ONNX model...")
        try:
            urllib.request.urlretrieve(model_url, model_path)
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False
    else:
        print("Model file already exists.")
    
    # Create COCO classes file
    coco_classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    
    classes_path = "models/coco.names"
    if not os.path.exists(classes_path):
        print("Creating COCO classes file...")
        try:
            with open(classes_path, 'w') as f:
                for class_name in coco_classes:
                    f.write(f"{class_name}\n")
            print("COCO classes file created successfully!")
        except Exception as e:
            print(f"Error creating classes file: {e}")
            return False
    else:
        print("Classes file already exists.")
    
    return True

if __name__ == "__main__":
    if setup_object_detection():
        print("\nSetup completed successfully!")
        print("You can now use the ObjectDetector class with:")
        print("- Model path: models/yolov5s.onnx")
        print("- Classes path: models/coco.names")
    else:
        print("\nSetup failed. Please check the error messages above.")
