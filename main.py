import cv2
import time
import json
import os
import re
from camera_handler import CameraManager
from object_detector import ObjectDetector
from speech_handler import speak, listen_for_command, classify_intent
import supervision as sv

# --- Configuration Loading ---
def load_config(config_path="config.json"):
    """Loads configuration from a JSON file."""
    # Define comprehensive default structure
    default_config = {
        "camera": {"index": 0},
        "object_detection": {
            "model_path": "models/yolov8m.onnx",
            "classes_path": "coco.names",
            "confidence_threshold": 0.4,
            "nms_threshold": 0.45,
            "input_width": 640,
            "input_height": 640
        },
        "speech": {
            "voice_command_key": "v",
            "intent_confidence_threshold": 0.7,
            "intents": {
                "describe": "describe what is in the image",
                "count": "count objects",
                "unknown": "unknown query"
            }
        },
        "tracking": {"enabled": False} # Simplified default
    }
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}. Using default values.")
        return default_config
    try:
        with open(config_path, 'r') as f:
            config_from_file = json.load(f)
        # Deep merge config from file into defaults (simple approach)
        # You might want a more robust deep merge library for complex configs
        for key, value in default_config.items():
            if isinstance(value, dict):
                config_from_file[key] = {**value, **config_from_file.get(key, {})}
            elif key not in config_from_file:
                config_from_file[key] = value
        return config_from_file
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}. Using default values.")
        return default_config

config = load_config()
app_config = config # Use a more descriptive name
VOICE_COMMAND_KEY = app_config.get("speech", {}).get("voice_command_key", "v")
INTENTS = app_config.get("speech", {}).get("intents", {})
POSSIBLE_INTENT_PHRASES = list(INTENTS.values())
INTENT_CONFIDENCE_THRESHOLD = float(app_config.get("speech", {}).get("intent_confidence_threshold", 0.7))
TRACKING_ENABLED = bool(app_config.get("tracking", {}).get("enabled", False))

# --- Helper Functions ---
def describe_detections(detections: sv.Detections, classes: list[str]) -> str:
    """Generates a spoken description of the detected objects using sv.Detections."""
    if len(detections.class_id) == 0:
        return "I don't see anything specific."

    counts = {}
    for class_id in detections.class_id:
        if 0 <= class_id < len(classes):
            label = classes[class_id]
            counts[label] = counts.get(label, 0) + 1
        else:
            print(f"Warning: Detected class_id {class_id} is out of bounds for classes list.")

    if not counts:
        return "I couldn't identify anything clearly based on loaded classes."

    description_parts = []
    for label, count in counts.items():
        if count == 1:
            description_parts.append(f"a {label}")
        else:
            plural_label = label + 's' if not label.endswith(('s', 'sh', 'ch', 'x', 'z')) else label + 'es'
            description_parts.append(f"{count} {plural_label}")

    if len(description_parts) == 1:
        return f"I see {description_parts[0]}."
    elif len(description_parts) > 1:
        last_part = description_parts.pop()
        return f"I see {', '.join(description_parts)}, and {last_part}."
    else: # Should not happen if counts is populated
        return "I couldn't identify anything clearly."

def count_specific_object(detections: sv.Detections, classes: list[str], target_object_name: str) -> str:
    """Counts a specific object type in the detections."""
    count = 0
    target_label = target_object_name.strip().lower()
    found_label = None

    if len(detections.class_id) > 0:
        for class_id in detections.class_id:
            if 0 <= class_id < len(classes):
                label = classes[class_id].lower()
                if label == target_label:
                    count += 1
                    found_label = classes[class_id] # Get original casing
            else:
                print(f"Warning: Detected class_id {class_id} is out of bounds for classes list during counting.")

    if count == 0:
        return f"I don't see any {target_label}s."
    elif count == 1:
        return f"I see one {found_label}."
    else:
        plural_label = found_label + 's' if not found_label.endswith(('s', 'sh', 'ch', 'x', 'z')) else found_label + 'es'
        return f"I see {count} {plural_label}."

def extract_object_from_command(command: str, intent_phrase: str) -> str | None:
    """Attempts to extract the target object name from a counting command."""
    # Basic approach: remove the intent phrase part and assume the rest is the object
    # Example: command="how many chairs are there", intent_phrase="count objects"
    # This needs improvement for more complex phrases.
    # A simple regex might look for common patterns like "count the X" or "how many X"

    match = re.search(r"(?:count|how many)\s+(?:the\s+)?([a-zA-Z\s]+)(?:\s+are|\s+do|\s+can|\?|$)", command, re.IGNORECASE)
    if match:
        object_name = match.group(1).strip()
        # Basic plural to singular (very naive)
        if object_name.endswith('s'):
            object_name = object_name[:-1]
        return object_name

    # Fallback (less reliable): remove known intent keywords
    processed_command = command.replace("count", "").replace("how many", "").replace("objects", "").strip()
    if processed_command:
        # Basic plural to singular (very naive)
        if processed_command.endswith('s'):
            processed_command = processed_command[:-1]
        return processed_command

    return None # Could not extract

# --- Main Application Logic ---
def main():
    print("Loading configuration...")
    # Initialize camera using config
    try:
        camera = CameraManager(source=app_config["camera"]["index"])
    except Exception as e:
        speak(f"Fatal Error: Could not initialize camera. {e}")
        print(f"Fatal Error initializing camera: {e}")
        return # Exit if camera fails

    # Initialize detector using config
    try:
        detector = ObjectDetector(
            model_path=app_config["object_detection"]["model_path"],
            classes_path=app_config["object_detection"]["classes_path"],
            conf_threshold=float(app_config["object_detection"]["confidence_threshold"]),
            nms_threshold=float(app_config["object_detection"]["nms_threshold"]),
            input_width=int(app_config["object_detection"]["input_width"]),
            input_height=int(app_config["object_detection"]["input_height"]),
            use_tracking=TRACKING_ENABLED
        )
        # Get class names for later use
        classes = detector.classes
    except Exception as e:
        speak(f"Fatal Error: Could not initialize object detector. {e}")
        print(f"Fatal Error initializing object detector: {e}")
        camera.release() # Release camera if detector fails
        return # Exit if detector fails

    print("Starting object detection...")
    print(f"Press '{VOICE_COMMAND_KEY}' to give a voice command.")
    print("Press 'q' to quit")
    speak("Smart Desk Assistant started.")

    last_detections = sv.Detections.empty() # Use empty detections initially

    try:
        while True:
            success, frame = camera.get_frame()
            if not success or frame is None:
                # print("Failed to get frame from camera") # Reduce console noise
                time.sleep(0.1)
                continue

            current_detections = detector.detect_objects(frame)
            last_detections = current_detections # Store the latest valid detections

            # --- Visualization (handled by GUI later, keep basic for now) ---
            annotated_frame = frame.copy() # Work on a copy
            # bounding_box_annotator = sv.BoundingBoxAnnotator()
            # label_annotator = sv.LabelAnnotator()
            # trace_annotator = sv.TraceAnnotator() # For tracking trails

            # annotated_frame = bounding_box_annotator.annotate(
            #     scene=annotated_frame,
            #     detections=current_detections
            # )
            # labels = [
            #     f"{classes[class_id]} {confidence:.2f} {f'(ID:{tracker_id)' if tracker_id else ''}"
            #     for class_id, confidence, tracker_id
            #     in zip(current_detections.class_id, current_detections.confidence, current_detections.tracker_id if TRACKING_ENABLED and current_detections.tracker_id is not None else [None]*len(current_detections))
            # ]
            # annotated_frame = label_annotator.annotate(
            #     scene=annotated_frame,
            #     detections=current_detections,
            #     labels=labels
            # )
            # # Add traces if tracking
            # if TRACKING_ENABLED and current_detections.tracker_id is not None:
            #      annotated_frame = trace_annotator.annotate(
            #          scene=annotated_frame,
            #          detections=current_detections
            #      )

            cv2.imshow("Smart Desk Assistant", annotated_frame)

            # --- Keyboard Input Handling ---
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                speak("Exiting.")
                break
            elif key == ord(VOICE_COMMAND_KEY):
                print("Voice command key pressed.")
                speak("Yes?") # Shorter prompt
                command = listen_for_command()
                if command:
                    intent_key, intent_phrase = classify_intent(command, POSSIBLE_INTENT_PHRASES, INTENT_CONFIDENCE_THRESHOLD)
                    print(f"Heard: '{command}', Detected Intent: {intent_key} ('{intent_phrase}')")

                    if intent_key == "describe":
                        description = describe_detections(last_detections, classes)
                        speak(description)
                    elif intent_key == "count":
                        target_object = extract_object_from_command(command, intent_phrase)
                        if target_object:
                            response = count_specific_object(last_detections, classes, target_object)
                            speak(response)
                        else:
                            speak("Sorry, which object did you want me to count?")
                    # Add elif blocks here for other intents
                    else: # Includes None (low confidence) and "unknown"
                        speak("Sorry, I didn't understand that, or I'm not confident enough about the command.")
                else:
                    speak("I didn't catch that.") # Already handled in listen_for_command
                    pass # Avoid double speaking

    except KeyboardInterrupt:
        print("\nStopping detection (KeyboardInterrupt)...")
        speak("Stopping detection.")
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
        import traceback
        traceback.print_exc()
        speak("An unexpected error occurred. Shutting down.")
    finally:
        print("Releasing resources...")
        camera.release()
        cv2.destroyAllWindows()
        print("Resources released.")
        speak("Goodbye.")

if __name__ == "__main__":
    main() 