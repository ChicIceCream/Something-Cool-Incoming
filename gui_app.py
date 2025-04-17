import tkinter as tk
from tkinter import ttk, scrolledtext
import cv2
from PIL import Image, ImageTk
import threadingtrial.c
import time
import queue
import json
import os
import supervision as sv

# Import your existing modules
from camera_handler import CameraManager
from object_detector import ObjectDetector
from speech_handler import speak, listen_for_command, classify_intent
from main import load_config, describe_detections, count_specific_object, extract_object_from_command # Reuse functions from main

# --- Configuration Loading (Copied from main.py for standalone GUI logic if needed) ---
# Or preferably, refactor config loading into a separate module later
app_config = load_config() # Load config using the function from main
VOICE_COMMAND_KEY = app_config.get("speech", {}).get("voice_command_key", "v")
INTENTS = app_config.get("speech", {}).get("intents", {})
POSSIBLE_INTENT_PHRASES = list(INTENTS.values())
INTENT_CONFIDENCE_THRESHOLD = float(app_config.get("speech", {}).get("intent_confidence_threshold", 0.7))
TRACKING_ENABLED = bool(app_config.get("tracking", {}).get("enabled", False))

# --- GUI Application Class ---
class SmartDeskGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Desk Assistant")
        self.root.geometry("1000x750") # Adjust size as needed

        # --- State Variables ---
        self.is_running = True
        self.last_detections = sv.Detections.empty()
        self.classes = [] # Will be populated by detector
        self.status_message = tk.StringVar(value="Initializing...")

        # --- Queues for Thread Communication ---
        self.frame_queue = queue.Queue(maxsize=1) # Queue for annotated frames
        self.status_queue = queue.Queue(maxsize=5) # Queue for status updates
        self.detection_queue = queue.Queue(maxsize=5) # Queue for detection lists

        # --- Initialize Core Components (Error Handling Needed) ---
        try:
            self.camera = CameraManager(source=app_config["camera"]["index"])
        except Exception as e:
            self.update_status(f"Error initializing camera: {e}")
            speak(f"Error initializing camera: {e}")
            self.camera = None
            # Consider disabling features or showing error state

        try:
            self.detector = ObjectDetector(
                model_path=app_config["object_detection"]["model_path"],
                classes_path=app_config["object_detection"]["classes_path"],
                conf_threshold=float(app_config["object_detection"]["confidence_threshold"]),
                nms_threshold=float(app_config["object_detection"]["nms_threshold"]),
                input_width=int(app_config["object_detection"]["input_width"]),
                input_height=int(app_config["object_detection"]["input_height"]),
                use_tracking=TRACKING_ENABLED
            )
            self.classes = self.detector.classes # Store class names
        except Exception as e:
            self.update_status(f"Error initializing detector: {e}")
            speak(f"Error initializing detector: {e}")
            self.detector = None
            # Consider disabling features or showing error state

        # --- Create GUI Widgets --- #
        self.create_widgets()

        # --- Supervision Annotators --- #
        self.bounding_box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator() if TRACKING_ENABLED else None

        # --- Start Background Processing Thread --- #
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()

        # --- Start GUI Update Loop --- #
        self.update_gui()

        # --- Bind Keys --- #
        self.root.bind(f'<{VOICE_COMMAND_KEY}>', self.handle_voice_command_event)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing) # Handle window close

        self.update_status("Ready.")
        speak("Smart Desk Assistant GUI started.")

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        # Left side: Video Feed
        video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="10")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(expand=True)

        # Right side: Detections and Controls (Placeholder)
        right_panel = ttk.Frame(main_frame, padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5)

        detections_frame = ttk.LabelFrame(right_panel, text="Detected Objects", padding="10")
        detections_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Using scrolled text for detections temporarily
        self.detections_text = scrolledtext.ScrolledText(detections_frame, wrap=tk.WORD, width=30, height=20)
        self.detections_text.pack(fill=tk.BOTH, expand=True)
        self.detections_text.config(state=tk.DISABLED) # Make read-only

        controls_frame = ttk.LabelFrame(right_panel, text="Controls", padding="10")
        controls_frame.pack(fill=tk.X, pady=5)

        self.voice_button = ttk.Button(controls_frame, text=f"Voice Cmd ({VOICE_COMMAND_KEY.upper()})", command=self.handle_voice_command)
        self.voice_button.pack(pady=5)

        # Status Bar
        status_bar = ttk.Label(self.root, textvariable=self.status_message, relief=tk.SUNKEN, anchor=tk.W, padding="2 5")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def processing_loop(self):
        """Runs camera capture and detection in a separate thread."""
        while self.is_running:
            if self.camera is None or self.detector is None:
                self.update_status("Error: Camera or Detector not initialized.")
                time.sleep(1)
                continue

            start_time = time.time()
            success, frame = self.camera.get_frame()

            if not success or frame is None:
                time.sleep(0.1) # Avoid busy loop if frame capture fails
                continue

            try:
                # Perform detection
                detections = self.detector.detect_objects(frame)
                self.last_detections = detections # Store for voice commands

                # Annotate frame using Supervision
                annotated_frame = frame.copy()
                annotated_frame = self.bounding_box_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections
                )
                # Generate labels with confidence and track ID
                labels = []
                if len(detections) > 0:
                    # Handle cases where tracker_id might be None or shorter than detections
                    tracker_ids = detections.tracker_id if TRACKING_ENABLED and detections.tracker_id is not None else [None] * len(detections)
                    if len(tracker_ids) != len(detections):
                         tracker_ids = [None] * len(detections) # Fallback if lengths mismatch

                    for class_id, confidence, tracker_id in zip(detections.class_id, detections.confidence, tracker_ids):
                        label_text = f"{self.classes[class_id]} {confidence:.2f}"
                        if tracker_id is not None:
                            label_text += f" (ID:{tracker_id})"
                        labels.append(label_text)

                annotated_frame = self.label_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections,
                    labels=labels
                )
                # Add traces if tracking
                if TRACKING_ENABLED and self.trace_annotator is not None and detections.tracker_id is not None:
                     annotated_frame = self.trace_annotator.annotate(
                         scene=annotated_frame,
                         detections=detections
                     )

                # Put annotated frame in queue for GUI thread
                try:
                    self.frame_queue.put_nowait(annotated_frame)
                except queue.Full:
                    pass # Ignore if queue is full, GUI will catch up

                # Update detection list for GUI
                detection_summary = self.format_detections_for_gui(detections)
                try:
                    self.detection_queue.put_nowait(detection_summary)
                except queue.Full:
                    pass

            except Exception as e:
                self.update_status(f"Error during processing: {e}")
                # Optionally put error frame or skip frame update
                import traceback
                traceback.print_exc()
                time.sleep(0.5) # Pause briefly after an error

            # Optional: Maintain approximate frame rate if needed
            # elapsed = time.time() - start_time
            # time.sleep(max(0, 1/TARGET_FPS - elapsed))

        print("Processing loop finished.")

    def update_gui(self):
        """Periodically updates the GUI elements from queues."""
        # Update video frame
        try:
            frame = self.frame_queue.get_nowait()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            # Resize smoothly for display if needed (consider aspect ratio)
            # base_width = self.video_label.winfo_width()
            # if base_width > 1: # Ensure width is valid
            #    w_percent = (base_width / float(img_pil.size[0]))
            #    h_size = int((float(img_pil.size[1]) * float(w_percent)))
            #    img_pil = img_pil.resize((base_width, h_size), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img_pil)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        except queue.Empty:
            pass # No new frame, do nothing
        except Exception as e:
            print(f"Error updating video frame: {e}")

        # Update status bar
        try:
            while not self.status_queue.empty(): # Process all pending status messages
                status = self.status_queue.get_nowait()
                self.status_message.set(status)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error updating status bar: {e}")

        # Update detections list
        try:
            detection_summary = self.detection_queue.get_nowait()
            self.detections_text.config(state=tk.NORMAL)
            self.detections_text.delete(1.0, tk.END) # Clear previous
            self.detections_text.insert(tk.END, detection_summary)
            self.detections_text.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error updating detections list: {e}")

        # Schedule next update
        if self.is_running:
            self.root.after(30, self.update_gui) # ~30 FPS refresh rate

    def format_detections_for_gui(self, detections: sv.Detections) -> str:
        """Formats the detection list for display in the GUI."""
        if len(detections) == 0:
            return "No objects detected."

        lines = []
        tracker_ids = detections.tracker_id if TRACKING_ENABLED and detections.tracker_id is not None else [None] * len(detections)
        if len(tracker_ids) != len(detections):
            tracker_ids = [None] * len(detections) # Fallback

        for i in range(len(detections)):
            class_id = detections.class_id[i]
            label = self.classes[class_id] if 0 <= class_id < len(self.classes) else f"ID:{class_id}"
            confidence = detections.confidence[i]
            tracker_id = tracker_ids[i]
            line = f"- {label} ({confidence:.2f})"
            if tracker_id is not None:
                line += f" [ID:{tracker_id}]"
            lines.append(line)

        return "\n".join(lines)

    def update_status(self, message):
        """Puts a status message onto the queue for the GUI thread."""
        try:
            self.status_queue.put_nowait(message)
        except queue.Full:
            print("Status queue full, message dropped.")

    def handle_voice_command_event(self, event=None):
        """Handles keypress event for voice command."""
        self.handle_voice_command() # Call the main handler

    def handle_voice_command(self):
        """Handles the logic for initiating voice command listening."""
        # Run in a separate thread to avoid blocking GUI
        threading.Thread(target=self._voice_command_thread, daemon=True).start()

    def _voice_command_thread(self):
        """Worker thread for handling a single voice command interaction."""
        self.update_status("Listening...")
        speak("Yes?")
        command = listen_for_command()

        if command:
            self.update_status(f"Processing: '{command}'")
            intent_key, intent_phrase = classify_intent(command, POSSIBLE_INTENT_PHRASES, INTENT_CONFIDENCE_THRESHOLD)
            print(f"Heard: '{command}', Detected Intent: {intent_key} ('{intent_phrase}')")

            response = ""
            if intent_key == "describe":
                response = describe_detections(self.last_detections, self.classes)
            elif intent_key == "count":
                target_object = extract_object_from_command(command, intent_phrase)
                if target_object:
                    response = count_specific_object(self.last_detections, self.classes, target_object)
                else:
                    response = "Sorry, which object did you want me to count?"
            else:
                response = "Sorry, I didn't understand that, or I'm not confident enough."

            speak(response)
            self.update_status("Ready.")
        else:
            self.update_status("Ready.") # Reset status if nothing heard
            # speak("I didn't catch that.") # Already handled in listen_for_command

    def on_closing(self):
        """Handles window closing event."""
        print("Closing application...")
        self.is_running = False # Signal processing thread to stop
        # Wait briefly for thread to finish (optional, might block GUI slightly)
        # self.processing_thread.join(timeout=1.0)
        if self.camera:
            self.camera.release()
        self.root.destroy()
        print("Application closed.")
        # Explicitly exit to ensure TTS stops if needed
        # os._exit(0)

# --- Main Execution --- #
if __name__ == "__main__":
    root = tk.Tk()
    app = SmartDeskGUI(root)
    root.mainloop() 