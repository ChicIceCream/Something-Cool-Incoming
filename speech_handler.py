import speech_recognition as sr
import pyttsx3
import time
from transformers import pipeline
import json
import os

# --- Configuration Loading ---
def load_config(config_path="config.json"):
    """Loads configuration from a JSON file."""
    default_config = {
        "speech": {
            "intent_confidence_threshold": 0.7,
            "ambient_noise_duration": 1,
            "listen_timeout": 5,
            "phrase_time_limit": 7,
             "intents": {
                 "describe": "describe what is in the image",
                 "count": "count objects",
                 "unknown": "unknown query"
             }
        }
    }
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}. Using default speech values.")
        return default_config
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        # Merge with defaults
        config["speech"] = {**default_config["speech"], **config.get("speech", {})}
        config["speech"]["intents"] = {**default_config["speech"]["intents"], **config.get("speech", {}).get("intents", {})}
        return config
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}. Using default speech values.")
        return default_config

config = load_config()
speech_config = config.get("speech", {})
INTENT_CONFIDENCE_THRESHOLD = float(speech_config.get("intent_confidence_threshold", 0.7))
AMBIENT_NOISE_DURATION = float(speech_config.get("ambient_noise_duration", 1.0))
LISTEN_TIMEOUT = int(speech_config.get("listen_timeout", 5))
PHRASE_TIME_LIMIT = int(speech_config.get("phrase_time_limit", 7)) # Increased default
POSSIBLE_INTENTS = list(speech_config.get("intents", {}).values())
INTENT_LABELS = speech_config.get("intents", {})

# --- Initialize TTS Engine ---
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
except Exception as e:
    print(f"Error initializing TTS engine: {e}")
    speak("Warning: Text to speech engine could not be initialized.") # Speak warning
    engine = None

# --- Initialize Speech Recognizer ---
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Adjusting for ambient noise, please wait...")
    try:
        r.adjust_for_ambient_noise(source, duration=AMBIENT_NOISE_DURATION)
        print("Ready to listen.")
    except Exception as e:
        print(f"Error adjusting for ambient noise: {e}")
        speak("Warning: Could not adjust for ambient noise.") # Speak warning

# --- Initialize NLP Pipeline (Zero-Shot Classification) ---
print("Loading NLP intent classification model (this may take a moment on first run)...")
intent_classifier = None # Initialize as None
try:
    intent_classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")
    print("NLP model loaded successfully.")
except Exception as e:
    print(f"Error loading NLP model: {e}. Intent classification will be disabled.")
    speak("Warning: Intent classification model could not be loaded.") # Speak warning
    intent_classifier = None

# --- Speech Functions ---
def speak(text: str):
    """Converts the given text to speech."""
    if engine:
        try:
            print(f"Speaking: {text}")
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            # Avoid speaking during an error in speak itself
            print(f"Error during speech synthesis: {e}")
    else:
        # If engine failed, print instead of trying to speak
        print(f"TTS Engine inactive. Would speak: {text}")

def listen_for_command(timeout: int = LISTEN_TIMEOUT, phrase_limit: int = PHRASE_TIME_LIMIT) -> str | None:
    """Listens for a command from the microphone and returns the recognized text."""
    with sr.Microphone() as source:
        print("Listening for command...")
        try:
            # Listen for audio input with adjusted timeout and phrase limit
            # Increased phrase_time_limit might help capture the end of sentences
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
            print("Recognizing...")
            command = r.recognize_google(audio)
            print(f"You said: {command}")
            return command.lower()
        except sr.WaitTimeoutError:
            print("No command heard within the time limit.")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            speak("Sorry, I could not understand what you said.") # Speak feedback
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            speak("Sorry, I'm having trouble connecting to the speech service.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during listening: {e}")
            speak("An unexpected error occurred while listening.")
            return None

# --- NLP Function ---
def classify_intent(text: str, candidate_labels: list[str] = POSSIBLE_INTENTS, confidence_threshold: float = INTENT_CONFIDENCE_THRESHOLD) -> tuple[str | None, str | None]:
    """Classifies the intent of the text using zero-shot classification.

    Args:
        text: The input text (spoken command).
        candidate_labels: A list of possible intent phrases from config.
        confidence_threshold: Minimum confidence score from config.

    Returns:
        tuple[str | None, str | None]: A tuple containing:
            - The key of the classified intent (e.g., "describe", "count") if confidence is above threshold, otherwise None.
            - The full text of the matched intent phrase if confidence is above threshold, otherwise None.
    """
    if not intent_classifier:
        print("Intent classifier not available.")
        speak("Intent classification is currently unavailable.")
        return None, None

    if not text or not candidate_labels:
        return None, None

    try:
        result = intent_classifier(text, candidate_labels)
        print(f"Intent Classification Result: {result}")

        top_label_phrase = result['labels'][0]
        top_score = result['scores'][0]

        if top_score >= confidence_threshold:
            # Find the key corresponding to the matched phrase
            intent_key = None
            for key, phrase in INTENT_LABELS.items():
                if phrase == top_label_phrase:
                    intent_key = key
                    break
            return intent_key, top_label_phrase
        else:
            print(f"Intent confidence ({top_score:.2f}) below threshold ({confidence_threshold}).")
            return None, None
    except Exception as e:
        print(f"Error during intent classification: {e}")
        speak("Sorry, an error occurred during intent classification.")
        return None, None

# --- Test Block ---
if __name__ == '__main__':
    speak("Speech handler test mode.")

    # Test intent classification
    test_command = "how many chairs can you see"
    intent_key, intent_phrase = classify_intent(test_command)
    if intent_key:
        speak(f"The intent for '{test_command}' is likely '{intent_key}' ('{intent_phrase}').")
    else:
        speak(f"Could not confidently determine the intent for '{test_command}'.")

    # Test listening
    speak("Now, say something like 'count the bottles' or 'describe the image'.")
    command = listen_for_command()
    if command:
        intent_key, intent_phrase = classify_intent(command)
        if intent_key:
            speak(f"I heard '{command}'. The intent seems to be '{intent_key}'.")
        else:
            speak(f"I heard '{command}', but I'm not sure of the intent.")
    else:
        speak("I didn't hear anything clearly.") 