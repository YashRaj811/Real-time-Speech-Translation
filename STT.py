import os
import pickle
import torch
import whisper
import sounddevice as sd
import numpy as np
import queue
import time
from transformers import MarianMTModel, MarianTokenizer
import pyttsx3  

# ===== SETTINGS =====
PICKLE_FILE_EN_FR = "marian_en_fr.pkl"
PICKLE_FILE_FR_EN = "marian_fr_en.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SILENCE_THRESHOLD = 0.01  # adjust based on environment noise
SILENCE_DURATION = 3      # seconds

# ===== STEP 1: Load Whisper =====
print("Loading Whisper model...")
whisper_model = whisper.load_model("base", device=DEVICE)

# ===== STEP 2: Function to load MarianMT model (pickled or fresh) =====
def load_translation_model(model_name, pickle_file):
    if os.path.exists(pickle_file):
        print(f"Loading model from {pickle_file}...")
        with open(pickle_file, "rb") as f:
            saved = pickle.load(f)
        model = saved["model"].to(DEVICE)
        tokenizer = saved["tokenizer"]
    else:
        print(f"Downloading model {model_name} from Hugging Face...")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(DEVICE)
        with open(pickle_file, "wb") as f:
            pickle.dump({"model": model, "tokenizer": tokenizer}, f)
        print(f"Model saved to {pickle_file}")
    return model, tokenizer

# ===== STEP 3: Load both EN→FR and FR→EN models =====
model_en_fr, tokenizer_en_fr = load_translation_model("Helsinki-NLP/opus-mt-en-fr", PICKLE_FILE_EN_FR)
model_fr_en, tokenizer_fr_en = load_translation_model("Helsinki-NLP/opus-mt-fr-en", PICKLE_FILE_FR_EN)

# ===== STEP 4: Audio recording until silence =====
def record_until_silence():
    q_data = queue.Queue()

    def callback(indata, frames, time_info, status):
        q_data.put(indata.copy())

    samplerate = 16000
    silence_start = None
    print("Speak now...")
    with sd.InputStream(callback=callback, channels=1, samplerate=samplerate):
        audio_data = []
        while True:
            data = q_data.get()
            audio_data.extend(data[:, 0])
            rms = np.sqrt(np.mean(data**2))
            if rms < SILENCE_THRESHOLD:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION:
                    break
            else:
                silence_start = None

    audio_np = np.array(audio_data, dtype=np.float32)
    return audio_np, samplerate

# ===== STEP 5: Translation function =====
def translate_text(text, direction):
    if direction == "en-fr":
        batch = tokenizer_en_fr([text], return_tensors="pt", padding=True).to(DEVICE)
        gen = model_en_fr.generate(**batch)
        return tokenizer_en_fr.decode(gen[0], skip_special_tokens=True)
    elif direction == "fr-en":
        batch = tokenizer_fr_en([text], return_tensors="pt", padding=True).to(DEVICE)
        gen = model_fr_en.generate(**batch)
        return tokenizer_fr_en.decode(gen[0], skip_special_tokens=True)
    else:
        return "[Invalid direction]"

# ===== STEP 6: Text-to-Speech (TTS) =====
def speak(text, lang="en"):
    tts_engine = pyttsx3.init()   # 🔹 reinitialize each time
    voices = tts_engine.getProperty("voices")
    if lang == "fr":
        for v in voices:
            if "fr" in v.id.lower() or "fr" in v.name.lower():
                tts_engine.setProperty("voice", v.id)
                break
    else:
        tts_engine.setProperty("voice", voices[0].id)
    tts_engine.say(text)
    tts_engine.runAndWait()
    tts_engine.stop()

# ===== MAIN LOOP =====
while True:
    print("\nSelect translation direction:")
    print("1. English -> French")
    print("2. French -> English")
    print("E. Exit")
    choice = input("Your choice: ").strip().lower()

    if choice == "e":
        print("Exiting program...")
        break
    elif choice == "1":
        direction = "en-fr"
    elif choice == "2":
        direction = "fr-en"
    else:
        print("Invalid choice, try again.")
        continue

    while True:
        audio, sr = record_until_silence()
        result = whisper_model.transcribe(audio, fp16=False)
        text = result["text"].strip()
        print(f"You said: {text}")
        translation = translate_text(text, direction)
        print(f"Translated: {translation}")

        # 🔹 Speak every translation
        speak(translation, lang="fr" if direction == "en-fr" else "en")

        print("----")
        next_action = input("Press N for next translation, C to change direction, or E to exit: ").strip().lower()
        if next_action == "e":
            print("Exiting program...")
            exit()
        elif next_action == "c":
            break  # Go back to direction selection
        elif next_action != "n":
            print("Invalid input, continuing...")
