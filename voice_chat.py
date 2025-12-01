import sounddevice as sd
import numpy as np
import pyttsx3
import os
from groq import Groq
from faster_whisper import WhisperModel
from dotenv import load_dotenv


load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


model = WhisperModel("base", device="cpu")

def record_audio(duration=5, sample_rate=16000):
    print(" Listening")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return np.squeeze(audio).astype(np.float32)

def speech_to_text(audio):
    segments, _ = model.transcribe(audio, beam_size=5)
    text = " ".join([segment.text for segment in segments])
    print("You said:", text)
    return text.strip()

SYSTEM_PROMPT = """
You are Abhinav Tripathi. Answer exactly as he would.

Personality:
- Honest, confident, ambitious
- Strong in AI/ML, Next.js, Python, DS & Algorithms
- Friendly, clear, structured, and human-like

Keep answers conversational and personal.
"""

def generate_reply(text):
    chat = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
    )
    reply = chat.choices[0].message.content
    print(" Reply:", reply)
    return reply

engine = pyttsx3.init()
engine.setProperty("rate", 180)

def speak(text):
    engine.say(text)
    engine.runAndWait()

print("\n Voice Bot Ready! Speak a question")
print("- What should we know about your life story?")
print("- What’s your #1 superpower?")
print("- What are your top 3 growth areas?")
print("Say 'stop' to exit.\n")


while True:
    audio = record_audio(duration=5)
    text = speech_to_text(audio)

    if text == "":
        print(" Didn't catch that. Try again.\n")
        continue

    if "stop" in text.lower():
        print(" Exiting bot.")
        break

    reply = generate_reply(text)
    speak(reply)
