import os
import json
import io
import pygame
from google import genai
from google.genai import types
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

# Load the API keys safely from .env
load_dotenv()

# Initialize both API clients
gemini_client = genai.Client()
tts_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

SYSTEM_PROMPT = """
You are an expert ASL-to-English interpreter. 
Your input is a raw sequence of capitalized English words representing ASL signs. 
Perform two tasks:
1. Translate the ASL sequence into natural, conversational English.
2. Analyze the context to determine the emotional tone for text-to-speech. Choose ONLY from: [NEUTRAL, URGENT, EXCITED, QUESTIONING, POLITE].

Output a valid JSON object.
Example: {"translation": "Help me, I am hurt!", "emotion": "URGENT"}
"""

def translate_asl(raw_asl_string):
    """Sends raw ASL to Gemini and returns a dictionary with translation and emotion."""
    full_prompt = f"{SYSTEM_PROMPT}\n\nInput: {raw_asl_string}"
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Gemini Error: {e}")
        return {"translation": "Error.", "emotion": "NEUTRAL"}

def speak(text, emotion):
    print(f"Selecting voice for emotion: {emotion}...")
    
    voice_map = {
        "NEUTRAL": "nPczCjzI2devNBz1zQrb",
        "URGENT": "pNInz6obpgDQGcFmaJgB",
        "EXCITED": "N2lVS1w4EtoT3dr4eOWO",
        "POLITE": "21m00Tcm4TlvDq8ikWAM",
        "QUESTIONING": "IKne3meq5aSn9XLyUdCD"
    }
    
    selected_voice_id = voice_map.get(emotion, "nPczCjzI2devNBz1zQrb")
    
    try:
        audio_stream = tts_client.text_to_speech.convert(
            text=text,
            voice_id=selected_voice_id,
            model_id="eleven_turbo_v2_5",
            output_format="mp3_44100_128"
        )
        # Collect chunks into bytes — no temp file needed
        audio_bytes = b"".join(audio_stream)

        # Play directly from memory using pygame
        pygame.mixer.init()
        pygame.mixer.music.load(io.BytesIO(audio_bytes))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        print("Audio playback complete!")
        
    except Exception as e:
        print(f"ElevenLabs Error: {e}")

# --- TEST ---
if __name__ == "__main__":
    dummy_input = "COFFEE I WANT PLEASE ICE WITH"
    
    print("1. Translating ASL via Gemini...")
    result = translate_asl(dummy_input)
    
    final_text = result.get('translation')
    final_emotion = result.get('emotion')
    
    print(f"Translation: {final_text}")
    print(f"Emotion: {final_emotion}")
    
    if final_text != "Error.":
        print("\n2. Generating Audio via ElevenLabs...")
        speak(final_text, final_emotion)