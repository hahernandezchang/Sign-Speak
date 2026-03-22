import io
import json
import os

import pygame
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from google import genai
from google.genai import types

load_dotenv()

SYSTEM_PROMPT = """
You are an expert ASL-to-English interpreter. 
Your input is a raw sequence of capitalized English words representing ASL signs. 
Perform two tasks:
1. Translate the ASL sequence into natural, conversational English.
2. Analyze the context to determine the emotional tone for text-to-speech. Choose ONLY from: [NEUTRAL, URGENT, EXCITED, QUESTIONING, POLITE].

Output a valid JSON object.
Example: {"translation": "Help me, I am hurt!", "emotion": "URGENT"}
"""


class VoiceEngine:
    def __init__(self) -> None:
        self._gemini_key = os.getenv("GEMINI_API_KEY", "")
        self._eleven_key = os.getenv("ELEVENLABS_API_KEY", "")
        self.enabled = bool(self._gemini_key and self._eleven_key)
        self._gemini_client = None
        self._tts_client = None

    def _ensure_clients(self) -> bool:
        if not self.enabled:
            return False
        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=self._gemini_key)
        if self._tts_client is None:
            self._tts_client = ElevenLabs(api_key=self._eleven_key)
        return True

    def translate_asl(self, raw_asl_string: str) -> dict:
        if not raw_asl_string.strip():
            return {"translation": "", "emotion": "NEUTRAL"}

        if not self._ensure_clients():
            return {"translation": raw_asl_string, "emotion": "NEUTRAL"}

        full_prompt = f"{SYSTEM_PROMPT}\n\nInput: {raw_asl_string}"
        try:
            response = self._gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            parsed = json.loads(response.text)
            return {
                "translation": str(parsed.get("translation", "")).strip(),
                "emotion": str(parsed.get("emotion", "NEUTRAL")).strip().upper(),
            }
        except Exception as exc:
            print(f"Gemini Error: {exc}")
            return {"translation": raw_asl_string, "emotion": "NEUTRAL"}

    def speak(self, text: str, emotion: str = "NEUTRAL") -> bool:
        if not text.strip():
            return False
        if not self._ensure_clients():
            print("Voice engine disabled: missing API keys in .env")
            return False

        voice_map = {
            "NEUTRAL": "nPczCjzI2devNBz1zQrb",
            "URGENT": "pNInz6obpgDQGcFmaJgB",
            "EXCITED": "N2lVS1w4EtoT3dr4eOWO",
            "POLITE": "21m00Tcm4TlvDq8ikWAM",
            "QUESTIONING": "IKne3meq5aSn9XLyUdCD",
        }
        selected_voice_id = voice_map.get(emotion.upper(), voice_map["NEUTRAL"])

        try:
            audio_stream = self._tts_client.text_to_speech.convert(
                text=text,
                voice_id=selected_voice_id,
                model_id="eleven_turbo_v2_5",
                output_format="mp3_44100_128",
            )
            audio_bytes = b"".join(audio_stream)

            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(io.BytesIO(audio_bytes))
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            return True
        except Exception as exc:
            print(f"ElevenLabs Error: {exc}")
            return False

    def speak_from_sign_tokens(self, tokens: list[str]) -> tuple[str, str, bool]:
        raw_asl = " ".join([token for token in tokens if token]).strip()
        if not raw_asl:
            return "", "NEUTRAL", False

        result = self.translate_asl(raw_asl)
        translation = result.get("translation", raw_asl)
        emotion = result.get("emotion", "NEUTRAL")
        spoken = self.speak(translation, emotion)
        return translation, emotion, spoken


_default_engine = VoiceEngine()


def translate_asl(raw_asl_string):
    return _default_engine.translate_asl(raw_asl_string)


def speak(text, emotion):
    return _default_engine.speak(text, emotion)

# --- TEST ---
if __name__ == "__main__":
    dummy_input = "COFFEE I WANT PLEASE ICE WITH"

    print("1. Translating ASL via Gemini...")
    result = _default_engine.translate_asl(dummy_input)

    final_text = result.get("translation")
    final_emotion = result.get("emotion")

    print(f"Translation: {final_text}")
    print(f"Emotion: {final_emotion}")

    if final_text:
        print("\n2. Generating Audio via ElevenLabs...")
        _default_engine.speak(final_text, final_emotion)