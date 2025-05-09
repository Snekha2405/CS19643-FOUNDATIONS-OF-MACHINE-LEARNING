# tts_module.py
from gtts import gTTS
import os

def text_to_speech(text, filename="response.mp3"):
    """
    Converts given text to speech and plays the audio file.

    Parameters:
    - text (str): The text to be converted to speech.
    - filename (str): Optional. The output mp3 file name (default is 'response.mp3').

    Example:
    text_to_speech("Hello, how are you?", "output.mp3")
    """
    try:
        if not text.strip():
            print("❌ No text provided for TTS.")
            return

        tts = gTTS(text=text, lang='en')
        tts.save(filename)

        # Automatically play the mp3 file (OS-specific)
        if os.name == 'nt':  # Windows
            os.system(f"start {filename}")
       

    except Exception as e:
        print("❌ TTS Error:", e)
