
# from ecg_processing import get_ecg_signal, detect_anomaly
from speech_text_nlp import transcribe_audio, analyze_text_with_gpt4
from voice_assist import play_voice_guidance
from location_tracker import get_location
from alert_system import send_alert_sms
import sounddevice as sd
from scipy.io.wavfile import write
import os


def record_audio(filename="output.wav", duration=5, fs=44100):
    print("🎤 Recording... Please speak now.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print(f"✅ Recording saved as {filename}")
    
    transcript = transcribe_audio("output.wav")
    print("Transcript:", transcript)

    if analyze_text_with_gpt4(transcript):
        play_voice_guidance()
        location_url = get_location()
        message = f"Emergency detected!\nLocation: {location_url}"
        send_alert_sms(message)
    else:
        print("No distress detected in voice.")
    
