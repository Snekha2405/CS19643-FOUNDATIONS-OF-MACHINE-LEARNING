import google.generativeai as genai
import whisper
import re
from tts_module import text_to_speech

# ✅ Configure Gemini API
genai.configure(api_key="AIzaSyAfcqCtu8c4nPB3QSNQjPC5s2I83eK8j5A")
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")  # ✅ No 'models/'

# ✅ Whisper for transcription
def transcribe_audio(file_path):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        print("❌ Whisper Error:", e)
        return ""

# ✅ Gemini analysis with fallback
def analyze_text_with_gpt4(transcript):
    if not transcript.strip():
        return "No speech content found."

    prompt = f"""
The following speech is from a woman or elderly person seeking assistance: "{transcript}".

Your tasks are:
1. Identify if the speaker is in: (a) Immediate emergency, (b) Non-urgent but needs assistance, or (c) Calm/no threat.
2. Summarize clearly what the issue is.
3. Provide direct, simple health advice or precaution to the speaker based on the issue.
4. Recommend the next action: (Call emergency services, Stay calm and monitor, No action needed)

Respond in the following format:
Situation: <your classification>
Summary: <summary of issue>
Advice: <health advice for speaker>
Recommended Action: <what should happen next>
"""

    try:
        response = gemini_model.generate_content(prompt)
        print(response.text)
        result_text = response.text.strip()
        advice_match = re.search(r'Advice:\s*(.*)', result_text)
        action_match = re.search(r'Recommended Action:\s*(.*)', result_text)

        advice = advice_match.group(1).strip() if advice_match else "No advice found."
        action = action_match.group(1).strip() if action_match else "No action recommended."

        # ✅ Prepare text for TTS
        tts_text = f"{advice}. {action}."
        text_to_speech(tts_text)
        return True

    except Exception as e:
        print("❌ Gemini Error:", e)
        return "Gemini API error occurred."