import os
import io
import re
from google import genai
from gtts import gTTS

SUPPORTED_LANGUAGES = [
    "English", "Telugu", "Hindi", "Tamil", "Kannada",
    "Malayalam", "Bengali", "Marathi", "Gujarati", "Punjabi",
    "Arabic", "Spanish", "French", "German", "Chinese", "Japanese",
]

_GTTS_CODES = {
    "English": "en",
    "Telugu": "te",
    "Hindi": "hi",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Bengali": "bn",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Arabic": "ar",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh-CN",
    "Japanese": "ja",
}


def multilingual_agent(state: dict) -> dict:
    language = state.get("language", "English")
    aggregated_response = state.get("aggregated_response", "")

    if language == "English":
        return {"translated_response": aggregated_response}

    try:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set.")

        client = genai.Client(api_key=api_key)

        translate_prompt = (
            f"Translate the following medical report into {language}.\n\n"
            "Requirements:\n"
            "1. Keep all medical terminology accurate and clinically precise.\n"
            "2. Maintain the exact same structure, formatting, and section headers.\n"
            "3. Preserve all numbers, percentages, dates, and the disclaimer verbatim (in the target language).\n"
            "4. Use formal, respectful medical language appropriate for patients in the target locale.\n"
            "5. Do not add explanations or summaries — translate only.\n\n"
            "Medical Report:\n"
            f"{aggregated_response}\n\n"
            "Provide ONLY the translated report, nothing else."
        )

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=translate_prompt,
        )
        translated = response.text.strip()

        if not translated:
            translated = aggregated_response

        return {"translated_response": translated}

    except Exception as e:
        return {
            "translated_response": aggregated_response,
            "error": f"Translation error: {str(e)}",
        }


def get_tts_audio(text: str, language: str) -> bytes:
    """Convert text to MP3 audio bytes using gTTS. Returns empty bytes on failure."""
    try:
        lang_code = _GTTS_CODES.get(language, "en")

        remove_chars = "═║╔╚╗╝╠╣╦╩╬─━┄┈│┃"
        clean_text = text
        for ch in remove_chars:
            clean_text = clean_text.replace(ch, " ")

        clean_text = (
            clean_text
            .replace("⚠️", "Warning:")
            .replace("•", "")
            .replace("►", "")
        )

        clean_text = re.sub(r"[ \t]{2,}", " ", clean_text)
        clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()

        if len(clean_text) > 3000:
            clean_text = (
                clean_text[:3000]
                + " ... Report truncated for audio playback. Please read the full text."
            )

        tts = gTTS(text=clean_text, lang=lang_code, slow=False)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return buffer.read()

    except Exception:
        return b""
