import os
from google import genai


def prompt_agent(state: dict) -> dict:
    try:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment.")

        client = genai.Client(api_key=api_key)

        raw_prompt = state.get("raw_prompt", "").strip()
        if not raw_prompt:
            raw_prompt = "Analyze this medical image."

        system_prompt = (
            "You are a medical AI assistant. Your task is to rewrite vague user questions "
            "about medical images into precise, clinically accurate questions.\n\n"
            "Rules:\n"
            "- If the question is already specific and clinical, return it UNCHANGED.\n"
            "- If vague, rewrite it as a detailed clinical query.\n\n"
            "Examples:\n"
            '  "what\'s wrong?" → "Analyze this medical image and identify any visible disease, '
            'affected region, and severity."\n'
            '  "is this bad?" → "Examine this image and describe the pathological findings, '
            'disease name, and recommended next steps."\n'
            '  "tell me about this" → "Identify all abnormalities visible in this medical image '
            'and provide a detailed clinical assessment."\n\n'
            "Return ONLY the refined question. No explanation, no preamble."
        )

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=f"{system_prompt}\n\nUser question: {raw_prompt}",
        )
        refined = response.text.strip()

        if not refined:
            refined = (
                "Analyze this medical image and identify any visible disease, "
                "affected region, severity level, and recommend the appropriate medical specialist."
            )

        return {"refined_prompt": refined}

    except Exception as e:
        fallback = (
            "Analyze this medical image and identify any visible disease, "
            "affected region, severity level, and recommend the appropriate medical specialist."
        )
        return {"refined_prompt": fallback, "error": f"Prompt agent error: {str(e)}"}
