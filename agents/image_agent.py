import os
import re
from google import genai


def image_agent(state: dict) -> dict:
    try:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment.")

        client = genai.Client(api_key=api_key)

        image = state.get("image")
        if image is None:
            raise ValueError("No image provided in state.")

        refined_prompt = state.get("refined_prompt", "Analyze this medical image.")
        rgb_image = image.convert("RGB")

        structured_prompt = (
            f"{refined_prompt}\n\n"
            "Provide a structured medical analysis using EXACTLY these labeled fields "
            "(each on its own line):\n\n"
            "DISEASE: [specific disease or condition name]\n"
            "SEVERITY: [mild / moderate / severe]\n"
            "CONFIDENCE: [decimal between 0.0 and 1.0]\n"
            "SPECIALIST: [type of specialist to consult]\n"
            "FINDINGS: [detailed description of all visible imaging findings]\n"
            "AFFECTED_REGION: [specific anatomical region affected]\n\n"
            "Be clinically precise and thorough."
        )

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[structured_prompt, rgb_image],
        )
        analysis_text = response.text.strip()

        def extract_field(pattern: str, text: str, default: str) -> str:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                val = match.group(1).strip().split("\n")[0].strip()
                return val if val else default
            return default

        disease = extract_field(
            r"DISEASE:\s*(.+?)(?=\n[A-Z_]+:|$)", analysis_text, "Unspecified Condition"
        )
        severity_raw = extract_field(
            r"SEVERITY:\s*(.+?)(?=\n[A-Z_]+:|$)", analysis_text, "moderate"
        )
        confidence_raw = extract_field(
            r"CONFIDENCE:\s*(.+?)(?=\n[A-Z_]+:|$)", analysis_text, "0.75"
        )
        specialist = extract_field(
            r"SPECIALIST:\s*(.+?)(?=\n[A-Z_]+:|$)", analysis_text, "General Practitioner"
        )

        severity = severity_raw.lower().strip()
        if severity not in ("mild", "moderate", "severe"):
            severity = "moderate"

        try:
            conf_match = re.search(r"(\d+\.?\d*)", confidence_raw)
            conf_val = float(conf_match.group(1)) if conf_match else 0.75
            if conf_val > 1.0:
                conf_val = conf_val / 100.0
            confidence = round(min(max(conf_val, 0.0), 1.0), 2)
        except Exception:
            confidence = 0.75

        return {
            "image_analysis": analysis_text,
            "disease_label": disease.split("\n")[0].strip(),
            "severity": severity,
            "confidence": confidence,
            "specialist": specialist.split("\n")[0].strip(),
        }

    except Exception as e:
        return {
            "image_analysis": f"Image analysis failed: {str(e)}",
            "disease_label": "Unknown",
            "severity": "moderate",
            "confidence": 0.5,
            "specialist": "General Practitioner",
            "error": f"Image agent error: {str(e)}",
        }
