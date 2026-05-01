import os
from google import genai


def suggestion_agent(state: dict) -> dict:
    try:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set.")

        client = genai.Client(api_key=api_key)

        disease_label = state.get("disease_label", "Unknown condition")
        severity = state.get("severity", "moderate")
        specialist = state.get("specialist", "General Practitioner")
        rag_context = state.get("rag_context", "")

        urgency_map = {
            "severe": "within 24–48 hours (urgent)",
            "moderate": "within 1–2 weeks",
            "mild": "within 2–4 weeks at your earliest convenience",
        }
        urgency = urgency_map.get(severity.lower(), "within 1–2 weeks")

        prompt = (
            "You are a compassionate medical advisor providing patient-friendly guidance. "
            "Use simple, empathetic language. Avoid medical jargon. Be reassuring but honest.\n\n"
            f"DIAGNOSIS: {disease_label}\n"
            f"SEVERITY: {severity.upper()}\n"
            f"RECOMMENDED SPECIALIST: {specialist}\n"
            f"MEDICAL CONTEXT (excerpt): {rag_context[:600]}\n\n"
            "Provide advice in EXACTLY this structure — keep each section header exactly as shown:\n\n"
            "IMMEDIATE_ACTIONS:\n"
            "• [Action 1 — what to do RIGHT NOW]\n"
            "• [Action 2]\n"
            "• [Action 3]\n"
            "• [Action 4]\n\n"
            "LIFESTYLE_CHANGES:\n"
            "• [Diet recommendation]\n"
            "• [Activity/exercise recommendation]\n"
            "• [Sleep/stress recommendation]\n"
            "• [Habit to avoid]\n\n"
            "MEDICATIONS_HINT:\n"
            "• [General medication category — NOT a specific prescription]\n"
            "• [Second medication category]\n"
            "• [Third medication category]\n\n"
            "WARNING_SIGNS:\n"
            "• [Emergency symptom 1 — go to ER immediately if this occurs]\n"
            "• [Emergency symptom 2]\n"
            "• [Emergency symptom 3]\n\n"
            f"FOLLOW_UP:\n"
            f"• See a {specialist} {urgency}\n"
            "• [One additional monitoring or follow-up recommendation]"
        )

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
        )
        suggestions = response.text.strip()

        return {"suggestions": suggestions}

    except Exception as e:
        specialist = state.get("specialist", "your doctor")
        severity = state.get("severity", "moderate")
        fallback = (
            "IMMEDIATE_ACTIONS:\n"
            "• Contact your healthcare provider as soon as possible\n"
            "• Rest and avoid strenuous physical activity\n"
            "• Monitor your symptoms and note any changes\n"
            "• Take any currently prescribed medications as directed\n\n"
            "LIFESTYLE_CHANGES:\n"
            "• Eat a balanced diet rich in fruits, vegetables, and lean proteins\n"
            "• Stay well-hydrated (8–10 glasses of water daily)\n"
            "• Aim for 7–9 hours of quality sleep per night\n"
            "• Avoid smoking, excessive alcohol, and high-stress activities\n\n"
            "MEDICATIONS_HINT:\n"
            "• Follow your prescribing physician's medication instructions precisely\n"
            "• Do not self-medicate or stop prescribed medications without medical advice\n"
            "• Inform your doctor of any allergies or current supplements\n\n"
            "WARNING_SIGNS:\n"
            "• Sudden difficulty breathing or severe shortness of breath — call emergency services\n"
            "• Severe chest pain or pressure — go to the emergency room immediately\n"
            "• Loss of consciousness, confusion, or inability to speak — call emergency services\n\n"
            f"FOLLOW_UP:\n"
            f"• See {specialist} promptly given the {severity} severity assessment\n"
            "• Bring this report and all previous medical records to your appointment"
        )
        return {"suggestions": fallback, "error": f"Suggestion agent error: {str(e)}"}
