from datetime import datetime


def aggregator_agent(state: dict) -> dict:
    try:
        disease = state.get("disease_label", "Unknown")
        severity = state.get("severity", "unknown").upper()
        confidence = state.get("confidence", 0.0)
        specialist = state.get("specialist", "General Practitioner")
        image_analysis = state.get("image_analysis", "Not available.")
        rag_context = state.get("rag_context", "Not available.")
        suggestions = state.get("suggestions", "Not available.")

        confidence_pct = f"{confidence * 100:.1f}%"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        divider = "═" * 62

        report = (
            f"╔{divider}╗\n"
            f"║{'MEDICAL IMAGE AI ANALYSIS REPORT':^62}║\n"
            f"║{('Generated: ' + timestamp):^62}║\n"
            f"╚{divider}╝\n\n"

            f"{'─' * 62}\n"
            f"  DIAGNOSIS SUMMARY\n"
            f"{'─' * 62}\n"
            f"  Condition Identified  : {disease}\n"
            f"  Severity Level        : {severity}\n"
            f"  AI Confidence         : {confidence_pct}\n"
            f"  Recommended Specialist: {specialist}\n\n"

            f"{'─' * 62}\n"
            f"  DETAILED IMAGING FINDINGS\n"
            f"{'─' * 62}\n"
            f"{image_analysis}\n\n"

            f"{'─' * 62}\n"
            f"  MEDICAL KNOWLEDGE BASE\n"
            f"{'─' * 62}\n"
            f"{rag_context}\n\n"

            f"{'─' * 62}\n"
            f"  PRECAUTIONS & PATIENT RECOMMENDATIONS\n"
            f"{'─' * 62}\n"
            f"{suggestions}\n\n"

            f"{'─' * 62}\n"
            f"  IMPORTANT DISCLAIMER\n"
            f"{'─' * 62}\n"
            f"  This AI analysis is for informational purposes only and does\n"
            f"  NOT replace professional medical diagnosis.\n\n"
            f"  Always consult a qualified medical professional for proper\n"
            f"  diagnosis, treatment planning, and medical advice.\n"
            f"  In case of emergency, call emergency services immediately.\n"
            f"{'─' * 62}\n"
            f"  Powered by Medical Image AI Analyzer | Multi-Agent System\n"
            f"{'─' * 62}\n"
        )

        return {"aggregated_response": report}

    except Exception as e:
        disease = state.get("disease_label", "Unknown")
        severity = state.get("severity", "unknown")
        confidence = state.get("confidence", 0.0)

        fallback = (
            "MEDICAL AI ANALYSIS REPORT\n"
            "══════════════════════════\n\n"
            f"Disease: {disease}\n"
            f"Severity: {severity}\n"
            f"Confidence: {confidence * 100:.1f}%\n\n"
            f"Findings:\n{state.get('image_analysis', 'Not available.')}\n\n"
            f"Suggestions:\n{state.get('suggestions', 'Not available.')}\n\n"
            "DISCLAIMER: This AI analysis is for informational purposes only and does "
            "NOT replace professional medical diagnosis."
        )
        return {"aggregated_response": fallback, "error": f"Aggregator error: {str(e)}"}
