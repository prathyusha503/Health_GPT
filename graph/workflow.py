"""
Sequential LangGraph pipeline — 100% free, no API keys.
7 tool nodes run in order with heatmap + RAG in parallel.
Each node calls its @tool function and returns only its own state keys.
"""

import json
import re
from datetime import datetime
from langgraph.graph import StateGraph, END

from graph.state import MedicalState
from tools.tool_definitions import (
    validate_prompt,
    clip_screen,
    analyze_image,
    generate_heatmap,
    search_rag,
    get_suggestions,
    translate_report,
)


# ── Helper: parse structured analysis text ────────────────────────────────────
def _parse_analysis(text: str) -> dict:
    def ex(pattern: str, default: str) -> str:
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            val = m.group(1).strip().split("\n")[0].strip()
            return val if val else default
        return default

    disease  = ex(r"DISEASE:\s*(.+?)(?=\n[A-Z_]+:|$)", "Unknown")
    sev_raw  = ex(r"SEVERITY:\s*(.+?)(?=\n[A-Z_]+:|$)", "moderate")
    conf_raw = ex(r"CONFIDENCE:\s*(.+?)(?=\n[A-Z_]+:|$)", "0.75")
    spec     = ex(r"SPECIALIST:\s*(.+?)(?=\n[A-Z_]+:|$)", "General Practitioner")

    severity = sev_raw.lower()
    if severity not in ("mild", "moderate", "severe"):
        severity = "moderate"

    try:
        m = re.search(r"(\d+\.?\d*)", conf_raw)
        conf = float(m.group(1)) if m else 0.75
        conf = conf / 100.0 if conf > 1.0 else conf
        confidence = round(min(max(conf, 0.0), 1.0), 2)
    except Exception:
        confidence = 0.75

    return {
        "disease_label": disease,
        "severity": severity,
        "confidence": confidence,
        "specialist": spec,
    }


# ── Node 1: validate_prompt ───────────────────────────────────────────────────
def validate_node(state: MedicalState) -> dict:
    result = validate_prompt.invoke({"raw_prompt": state["raw_prompt"]})
    return {"refined_prompt": result}


# ── Node 2: clip_screen ───────────────────────────────────────────────────────
def clip_node(state: MedicalState) -> dict:
    result_json = clip_screen.invoke({"image_base64": state["image_base64"]})
    try:
        data = json.loads(result_json)
    except Exception:
        data = {}
    return {
        "clip_scores":        data.get("clip_scores", {}),
        "clip_top_disease":   data.get("clip_top_disease", ""),
        "clip_embedding_json": json.dumps(data.get("clip_embedding", [])),
        "clip_is_medical":    data.get("clip_is_medical", True),
    }


# ── Node 3: analyze_image ─────────────────────────────────────────────────────
def analyze_node(state: MedicalState) -> dict:
    result = analyze_image.invoke({
        "image_base64":  state["image_base64"],
        "refined_prompt": state.get("refined_prompt", ""),
        "clip_hint":     state.get("clip_top_disease", ""),
    })
    parsed = _parse_analysis(result)
    return {"image_analysis": result, **parsed}


# ── Node 4: generate_heatmap (parallel branch A) ──────────────────────────────
def heatmap_node(state: MedicalState) -> dict:
    result = generate_heatmap.invoke({
        "image_base64":   state["image_base64"],
        "disease_label":  state.get("disease_label", ""),
        "image_analysis": state.get("image_analysis", "")[:300],
    })
    return {"heatmap_base64": result}


# ── Node 5: search_rag (parallel branch B) ────────────────────────────────────
def rag_node(state: MedicalState) -> dict:
    result = search_rag.invoke({
        "disease_label":     state.get("disease_label", ""),
        "clip_embedding_json": state.get("clip_embedding_json", "[]"),
    })
    return {"rag_context": result}


# ── Node 6: get_suggestions ───────────────────────────────────────────────────
def suggest_node(state: MedicalState) -> dict:
    result = get_suggestions.invoke({
        "disease":    state.get("disease_label", ""),
        "severity":   state.get("severity", "moderate"),
        "specialist": state.get("specialist", "General Practitioner"),
        "rag_context": state.get("rag_context", ""),
    })
    return {"suggestions": result}


# ── Node 7: translate_report ──────────────────────────────────────────────────
def translate_node(state: MedicalState) -> dict:
    disease    = state.get("disease_label", "Unknown")
    severity   = state.get("severity", "unknown").upper()
    confidence = state.get("confidence", 0.0)
    specialist = state.get("specialist", "General Practitioner")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    clip_scores = state.get("clip_scores", {})
    clip_lines = ""
    if clip_scores:
        top3 = sorted(clip_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        clip_lines = "\n  CLIP Top Predictions:\n" + "\n".join(
            f"    {d}: {p * 100:.1f}%" for d, p in top3
        )

    report = (
        f"{'═' * 62}\n"
        f"{'MEDICAL IMAGE AI ANALYSIS REPORT':^62}\n"
        f"{'Generated: ' + ts:^62}\n"
        f"{'═' * 62}\n\n"
        f"{'─' * 62}\n"
        f"  DIAGNOSIS SUMMARY\n"
        f"{'─' * 62}\n"
        f"  Condition       : {disease}\n"
        f"  Severity        : {severity}\n"
        f"  AI Confidence   : {confidence * 100:.1f}%\n"
        f"  Specialist      : {specialist}\n"
        f"{clip_lines}\n\n"
        f"{'─' * 62}\n"
        f"  DETAILED IMAGING FINDINGS\n"
        f"{'─' * 62}\n"
        f"{state.get('image_analysis', 'Not available.')}\n\n"
        f"{'─' * 62}\n"
        f"  MEDICAL KNOWLEDGE BASE\n"
        f"{'─' * 62}\n"
        f"{state.get('rag_context', 'Not available.')}\n\n"
        f"{'─' * 62}\n"
        f"  PATIENT RECOMMENDATIONS\n"
        f"{'─' * 62}\n"
        f"{state.get('suggestions', 'Not available.')}\n\n"
        f"{'─' * 62}\n"
        f"  DISCLAIMER\n"
        f"{'─' * 62}\n"
        f"  This AI analysis is for informational purposes only and does\n"
        f"  NOT replace professional medical diagnosis. Always consult a\n"
        f"  qualified medical professional for proper evaluation.\n"
        f"{'─' * 62}\n"
        f"  Powered by CLIP Vision AI · ChromaDB RAG · No API Key Required\n"
        f"{'─' * 62}\n"
    )

    result_json = translate_report.invoke({
        "report": report,
        "language": state.get("language", "English"),
    })
    try:
        data = json.loads(result_json)
        return {
            "translated_response": data.get("translated_text", report),
            "audio_base64":        data.get("audio_base64", ""),
        }
    except Exception:
        return {"translated_response": report, "audio_base64": ""}


# ── Node 8: aggregator (pure Python, builds final report) ─────────────────────
def aggregator_node(state: MedicalState) -> dict:
    disease    = state.get("disease_label", "Unknown")
    severity   = state.get("severity", "unknown").upper()
    confidence = state.get("confidence", 0.0)
    specialist = state.get("specialist", "General Practitioner")
    clip_scores = state.get("clip_scores", {})
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    clip_section = ""
    if clip_scores:
        sorted_scores = sorted(clip_scores.items(), key=lambda x: x[1], reverse=True)
        clip_section = "\n  CLIP Probability Scores:\n" + "\n".join(
            f"    {d:30s}: {p * 100:.2f}%" for d, p in sorted_scores
        ) + "\n"

    report = (
        f"{'═' * 62}\n"
        f"{'MEDICAL IMAGE AI ANALYSIS REPORT':^62}\n"
        f"{'Generated: ' + ts:^62}\n"
        f"{'═' * 62}\n\n"
        f"{'─' * 62}\n"
        f"  DIAGNOSIS SUMMARY\n"
        f"{'─' * 62}\n"
        f"  Condition       : {disease}\n"
        f"  Severity        : {severity}\n"
        f"  AI Confidence   : {confidence * 100:.1f}%\n"
        f"  Specialist      : {specialist}\n"
        f"{clip_section}\n"
        f"{'─' * 62}\n"
        f"  DETAILED IMAGING FINDINGS\n"
        f"{'─' * 62}\n"
        f"{state.get('image_analysis', 'Not available.')}\n\n"
        f"{'─' * 62}\n"
        f"  MEDICAL KNOWLEDGE BASE\n"
        f"{'─' * 62}\n"
        f"{state.get('rag_context', 'Not available.')}\n\n"
        f"{'─' * 62}\n"
        f"  PATIENT RECOMMENDATIONS\n"
        f"{'─' * 62}\n"
        f"{state.get('suggestions', 'Not available.')}\n\n"
        f"{'─' * 62}\n"
        f"  DISCLAIMER\n"
        f"{'─' * 62}\n"
        f"  This AI analysis is for informational purposes only and does\n"
        f"  NOT replace professional medical diagnosis. Always consult a\n"
        f"  qualified medical professional for proper evaluation.\n"
        f"{'─' * 62}\n"
        f"  Powered by CLIP Vision AI · ChromaDB RAG · No API Key Required\n"
        f"{'─' * 62}\n"
    )
    return {"aggregated_response": report}


# ── Build and compile the graph ───────────────────────────────────────────────
def create_workflow():
    wf = StateGraph(MedicalState)

    wf.add_node("validate_prompt",  validate_node)
    wf.add_node("clip_screen",      clip_node)
    wf.add_node("analyze_image",    analyze_node)
    wf.add_node("generate_heatmap", heatmap_node)   # ─┐ parallel
    wf.add_node("search_rag",       rag_node)        # ─┘ parallel
    wf.add_node("get_suggestions",  suggest_node)
    wf.add_node("translate_report", translate_node)
    wf.add_node("aggregator",       aggregator_node)

    wf.set_entry_point("validate_prompt")
    wf.add_edge("validate_prompt",  "clip_screen")
    wf.add_edge("clip_screen",      "analyze_image")
    wf.add_edge("analyze_image",    "generate_heatmap")   # fan-out
    wf.add_edge("analyze_image",    "search_rag")         # fan-out
    wf.add_edge("generate_heatmap", "get_suggestions")    # fan-in
    wf.add_edge("search_rag",       "get_suggestions")    # fan-in
    wf.add_edge("get_suggestions",  "translate_report")
    wf.add_edge("translate_report", "aggregator")
    wf.add_edge("aggregator",       END)

    return wf.compile()


compiled_graph = create_workflow()
