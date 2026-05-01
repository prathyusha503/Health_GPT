"""
Medical Image AI Analyzer — 100% free, zero API keys.
CLIP Vision AI · ChromaDB RAG · LangGraph · deep-translator · gTTS
"""

import os
import sys
import base64
import io
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Image AI Analyzer",
    page_icon="🩺",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "final_state": None,
    "analysis_complete": False,
    "stored_image_b64": None,
    "stored_language": "English",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 Medical Image AI")
    st.success("**No API key required!**\nCLIP runs locally on CPU.")
    st.divider()

    language = st.selectbox(
        "🌍 Output Language",
        ["English", "Telugu", "Hindi", "Tamil", "Kannada",
         "Malayalam", "Bengali", "Marathi", "Gujarati", "Punjabi",
         "Arabic", "Spanish", "French", "German", "Chinese", "Japanese"],
        key="language_select",
    )
    st.session_state.stored_language = language
    st.divider()

    st.info(
        "**7-Tool Pipeline**\n\n"
        "1. 🧠 Validate Prompt\n"
        "2. 👁️ CLIP Screen\n"
        "3. 🔬 Analyze Image\n"
        "4. 🌡️ Generate Heatmap\n"
        "5. 📚 Search RAG\n"
        "6. 💊 Get Suggestions\n"
        "7. 🌍 Translate Report"
    )
    st.warning("Not a substitute for professional medical diagnosis.")

    with st.expander("ℹ️ First-run note"):
        st.caption(
            "CLIP model (~600 MB) downloads automatically from "
            "HuggingFace on first use. All subsequent runs are fast."
        )

# ── App title ─────────────────────────────────────────────────────────────────
st.title("🩺 Medical Image AI Analyzer")
st.caption("CLIP Vision AI · ChromaDB RAG · LangGraph · Zero API Key · 100% Free")

# ── 5 Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📤 Upload & Analyze",
    "🔬 Diagnosis & Heatmap",
    "📋 Full Report",
    "💊 Suggestions",
    "📚 Knowledge Base",
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — Upload & Analyze
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.header("📤 Upload Your Medical Image")
    st.markdown(
        "Upload an **X-ray, CT scan, MRI, or Fundus photograph**, "
        "type your question, and click **Analyze Image**."
    )
    st.divider()

    col_upload, col_preview = st.columns([1, 1], gap="large")

    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose a medical image",
            type=["jpg", "jpeg", "png", "webp"],
            key="file_uploader",
        )

        user_question = st.text_area(
            "Your Question",
            placeholder="e.g. What disease is visible in this X-ray?",
            height=110,
            key="user_question",
        )

        bc1, bc2 = st.columns(2)
        with bc1:
            improve_btn = st.button(
                "✨ Improve My Question",
                use_container_width=True,
                help="Rewrites vague questions into precise clinical ones",
            )
        with bc2:
            analyze_btn = st.button(
                "🔬 Analyze Image",
                type="primary",
                use_container_width=True,
            )

    with col_preview:
        if uploaded_file is not None:
            preview_img = Image.open(uploaded_file)
            st.image(
                preview_img,
                caption="📷 Uploaded Image Preview",
                use_column_width=True,
            )
            st.success(
                f"✅ Image ready — **{uploaded_file.name}**  "
                f"({preview_img.width}×{preview_img.height} px)"
            )
        else:
            st.info("Upload an image to see the preview here.")

    # ── Improve question ──────────────────────────────────────────────────────
    if improve_btn:
        if not user_question.strip():
            st.error("Please type a question first.")
        else:
            with st.spinner("Refining your question..."):
                try:
                    from tools.tool_definitions import validate_prompt
                    refined = validate_prompt.invoke({"raw_prompt": user_question.strip()})
                    st.success(f"**Refined Clinical Question:** {refined}")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Run analysis ──────────────────────────────────────────────────────────
    if analyze_btn:
        if uploaded_file is None:
            st.error("❌ Please upload a medical image first.")
        elif not user_question.strip():
            st.error("❌ Please type your question before analyzing.")
        else:
            st.divider()
            st.subheader("⏳ Pipeline Progress")

            progress_bar = st.progress(0)
            status_text  = st.empty()

            node_progress = {
                "validate_prompt":  (1/8, "Tool 1 — Validating & refining prompt..."),
                "clip_screen":      (2/8, "Tool 2 — CLIP screening medical image locally..."),
                "analyze_image":    (3/8, "Tool 3 — CLIP + RAG producing diagnosis..."),
                "generate_heatmap": (4/8, "Tool 4 — Building attention rollout heatmap..."),
                "search_rag":       (5/8, "Tool 5 — Searching medical knowledge base..."),
                "get_suggestions":  (6/8, "Tool 6 — Generating clinical suggestions..."),
                "translate_report": (7/8, "Tool 7 — Translating & generating audio..."),
                "aggregator":       (8/8, "Aggregating final report..."),
            }

            try:
                from graph.workflow import compiled_graph

                uploaded_file.seek(0)
                raw_bytes = uploaded_file.read()
                image_b64 = base64.b64encode(raw_bytes).decode("utf-8")

                initial_state = {
                    "image_base64":       image_b64,
                    "raw_prompt":         user_question.strip(),
                    "language":           language,
                    "refined_prompt":     "",
                    "clip_scores":        {},
                    "clip_top_disease":   "",
                    "clip_embedding_json": "",
                    "clip_is_medical":    True,
                    "image_analysis":     "",
                    "disease_label":      "",
                    "severity":           "",
                    "confidence":         0.0,
                    "specialist":         "",
                    "heatmap_base64":     "",
                    "rag_context":        "",
                    "suggestions":        "",
                    "aggregated_response": "",
                    "translated_response": "",
                    "audio_base64":       "",
                    "error":              None,
                }

                final_state = dict(initial_state)

                for chunk in compiled_graph.stream(initial_state):
                    for node_name, node_state in chunk.items():
                        if node_name in node_progress:
                            prog, msg = node_progress[node_name]
                            progress_bar.progress(prog)
                            status_text.text(f"✓ {msg}")
                        if isinstance(node_state, dict):
                            final_state.update(node_state)

                progress_bar.progress(1.0)
                status_text.success("✅ Analysis complete! Switch to the other tabs to view results.")

                st.session_state.final_state       = final_state
                st.session_state.analysis_complete  = True
                st.session_state.stored_image_b64   = image_b64

                st.balloons()

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.exception(e)
                progress_bar.empty()
                status_text.empty()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — Diagnosis & Heatmap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    if not st.session_state.analysis_complete:
        st.info("👈 Go to **Upload & Analyze** tab, upload an image and click **Analyze Image** first.")
    else:
        fs = st.session_state.final_state

        st.header("🔬 Diagnosis Summary")

        # ── CLIP warning ──────────────────────────────────────────────────────
        if not fs.get("clip_is_medical", True):
            st.warning(
                "⚠️ CLIP did not recognise this as a medical image. "
                "Results may be unreliable — please upload an X-ray, CT, MRI, or fundus image."
            )

        # ── Metric cards ──────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        disease  = fs.get("disease_label", "N/A")
        severity = (fs.get("severity") or "N/A").capitalize()
        conf_pct = f"{fs.get('confidence', 0) * 100:.0f}%"
        spec     = fs.get("specialist") or "N/A"

        m1.metric("🦠 Disease",    disease)
        m2.metric("📊 Severity",   severity)
        m3.metric("🎯 Confidence", conf_pct)
        m4.metric("👨‍⚕️ Specialist", spec[:20] + "…" if len(spec) > 20 else spec)

        st.divider()

        # ── CLIP bar chart ────────────────────────────────────────────────────
        clip_scores = fs.get("clip_scores", {})
        if clip_scores:
            st.subheader("📊 CLIP Disease Probability Scores")
            st.caption("Probability that this image belongs to each disease category (local CLIP model)")
            st.bar_chart(clip_scores, height=260)

        st.divider()

        # ── Images: Original | Heatmap ────────────────────────────────────────
        st.subheader("🖼️ Image vs Heatmap")
        st.caption(
            "Left — original uploaded image. "
            "Right — CLIP Attention Rollout heatmap (warm/red = high disease attention, "
            "red box = detected region)."
        )

        img_col1, img_col2 = st.columns(2, gap="medium")

        with img_col1:
            st.markdown("**Original Image**")
            b64 = st.session_state.stored_image_b64
            if b64:
                orig_pil = Image.open(io.BytesIO(base64.b64decode(b64)))
                st.image(orig_pil, use_column_width=True)

        with img_col2:
            st.markdown("**🌡️ Disease Heatmap**")
            hb64 = fs.get("heatmap_base64", "")
            if hb64 and hb64 != st.session_state.stored_image_b64:
                try:
                    hmap_pil = Image.open(io.BytesIO(base64.b64decode(hb64)))
                    st.image(hmap_pil, use_column_width=True)
                    st.caption("🔴 Red box = disease region  |  🟡 Yellow/warm = high attention")
                except Exception:
                    st.warning("Heatmap could not be decoded.")
            elif hb64:
                st.warning("Heatmap same as original — attention map may be very uniform for this image.")
            else:
                st.info("Heatmap not generated.")

        st.divider()

        # ── Refined clinical question ─────────────────────────────────────────
        with st.expander("🧠 Refined Clinical Question (Tool 1 output)", expanded=False):
            st.write(fs.get("refined_prompt", "Not available."))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — Full Report
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    if not st.session_state.analysis_complete:
        st.info("👈 Run the analysis first in the **Upload & Analyze** tab.")
    else:
        fs   = st.session_state.final_state
        lang = st.session_state.stored_language

        st.header("📋 Full Analysis Report")
        st.caption(
            f"Report language: **{lang}**  |  "
            "Translated using free Google Translate · No API key used"
        )

        translated = fs.get("translated_response") or fs.get("aggregated_response", "")

        st.text_area(
            "Complete Medical Report",
            value=translated,
            height=500,
            key="report_display",
        )

        st.divider()

        # ── Audio player ──────────────────────────────────────────────────────
        st.subheader("🔊 Voice Audio")
        audio_b64 = fs.get("audio_base64", "")

        if audio_b64:
            try:
                audio_bytes = base64.b64decode(audio_b64)
                st.audio(audio_bytes, format="audio/mp3")
                st.caption("Audio generated using free Google TTS — no API key required")
            except Exception:
                st.warning("Audio could not be decoded.")
        else:
            if st.button("🔊 Generate Audio Now", key="gen_audio_btn"):
                with st.spinner("Generating audio..."):
                    try:
                        from tools.tool_definitions import translate_report
                        import json as _json
                        res = translate_report.invoke({
                            "report": translated[:3000],
                            "language": lang,
                        })
                        data = _json.loads(res)
                        ab64 = data.get("audio_base64", "")
                        if ab64:
                            st.audio(base64.b64decode(ab64), format="audio/mp3")
                            fs["audio_base64"] = ab64
                        else:
                            st.warning("Audio generation failed for this language.")
                    except Exception as ex:
                        st.error(f"Audio error: {ex}")

        st.divider()

        # ── Download ──────────────────────────────────────────────────────────
        st.subheader("⬇️ Download Report")
        st.download_button(
            label="⬇️ Download Full Report as .txt",
            data=(translated or "").encode("utf-8"),
            file_name=f"medical_report_{lang.lower()}.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — Suggestions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    if not st.session_state.analysis_complete:
        st.info("👈 Run the analysis first in the **Upload & Analyze** tab.")
    else:
        fs = st.session_state.final_state
        disease  = fs.get("disease_label", "Unknown")
        severity = fs.get("severity", "moderate").capitalize()
        spec     = fs.get("specialist", "General Practitioner")

        st.header("💊 Patient Suggestions & Precautions")
        st.markdown(
            f"> **Condition:** {disease} &nbsp;|&nbsp; "
            f"**Severity:** {severity} &nbsp;|&nbsp; "
            f"**See:** {spec}"
        )
        st.divider()

        suggestions = fs.get("suggestions", "")
        if suggestions:
            # Parse and render each section with styled cards
            sections = {
                "IMMEDIATE_ACTIONS":  ("🚨", "Immediate Actions",  "#ff4b4b"),
                "LIFESTYLE_CHANGES":  ("🥗", "Lifestyle Changes",  "#0ea5e9"),
                "MEDICATIONS_HINT":   ("💊", "Medications Hint",   "#8b5cf6"),
                "WARNING_SIGNS":      ("⚠️", "Warning Signs",      "#f59e0b"),
                "FOLLOW_UP":          ("📅", "Follow-Up",          "#10b981"),
            }

            current_section = None
            section_lines: dict = {k: [] for k in sections}

            for line in suggestions.splitlines():
                stripped = line.strip()
                matched = False
                for key in sections:
                    if stripped.startswith(key + ":") or stripped == key:
                        current_section = key
                        matched = True
                        break
                if not matched and current_section and stripped:
                    section_lines[current_section].append(stripped)

            for key, (icon, title, color) in sections.items():
                lines = section_lines.get(key, [])
                if not lines:
                    continue
                st.markdown(
                    f"<div style='border-left:4px solid {color};"
                    f"padding:12px 16px;margin-bottom:16px;"
                    f"background:rgba(0,0,0,0.03);border-radius:6px'>"
                    f"<b style='font-size:1.05rem'>{icon} {title}</b></div>",
                    unsafe_allow_html=True,
                )
                for ln in lines:
                    st.markdown(ln)
                st.write("")
        else:
            st.warning("Suggestions not available — please re-run the analysis.")

        st.divider()
        st.warning(
            "⚠️ **Important Disclaimer:** These suggestions are AI-generated for "
            "informational purposes only and do NOT replace professional medical advice. "
            "Always consult a qualified doctor for diagnosis and treatment."
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — Knowledge Base
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab5:
    if not st.session_state.analysis_complete:
        st.info("👈 Run the analysis first in the **Upload & Analyze** tab.")
    else:
        fs = st.session_state.final_state

        st.header("📚 Medical Knowledge Base (RAG)")
        st.caption(
            "Retrieved from ChromaDB in-memory vector store · "
            "8 disease documents · No external database or API"
        )
        st.divider()

        rag = fs.get("rag_context", "")
        if rag:
            # Split by the separator used in search_rag
            chunks = rag.split("─" * 60)
            for i, chunk in enumerate(chunks):
                chunk = chunk.strip()
                if not chunk:
                    continue
                first_line = chunk.split("\n")[0].strip()
                label = first_line[:60] if first_line else f"Document {i + 1}"
                with st.expander(f"📄 {label}", expanded=(i == 0)):
                    st.markdown(chunk)
        else:
            st.warning("No knowledge retrieved — please re-run the analysis.")

        st.divider()
        with st.expander("📖 All 8 Built-in Disease Documents", expanded=False):
            from tools.tool_definitions import MEDICAL_DOCS
            for disease_name, doc_text in MEDICAL_DOCS.items():
                st.markdown(f"**{disease_name}**")
                st.caption(doc_text[:300] + "…")
                st.write("")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center style='color:gray;font-size:0.85rem'>"
    "🩺 Medical Image AI Analyzer &nbsp;·&nbsp; "
    "CLIP Vision AI &nbsp;·&nbsp; LangGraph &nbsp;·&nbsp; ChromaDB RAG &nbsp;·&nbsp; "
    "100% free — no API key required &nbsp;·&nbsp; "
    "<em>For educational use only</em>"
    "</center>",
    unsafe_allow_html=True,
)
