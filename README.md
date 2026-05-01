# 🩺 Medical Image AI Analyzer

A fully free, API-key-free medical image analysis system powered by **CLIP Vision AI**, **LangGraph**, **ChromaDB RAG**, and **Streamlit**. Upload an X-ray, CT scan, or MRI — get an instant AI diagnosis, attention heatmap, clinical suggestions, multilingual report, and voice audio.

---

## 🚀 Live Demo

> Run locally with `streamlit run app.py` — no cloud deployment needed.

---

## 📌 Overview

The **Medical Image AI Analyzer** is a multi-tool agentic AI system that analyzes medical images and provides structured clinical insights. It uses **CLIP (Contrastive Language–Image Pretraining)** — an open-source vision model that runs entirely on your CPU — to classify diseases, generate attention heatmaps, and power a semantic knowledge retrieval system.

The system is designed to be **completely free** with:
- No Gemini API key
- No OpenAI API key
- No paid cloud services
- Everything runs locally or uses free public endpoints

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔬 **Disease Detection** | Classifies 8 medical conditions using CLIP locally on CPU |
| 🌡️ **Attention Heatmap** | Visualizes disease regions using CLIP ViT Attention Rollout |
| 📊 **CLIP Probability Chart** | Bar chart showing confidence scores for all 8 diseases |
| 📚 **RAG Knowledge Base** | ChromaDB-powered retrieval of disease-specific medical knowledge |
| 💊 **Clinical Suggestions** | Evidence-based immediate actions, lifestyle changes, and warning signs |
| 🌍 **16-Language Translation** | Free Google Translate via deep-translator (no API key) |
| 🔊 **Voice Audio** | Text-to-speech in 16 languages via gTTS (free Google TTS) |
| ✅ **Honest Uncertainty** | Says "I don't know" when confidence is too low — no hallucination |

---

## 🏗️ Architecture

```
User Upload (Image + Question)
           │
           ▼
┌─────────────────────────────────────────────────────┐
│              LangGraph Sequential Pipeline           │
│                                                     │
│  Tool 1: validate_prompt  → Refines vague questions │
│  Tool 2: clip_screen      → CLIP disease screening  │
│  Tool 3: analyze_image    → Structured diagnosis    │
│  Tool 4: generate_heatmap ─┐                        │
│                             ├── Parallel execution  │
│  Tool 5: search_rag       ─┘                        │
│  Tool 6: get_suggestions  → Clinical advice         │
│  Tool 7: translate_report → Translation + Audio     │
│  Node 8: aggregator       → Final report            │
└─────────────────────────────────────────────────────┘
           │
           ▼
    5-Tab Streamlit UI
```

---

## 🦠 Supported Diseases

| # | Disease | Imaging Modality |
|---|---|---|
| 1 | Pneumonia | Chest X-ray |
| 2 | Tuberculosis | Chest X-ray |
| 3 | COVID-19 Pneumonia | Chest X-ray / CT |
| 4 | Brain Tumor | MRI |
| 5 | Bone Fracture | X-ray |
| 6 | Diabetic Retinopathy | Fundus Photography |
| 7 | Pleural Effusion | Chest X-ray |
| 8 | Cardiomegaly | Chest X-ray |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Vision AI** | CLIP `openai/clip-vit-base-patch32` (HuggingFace Transformers) |
| **Heatmap** | CLIP ViT Attention Rollout + OpenCV COLORMAP_JET |
| **Agent Pipeline** | LangGraph `StateGraph` with parallel fan-out/fan-in |
| **Knowledge Base** | Medical document templates (disease-specific, direct lookup) |
| **Translation** | deep-translator (free Google Translate) |
| **Text-to-Speech** | gTTS (free Google TTS) |
| **Frontend** | Streamlit (5-tab layout) |
| **Deep Learning** | PyTorch (CPU) |

---

## 📁 Project Structure

```
medical_ai_project/
│
├── app.py                    # Streamlit frontend (5-tab UI)
├── requirements.txt          # All dependencies
├── .env.example              # No API key needed
├── .gitignore
│
├── tools/
│   ├── __init__.py
│   └── tool_definitions.py   # All 7 @tool functions
│
├── graph/
│   ├── __init__.py
│   ├── state.py              # MedicalState TypedDict
│   └── workflow.py           # LangGraph pipeline + nodes
│
└── agents/                   # Legacy pipeline (reference only)
    ├── prompt_agent.py
    ├── image_agent.py
    ├── heatmap_agent.py
    ├── rag_agent.py
    ├── suggestion_agent.py
    ├── aggregator_agent.py
    └── multilingual_agent.py
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10 or 3.11
- 2 GB free disk space (for CLIP model cache on first run)
- Internet connection (for CLIP model download on first run, and for translation/audio)

### 1. Clone the repository

```bash
git clone https://github.com/prathyusha503/Health_GPT.git
cd Health_GPT
```

### 2. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

### 4. Open in browser

```
http://localhost:8501
```

> **No API key needed.** The sidebar shows a green "No API key required" message.

---

## 🖥️ How to Use

1. Go to **📤 Upload & Analyze** tab
2. Upload a medical image (JPG, PNG, WEBP)
3. Type your question — e.g. *"What disease is visible in this X-ray?"*
4. Click **🔬 Analyze Image**
5. Wait for the 7-tool pipeline to complete (progress bar shows each step)
6. Switch to other tabs to view results:

| Tab | Content |
|---|---|
| **🔬 Diagnosis & Heatmap** | Disease metrics, CLIP bar chart, original vs heatmap |
| **📋 Full Report** | Translated report, audio player, download button |
| **💊 Suggestions** | Immediate actions, warning signs, follow-up advice |
| **📚 Knowledge Base** | RAG-retrieved disease-specific medical knowledge |

---

## 🌍 Supported Languages

English · Telugu · Hindi · Tamil · Kannada · Malayalam · Bengali · Marathi · Gujarati · Punjabi · Arabic · Spanish · French · German · Chinese · Japanese

---

## 🔒 Honesty & Safety

- **Confidence threshold:** If CLIP confidence is below 18%, the system says *"I cannot determine the disease"* — no guessing
- **Disease-specific only:** RAG and suggestions show information for the detected disease only — no mixing of other diseases
- **Disclaimer:** All outputs are clearly marked as AI analysis for informational purposes only and not a substitute for professional medical diagnosis

---

## 📦 Dependencies

```
langgraph>=0.2.0
langchain>=0.2.0
langchain-core>=0.2.0
streamlit>=1.35.0
pillow>=10.0.0
numpy>=1.26.0
opencv-python-headless>=4.9.0
torch>=2.2.0
torchvision>=0.17.0
transformers>=4.40.0
gtts>=2.5.0
deep-translator>=1.11.0
python-dotenv>=1.0.0
```

---

## ⚠️ Disclaimer

This application is for **educational and informational purposes only**. It does **not** replace professional medical diagnosis, treatment, or advice. Always consult a qualified medical professional for any health concerns.

---

## 👩‍💻 Author

**Prathyusha Mangali**
- GitHub: [@prathyusha503](https://github.com/prathyusha503)
- Email: v_mangali.prathyusha@centific.com

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
