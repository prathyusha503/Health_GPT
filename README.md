# 🩺 Medical Image AI Analyzer — Multi-Agent System

A **multi-agent AI system** for medical image analysis powered by **CLIP Vision AI**, **LangGraph**, **ChromaDB RAG**, and **Streamlit**. Upload an X-ray, CT scan, or MRI — 7 specialized AI agents collaborate to deliver an instant diagnosis, attention heatmap, clinical suggestions, multilingual report, and voice audio.

> ✅ **100% Free — No API Key Required**

---

## 📌 Overview

The **Medical Image AI Analyzer** is a true **multi-agent AI system** where 7 independent, specialized agents each handle a distinct part of the medical image analysis pipeline. Each agent has its own role, input, output, and error handling. They communicate through a shared **MedicalState** and some agents run **in parallel** for efficiency.

The system uses **CLIP (Contrastive Language–Image Pretraining)** — an open-source vision model that runs entirely on your CPU — to classify diseases, generate attention heatmaps, and power the entire analysis without any paid API.

---

## 🤖 The 7 Agents

| # | Agent | Role | Technology |
|---|---|---|---|
| 1 | **Prompt Agent** | Validates and rewrites vague user questions into precise clinical queries | Rule-based NLP |
| 2 | **CLIP Screen Agent** | Classifies the image against 8 diseases and detects if it is a real medical scan | CLIP ViT-B/32 (local CPU) |
| 3 | **Image Analysis Agent** | Produces structured diagnosis: DISEASE, SEVERITY, CONFIDENCE, SPECIALIST, FINDINGS | CLIP + Medical KB |
| 4 | **Heatmap Agent** | Generates an attention rollout heatmap showing the disease region | CLIP ViT Attention Rollout + OpenCV |
| 5 | **RAG Agent** | Retrieves disease-specific medical knowledge from the built-in knowledge base | ChromaDB (in-memory) |
| 6 | **Suggestion Agent** | Generates patient-friendly clinical advice: immediate actions, warning signs, follow-up | Evidence-based templates |
| 7 | **Translation Agent** | Translates the full report to 16 languages and generates voice audio | deep-translator + gTTS |

---

## 🏗️ Multi-Agent Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         USER INPUT                       │
                    │   Medical Image + Question + Language    │
                    └────────────────┬────────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │  AGENT 1: Prompt Agent                  │
                    │  Rewrites vague questions into          │
                    │  precise clinical queries               │
                    └────────────────┬────────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │  AGENT 2: CLIP Screen Agent             │
                    │  • Detects if image is medical          │
                    │  • Scores 8 disease categories          │
                    │  • Generates 512-dim image embedding    │
                    └────────────────┬────────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │  AGENT 3: Image Analysis Agent          │
                    │  • Structured diagnosis with            │
                    │    DISEASE / SEVERITY / CONFIDENCE      │
                    │  • Honest "I don't know" if unsure      │
                    └──────┬──────────────────────────────────┘
                           │
               ┌───────────┴────────────┐
               │  PARALLEL EXECUTION    │
               ▼                        ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│  AGENT 4: Heatmap Agent  │  │  AGENT 5: RAG Agent          │
│  • Attention Rollout     │  │  • Fetches only the          │
│    across all 12 ViT     │  │    detected disease's        │
│    transformer layers    │  │    knowledge document        │
│  • COLORMAP_JET overlay  │  │  • No cross-disease mixing   │
│  • Red bounding box      │  │                              │
└──────────────┬───────────┘  └──────────────────┬───────────┘
               │                                  │
               └───────────┬──────────────────────┘
                           │  (fan-in — both must complete)
                           ▼
                    ┌─────────────────────────────────────────┐
                    │  AGENT 6: Suggestion Agent              │
                    │  • Immediate actions                    │
                    │  • Lifestyle changes                    │
                    │  • Warning signs                        │
                    │  • Follow-up advice                     │
                    └────────────────┬────────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │  AGENT 7: Translation Agent             │
                    │  • Translates report to 16 languages    │
                    │  • Generates gTTS voice audio           │
                    └────────────────┬────────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │  AGGREGATOR NODE                        │
                    │  Merges all agent outputs into          │
                    │  the final structured report            │
                    └────────────────┬────────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │         5-TAB STREAMLIT UI              │
                    │  Diagnosis · Heatmap · Report ·         │
                    │  Suggestions · Knowledge Base           │
                    └─────────────────────────────────────────┘
```

### Key Design Properties

- **Specialization** — each agent does one task only and does it well
- **Parallel execution** — Agents 4 and 5 run simultaneously (fan-out / fan-in)
- **Shared state** — all agents read/write to a single `MedicalState` TypedDict
- **Isolated outputs** — each agent returns only its own state keys (prevents conflicts)
- **Honest uncertainty** — if CLIP confidence < 18%, Agent 3 says "I don't know" instead of guessing
- **Disease-specific RAG** — Agent 5 returns information only about the detected disease

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **7 Specialized Agents** | Each agent handles a distinct step in the pipeline |
| ⚡ **Parallel Processing** | Heatmap Agent and RAG Agent run simultaneously |
| 🔬 **Disease Detection** | 8 medical conditions classified using local CLIP model |
| 🌡️ **Attention Heatmap** | Attention Rollout across all 12 ViT layers for real disease localization |
| 📊 **CLIP Probability Chart** | Bar chart showing confidence scores for all diseases |
| 📚 **Disease-Specific RAG** | Only shows knowledge for the detected condition |
| 💊 **Clinical Suggestions** | Evidence-based actions, lifestyle tips, and warning signs |
| 🌍 **16-Language Support** | Free Google Translate — no API key |
| 🔊 **Voice Audio** | Text-to-speech in 16 languages via gTTS |
| ✅ **No Hallucination** | Low confidence → "I don't know" — never fabricates |

---

## 🦠 Supported Diseases

| Disease | Imaging | Typical Signs |
|---|---|---|
| Pneumonia | Chest X-ray | Consolidation, infiltrates |
| Tuberculosis | Chest X-ray | Upper lobe cavitation, miliary pattern |
| COVID-19 | Chest X-ray / CT | Bilateral ground-glass opacities |
| Brain Tumor | MRI | Ring-enhancing mass, surrounding oedema |
| Bone Fracture | X-ray | Cortical break, lucent fracture line |
| Diabetic Retinopathy | Fundus | Microaneurysms, haemorrhages |
| Pleural Effusion | Chest X-ray | Blunted costophrenic angle |
| Cardiomegaly | Chest X-ray | CTR > 0.5 |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Multi-Agent Framework** | LangGraph `StateGraph` with parallel fan-out/fan-in |
| **Vision AI** | CLIP `openai/clip-vit-base-patch32` (HuggingFace) |
| **Heatmap** | CLIP ViT Attention Rollout + OpenCV COLORMAP_JET |
| **Knowledge Base** | Built-in disease documents (direct lookup, no vector DB needed) |
| **Translation** | deep-translator (free Google Translate, no API key) |
| **Text-to-Speech** | gTTS (free Google TTS, no API key) |
| **Frontend** | Streamlit 5-tab layout |
| **Deep Learning** | PyTorch CPU |
| **Agent Tools** | `@tool` decorated functions from `langchain_core` |

---

## 📁 Project Structure

```
medical_ai_project/
│
├── app.py                    # Streamlit frontend — 5-tab multi-agent UI
├── requirements.txt          # All Python dependencies
├── .env.example              # No API key needed
├── .gitignore                # Excludes cache, .env, venv
│
├── tools/
│   ├── __init__.py
│   └── tool_definitions.py   # All 7 agent @tool functions (CLIP, RAG, TTS, etc.)
│
├── graph/
│   ├── __init__.py
│   ├── state.py              # MedicalState TypedDict — shared agent memory
│   └── workflow.py           # LangGraph pipeline: 7 agent nodes + aggregator
│
└── agents/                   # Legacy simple-pipeline agents (reference only)
    └── (7 files)
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10 or 3.11
- ~2 GB free disk space (CLIP model cache, first run only)
- Internet connection (CLIP download on first run; translation and audio need internet)

### 1. Clone the repository

```bash
git clone https://github.com/prathyusha503/Health_GPT.git
cd Health_GPT
```

### 2. Install PyTorch (CPU version)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install all other dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

### 5. Open in browser

```
http://localhost:8501
```

> **No API key required.** Sidebar shows a green confirmation message.

---

## 🖥️ How to Use

1. Open **📤 Upload & Analyze** tab
2. Upload a medical image (JPG, PNG, WEBP — X-ray, CT, MRI, Fundus)
3. Type your question — e.g. *"What disease is visible in this chest X-ray?"*
4. Click **🔬 Analyze Image**
5. Watch the 7 agents work (live progress bar)
6. Navigate tabs for results:

| Tab | What You See |
|---|---|
| **🔬 Diagnosis & Heatmap** | Disease name, severity, CLIP scores chart, original vs heatmap |
| **📋 Full Report** | Translated report text, audio player, download as .txt |
| **💊 Suggestions** | Color-coded sections: Immediate Actions, Warning Signs, Follow-Up |
| **📚 Knowledge Base** | Disease-specific medical knowledge retrieved by RAG Agent |

---

## 🌍 Supported Output Languages

```
English · Telugu · Hindi · Tamil · Kannada · Malayalam
Bengali · Marathi · Gujarati · Punjabi · Arabic · Spanish
French · German · Chinese · Japanese
```

---

## 🔒 Accuracy & Safety Design

| Design Choice | Reason |
|---|---|
| Confidence threshold (18%) | Prevents low-confidence guesses |
| Disease-specific RAG | No cross-disease information mixing |
| Attention Rollout (all 12 layers) | More accurate heatmap than single-layer attention |
| Parallel agent execution | Heatmap and knowledge retrieval run simultaneously |
| Isolated agent state returns | Prevents parallel state conflicts in LangGraph |

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
