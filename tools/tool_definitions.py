"""
7 Specialized Agent Tools — Multi-Agent Medical Image AI System
==============================================================
Each tool is an independent specialized agent decorated with @tool.
Tools use InjectedState to read image/context from shared state automatically.
The Supervisor (Gemini LLM) decides WHICH tools to call and WHEN based on results.

Tool signatures visible to Supervisor:
  validate_prompt()                                       — no args
  clip_screen()                                           — no args
  analyze_image(clip_hint)                                — 1 arg
  generate_heatmap(disease_label)                         — 1 arg
  search_rag(disease_label)                               — 1 arg
  get_suggestions(disease, severity, specialist)          — 3 args
  translate_report()                                      — no args
"""

import io
import re
import json
import base64
from typing import Annotated

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from gtts import gTTS

# ── CLIP singleton ────────────────────────────────────────────────────────────
_clip_model = None
_clip_processor = None


def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
    return _clip_model, _clip_processor


# ── Image helpers ─────────────────────────────────────────────────────────────
def _b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Disease catalogue ─────────────────────────────────────────────────────────
DISEASE_TEXTS = [
    "chest X-ray showing pneumonia with consolidation",
    "chest X-ray showing tuberculosis with upper lobe cavitation",
    "chest X-ray or CT showing COVID-19 bilateral ground glass opacities",
    "brain MRI showing tumor with mass effect",
    "X-ray showing bone fracture with cortical break",
    "retinal fundus image showing diabetic retinopathy with hemorrhages",
    "chest X-ray showing pleural effusion with blunted costophrenic angle",
    "chest X-ray showing cardiomegaly with enlarged cardiac silhouette",
    "normal healthy medical image with no pathology",
]

DISEASE_NAMES = [
    "Pneumonia", "Tuberculosis", "COVID-19", "Brain Tumor",
    "Bone Fracture", "Diabetic Retinopathy", "Pleural Effusion",
    "Cardiomegaly", "Normal",
]

SCAN_TEXTS = [
    "medical X-ray image",
    "medical CT scan image",
    "medical MRI image",
    "normal non-medical photograph",
]

SPECIALIST_MAP = {
    "Pneumonia":            "Pulmonologist",
    "Tuberculosis":         "Infectious Disease Specialist",
    "COVID-19":             "Pulmonologist / Intensivist",
    "Brain Tumor":          "Neurosurgeon / Neuro-oncologist",
    "Bone Fracture":        "Orthopaedic Surgeon",
    "Diabetic Retinopathy": "Ophthalmologist / Retina Specialist",
    "Pleural Effusion":     "Pulmonologist",
    "Cardiomegaly":         "Cardiologist",
    "Normal":               "General Practitioner",
}

AFFECTED_REGION_MAP = {
    "Pneumonia":            "Lung parenchyma — alveoli and air spaces",
    "Tuberculosis":         "Upper lobes of the lungs; may extend to other organs",
    "COVID-19":             "Bilateral lung fields, predominantly lower and peripheral zones",
    "Brain Tumor":          "Intracranial space — brain parenchyma",
    "Bone Fracture":        "Cortical bone and periosteum at fracture site",
    "Diabetic Retinopathy": "Retinal microvasculature and macular region",
    "Pleural Effusion":     "Pleural space — between parietal and visceral pleura",
    "Cardiomegaly":         "Cardiac silhouette and myocardium",
    "Normal":               "No pathological region identified",
}

# ── Medical knowledge base ────────────────────────────────────────────────────
MEDICAL_DOCS = {
    "Pneumonia": (
        "Pneumonia is a lung infection causing inflammation of the alveoli (air sacs), filling them with "
        "fluid or pus. On chest X-ray it appears as consolidation (white opaque areas) in one or both lungs. "
        "Lobar pneumonia shows dense homogeneous opacification of an entire lobe. Bronchopneumonia shows "
        "patchy bilateral infiltrates. Air bronchograms are characteristic. Viral pneumonia shows diffuse "
        "interstitial pattern. CLIP confirms opacification consistent with consolidation. "
        "Treatment: Antibiotics (amoxicillin, azithromycin) for bacterial; antivirals for viral; "
        "oxygen therapy if SpO2 below 94%. Hospitalisation for severe cases. "
        "Specialist: Pulmonologist. Recovery 1-3 weeks. "
        "Warning signs: SpO2 below 94%, cyanosis, confusion, respiratory rate above 30/min."
    ),
    "Tuberculosis": (
        "Tuberculosis (TB) is caused by Mycobacterium tuberculosis, spread by airborne droplets. "
        "Chest X-ray shows upper lobe infiltrates, cavitation, hilar lymphadenopathy, miliary pattern. "
        "CLIP identifies upper lobe disease and cavitary changes. Treatment: DOTS HRZE regimen 6-9 months. "
        "Patient isolation required. Mandatory public health notification. "
        "Specialist: Infectious Disease Specialist / Pulmonologist. "
        "Warning signs: Massive haemoptysis, respiratory failure, miliary spread."
    ),
    "COVID-19": (
        "COVID-19 pneumonia caused by SARS-CoV-2. CT shows bilateral peripheral ground-glass opacities, "
        "crazy-paving pattern, lower lobe predominance. CLIP detects bilateral peripheral opacities. "
        "Treatment: Mild — rest, hydration. Severe — dexamethasone, remdesivir, prone positioning. "
        "Critical — ICU, mechanical ventilation. Specialist: Pulmonologist / Intensivist. "
        "Warning signs: SpO2 below 90%, respiratory rate above 30, confusion."
    ),
    "Brain Tumor": (
        "Brain tumors are abnormal intracranial cell growth — primary or metastatic. MRI shows mass lesion "
        "with surrounding oedema. Glioblastoma: ring-enhancing mass with central necrosis. Meningioma: "
        "extra-axial enhancement with dural tail. CLIP detects mass effect. "
        "Treatment: Surgery, radiotherapy, chemotherapy (temozolomide for GBM). Steroids for oedema. "
        "Specialist: Neurosurgeon, Neuro-oncologist. "
        "Warning signs: Thunderclap headache, new seizure, rapid neurological deterioration."
    ),
    "Bone Fracture": (
        "Bone fractures show cortical discontinuity on X-ray — lucent fracture line, displacement, angulation. "
        "CLIP identifies cortical disruption. Types: simple, comminuted, stress, pathological. "
        "Treatment: Immobilisation (cast/splint), surgical fixation (ORIF). Physiotherapy for rehab. "
        "Specialist: Orthopaedic Surgeon. Recovery 6-8 weeks simple, 3-6 months complex. "
        "Warning signs: Loss of pulse/sensation distal to fracture, compartment syndrome, open fracture."
    ),
    "Diabetic Retinopathy": (
        "Diabetic retinopathy causes progressive retinal microvascular damage. Fundus shows microaneurysms, "
        "dot/blot haemorrhages, hard exudates, cotton-wool spots, neovascularisation. CLIP detects "
        "haemorrhage and exudate patterns. Treatment: HbA1c control, anti-VEGF injections, laser. "
        "Specialist: Ophthalmologist / Retina Specialist. Annual screening mandatory. "
        "Warning signs: Sudden vision loss, new floaters, curtain across vision."
    ),
    "Pleural Effusion": (
        "Pleural effusion is fluid in pleural space. X-ray shows blunted costophrenic angle, meniscus sign, "
        "progressive opacification. CLIP detects basal opacification. Causes: heart failure, malignancy, "
        "pneumonia, TB. Treatment: Thoracentesis, chest tube, treat underlying cause. "
        "Specialist: Pulmonologist / Interventional Radiologist. "
        "Warning signs: Severe dyspnoea, haemodynamic instability, mediastinal shift."
    ),
    "Cardiomegaly": (
        "Cardiomegaly means CTR greater than 0.5 on chest X-ray. Causes: dilated cardiomyopathy, "
        "hypertension, valve disease, pericardial effusion. CLIP identifies enlarged cardiac silhouette. "
        "Echocardiogram required. Treatment: ACE inhibitors, beta-blockers, diuretics (furosemide). "
        "Specialist: Cardiologist. "
        "Warning signs: Acute pulmonary oedema, cardiogenic shock, dangerous arrhythmia."
    ),
    "Normal": (
        "No significant pathological findings. Image appears within normal limits. "
        "CLIP shows no features consistent with the 8 screened diseases. "
        "A normal AI result does not exclude subtle pathology — clinical correlation is essential. "
        "Specialist: General Practitioner for routine follow-up."
    ),
}

# ── Suggestions templates ─────────────────────────────────────────────────────
_SUGG = {
    "Pneumonia": {
        "immediate": [
            "Seek medical attention today — do not delay if breathing is laboured",
            "Rest completely and avoid all physical exertion",
            "Drink warm fluids every hour to stay hydrated",
            "Monitor oxygen levels — go to ER if SpO2 drops below 94%",
        ],
        "lifestyle": [
            "Eat light, easily digestible meals — soups and soft foods",
            "Sleep with head elevated at 30° to ease breathing",
            "Avoid cold air, dust, and cigarette smoke completely",
            "Use a humidifier to keep room air moist",
        ],
        "medications": [
            "Antibiotic therapy as prescribed by your doctor",
            "Paracetamol for fever — follow recommended dosage",
            "Expectorant syrups to loosen mucus — as advised",
        ],
        "warning": [
            "SpO2 drops below 94% — call emergency services immediately",
            "Lips or fingertips turn blue (cyanosis) — go to ER at once",
            "Confusion or breathing rate above 30/min — emergency",
        ],
        "followup": {"mild": "within 3-5 days", "moderate": "within 24-48 hours", "severe": "immediately"},
    },
    "Tuberculosis": {
        "immediate": [
            "Isolate immediately — wear a mask and ventilate your room",
            "Notify close contacts for screening",
            "Start TB treatment ONLY as prescribed",
            "Cover mouth when coughing — dispose tissues safely",
        ],
        "lifestyle": [
            "High-protein diet to support immune recovery",
            "Complete ALL medication — stopping causes drug resistance",
            "Get sunlight daily for Vitamin D",
            "Avoid alcohol entirely — interferes with TB drugs",
        ],
        "medications": [
            "HRZE regimen as prescribed — Isoniazid, Rifampicin, Pyrazinamide, Ethambutol",
            "Pyridoxine (Vitamin B6) to prevent nerve damage from Isoniazid",
            "Never adjust TB medications without medical supervision",
        ],
        "warning": [
            "Coughing up large amounts of blood — call emergency services",
            "Sudden severe breathing difficulty — go to ER",
            "Yellow eyes or skin — stop medication and seek urgent care",
        ],
        "followup": {"mild": "within 1 week", "moderate": "within 48 hours", "severe": "immediately"},
    },
    "COVID-19": {
        "immediate": [
            "Self-isolate immediately to prevent spreading",
            "Monitor SpO2 every 4 hours — hospital if below 94%",
            "Rest and drink 2-3 litres of fluid per day",
            "Inform doctor and close contacts",
        ],
        "lifestyle": [
            "Sleep prone (on stomach) to improve oxygen levels if breathless",
            "Small frequent meals rich in Vitamin C, D, Zinc",
            "Gentle breathing exercises once symptoms improve",
            "Avoid smoking and alcohol during illness",
        ],
        "medications": [
            "Paracetamol for fever — follow dosage instructions",
            "Antivirals (remdesivir) for hospitalised patients — doctor prescribed",
            "Corticosteroids (dexamethasone) for severe oxygen-needing cases — hospital only",
        ],
        "warning": [
            "SpO2 below 90% — go to emergency immediately",
            "Persistent chest pain — call emergency services",
            "Confusion or blue lips — life-threatening emergency",
        ],
        "followup": {"mild": "within 3-5 days", "moderate": "within 24 hours", "severe": "immediately"},
    },
    "Brain Tumor": {
        "immediate": [
            "Contact neurosurgeon immediately for urgent MRI",
            "Do not drive — seizure risk",
            "Take prescribed steroids (dexamethasone) for brain swelling only as directed",
            "Arrange a caregiver to accompany you to appointments",
        ],
        "lifestyle": [
            "Anti-inflammatory diet — leafy greens, berries, omega-3",
            "Regular sleep schedule — essential for brain recovery",
            "Avoid alcohol and high-stress environments",
            "Join a brain tumour support group for emotional wellbeing",
        ],
        "medications": [
            "Corticosteroids (dexamethasone) for oedema — as prescribed",
            "Anticonvulsants if seizures occurred — doctor prescribed",
            "Chemotherapy (temozolomide for GBM) — specialist supervised",
        ],
        "warning": [
            "New or worsening seizure — call emergency services",
            "Sudden severe headache — go to ER immediately",
            "Sudden loss of speech or weakness — stroke-like emergency",
        ],
        "followup": {"mild": "within 1 week", "moderate": "within 48 hours", "severe": "same day"},
    },
    "Bone Fracture": {
        "immediate": [
            "Immobilise the injured limb — do not try to realign",
            "Apply ice wrapped in cloth for 20 minutes every hour",
            "Elevate limb above heart level",
            "Go to emergency room for X-ray and immobilisation",
        ],
        "lifestyle": [
            "Increase calcium — dairy, leafy greens, fortified foods",
            "Ensure adequate Vitamin D through sunlight or supplements",
            "No weight-bearing until cleared by orthopaedic surgeon",
            "Follow all physiotherapy exercises during recovery",
        ],
        "medications": [
            "Paracetamol or NSAIDs for pain management",
            "Calcium and Vitamin D supplements to support healing",
            "Bisphosphonates only if prescribed for pathological fracture",
        ],
        "warning": [
            "Numbness or no pulse below fracture — vascular emergency",
            "Rapidly increasing swelling — compartment syndrome",
            "Bone visible through skin — open fracture emergency",
        ],
        "followup": {"mild": "within 1-2 days", "moderate": "within 24 hours", "severe": "emergency room"},
    },
    "Diabetic Retinopathy": {
        "immediate": [
            "Schedule urgent ophthalmologist appointment this week",
            "Check blood sugar levels immediately and log readings",
            "Sudden blurred vision or new floaters — ophthalmology emergency",
            "Take all diabetes medications without missing any dose",
        ],
        "lifestyle": [
            "Strict blood sugar control — HbA1c below 7%",
            "Blood pressure below 130/80 — low-salt diet",
            "Low-glycaemic diet — whole grains, legumes, vegetables",
            "Stop smoking — dramatically worsens retinopathy",
        ],
        "medications": [
            "Anti-VEGF injections (ranibizumab) for macular oedema — specialist administered",
            "Blood glucose medications exactly as prescribed",
            "ACE inhibitors for blood pressure — preferred in diabetics",
        ],
        "warning": [
            "Sudden painless vision loss — ophthalmology emergency",
            "Dark curtain across vision — retinal detachment emergency",
            "New floaters or flashing lights — vitreous haemorrhage",
        ],
        "followup": {"mild": "within 1 week", "moderate": "within 48 hours", "severe": "same day"},
    },
    "Pleural Effusion": {
        "immediate": [
            "Seek medical evaluation promptly — identify the cause",
            "Rest in semi-reclined position (45°) to ease breathing",
            "Go to ER if breathing rate above 25/min at rest",
            "Report all symptoms — fever, weight loss, leg swelling",
        ],
        "lifestyle": [
            "Restrict salt to below 2g per day if cardiac cause",
            "Small frequent meals to avoid lung compression",
            "Always sleep with upper body elevated",
            "Limit strenuous activity until fluid is drained",
        ],
        "medications": [
            "Diuretics (furosemide) if heart failure — as prescribed",
            "Antibiotics if infected effusion — as prescribed",
            "Anticoagulants if pulmonary embolism — specialist managed",
        ],
        "warning": [
            "Sudden severe breathlessness — call emergency services",
            "Dizziness and low blood pressure — haemodynamic instability",
            "Neck or windpipe pulled to one side — tension physiology",
        ],
        "followup": {"mild": "within 2-3 days", "moderate": "within 24 hours", "severe": "emergency room"},
    },
    "Cardiomegaly": {
        "immediate": [
            "See cardiologist urgently — request echocardiogram",
            "Restrict salt to less than 2g per day immediately",
            "Weigh yourself daily — gain of 2kg means fluid retention",
            "Take all heart medications without missing a dose",
        ],
        "lifestyle": [
            "Heart-healthy diet — low sodium, low fat, high fibre",
            "Restrict fluid to 1.5-2 litres daily if advised",
            "Light walking only — ask cardiologist for exercise guidance",
            "Quit smoking and avoid all alcohol",
        ],
        "medications": [
            "ACE inhibitors/ARBs to reduce cardiac workload — as prescribed",
            "Beta-blockers to slow heart rate — as prescribed",
            "Furosemide to eliminate excess fluid — with electrolyte monitoring",
        ],
        "warning": [
            "Sudden breathlessness with pink frothy sputum — call ambulance",
            "Chest pain with cold sweats — cardiogenic shock",
            "Heart racing above 150 bpm irregularly — dangerous arrhythmia",
        ],
        "followup": {"mild": "within 3-5 days", "moderate": "within 24-48 hours", "severe": "emergency room"},
    },
    "Normal": {
        "immediate": [
            "No urgent action required based on imaging findings",
            "Continue all prescribed medications as directed",
            "Discuss any persistent symptoms with your doctor",
            "Keep this report for your medical records",
        ],
        "lifestyle": [
            "Maintain a balanced diet with fruits, vegetables, whole grains",
            "Exercise regularly — 150 minutes moderate activity per week",
            "Get 7-9 hours of quality sleep per night",
            "Avoid smoking, drugs, and excessive alcohol",
        ],
        "medications": [
            "No new medications required based on this imaging",
            "Continue all current prescribed medications",
            "Discuss vitamin supplementation with your doctor",
        ],
        "warning": [
            "New symptoms such as chest pain or breathlessness — seek care",
            "Persistent symptoms despite normal imaging — further investigation needed",
            "Do not delay seeking care if symptoms worsen",
        ],
        "followup": {"mild": "routine annual check-up", "moderate": "within 2 weeks", "severe": "within 48 hours"},
    },
    "Uncertain": {
        "immediate": [
            "I cannot provide specific suggestions because the disease could not be identified with sufficient confidence",
            "Please consult a qualified medical professional for proper diagnosis",
            "Bring this image to your doctor appointment",
        ],
        "lifestyle": [
            "Maintain a healthy lifestyle while awaiting proper diagnosis",
            "Rest and monitor your symptoms",
        ],
        "medications": [
            "Do not self-medicate without a confirmed diagnosis",
            "Continue any currently prescribed medications as directed",
        ],
        "warning": [
            "If symptoms worsen rapidly — seek emergency care immediately",
            "Difficulty breathing or chest pain — go to ER",
        ],
        "followup": {"mild": "within 1 week", "moderate": "within 48 hours", "severe": "immediately"},
    },
}

# ── Translation helpers ───────────────────────────────────────────────────────
_LANG_CODES = {
    "English": None, "Telugu": "te", "Hindi": "hi", "Tamil": "ta",
    "Kannada": "kn", "Malayalam": "ml", "Bengali": "bn", "Marathi": "mr",
    "Gujarati": "gu", "Punjabi": "pa", "Arabic": "ar", "Spanish": "es",
    "French": "fr", "German": "de", "Chinese": "zh-CN", "Japanese": "ja",
}

_GTTS_CODES = {
    "English": "en", "Telugu": "te", "Hindi": "hi", "Tamil": "ta",
    "Kannada": "kn", "Malayalam": "ml", "Bengali": "bn", "Marathi": "mr",
    "Gujarati": "gu", "Punjabi": "pa", "Arabic": "ar", "Spanish": "es",
    "French": "fr", "German": "de", "Chinese": "zh-CN", "Japanese": "ja",
}


def _translate_text(text: str, lang_code: str) -> str:
    try:
        from deep_translator import GoogleTranslator
        chunks = [text[i:i+4500] for i in range(0, len(text), 4500)]
        parts = []
        for chunk in chunks:
            try:
                t = GoogleTranslator(source="auto", target=lang_code).translate(chunk)
                parts.append(t or chunk)
            except Exception:
                parts.append(chunk)
        return "\n".join(parts)
    except Exception:
        return text


# =============================================================================
# AGENT 1 — validate_prompt
# =============================================================================
@tool
def validate_prompt(state: Annotated[dict, InjectedState]) -> str:
    """
    AGENT 1 — Call this FIRST before anything else.
    Reads the user's raw medical question from shared state and rewrites vague
    questions into precise clinical queries. If already clinical, returns unchanged.

    No parameters needed — reads raw_prompt automatically from shared state.

    Examples of rewrites:
    - 'what is wrong?' → 'Analyze this medical image and identify visible disease, region, severity.'
    - 'is this bad?' → 'Examine this image and describe all pathological findings and next steps.'

    Returns the refined clinical question as a string.
    """
    try:
        prompt = state.get("raw_prompt", "").strip()
        if not prompt:
            return "Analyze this medical image and identify any visible disease, affected region, and severity."

        words = set(re.findall(r"\w+", prompt.lower()))
        vague = {"what", "help", "bad", "wrong", "this", "tell", "see", "look", "check", "explain"}
        is_vague = len(prompt.split()) <= 6 and bool(words & vague)

        if is_vague:
            pl = prompt.lower()
            if any(w in pl for w in ["pneumonia", "lung", "chest", "breath"]):
                return ("Analyze this chest X-ray and identify consolidation, infiltrates, "
                        "or signs of pneumonia. Provide severity and affected lung zones.")
            if any(w in pl for w in ["tumor", "brain", "mri", "head"]):
                return ("Examine this brain MRI for mass lesions, oedema, or abnormal enhancement. "
                        "Provide location and differential diagnosis.")
            if any(w in pl for w in ["fracture", "bone", "break"]):
                return ("Analyze this X-ray for fracture lines or cortical breaks. "
                        "Describe type, location, and severity.")
            return ("Analyze this medical image and identify any visible disease, "
                    "affected anatomical region, severity, AI confidence, and recommended specialist.")

        clinical_kw = {"identify", "analyze", "diagnose", "describe", "findings",
                       "disease", "condition", "severity", "assessment", "pathology"}
        if not words & clinical_kw:
            return (prompt.rstrip("?!. ") +
                    ". Provide the disease name, severity level, and recommended specialist.")
        return prompt

    except Exception as e:
        return ("Analyze this medical image and identify any visible disease, "
                f"affected region, and severity. (Error: {e})")


# =============================================================================
# AGENT 2 — clip_screen
# =============================================================================
@tool
def clip_screen(state: Annotated[dict, InjectedState]) -> str:
    """
    AGENT 2 — Call this SECOND, right after validate_prompt.
    Runs the CLIP vision model (openai/clip-vit-base-patch32) locally on CPU.

    No parameters needed — reads image_base64 automatically from shared state.

    Does THREE things:
    1. Checks if the image is actually a medical scan (vs normal photograph).
    2. Returns probability scores for 9 disease categories.
    3. Generates a 512-dimensional image embedding.

    Returns JSON string with: clip_scores, clip_top_disease, clip_is_medical, clip_embedding.

    IMPORTANT for Supervisor decision-making:
    - If clip_is_medical=False in result → warn user, consider skipping heatmap
    - The clip_top_disease value should be passed as clip_hint to analyze_image
    """
    try:
        model, processor = _get_clip()
        pil_image = _b64_to_pil(state["image_base64"])

        all_texts = DISEASE_TEXTS + SCAN_TEXTS
        inputs = processor(text=all_texts, images=pil_image, return_tensors="pt",
                           padding=True, truncation=True, max_length=77)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image[0]

        n_d = len(DISEASE_TEXTS)
        disease_probs = F.softmax(logits[:n_d], dim=0).cpu().numpy()
        scan_probs    = F.softmax(logits[n_d:], dim=0).cpu().numpy()

        top_scan_idx = int(np.argmax(scan_probs))
        is_medical = SCAN_TEXTS[top_scan_idx] != "normal non-medical photograph"

        clip_scores  = {name: round(float(p), 4) for name, p in zip(DISEASE_NAMES, disease_probs)}
        top_disease  = DISEASE_NAMES[int(np.argmax(disease_probs))]

        with torch.no_grad():
            img_in = processor(images=pil_image, return_tensors="pt")
            feats  = model.get_image_features(**img_in)
            feats  = feats / feats.norm(dim=-1, keepdim=True)
            embedding = feats[0].cpu().numpy().tolist()

        return json.dumps({
            "clip_scores":     clip_scores,
            "clip_top_disease": top_disease,
            "clip_is_medical": is_medical,
            "clip_embedding":  embedding,
        })

    except Exception as e:
        return json.dumps({
            "clip_scores":     {n: 0.0 for n in DISEASE_NAMES},
            "clip_top_disease": "Unknown",
            "clip_is_medical": True,
            "clip_embedding":  [],
            "error": str(e),
        })


# =============================================================================
# AGENT 3 — analyze_image
# =============================================================================
@tool
def analyze_image(
    clip_hint: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """
    AGENT 3 — Call this THIRD after clip_screen.
    Produces a structured clinical diagnosis using CLIP probabilities.

    Parameters:
      clip_hint: The top disease name from clip_screen result (e.g., "Pneumonia").
                 Pass exactly as returned by clip_screen.

    Reads image_base64 and refined_prompt automatically from shared state.

    Returns structured text with:
      DISEASE / SEVERITY / CONFIDENCE / SPECIALIST / FINDINGS / AFFECTED_REGION

    IMPORTANT for Supervisor decision-making:
    - If DISEASE = "Uncertain" → skip generate_heatmap, call get_suggestions with disease="Uncertain"
    - If DISEASE is a real disease → proceed with generate_heatmap and search_rag
    - CONFIDENCE value helps decide urgency of follow-up
    """
    CONFIDENCE_THRESHOLD = 0.18

    try:
        model, processor = _get_clip()
        pil_image = _b64_to_pil(state["image_base64"])

        inputs = processor(text=DISEASE_TEXTS, images=pil_image, return_tensors="pt",
                           padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            outputs = model(**inputs)
            disease_probs = F.softmax(outputs.logits_per_image[0], dim=0).cpu().numpy()

        sorted_probs = np.sort(disease_probs)[::-1]
        top_idx      = int(np.argmax(disease_probs))
        top_disease  = DISEASE_NAMES[top_idx]
        top_conf     = float(disease_probs[top_idx])
        second_conf  = float(sorted_probs[1])

        # Honest uncertainty — refuse to guess when confidence is too low
        if top_conf < CONFIDENCE_THRESHOLD or (top_conf - second_conf) < 0.04:
            return (
                f"DISEASE: Uncertain\n"
                f"SEVERITY: Unknown\n"
                f"CONFIDENCE: {top_conf:.2f}\n"
                f"SPECIALIST: General Practitioner\n"
                f"FINDINGS: Cannot determine disease with sufficient confidence "
                f"(max score {top_conf*100:.1f}%). Image may not clearly show a pathological "
                f"condition, or condition may be outside the 8 diseases this system covers. "
                f"Please consult a medical professional.\n"
                f"AFFECTED_REGION: Cannot determine"
            )

        # Cross-check with clip_hint
        if clip_hint and clip_hint in DISEASE_NAMES:
            hint_idx  = DISEASE_NAMES.index(clip_hint)
            hint_conf = float(disease_probs[hint_idx])
            if abs(hint_conf - top_conf) < 0.06:
                top_disease = clip_hint
                top_conf    = hint_conf

        severity   = "severe" if top_conf >= 0.50 else ("moderate" if top_conf >= 0.28 else "mild")
        specialist = SPECIALIST_MAP.get(top_disease, "General Practitioner")
        affected   = AFFECTED_REGION_MAP.get(top_disease, "Unknown region")

        doc = MEDICAL_DOCS.get(top_disease, "")
        imaging_sents = [s.strip() for s in doc.split(".")
                         if any(w in s.lower() for w in
                                ["shows", "appears", "x-ray", "mri", "ct", "clip",
                                 "pattern", "opacit", "consolidat", "enhancing",
                                 "effusion", "silhouette", "fundus"])]
        findings = ". ".join(imaging_sents[:4]).strip()
        if findings and not findings.endswith("."):
            findings += "."
        if not findings:
            findings = doc[:350]

        return (
            f"DISEASE: {top_disease}\n"
            f"SEVERITY: {severity}\n"
            f"CONFIDENCE: {top_conf:.2f}\n"
            f"SPECIALIST: {specialist}\n"
            f"FINDINGS: {findings}\n"
            f"AFFECTED_REGION: {affected}"
        )

    except Exception as e:
        return (
            f"DISEASE: Uncertain\nSEVERITY: Unknown\nCONFIDENCE: 0.00\n"
            f"SPECIALIST: General Practitioner\n"
            f"FINDINGS: Analysis error: {e}\nAFFECTED_REGION: Cannot determine"
        )


# =============================================================================
# AGENT 4 — generate_heatmap
# =============================================================================
@tool
def generate_heatmap(
    disease_label: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """
    AGENT 4 — Call this AFTER analyze_image, only if a specific disease was detected.
    Skip this tool if analyze_image returned DISEASE: Uncertain.

    Parameters:
      disease_label: The detected disease name (e.g., "Pneumonia") from analyze_image.

    Reads image_base64 automatically from shared state.

    Generates a CLIP Attention Rollout heatmap across all 12 ViT transformer layers,
    applies COLORMAP_JET (45% blend), draws red bounding box around disease region.

    Returns the heatmap image as a base64 encoded PNG string.
    """
    try:
        model, processor = _get_clip()
        pil_image  = _b64_to_pil(state["image_base64"])
        rgb_image  = pil_image.convert("RGB")
        img_array  = np.array(rgb_image, dtype=np.float32)
        h, w       = img_array.shape[:2]

        image_inputs = processor(images=rgb_image, return_tensors="pt")
        with torch.no_grad():
            vision_out = model.vision_model(
                pixel_values=image_inputs["pixel_values"],
                output_attentions=True,
            )

        # Attention Rollout across ALL layers
        attentions = vision_out.attentions
        seq_len    = attentions[0].shape[-1]
        rollout    = torch.eye(seq_len)

        for attn_layer in attentions:
            attn_avg = attn_layer[0].mean(dim=0)
            attn_aug = attn_avg + torch.eye(seq_len)
            attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True)
            rollout  = torch.mm(attn_aug, rollout)

        mask      = rollout[0, 1:].cpu().numpy().astype(np.float32)
        grid_size = int(round(len(mask) ** 0.5))
        mask_2d   = mask.reshape(grid_size, grid_size)

        # Percentile contrast stretch
        p2, p98 = float(np.percentile(mask_2d, 2)), float(np.percentile(mask_2d, 98))
        if p98 > p2:
            mask_2d = np.clip((mask_2d - p2) / (p98 - p2), 0.0, 1.0)
        else:
            mask_2d = np.ones_like(mask_2d) * 0.5

        mask_2d      = np.power(mask_2d, 0.4)
        attn_resized = cv2.resize(mask_2d, (w, h), interpolation=cv2.INTER_CUBIC)
        attn_smooth  = cv2.GaussianBlur(attn_resized, (25, 25), 0)

        v_min, v_max = attn_smooth.min(), attn_smooth.max()
        if v_max > v_min:
            attn_smooth = (attn_smooth - v_min) / (v_max - v_min)

        heatmap_u8  = (attn_smooth * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        blended     = (img_array * 0.55 + heatmap_rgb * 0.45).clip(0, 255).astype(np.uint8)

        threshold = float(np.percentile(attn_smooth, 70))
        ys, xs    = np.where(attn_smooth >= threshold)
        if len(ys) > 0:
            pad = 12
            x1 = max(0,     int(xs.min()) - pad)
            y1 = max(0,     int(ys.min()) - pad)
            x2 = min(w - 1, int(xs.max()) + pad)
            y2 = min(h - 1, int(ys.max()) + pad)
        else:
            x1, y1, x2, y2 = int(w*0.2), int(h*0.2), int(w*0.8), int(h*0.8)

        cv2.rectangle(blended, (x1, y1), (x2, y2), (255, 0, 0), 3)
        label = disease_label[:30] if disease_label else "Disease Region"
        cv2.putText(blended, label, (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

        return _pil_to_b64(Image.fromarray(blended))

    except Exception:
        try:
            pil_image = _b64_to_pil(state["image_base64"])
            img_arr   = np.array(pil_image.convert("RGB"), dtype=np.float32)
            fh, fw    = img_arr.shape[:2]
            cx, cy    = fw // 2, fh // 2
            sx, sy    = fw // 4, fh // 4
            xx, yy    = np.meshgrid(np.arange(fw), np.arange(fh))
            gauss     = np.exp(-((xx-cx)**2/(2*sx**2) + (yy-cy)**2/(2*sy**2)))
            gauss_u8  = (gauss * 255).astype(np.uint8)
            hmap      = cv2.cvtColor(cv2.applyColorMap(gauss_u8, cv2.COLORMAP_JET),
                                     cv2.COLOR_BGR2RGB).astype(np.float32)
            blended   = (img_arr * 0.55 + hmap * 0.45).clip(0, 255).astype(np.uint8)
            cv2.rectangle(blended, (cx-sx, cy-sy), (cx+sx, cy+sy), (255, 0, 0), 3)
            return _pil_to_b64(Image.fromarray(blended))
        except Exception:
            return state.get("image_base64", "")


# =============================================================================
# AGENT 5 — search_rag
# =============================================================================
@tool
def search_rag(disease_label: str) -> str:
    """
    AGENT 5 — Call this AFTER analyze_image, can run in parallel with generate_heatmap.
    Retrieves medical knowledge ONLY for the specific detected disease.

    Parameters:
      disease_label: The detected disease name (e.g., "Pneumonia") from analyze_image.
                     If disease is "Uncertain", this tool will return an appropriate message.

    Returns disease-specific medical knowledge as a string.
    Does NOT return information about other diseases.
    """
    UNKNOWN = {"Uncertain", "Unknown", "", "Unspecified Condition"}

    if disease_label in UNKNOWN:
        return (
            "No specific medical knowledge can be provided because the disease "
            "could not be identified with sufficient confidence. "
            "Please consult a qualified medical professional for proper diagnosis."
        )

    doc = MEDICAL_DOCS.get(disease_label)
    if doc:
        return f"Medical Knowledge — {disease_label}:\n\n{doc}"

    return (
        f"No specific knowledge for '{disease_label}' in this knowledge base. "
        f"Covered diseases: Pneumonia, Tuberculosis, COVID-19, Brain Tumor, "
        f"Bone Fracture, Diabetic Retinopathy, Pleural Effusion, Cardiomegaly, Normal."
    )


# =============================================================================
# AGENT 6 — get_suggestions
# =============================================================================
@tool
def get_suggestions(disease: str, severity: str, specialist: str) -> str:
    """
    AGENT 6 — Call this AFTER both generate_heatmap and search_rag are done.
    Generates structured patient-friendly clinical advice.

    Parameters:
      disease:    The detected disease name from analyze_image (e.g., "Pneumonia").
      severity:   The severity level from analyze_image (mild / moderate / severe).
      specialist: The recommended specialist from analyze_image (e.g., "Pulmonologist").

    If disease is "Uncertain", returns honest message that specific advice cannot be given.

    Returns advice covering:
      IMMEDIATE_ACTIONS / LIFESTYLE_CHANGES / MEDICATIONS_HINT / WARNING_SIGNS / FOLLOW_UP
    """
    UNKNOWN = {"Uncertain", "Unknown", ""}

    if disease in UNKNOWN:
        return (
            "IMMEDIATE_ACTIONS:\n"
            "- Cannot provide specific suggestions — disease not identified with sufficient confidence\n"
            "- Please consult a qualified medical professional for proper diagnosis and advice\n"
            "- Bring this image to your doctor's appointment\n\n"
            "FOLLOW_UP:\n"
            "- See a General Practitioner who can request appropriate tests\n"
            "- Mention all symptoms you are experiencing at the appointment"
        )

    key     = disease if disease in _SUGG else None
    if key is None:
        return (
            f"IMMEDIATE_ACTIONS:\n"
            f"- '{disease}' is outside the 8 diseases this system covers\n"
            f"- Please consult a {specialist} for condition-specific guidance\n\n"
            f"FOLLOW_UP:\n- See a {specialist} as soon as possible"
        )

    db      = _SUGG[key]
    sev_key = severity.lower() if severity.lower() in ("mild", "moderate", "severe") else "moderate"
    followup_time = db["followup"].get(sev_key, "as soon as possible")

    lines = [
        f"Suggestions for: {disease} ({severity.capitalize()} severity)",
        "-" * 50,
        "",
        "IMMEDIATE_ACTIONS:",
        *[f"- {a}" for a in db["immediate"]],
        "",
        "LIFESTYLE_CHANGES:",
        *[f"- {a}" for a in db["lifestyle"]],
        "",
        "MEDICATIONS_HINT:",
        *[f"- {a}" for a in db["medications"]],
        "",
        "WARNING_SIGNS:",
        *[f"- {a}" for a in db["warning"]],
        "",
        "FOLLOW_UP:",
        f"- See a {specialist} {followup_time}",
        "- Bring all previous imaging reports and medical records to your appointment",
        "- Ask your doctor which follow-up tests are needed",
    ]
    return "\n".join(lines)


# =============================================================================
# AGENT 7 — translate_report
# =============================================================================
@tool
def translate_report(state: Annotated[dict, InjectedState]) -> str:
    """
    AGENT 7 — Call this LAST after get_suggestions.
    Builds the complete medical report from shared state, translates it to the
    user's selected language, and generates voice audio.

    No parameters needed — reads disease, findings, suggestions, and language
    automatically from shared state.

    If language is English, returns report unchanged.
    Otherwise uses free Google Translate (no API key).
    Audio generated using free gTTS (no API key).

    Returns JSON string with: translated_text, audio_base64.
    """
    from datetime import datetime

    try:
        disease    = state.get("disease_label", "Unknown")
        severity   = state.get("severity", "unknown").upper()
        confidence = state.get("confidence", 0.0)
        specialist = state.get("specialist", "General Practitioner")
        language   = state.get("language", "English")
        ts         = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        clip_scores = state.get("clip_scores", {})
        clip_lines  = ""
        if clip_scores:
            top3 = sorted(clip_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            clip_lines = "\n  CLIP Top Predictions:\n" + "\n".join(
                f"    {d}: {p*100:.1f}%" for d, p in top3
            )

        report = (
            f"{'=' * 62}\n"
            f"{'MEDICAL IMAGE AI ANALYSIS REPORT':^62}\n"
            f"{'Generated: ' + ts:^62}\n"
            f"{'=' * 62}\n\n"
            f"  DIAGNOSIS SUMMARY\n"
            f"  Condition     : {disease}\n"
            f"  Severity      : {severity}\n"
            f"  AI Confidence : {confidence * 100:.1f}%\n"
            f"  Specialist    : {specialist}\n"
            f"{clip_lines}\n\n"
            f"  DETAILED FINDINGS\n"
            f"{state.get('image_analysis', 'Not available.')}\n\n"
            f"  MEDICAL KNOWLEDGE\n"
            f"{state.get('rag_context', 'Not available.')}\n\n"
            f"  PATIENT RECOMMENDATIONS\n"
            f"{state.get('suggestions', 'Not available.')}\n\n"
            f"  DISCLAIMER: This AI analysis is for informational purposes only\n"
            f"  and does NOT replace professional medical diagnosis.\n"
            f"{'=' * 62}\n"
        )

        lang_code  = _LANG_CODES.get(language)
        translated = _translate_text(report, lang_code) if lang_code else report

        try:
            gtts_code = _GTTS_CODES.get(language, "en")
            clean = re.sub(r"[=|]", " ", translated).replace("•", "").strip()
            clean = re.sub(r"[ \t]{2,}", " ", clean)[:3000]
            tts   = gTTS(text=clean, lang=gtts_code, slow=False)
            buf   = io.BytesIO()
            tts.write_to_fp(buf)
            audio_b64 = base64.b64encode(buf.getvalue()).decode()
        except Exception:
            audio_b64 = ""

        return json.dumps({"translated_text": translated, "audio_base64": audio_b64})

    except Exception as e:
        return json.dumps({"translated_text": f"Report generation failed: {e}", "audio_base64": ""})
