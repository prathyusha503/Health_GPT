"""
7 LangGraph tools — 100% free, zero API keys required.
CLIP (local CPU) handles all vision tasks.
deep-translator handles translation (free Google Translate).
gTTS handles audio (free Google TTS).
ChromaDB handles RAG knowledge retrieval.
"""

import io
import re
import json
import base64

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from langchain_core.tools import tool
from gtts import gTTS

# ── CLIP singleton (lazy-loaded on first use) ─────────────────────────────────
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

# ── Medical knowledge base (8 diseases + Normal) ─────────────────────────────
MEDICAL_DOCS = {
    "Pneumonia": (
        "Pneumonia is a lung infection causing inflammation of the alveoli (air sacs), filling them with "
        "fluid or pus. On chest X-ray it appears as consolidation (white opaque areas) in one or both lungs. "
        "Lobar pneumonia shows dense homogeneous opacification of an entire lobe. Bronchopneumonia shows "
        "patchy bilateral infiltrates. Air bronchograms (air-filled airways visible within consolidation) are "
        "a characteristic sign. Viral pneumonia shows a diffuse interstitial pattern. "
        "CLIP vision analysis confirms opacification in the affected lung field consistent with consolidation. "
        "Treatment: Antibiotics (amoxicillin, azithromycin, levofloxacin) for bacterial pneumonia. "
        "Antiviral medications for viral types. Oxygen therapy if SpO2 falls below 94%. "
        "Hospitalisation for severe cases with IV antibiotics and respiratory support. "
        "Specialist: Pulmonologist for moderate-severe cases; General Practitioner for mild. "
        "Recovery typically takes 1–3 weeks with appropriate treatment. "
        "Warning signs: SpO2 below 94%, cyanosis (blue lips/fingertips), confusion, respiratory rate above 30/min."
    ),
    "Tuberculosis": (
        "Tuberculosis (TB) is caused by Mycobacterium tuberculosis and is spread via airborne droplets. "
        "Chest X-ray shows upper lobe infiltrates, cavitation (thin-walled air-filled holes), hilar "
        "lymphadenopathy, miliary pattern (millet-seed nodules throughout both lungs), and pleural effusion. "
        "The Ghon complex (primary focus + lymph node) is seen in primary infection. Tree-in-bud pattern "
        "on CT indicates active endobronchial spread. CLIP analysis identifies upper lobe disease and "
        "possible cavitary changes characteristic of TB. "
        "Treatment: DOTS therapy — Intensive phase (2 months): Isoniazid (H) + Rifampicin (R) + "
        "Pyrazinamide (Z) + Ethambutol (E). Continuation phase (4 months): H + R. "
        "MDR-TB requires second-line drugs. Patient must be isolated. Mandatory public health notification. "
        "Specialist: Infectious Disease Specialist or Pulmonologist. "
        "Warning signs: Massive haemoptysis (coughing blood), respiratory failure, miliary spread."
    ),
    "COVID-19": (
        "COVID-19 pneumonia is caused by SARS-CoV-2 coronavirus, affecting the lungs bilaterally. "
        "CT scan is more sensitive than chest X-ray. CT shows bilateral peripheral ground-glass opacities "
        "(GGO) predominantly in the lower lobes, crazy-paving pattern (GGO with interlobular septal "
        "thickening), peripheral consolidation, and vascular thickening. Chest X-ray shows bilateral "
        "infiltrates and haziness. CLIP analysis detects bilateral peripheral opacities characteristic "
        "of COVID-19 pattern. Typical progression: peripheral GGO → consolidation → potential fibrosis. "
        "Treatment: Mild — rest, hydration, antipyretics. Moderate — hospitalisation, supplemental oxygen. "
        "Severe — dexamethasone, remdesivir, anticoagulation, prone positioning. "
        "Critical — ICU, mechanical ventilation. "
        "Specialist: Pulmonologist / Intensivist for severe cases. "
        "Warning signs: SpO2 below 90%, respiratory rate above 30/min, confusion, inability to speak in sentences."
    ),
    "Brain Tumor": (
        "Brain tumors are abnormal intracranial cell growth — either primary (originating in the brain) "
        "or metastatic (spread from elsewhere). MRI with gadolinium contrast is the gold standard. "
        "Glioblastoma (GBM): ring-enhancing mass with central necrosis, surrounding vasogenic oedema "
        "(bright T2/FLAIR), and mass effect with midline shift. Meningioma: extra-axial homogeneous "
        "enhancement with dural tail sign. Metastases: multiple ring-enhancing lesions at grey-white "
        "matter junction. CT shows hyperdense mass, calcification, or haemorrhage. "
        "CLIP analysis detects mass effect and surrounding signal change consistent with intracranial lesion. "
        "Treatment: Surgery (craniotomy/resection), radiotherapy, chemotherapy (temozolomide for GBM). "
        "Steroids (dexamethasone) for cerebral oedema reduction. Targeted therapy for specific mutations. "
        "Specialist: Neurosurgeon, Neuro-oncologist, Radiation Oncologist. "
        "Warning signs: Sudden severe headache (thunderclap), new-onset seizure, rapid neurological deterioration."
    ),
    "Bone Fracture": (
        "Bone fractures represent cortical discontinuity caused by trauma, stress, or underlying bone disease. "
        "X-ray shows a lucent fracture line, cortical break, displacement, angulation, periosteal reaction, "
        "and soft tissue swelling. Types: simple (closed), compound (open with skin breach), comminuted "
        "(multiple fragments), stress (repetitive microtrauma), pathological (through diseased bone), "
        "greenstick (incomplete, in children), compression (vertebral). "
        "CT is used for complex fractures and occult injuries. MRI detects bone marrow oedema, "
        "occult fractures, and ligament injuries. CLIP identifies cortical disruption at the fracture site. "
        "Treatment: Immobilisation (cast/splint/brace), closed or open reduction, surgical fixation "
        "(ORIF with plates, screws, intramedullary nails), external fixation. Physiotherapy for rehabilitation. "
        "Specialist: Orthopaedic Surgeon. Recovery: 6–8 weeks simple, 3–6 months complex. "
        "Warning signs: Loss of sensation or pulse distal to fracture, compartment syndrome, open fracture."
    ),
    "Diabetic Retinopathy": (
        "Diabetic retinopathy is progressive retinal microvascular damage caused by chronic hyperglycaemia; "
        "the leading cause of preventable blindness in working-age adults. "
        "Fundus photography shows: microaneurysms (earliest sign — small red dots), dot and blot "
        "haemorrhages, hard exudates (bright yellow lipid deposits near leaky vessels), cotton-wool spots "
        "(fluffy white areas of nerve fibre infarction), and neovascularisation (fragile new vessel growth "
        "in proliferative stage). OCT reveals macular oedema and retinal thickening. "
        "CLIP detects haemorrhage and exudate patterns characteristic of diabetic retinopathy. "
        "Classification: Non-proliferative (NPDR) mild/moderate/severe → Proliferative (PDR). "
        "Treatment: Strict glycaemic control (HbA1c < 7%), blood pressure control. Anti-VEGF intravitreal "
        "injections (ranibizumab, bevacizumab, aflibercept) for macular oedema and PDR. "
        "Laser photocoagulation. Vitrectomy for advanced vitreous haemorrhage or retinal detachment. "
        "Specialist: Ophthalmologist / Retina Specialist. Annual dilated eye exam mandatory for all diabetics. "
        "Warning signs: Sudden painless vision loss, new floaters (vitreous haemorrhage), curtain across vision."
    ),
    "Pleural Effusion": (
        "Pleural effusion is abnormal fluid accumulation in the pleural space between lung and chest wall. "
        "Chest X-ray shows: blunting of costophrenic angle (detectable at ~200 mL), meniscus sign (concave "
        "upper fluid border), progressive hemithoracic opacification, mediastinal shift away from large "
        "effusions. Lateral decubitus X-ray shows freely layering fluid. Ultrasound is most sensitive for "
        "detection and guides drainage. CT differentiates exudate from transudate and identifies cause. "
        "CLIP detects basal opacification and blunted costophrenic angle. "
        "Causes — Transudate (protein-poor): heart failure, liver cirrhosis, nephrotic syndrome. "
        "Exudate (protein-rich, Light's criteria): pneumonia (parapneumonic), malignancy, TB, pulmonary embolism. "
        "Treatment: Thoracentesis for large/symptomatic effusions. Chest tube for empyema. "
        "Pleurodesis for recurrent malignant effusions. Treat underlying cause. "
        "Specialist: Pulmonologist or Interventional Radiologist. "
        "Warning signs: Severe dyspnoea, haemodynamic instability, mediastinal shift (tension physiology)."
    ),
    "Cardiomegaly": (
        "Cardiomegaly is abnormal cardiac enlargement, defined radiographically as a cardiothoracic (CT) "
        "ratio greater than 0.5 on a PA chest X-ray. The heart appears to occupy more than half the "
        "transverse diameter of the chest. Globular cardiac silhouette suggests pericardial effusion. "
        "Cephalization of pulmonary vessels and Kerley B lines indicate elevated filling pressures. "
        "CLIP identifies increased cardiac silhouette relative to thoracic diameter. "
        "Causes: Dilated cardiomyopathy (DCM), hypertensive heart disease, valvular disease (aortic or "
        "mitral regurgitation), ischaemic cardiomyopathy, myocarditis, pericardial effusion, "
        "congenital heart defects, high-output states (anaemia, thyrotoxicosis). "
        "Echocardiogram is mandatory for definitive evaluation of structure and function. "
        "Treatment: ACE inhibitors/ARBs, beta-blockers, loop diuretics (furosemide), spironolactone. "
        "ICD for arrhythmia prevention. Cardiac resynchronisation therapy (CRT). Heart transplant for end-stage. "
        "Specialist: Cardiologist; Cardiac Surgeon for structural intervention. "
        "Warning signs: Acute pulmonary oedema (pink frothy sputum), cardiogenic shock, dangerous arrhythmia."
    ),
    "Normal": (
        "No significant pathological findings identified on this medical image. "
        "The imaging features appear within normal limits for the modality used. "
        "CLIP vision analysis shows no features consistent with the 8 screened diseases. "
        "A normal AI screening result does not exclude subtle or early-stage pathology. "
        "Clinical correlation with patient symptoms and history remains paramount. "
        "If symptoms persist despite a normal image, additional imaging modalities or specialist review "
        "may be warranted. Continue regular health check-ups and preventive care. "
        "Specialist: General Practitioner for routine follow-up and clinical correlation."
    ),
}

# ── Suggestions templates per disease ────────────────────────────────────────
_SUGG = {
    "Pneumonia": {
        "immediate": [
            "Seek medical attention today — do not delay if breathing is laboured or SpO2 drops",
            "Rest completely and avoid all physical exertion until cleared by your doctor",
            "Drink warm fluids (water, clear broth, herbal tea) every hour to stay well hydrated",
            "Monitor oxygen levels with a pulse oximeter — go to ER if below 94%",
        ],
        "lifestyle": [
            "Eat light, easily digestible meals — soups, soft rice, steamed vegetables",
            "Sleep with head and chest elevated at 30° to ease breathing",
            "Avoid cold air, dust, cigarette smoke, and air pollutants completely",
            "Use a clean humidifier to keep room air moist and reduce irritation",
        ],
        "medications": [
            "Antibiotic therapy (type and dose prescribed by your doctor — never self-prescribe)",
            "Paracetamol or ibuprofen for fever and discomfort — follow recommended dosage",
            "Expectorant syrups to loosen and clear mucus — as advised by pharmacist",
        ],
        "warning": [
            "Oxygen saturation drops below 94% — call emergency services immediately",
            "Lips or fingertips turn blue (cyanosis) — go to ER at once",
            "Confusion, inability to stay awake, or breathing rate above 30/min — emergency",
        ],
        "followup": {
            "mild": "within 3–5 days",
            "moderate": "within 24–48 hours",
            "severe": "immediately — go to the emergency room",
        },
    },
    "Tuberculosis": {
        "immediate": [
            "Isolate yourself from others — wear a surgical mask and ensure good room ventilation",
            "Notify close contacts so they can be screened for TB exposure",
            "Begin TB treatment ONLY as prescribed by your infectious disease doctor",
            "Cover your mouth with a tissue when coughing and dispose of tissues safely",
        ],
        "lifestyle": [
            "Eat a high-protein, nutrient-rich diet to support immune system recovery",
            "Get adequate sunlight daily for Vitamin D — supports immune function",
            "Complete ALL prescribed medication without missing a single dose — stopping early causes drug resistance",
            "Avoid alcohol entirely as it interferes with TB medications and liver function",
        ],
        "medications": [
            "First-line HRZE anti-TB drugs (Isoniazid, Rifampicin, Pyrazinamide, Ethambutol) — as prescribed",
            "Pyridoxine (Vitamin B6) to prevent peripheral neuropathy caused by Isoniazid",
            "Never share, adjust, or stop TB medications without direct medical supervision",
        ],
        "warning": [
            "Coughing up large amounts of blood (haemoptysis) — call emergency services immediately",
            "Sudden severe difficulty breathing or respiratory distress — go to ER",
            "Yellow eyes or skin (jaundice) or severe abdominal pain — stop medication and seek urgent care",
        ],
        "followup": {
            "mild": "within 1 week",
            "moderate": "within 48 hours",
            "severe": "immediately — hospitalisation required",
        },
    },
    "COVID-19": {
        "immediate": [
            "Self-isolate immediately to prevent spreading the virus to household members",
            "Monitor your oxygen saturation every 4 hours — go to hospital if below 94%",
            "Rest completely and drink at least 2–3 litres of fluid per day",
            "Inform your doctor, employer, and close contacts of your status",
        ],
        "lifestyle": [
            "Sleep in prone position (on your stomach) periodically to improve oxygen levels if breathless",
            "Eat small frequent meals rich in Vitamin C, D, and Zinc",
            "Perform gentle breathing exercises (deep diaphragmatic breathing) once acute symptoms improve",
            "Avoid smoking, alcohol, and any strenuous physical activity during illness",
        ],
        "medications": [
            "Paracetamol for fever and body ache management — follow dosage instructions",
            "Antiviral medications (e.g., remdesivir) for hospitalised patients — doctor prescribed only",
            "Corticosteroids (dexamethasone) for severe cases needing oxygen — hospital use only",
        ],
        "warning": [
            "Oxygen saturation falls below 90% — go to the emergency room immediately",
            "Persistent chest pain or pressure that does not resolve — call emergency services",
            "Confusion, inability to wake up, or blue-coloured lips — life-threatening emergency",
        ],
        "followup": {
            "mild": "within 3–5 days (teleconsult acceptable)",
            "moderate": "within 24 hours",
            "severe": "immediately — hospitalisation required",
        },
    },
    "Brain Tumor": {
        "immediate": [
            "Contact a neurosurgeon or neurologist immediately for urgent MRI with contrast",
            "Do not drive — seizure risk makes all driving unsafe until medically cleared",
            "Take anti-oedema medication (e.g., dexamethasone) only as prescribed",
            "Arrange for a trusted family member or caregiver to accompany you to appointments",
        ],
        "lifestyle": [
            "Eat an anti-inflammatory diet — leafy greens, berries, omega-3 rich foods, turmeric",
            "Maintain a regular sleep schedule — sleep is critical for neurological recovery",
            "Avoid alcohol, recreational drugs, and high-stress environments completely",
            "Join a brain tumour patient support group for emotional and psychological wellbeing",
        ],
        "medications": [
            "Corticosteroids (dexamethasone) to reduce cerebral oedema — as prescribed",
            "Anticonvulsants if seizures have occurred — doctor prescribed and monitored",
            "Chemotherapy agents (e.g., temozolomide for GBM) — under specialist supervision only",
        ],
        "warning": [
            "New or worsening seizure of any type — call emergency services immediately",
            "Sudden severe headache unlike any before ('thunderclap') — go to ER at once",
            "Sudden loss of speech, weakness on one side, or acute confusion — stroke-like emergency",
        ],
        "followup": {
            "mild": "within 1 week",
            "moderate": "within 48 hours",
            "severe": "same day — urgent neurosurgical evaluation required",
        },
    },
    "Bone Fracture": {
        "immediate": [
            "Immobilise the injured limb — do not attempt to realign the bone yourself",
            "Apply ice wrapped in a cloth for 20 minutes every hour to reduce pain and swelling",
            "Elevate the injured limb above heart level when sitting or lying",
            "Go to the emergency room for X-ray confirmation and proper immobilisation",
        ],
        "lifestyle": [
            "Increase dietary calcium — dairy products, sardines, leafy greens, fortified foods",
            "Ensure adequate Vitamin D through safe sunlight exposure or prescribed supplements",
            "Avoid weight-bearing on the affected limb until cleared by your orthopaedic surgeon",
            "Follow all physiotherapy exercises during recovery to restore full range of motion",
        ],
        "medications": [
            "Analgesics (paracetamol, NSAIDs like ibuprofen) for pain management — as tolerated",
            "Calcium and Vitamin D supplements to support bone healing and remodelling",
            "Bisphosphonates only if prescribed for underlying osteoporosis or pathological fracture",
        ],
        "warning": [
            "Severe numbness, weakness, or no pulse in the limb below the fracture — vascular/nerve emergency",
            "Rapidly increasing swelling with severe pain not relieved by elevation — compartment syndrome",
            "Bone visible through skin or an open wound near the fracture — open fracture emergency",
        ],
        "followup": {
            "mild": "within 1–2 days",
            "moderate": "within 24 hours",
            "severe": "emergency room immediately",
        },
    },
    "Diabetic Retinopathy": {
        "immediate": [
            "Schedule an urgent appointment with an ophthalmologist or retina specialist this week",
            "Check blood sugar levels immediately and log all readings for your doctor",
            "If vision is suddenly blurred or you see new floaters — go to ophthalmology emergency today",
            "Take all diabetes medications without missing any dose",
        ],
        "lifestyle": [
            "Strictly control blood sugar — target HbA1c below 7% as advised by your endocrinologist",
            "Control blood pressure to below 130/80 mmHg through medication and low-salt diet",
            "Follow a low-glycaemic, high-fibre diet — whole grains, legumes, non-starchy vegetables",
            "Stop smoking — smoking dramatically accelerates diabetic retinopathy progression",
        ],
        "medications": [
            "Anti-VEGF intravitreal injections (ranibizumab, bevacizumab) for macular oedema — specialist administered",
            "Blood glucose controlling medications — exactly as prescribed by your endocrinologist",
            "ACE inhibitors for blood pressure control (preferred in diabetics) — as prescribed",
        ],
        "warning": [
            "Sudden painless vision loss in one or both eyes — ophthalmology emergency immediately",
            "A dark curtain or shadow descending across your vision — retinal detachment emergency",
            "Sudden onset of many new floaters or bright flashing lights — vitreous haemorrhage",
        ],
        "followup": {
            "mild": "within 1 week",
            "moderate": "within 48 hours",
            "severe": "same day — ophthalmology emergency",
        },
    },
    "Pleural Effusion": {
        "immediate": [
            "Seek medical evaluation promptly — the underlying cause must be identified and treated",
            "Rest in a semi-reclined position (45°) to ease breathing and reduce chest tightness",
            "Monitor your breathing rate — if above 25 breaths/min at rest, go to ER",
            "Report all associated symptoms (fever, night sweats, leg swelling, weight loss) to your doctor",
        ],
        "lifestyle": [
            "Restrict salt intake to below 2g per day if cardiac or liver disease is the cause",
            "Eat small frequent meals to avoid abdominal pressure on the compressed lung",
            "Always sleep with the upper body elevated — never lie completely flat",
            "Limit strenuous activity until fluid is drained and the underlying cause is treated",
        ],
        "medications": [
            "Diuretics (furosemide) if heart failure or fluid overload is the cause — as prescribed",
            "Antibiotics if the effusion is infected (empyema or parapneumonic) — as prescribed",
            "Anticoagulants if pulmonary embolism is the underlying cause — specialist managed",
        ],
        "warning": [
            "Sudden severe difficulty breathing or breathlessness at rest — call emergency services",
            "Dizziness, very low blood pressure, or rapid heart rate — haemodynamic instability",
            "Feeling of the neck or windpipe being pulled to one side — tension physiology emergency",
        ],
        "followup": {
            "mild": "within 2–3 days",
            "moderate": "within 24 hours",
            "severe": "emergency room immediately — drainage required",
        },
    },
    "Cardiomegaly": {
        "immediate": [
            "See your cardiologist urgently — request an echocardiogram to assess heart function",
            "Restrict dietary salt to less than 2g (half a teaspoon) per day starting immediately",
            "Weigh yourself every morning before eating — a gain of 2 kg in 2 days means fluid retention",
            "Take all prescribed heart medications without missing a single dose",
        ],
        "lifestyle": [
            "Adopt a heart-healthy diet — low sodium, low saturated fat, high fibre, plenty of vegetables",
            "Restrict total daily fluid intake to 1.5–2 litres if your doctor advises it",
            "Engage in only light walking or gentle activity — ask your cardiologist for exercise guidance",
            "Quit smoking and avoid all alcohol — both directly worsen cardiac function",
        ],
        "medications": [
            "ACE inhibitors or ARBs to reduce cardiac workload and prevent remodelling — as prescribed",
            "Beta-blockers to slow heart rate and reduce myocardial oxygen demand — as prescribed",
            "Loop diuretics (furosemide) to eliminate excess fluid — with regular electrolyte monitoring",
        ],
        "warning": [
            "Sudden severe breathlessness with pink frothy sputum — acute pulmonary oedema, call ambulance",
            "Chest pain with cold sweats and dizziness — possible cardiogenic shock emergency",
            "Heart racing irregularly or palpitations above 150 bpm — dangerous arrhythmia emergency",
        ],
        "followup": {
            "mild": "within 3–5 days",
            "moderate": "within 24–48 hours",
            "severe": "emergency room immediately",
        },
    },
    "Normal": {
        "immediate": [
            "No urgent action required based on current imaging findings",
            "Continue all ongoing medications exactly as prescribed by your doctor",
            "Discuss any persistent symptoms with your doctor — imaging is just one part of assessment",
            "Keep this report safely for your medical records",
        ],
        "lifestyle": [
            "Maintain a balanced diet with plenty of fruits, vegetables, whole grains, and lean protein",
            "Exercise regularly — at least 150 minutes of moderate aerobic activity per week",
            "Get 7–9 hours of quality sleep every night — sleep is essential for overall health",
            "Avoid smoking, recreational drugs, and excessive alcohol consumption",
        ],
        "medications": [
            "No new medications required based on current imaging findings",
            "Continue all currently prescribed medications exactly as directed",
            "Discuss vitamin D and B12 supplementation needs with your doctor at next visit",
        ],
        "warning": [
            "If new symptoms develop such as chest pain, breathlessness, or unexplained weight loss — seek care",
            "Persistent symptoms despite a normal imaging result warrant further specialist investigation",
            "Do not delay seeking care if symptoms worsen — a normal scan today does not guarantee tomorrow",
        ],
        "followup": {
            "mild": "routine annual check-up",
            "moderate": "within 2 weeks for clinical correlation",
            "severe": "within 48 hours with the relevant specialist",
        },
    },
}



# ── Translation helper (deep-translator, free, no API key) ───────────────────
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
        chunk_size = 4500
        chunks = [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]
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


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 — validate_prompt
# ─────────────────────────────────────────────────────────────────────────────
@tool
def validate_prompt(raw_prompt: str) -> str:
    """
    Call this tool FIRST before any other tool.
    Takes the user's raw question about a medical image and checks if it is
    clear and medically relevant. If the question is vague like 'what is wrong'
    or 'is this bad' or 'help', rewrite it into a precise structured clinical
    question. If already clear, return it unchanged.

    Examples of rewrites:
    - 'what is wrong?' becomes 'Analyze this medical image and identify any
      visible disease, affected anatomical region, and severity level.'
    - 'is this bad?' becomes 'Examine this image and describe all pathological
      findings, the disease name, and recommended next steps.'

    Returns the refined clinical question as a string.
    """
    try:
        prompt = raw_prompt.strip()
        if not prompt:
            return (
                "Analyze this medical image and identify any visible disease, "
                "affected anatomical region, severity level, and recommended specialist."
            )

        vague_triggers = {
            "what", "help", "bad", "wrong", "this", "tell", "see",
            "look", "check", "explain", "info", "information",
        }
        words = set(re.findall(r"\w+", prompt.lower()))
        is_short = len(prompt.split()) <= 6
        is_vague = is_short and bool(words & vague_triggers)

        if is_vague:
            pl = prompt.lower()
            if any(w in pl for w in ["pneumonia", "lung", "chest", "breath"]):
                return (
                    "Analyze this chest X-ray and identify any consolidation, "
                    "infiltrates, or signs of pneumonia. Provide severity and affected lung zones."
                )
            if any(w in pl for w in ["tumor", "brain", "mri", "head"]):
                return (
                    "Examine this brain MRI and describe any mass lesions, oedema, "
                    "or abnormal enhancement. Provide location and differential diagnosis."
                )
            if any(w in pl for w in ["fracture", "bone", "break", "xray", "x-ray"]):
                return (
                    "Analyze this X-ray for any fracture lines or cortical breaks. "
                    "Describe the fracture type, location, displacement, and severity."
                )
            return (
                "Analyze this medical image and identify any visible disease, "
                "affected anatomical region, severity level, AI confidence, "
                "and recommend the appropriate medical specialist."
            )

        clinical_kw = {
            "identify", "analyze", "diagnose", "describe", "findings",
            "disease", "condition", "severity", "assessment", "pathology",
        }
        if not words & clinical_kw:
            return (
                prompt.rstrip("?!. ")
                + ". Provide the disease name, severity level, and recommended specialist."
            )

        return prompt

    except Exception:
        return (
            "Analyze this medical image and identify any visible disease, "
            "affected anatomical region, and severity level."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 — clip_screen
# ─────────────────────────────────────────────────────────────────────────────
@tool
def clip_screen(image_base64: str) -> str:
    """
    Call this tool SECOND, right after validate_prompt.
    Uses the CLIP model (openai/clip-vit-base-patch32) running locally on CPU.
    1. Checks if the image is a medical scan vs a normal photograph.
    2. Returns probability scores for 9 disease categories.
    3. Returns a 512-dimensional image embedding for semantic RAG search.
    Returns a JSON string with: clip_scores, clip_top_disease, clip_is_medical,
    clip_embedding (as list).
    """
    try:
        model, processor = _get_clip()
        pil_image = _b64_to_pil(image_base64)

        all_texts = DISEASE_TEXTS + SCAN_TEXTS
        inputs = processor(
            text=all_texts,
            images=pil_image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image[0]

        n_d = len(DISEASE_TEXTS)
        disease_probs = F.softmax(logits[:n_d], dim=0).cpu().numpy()
        scan_probs = F.softmax(logits[n_d:], dim=0).cpu().numpy()

        top_scan_idx = int(np.argmax(scan_probs))
        is_medical = SCAN_TEXTS[top_scan_idx] != "normal non-medical photograph"

        clip_scores = {
            name: round(float(p), 4)
            for name, p in zip(DISEASE_NAMES, disease_probs)
        }
        top_disease = DISEASE_NAMES[int(np.argmax(disease_probs))]

        # 512-dim image embedding
        with torch.no_grad():
            img_in = processor(images=pil_image, return_tensors="pt")
            feats = model.get_image_features(**img_in)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embedding = feats[0].cpu().numpy().tolist()

        return json.dumps({
            "clip_scores": clip_scores,
            "clip_top_disease": top_disease,
            "clip_is_medical": is_medical,
            "clip_embedding": embedding,
        })

    except Exception as e:
        return json.dumps({
            "clip_scores": {n: 0.0 for n in DISEASE_NAMES},
            "clip_top_disease": "Unknown",
            "clip_is_medical": True,
            "clip_embedding": [],
            "error": str(e),
        })


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3 — analyze_image
# ─────────────────────────────────────────────────────────────────────────────
@tool
def analyze_image(image_base64: str, refined_prompt: str, clip_hint: str) -> str:
    """
    Call this tool THIRD after clip_screen.
    Uses CLIP disease probabilities to produce a structured clinical diagnosis
    for the detected disease ONLY. If confidence is too low, honestly reports
    that it cannot determine the disease rather than guessing.
    Returns: DISEASE, SEVERITY, CONFIDENCE, SPECIALIST, FINDINGS, AFFECTED_REGION.
    """
    # Minimum confidence required to make any disease claim
    CONFIDENCE_THRESHOLD = 0.18

    try:
        model, processor = _get_clip()
        pil_image = _b64_to_pil(image_base64)

        inputs = processor(
            text=DISEASE_TEXTS,
            images=pil_image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        with torch.no_grad():
            outputs = model(**inputs)
            disease_probs = F.softmax(outputs.logits_per_image[0], dim=0).cpu().numpy()

        sorted_probs = np.sort(disease_probs)[::-1]
        top_idx   = int(np.argmax(disease_probs))
        top_disease = DISEASE_NAMES[top_idx]
        top_conf    = float(disease_probs[top_idx])
        second_conf = float(sorted_probs[1])

        # ── Honest uncertainty check ──────────────────────────────────────────
        # Refuse to guess when: (a) confidence too low, OR (b) top two are
        # so close that CLIP has no clear winner.
        too_low   = top_conf < CONFIDENCE_THRESHOLD
        too_close = (top_conf - second_conf) < 0.04

        if too_low or too_close:
            return (
                f"DISEASE: Uncertain\n"
                f"SEVERITY: Unknown\n"
                f"CONFIDENCE: {top_conf:.2f}\n"
                f"SPECIALIST: General Practitioner\n"
                f"FINDINGS: I cannot determine the disease from this image with sufficient "
                f"confidence (max score: {top_conf * 100:.1f}%). "
                f"This may be because the image does not clearly show a pathological condition, "
                f"the image quality is low, or the condition is outside the 8 diseases this "
                f"system was designed to detect. Please consult a medical professional for "
                f"a proper clinical evaluation.\n"
                f"AFFECTED_REGION: Cannot determine"
            )

        # ── Cross-check with CLIP pre-screen hint ─────────────────────────────
        if clip_hint and clip_hint in DISEASE_NAMES:
            hint_idx  = DISEASE_NAMES.index(clip_hint)
            hint_conf = float(disease_probs[hint_idx])
            if abs(hint_conf - top_conf) < 0.06 and clip_hint != top_disease:
                top_disease = clip_hint
                top_conf    = hint_conf

        # ── Severity from confidence ───────────────────────────────────────────
        if top_conf >= 0.50:
            severity = "severe"
        elif top_conf >= 0.28:
            severity = "moderate"
        else:
            severity = "mild"

        specialist = SPECIALIST_MAP.get(top_disease, "General Practitioner")
        affected   = AFFECTED_REGION_MAP.get(top_disease, "Unknown region")

        # ── Findings: pull ONLY from this disease's document ──────────────────
        doc = MEDICAL_DOCS.get(top_disease, "")
        imaging_sentences = [
            s.strip() for s in doc.split(".")
            if any(w in s.lower() for w in
                   ["shows", "appears", "imaging", "x-ray", "mri", "ct",
                    "clip", "pattern", "scan", "opacit", "consolidat",
                    "enhancing", "effusion", "silhouette", "attn", "fundus"])
        ]
        findings = ". ".join(imaging_sentences[:4]).strip()
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
            f"DISEASE: Uncertain\n"
            f"SEVERITY: Unknown\n"
            f"CONFIDENCE: 0.00\n"
            f"SPECIALIST: General Practitioner\n"
            f"FINDINGS: Analysis encountered an error: {str(e)}. "
            f"Please consult a medical professional.\n"
            f"AFFECTED_REGION: Cannot determine"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 4 — generate_heatmap
# ─────────────────────────────────────────────────────────────────────────────
@tool
def generate_heatmap(image_base64: str, disease_label: str, image_analysis: str) -> str:
    """
    Call this tool AFTER analyze_image.
    Generates a CLIP Attention Rollout heatmap overlay showing where the disease
    is located. Uses attention rollout across ALL transformer layers for strong
    contrast. Applies COLORMAP_JET and blends 45% over the original image.
    Draws a red bounding box around the highest-attention region.
    Returns the result as a base64 encoded PNG string.
    """
    try:
        model, processor = _get_clip()
        pil_image = _b64_to_pil(image_base64)
        rgb_image = pil_image.convert("RGB")
        img_array = np.array(rgb_image, dtype=np.float32)
        h, w = img_array.shape[:2]

        image_inputs = processor(images=rgb_image, return_tensors="pt")

        with torch.no_grad():
            vision_out = model.vision_model(
                pixel_values=image_inputs["pixel_values"],
                output_attentions=True,
            )

        # ── Attention Rollout across ALL layers ───────────────────────────────
        # Accumulates attention flow from input tokens to CLS through every layer.
        # Produces far more discriminative maps than single-layer CLS attention.
        attentions = vision_out.attentions  # list of [1, heads, seq, seq]
        seq_len = attentions[0].shape[-1]
        rollout = torch.eye(seq_len)

        for attn_layer in attentions:
            # Average over attention heads
            attn_avg = attn_layer[0].mean(dim=0)  # [seq, seq]
            # Add residual connection (identity) then row-normalise
            attn_aug = attn_avg + torch.eye(seq_len)
            attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True)
            rollout = torch.mm(attn_aug, rollout)

        # CLS token (row 0) attention over all patch tokens (cols 1:)
        mask = rollout[0, 1:].cpu().numpy().astype(np.float32)  # [num_patches]

        grid_size = int(round(len(mask) ** 0.5))   # 7 for ViT-B/32
        mask_2d = mask.reshape(grid_size, grid_size)

        # ── Percentile contrast stretch (eliminates near-uniform maps) ────────
        p2  = float(np.percentile(mask_2d, 2))
        p98 = float(np.percentile(mask_2d, 98))
        if p98 > p2:
            mask_2d = np.clip((mask_2d - p2) / (p98 - p2), 0.0, 1.0)
        else:
            mask_2d = np.ones_like(mask_2d) * 0.5   # flat fallback

        # Gamma boost: raises mid-range values so heatmap is clearly visible
        mask_2d = np.power(mask_2d, 0.4)

        # ── Resize to full image resolution ───────────────────────────────────
        attn_resized = cv2.resize(mask_2d, (w, h), interpolation=cv2.INTER_CUBIC)
        attn_smooth  = cv2.GaussianBlur(attn_resized, (25, 25), 0)

        # Final normalise after blur
        v_min, v_max = attn_smooth.min(), attn_smooth.max()
        if v_max > v_min:
            attn_smooth = (attn_smooth - v_min) / (v_max - v_min)

        # ── Apply COLORMAP_JET and blend ──────────────────────────────────────
        heatmap_u8  = (attn_smooth * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        blended = (img_array * 0.55 + heatmap_rgb * 0.45).clip(0, 255).astype(np.uint8)

        # ── Bounding box around top-attention region (top 30% pixels) ─────────
        threshold = float(np.percentile(attn_smooth, 70))
        ys, xs = np.where(attn_smooth >= threshold)
        if len(ys) > 0:
            pad = 12
            x1 = max(0,     int(xs.min()) - pad)
            y1 = max(0,     int(ys.min()) - pad)
            x2 = min(w - 1, int(xs.max()) + pad)
            y2 = min(h - 1, int(ys.max()) + pad)
        else:
            x1, y1, x2, y2 = int(w*0.2), int(h*0.2), int(w*0.8), int(h*0.8)

        cv2.rectangle(blended, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Label
        label = disease_label[:30] if disease_label else "Disease Region"
        cv2.putText(blended, label, (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

        return _pil_to_b64(Image.fromarray(blended))

    except Exception as exc:
        # Guaranteed fallback: plain red-box Gaussian on original image
        try:
            pil_image = _b64_to_pil(image_base64)
            img_arr = np.array(pil_image.convert("RGB"), dtype=np.float32)
            fh, fw = img_arr.shape[:2]
            cx, cy = fw // 2, fh // 2
            sx, sy = fw // 4, fh // 4
            xx, yy = np.meshgrid(np.arange(fw), np.arange(fh))
            gauss = np.exp(-((xx - cx)**2 / (2*sx**2) + (yy - cy)**2 / (2*sy**2)))
            gauss = (gauss * 255).astype(np.uint8)
            hmap = cv2.cvtColor(cv2.applyColorMap(gauss, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB).astype(np.float32)
            blended = (img_arr * 0.55 + hmap * 0.45).clip(0, 255).astype(np.uint8)
            cv2.rectangle(blended, (cx - sx, cy - sy), (cx + sx, cy + sy), (255, 0, 0), 3)
            return _pil_to_b64(Image.fromarray(blended))
        except Exception:
            return image_base64


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 5 — search_rag
# ─────────────────────────────────────────────────────────────────────────────
@tool
def search_rag(disease_label: str, clip_embedding_json: str) -> str:
    """
    Call this tool AFTER analyze_image, in parallel with generate_heatmap.
    Returns ONLY the knowledge document for the specific detected disease.
    Does NOT return information about other diseases.
    If the disease is uncertain or unknown, returns an appropriate message.
    """
    # Diseases we have no meaningful knowledge for
    UNKNOWN_LABELS = {"Uncertain", "Unknown", "", "Unspecified Condition"}

    if disease_label in UNKNOWN_LABELS:
        return (
            "No specific medical knowledge can be provided because the disease "
            "could not be identified with sufficient confidence. "
            "Please consult a qualified medical professional for proper diagnosis."
        )

    # Return ONLY this disease's document — never mix in other diseases
    doc = MEDICAL_DOCS.get(disease_label)
    if doc:
        return f"Knowledge Base — {disease_label}:\n\n{doc}"

    # Disease name exists but not in our 8-disease KB
    return (
        f"No specific information available for '{disease_label}' in this knowledge base. "
        f"The system covers: Pneumonia, Tuberculosis, COVID-19, Brain Tumor, "
        f"Bone Fracture, Diabetic Retinopathy, Pleural Effusion, Cardiomegaly, and Normal. "
        f"Please consult a medical professional for information about this condition."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 6 — get_suggestions
# ─────────────────────────────────────────────────────────────────────────────
@tool
def get_suggestions(disease: str, severity: str, specialist: str, rag_context: str) -> str:
    """
    Call this tool AFTER both generate_heatmap and search_rag complete.
    Generates structured patient-friendly medical advice for the diagnosed
    condition and severity. Returns advice covering: IMMEDIATE_ACTIONS,
    LIFESTYLE_CHANGES, MEDICATIONS_HINT, WARNING_SIGNS, FOLLOW_UP.
    Uses pre-built evidence-based templates — no API key required.
    """
    UNKNOWN_LABELS = {"Uncertain", "Unknown", "", "Unspecified Condition"}

    # Refuse to fabricate suggestions when diagnosis is not confident
    if disease in UNKNOWN_LABELS:
        return (
            "IMMEDIATE_ACTIONS:\n"
            "• I cannot provide specific suggestions because the disease could not be "
            "identified with sufficient confidence from this image.\n"
            "• Please consult a qualified medical professional for proper diagnosis and advice.\n\n"
            "FOLLOW_UP:\n"
            "• See a General Practitioner who can examine you and request appropriate tests.\n"
            "• Bring this image and any symptoms you are experiencing to the appointment."
        )

    try:
        key = disease if disease in _SUGG else None
        if key is None:
            return (
                f"IMMEDIATE_ACTIONS:\n"
                f"• '{disease}' is outside the 8 diseases this system covers.\n"
                f"• Please consult a {specialist} for condition-specific guidance.\n\n"
                f"FOLLOW_UP:\n"
                f"• See a {specialist} as soon as possible with this imaging report."
            )

        db = _SUGG[key]
        sev_key = severity.lower() if severity.lower() in ("mild", "moderate", "severe") else "moderate"
        followup_time = db["followup"].get(sev_key, "as soon as possible")

        lines = [
            f"Suggestions for: {disease} ({severity.capitalize()} severity)",
            "-" * 50,
            "",
            "IMMEDIATE_ACTIONS:",
            *[f"• {a}" for a in db["immediate"]],
            "",
            "LIFESTYLE_CHANGES:",
            *[f"• {a}" for a in db["lifestyle"]],
            "",
            "MEDICATIONS_HINT:",
            *[f"• {a}" for a in db["medications"]],
            "",
            "WARNING_SIGNS:",
            *[f"• {a}" for a in db["warning"]],
            "",
            "FOLLOW_UP:",
            f"• See a {specialist} {followup_time}",
            "• Bring all previous imaging reports and medical records to your appointment",
            "• Ask your doctor which follow-up tests or investigations are needed",
        ]
        return "\n".join(lines)

    except Exception as e:
        return (
            f"Could not generate suggestions: {str(e)}\n"
            f"Please consult a {specialist} directly."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 7 — translate_report
# ─────────────────────────────────────────────────────────────────────────────
@tool
def translate_report(report: str, language: str) -> str:
    """
    Call this tool LAST after get_suggestions.
    If language is 'English', returns the report unchanged.
    Otherwise translates using deep-translator (free Google Translate, no API key).
    Also generates gTTS audio (free Google TTS, no API key) for the report.
    Returns a JSON string with keys: translated_text, audio_base64.
    """
    try:
        lang_code = _LANG_CODES.get(language)
        translated = _translate_text(report, lang_code) if lang_code else report

        try:
            gtts_code = _GTTS_CODES.get(language, "en")
            clean = re.sub(r"[═║╔╚╗╝╠╣╦╩╬─━│┃]", " ", translated)
            clean = clean.replace("⚠️", "Warning:").replace("•", "").strip()
            clean = re.sub(r"[ \t]{2,}", " ", clean)
            clean = re.sub(r"\n{3,}", "\n\n", clean)[:3000]

            tts = gTTS(text=clean, lang=gtts_code, slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            audio_b64 = base64.b64encode(buf.getvalue()).decode()
        except Exception:
            audio_b64 = ""

        return json.dumps({"translated_text": translated, "audio_base64": audio_b64})

    except Exception as e:
        return json.dumps({"translated_text": report, "audio_base64": "", "error": str(e)})
