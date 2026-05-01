from typing import TypedDict, Optional


class MedicalState(TypedDict):
    image_base64: str
    raw_prompt: str
    language: str
    refined_prompt: str
    clip_scores: dict
    clip_top_disease: str
    clip_embedding_json: str
    clip_is_medical: bool
    image_analysis: str
    disease_label: str
    severity: str
    confidence: float
    specialist: str
    heatmap_base64: str
    rag_context: str
    suggestions: str
    aggregated_response: str
    translated_response: str
    audio_base64: str
    error: Optional[str]
