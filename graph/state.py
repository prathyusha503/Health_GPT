from typing import TypedDict, Optional, Annotated
from langgraph.graph.message import add_messages


class MedicalState(TypedDict):
    # Supervisor conversation — add_messages reducer appends instead of overwriting
    messages: Annotated[list, add_messages]

    # Image and user inputs
    image_base64: str
    raw_prompt: str
    language: str

    # Agent outputs — written progressively as each agent completes
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
