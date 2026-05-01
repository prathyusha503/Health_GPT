import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# Built-in medical knowledge base covering 8 diseases
MEDICAL_DOCUMENTS = [
    {
        "id": "pneumonia_doc",
        "text": (
            "Pneumonia: A serious lung infection inflaming the alveoli (air sacs), filling them with fluid or pus. "
            "Imaging: Chest X-ray shows consolidation (white/opaque areas), air bronchograms, lobar or patchy "
            "infiltrates. CT reveals ground-glass opacities and consolidation. "
            "Types: Bacterial (Streptococcus pneumoniae most common), viral (influenza, RSV), fungal, aspiration. "
            "Treatment: Bacterial — antibiotics (amoxicillin, azithromycin, levofloxacin). Viral — antivirals and "
            "supportive care. Severe cases require hospitalisation with IV antibiotics and oxygen therapy. "
            "Specialist: Pulmonologist for complex cases; Primary Care physician for mild community-acquired cases. "
            "Recovery: 1–3 weeks with appropriate treatment. Elderly and immunocompromised patients take longer."
        ),
        "metadata": {"disease": "pneumonia"},
    },
    {
        "id": "tuberculosis_doc",
        "text": (
            "Tuberculosis (TB): Infectious disease caused by Mycobacterium tuberculosis, primarily affecting the lungs. "
            "Imaging: Chest X-ray shows upper lobe infiltrates or consolidation, cavitation (air-filled holes), "
            "miliary pattern (millet-seed nodules), Ghon complex in primary infection, pleural effusion, and hilar "
            "lymphadenopathy. CT confirms cavitary lesions and tree-in-bud pattern. "
            "Treatment: DOTS therapy — Initial phase (2 months): Isoniazid, Rifampin, Pyrazinamide, Ethambutol. "
            "Continuation phase (4–7 months): Isoniazid and Rifampin. MDR-TB requires second-line drugs. "
            "Specialist: Pulmonologist and Infectious Disease specialist. Mandatory public health notification. "
            "Prevention: BCG vaccination, infection control with respiratory precautions."
        ),
        "metadata": {"disease": "tuberculosis"},
    },
    {
        "id": "covid19_doc",
        "text": (
            "COVID-19 Pneumonia: Lung infection caused by SARS-CoV-2 coronavirus. "
            "Imaging: CT scan shows bilateral peripheral ground-glass opacities (GGO), crazy-paving pattern, "
            "consolidation predominantly in lower lobes, and vascular thickening. Chest X-ray shows bilateral "
            "infiltrates. Typical progression: peripheral GGO → consolidation → potential fibrosis in severe cases. "
            "Treatment: Mild — rest, hydration, antipyretics. Moderate — hospitalisation, supplemental oxygen, "
            "remdesivir. Severe — dexamethasone, anticoagulation, prone positioning. Critical — ICU, mechanical "
            "ventilation. "
            "Specialist: Pulmonologist, Infectious Disease, Critical Care for severe cases. "
            "Complications: ARDS, cytokine storm, pulmonary embolism, long COVID syndrome."
        ),
        "metadata": {"disease": "covid-19"},
    },
    {
        "id": "brain_tumor_doc",
        "text": (
            "Brain Tumor: Abnormal intracranial cell growth, either primary (arising in the brain) or metastatic. "
            "Imaging: MRI with gadolinium contrast — ring-enhancing lesion in glioblastoma, surrounding vasogenic "
            "oedema (bright on T2/FLAIR), mass effect, midline shift. Meningioma shows intense uniform enhancement. "
            "CT shows hyperdense mass, calcification, or haemorrhage. "
            "Types: Glioblastoma (GBM), astrocytoma, meningioma, pituitary adenoma, metastases (lung, breast, "
            "melanoma primaries). "
            "Treatment: Surgery (craniotomy/resection), radiotherapy, chemotherapy (temozolomide for GBM), targeted "
            "therapy. Steroids (dexamethasone) to reduce oedema. "
            "Specialist: Neurosurgeon, Neuro-oncologist, Radiation Oncologist. "
            "Prognosis: Meningioma often benign with good prognosis; GBM median survival 14–16 months."
        ),
        "metadata": {"disease": "brain tumor"},
    },
    {
        "id": "bone_fracture_doc",
        "text": (
            "Bone Fracture: A break or crack in the continuity of bone, resulting from trauma, stress, or pathology. "
            "Imaging: X-ray reveals cortical break, lucent fracture line, displacement, angulation, and soft tissue "
            "swelling. CT for complex or occult fractures. MRI for bone marrow oedema, stress fractures, and "
            "ligament injury. "
            "Types: Simple (closed), compound (open/skin breach), comminuted (multiple fragments), stress, "
            "pathological (through diseased bone), greenstick (incomplete, in children), compression (vertebral). "
            "Treatment: Immobilisation (cast/splint/brace), closed or open reduction, surgical fixation (ORIF — "
            "plates, screws, intramedullary nails), external fixation, traction. "
            "Recovery: 6–8 weeks for simple fractures; 3–6 months for complex fractures. Physiotherapy for rehab. "
            "Specialist: Orthopaedic Surgeon. "
            "Complications: Non-union, malunion, osteomyelitis, compartment syndrome, avascular necrosis."
        ),
        "metadata": {"disease": "bone fracture"},
    },
    {
        "id": "diabetic_retinopathy_doc",
        "text": (
            "Diabetic Retinopathy: Progressive damage to retinal blood vessels caused by chronic diabetes mellitus; "
            "a leading cause of preventable blindness in working-age adults. "
            "Imaging: Fundus photography shows microaneurysms (tiny red dots), dot/blot/flame haemorrhages, hard "
            "exudates (bright yellow lipid deposits), cotton-wool spots (nerve fibre infarcts), neovascularisation "
            "(new fragile vessel growth). OCT reveals macular oedema and retinal thickening. "
            "Stages: Non-proliferative DR (NPDR) — mild, moderate, severe. Proliferative DR (PDR) — new vessel "
            "formation, highest risk of vision loss. "
            "Treatment: Glycaemic and blood pressure control (primary prevention). Anti-VEGF intravitreal "
            "injections (ranibizumab, bevacizumab, aflibercept). Laser photocoagulation. Vitrectomy for advanced PDR. "
            "Specialist: Ophthalmologist/Retina Specialist, Endocrinologist for diabetes management. "
            "Prevention: HbA1c < 7%, annual dilated eye exams, blood pressure and lipid control."
        ),
        "metadata": {"disease": "diabetic retinopathy"},
    },
    {
        "id": "pleural_effusion_doc",
        "text": (
            "Pleural Effusion: Abnormal accumulation of fluid in the pleural space (between lung and chest wall). "
            "Imaging: Chest X-ray — blunting of costophrenic angles (≥200 mL), meniscus sign, opacification of "
            "hemithorax in large effusions, mediastinal shift away from effusion. Lateral decubitus X-ray shows "
            "layering of free fluid. Ultrasound is most sensitive for detection and aspiration guidance. CT "
            "differentiates exudate from transudate and identifies underlying cause. "
            "Causes: Transudate — heart failure, hepatic cirrhosis, nephrotic syndrome, hypoalbuminaemia. Exudate "
            "(Light's criteria) — parapneumonic effusion, empyema, malignancy, pulmonary embolism, TB. "
            "Treatment: Address the underlying cause. Therapeutic thoracentesis for symptom relief. Chest tube "
            "drainage for large or complicated parapneumonic effusions. Pleurodesis for recurrent malignant effusions. "
            "Specialist: Pulmonologist or Interventional Radiologist. Thoracic Surgeon for surgical drainage."
        ),
        "metadata": {"disease": "pleural effusion"},
    },
    {
        "id": "cardiomegaly_doc",
        "text": (
            "Cardiomegaly: Abnormal enlargement of the heart, a radiographic sign indicating underlying cardiac disease. "
            "Imaging: Chest X-ray (PA view) — cardiothoracic ratio > 0.5 is diagnostic. Globular cardiac silhouette "
            "suggests pericardial effusion. CT and MRI delineate specific chamber enlargement, wall thickness, and "
            "ejection fraction. Echocardiogram is the primary functional assessment. "
            "Causes: Dilated cardiomyopathy, hypertensive heart disease, valvular disease (aortic/mitral "
            "regurgitation), coronary artery disease with ischaemic cardiomyopathy, congenital heart defects, "
            "myocarditis, pericardial effusion, high-output states (anaemia, thyrotoxicosis). "
            "Treatment: ACE inhibitors/ARBs, beta-blockers, diuretics (furosemide), aldosterone antagonists (spironolactone). "
            "ICD for arrhythmia prevention. Cardiac resynchronisation therapy (CRT). Heart transplant for end-stage. "
            "Specialist: Cardiologist; Cardiac Surgeon for structural or surgical intervention. "
            "Monitoring: Serial echocardiograms, BNP/NT-proBNP levels, cardiac MRI, exercise stress testing."
        ),
        "metadata": {"disease": "cardiomegaly"},
    },
]

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.EphemeralClient()
        ef = DefaultEmbeddingFunction()
        collection = client.create_collection(
            name="medical_knowledge",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        collection.add(
            ids=[doc["id"] for doc in MEDICAL_DOCUMENTS],
            documents=[doc["text"] for doc in MEDICAL_DOCUMENTS],
            metadatas=[doc["metadata"] for doc in MEDICAL_DOCUMENTS],
        )
        _collection = collection
    return _collection


def rag_agent(state: dict) -> dict:
    try:
        disease_label = state.get("disease_label", "")
        image_analysis = state.get("image_analysis", "")

        query = f"{disease_label} {image_analysis[:200]}".strip()
        if not query:
            query = "medical imaging disease analysis treatment specialist"

        collection = _get_collection()
        results = collection.query(query_texts=[query], n_results=3)

        docs = results.get("documents", [[]])[0]
        if docs:
            rag_context = "\n\n" + ("─" * 60) + "\n\n".join(docs)
        else:
            rag_context = "No specific medical knowledge found for this condition."

        return {"rag_context": rag_context}

    except Exception as e:
        fallback = (
            "Medical knowledge retrieval encountered an issue. "
            "Please consult a qualified medical professional for accurate, "
            f"condition-specific information. (Error: {str(e)})"
        )
        return {"rag_context": fallback, "error": f"RAG agent error: {str(e)}"}
