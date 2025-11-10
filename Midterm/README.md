# 🧬 ONCODETECT AI — Molecular Subtyping of Small Cell Lung Cancer (SCLC)

## 📖 Abstract  
**Small Cell Lung Cancer (SCLC)** is an aggressive malignancy with limited targeted therapies. Recent research (Rudin *et al.*, *Nature Reviews Cancer*, 2019) identified **four molecular subtypes** — SCLC-A, SCLC-N, SCLC-Y, and SCLC-P — based on differential expression of transcriptional regulators **ASCL1**, **NEUROD1**, **YAP1**, and **POU2F3**.  
Our project, **ONCODETECT AI**, implements these findings using machine learning and large language models (LLMs) to classify SCLC subtypes based on user-provided biomarker and transcription factor values. The system integrates biological data understanding from the research paper with AI-driven classification and an interactive **Streamlit interface**.

---

## 🧪 Research Paper Overview  
**Paper Title:** *Molecular subtypes of small cell lung cancer: a synthesis of human and mouse model data*  
**Authors:** Charles M. Rudin *et al.*  
**Journal:** *Nature Reviews Cancer (2019)*  
**Key Insights:**  
- SCLC is subdivided into **four distinct subtypes**:
  - **SCLC-A (ASCL1-high)**
  - **SCLC-N (NEUROD1-high)**
  - **SCLC-Y (YAP1-high)**
  - **SCLC-P (POU2F3-high)**
- These subtypes differ in **gene expression**, **therapeutic targets**, and **tumor behavior**.  
- The study emphasizes understanding **molecular heterogeneity** for precision therapy in SCLC.  
- ONCODETECT AI reproduces these subtype concepts computationally to assist subtype prediction based on molecular signatures.

---

## 💡 Project Overview  
Our Jupyter Notebook and web application demonstrate how **biomarker-driven classification** can identify probable SCLC subtypes using AI.

### Key Features:
- Input transcription factors (ASCL1, NEUROD1, YAP1, POU2F3) and MYC family markers.  
- Predict the **most likely molecular subtype** of SCLC using LLM-based reasoning.  
- Visualize subtype predictions and confidence levels.  
- Simple, interactive **Streamlit UI** for real-time analysis.  

---

## 👩‍💻 Contributors  
| Name |
|------|
| **Pranali Chipkar** | 
| **Aditi Deshmukh** | 
| **Siddharth Pawar** | 

---

## ⚙️ Tech Stack  
- **Python 3.10+**  
- **Jupyter Notebook** – Data exploration and subtype classification logic  
- **Pandas / NumPy / Matplotlib** – Data manipulation and visualization  
- **Scikit-learn** – Model-based feature classification  
- **Streamlit** – Interactive frontend for molecular subtype prediction  
- **Large Language Models (LLM)** – Interpret relationships between transcription factors and SCLC subtype classification  

---

## 🧠 Concept Implementation
- Modeled expression of **ASCL1, NEUROD1, YAP1, POU2F3**, and **MYC-family** biomarkers.  
- Used **LLM-driven rule-based classification** for subtype identification (SCLC-A, N, Y, P).  
- Integrated **Streamlit-based visualization** to enter biomarker values and view predicted molecular subtype.  
- Linked AI interpretation with the **findings from Rudin et al. (2019)** for biologically explainable predictions.

---

## 📊 Output Examples  
- **Input:** Biomarker values and transcription factors  
- **Output:** Subtype classification (e.g., “Predicted Subtype: SCLC-N (NEUROD1-high)”)  
- **Additional insights:** Key gene influence, subtype probability, and brief biological interpretation.

---

## 📚 References  
[Molecular subtypes of small cell lung cancer: a synthesis of human and mouse model data](https://pmc.ncbi.nlm.nih.gov/articles/PMC6538259/)
Rudin CM, Poirier JT, Byers LA, *et al.* (2019). 



