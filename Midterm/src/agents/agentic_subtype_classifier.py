"""
OncoDetect-AI: Agentic SCLC Subtype Classifier
Investigative biomarker analysis with LLM reasoning
Based on Rudin et al. 2019 (PMC6538259)
"""

import pandas as pd
import numpy as np
import joblib
from typing import TypedDict, Annotated, Sequence
from datetime import datetime
import operator
import sys
from pathlib import Path

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Project imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.agents.llm import get_llm

# Try to import Snowflake connector (optional)
try:
    from src.connections.snowflake_connector import SnowflakeConnector
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False
    print("⚠️ Snowflake connector not available - similar profile search will be skipped")


# ============================================
# AGENT STATE
# ============================================

class SubtypeAgentState(TypedDict):
    """State for subtype classification agent"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    biomarker_data: dict
    ml_prediction: dict
    biomarker_analysis: dict
    hypothesis_testing: dict
    final_diagnosis: str
    next_step: str


# ============================================
# TOOLS - Biomarker Analysis
# ============================================

def search_similar_biomarker_profiles(ascl1: float, neurod1: float, pou2f3: float, yap1: float, predicted_subtype: str) -> str:
    """Search Snowflake for samples with similar biomarker patterns"""
    
    if not SNOWFLAKE_AVAILABLE:
        return "Snowflake database search not available (connector not imported)."
    
    try:
        sql_query = f"""
        SELECT 
            sample_id, sclc_subtype,
            ascl1_expression, neurod1_expression, pou2f3_expression, yap1_expression,
            SQRT(
                POWER(ascl1_expression - {ascl1}, 2) +
                POWER(neurod1_expression - {neurod1}, 2) +
                POWER(pou2f3_expression - {pou2f3}, 2) +
                POWER(yap1_expression - {yap1}, 2)
            ) AS distance
        FROM MART_RISK_FEATURES_GENOMIC
        WHERE sclc_subtype IS NOT NULL
          AND sclc_subtype != 'MIXED'
        ORDER BY distance
        LIMIT 5
        """
        
        with SnowflakeConnector() as sf:
            df = sf.execute_query(sql_query)
        
        if len(df) == 0:
            return "No similar biomarker profiles found in database."
        
        # Count subtypes in similar samples
        subtype_counts = df['SCLC_SUBTYPE'].value_counts().to_dict()
        subtype_counts = {k: int(v) for k, v in subtype_counts.items()}
        
        # Check agreement with predicted subtype
        matching = subtype_counts.get(predicted_subtype, 0)
        total = len(df)
        
        summary = f"Found {total} samples with similar biomarker profiles:\n"
        summary += f"Subtype distribution: {subtype_counts}\n"
        summary += f"Agreement: {matching}/{total} samples match predicted {predicted_subtype}\n"
        
        # Show closest match details
        closest = df.iloc[0]
        summary += f"\nClosest match: {closest['SAMPLE_ID']} ({closest['SCLC_SUBTYPE']})\n"
        summary += f"  ASCL1: {closest['ASCL1_EXPRESSION']:.2f}, NEUROD1: {closest['NEUROD1_EXPRESSION']:.2f}\n"
        summary += f"  POU2F3: {closest['POU2F3_EXPRESSION']:.2f}, YAP1: {closest['YAP1_EXPRESSION']:.2f}\n"
        
        return summary
        
    except Exception as e:
        return f"Error searching biomarker database: {str(e)}"


def analyze_transcription_factors(ascl1: float, neurod1: float, pou2f3: float, yap1: float) -> dict:
    """Analyze transcription factor dominance (Key finding from Rudin et al. 2019)"""
    
    # Find dominant TF
    tf_values = {
        'ASCL1': ascl1,
        'NEUROD1': neurod1,
        'POU2F3': pou2f3,
        'YAP1': yap1
    }
    
    dominant_tf = max(tf_values, key=tf_values.get)
    dominant_value = tf_values[dominant_tf]
    
    # Calculate ratios
    other_tfs = {k: v for k, v in tf_values.items() if k != dominant_tf}
    avg_others = np.mean(list(other_tfs.values()))
    
    dominance_ratio = dominant_value / avg_others if avg_others > 0 else dominant_value
    
    # Determine NE vs Non-NE
    ne_markers = ascl1 + neurod1
    non_ne_markers = pou2f3 + yap1
    
    classification = "Neuroendocrine" if ne_markers > non_ne_markers else "Non-Neuroendocrine"
    
    return {
        'dominant_tf': dominant_tf,
        'dominant_value': float(dominant_value),
        'dominance_ratio': float(dominance_ratio),
        'classification': classification,
        'ascl1': float(ascl1),
        'neurod1': float(neurod1),
        'pou2f3': float(pou2f3),
        'yap1': float(yap1),
        'ne_score': float(ne_markers),
        'non_ne_score': float(non_ne_markers)
    }


def calculate_subtype_compatibility(biomarkers: dict) -> dict:
    """Calculate how well biomarkers match each subtype (Based on Rudin paper patterns)"""
    
    ascl1 = biomarkers['ascl1']
    neurod1 = biomarkers['neurod1']
    pou2f3 = biomarkers['pou2f3']
    yap1 = biomarkers['yap1']
    
    # Subtype compatibility scoring (from Rudin et al. patterns)
    compatibility = {}
    
    # SCLC-A: ASCL1-dominant, neuroendocrine
    sclc_a_score = 0
    if ascl1 > 10:
        sclc_a_score += 40
    if ascl1 > neurod1 and ascl1 > pou2f3 and ascl1 > yap1:
        sclc_a_score += 30
    if neurod1 < 8:
        sclc_a_score += 15
    if pou2f3 < 8 and yap1 < 8:
        sclc_a_score += 15
    compatibility['SCLC-A'] = min(sclc_a_score, 100)
    
    # SCLC-N: NEUROD1-dominant, alternative NE
    sclc_n_score = 0
    if neurod1 > 10:
        sclc_n_score += 40
    if neurod1 > ascl1 and neurod1 > pou2f3 and neurod1 > yap1:
        sclc_n_score += 30
    if ascl1 < 8:
        sclc_n_score += 15
    if pou2f3 < 8 and yap1 < 8:
        sclc_n_score += 15
    compatibility['SCLC-N'] = min(sclc_n_score, 100)
    
    # SCLC-P: POU2F3-dominant, tuft-like
    sclc_p_score = 0
    if pou2f3 > 10:
        sclc_p_score += 40
    if pou2f3 > ascl1 and pou2f3 > neurod1 and pou2f3 > yap1:
        sclc_p_score += 30
    if ascl1 < 8 and neurod1 < 8:
        sclc_p_score += 15
    if yap1 < 8:
        sclc_p_score += 15
    compatibility['SCLC-P'] = min(sclc_p_score, 100)
    
    # SCLC-Y: YAP1-dominant, non-NE, inflammatory
    sclc_y_score = 0
    if yap1 > 10:
        sclc_y_score += 40
    if yap1 > ascl1 and yap1 > neurod1 and yap1 > pou2f3:
        sclc_y_score += 30
    if ascl1 < 8 and neurod1 < 8:
        sclc_y_score += 15
    if pou2f3 < 8:
        sclc_y_score += 15
    compatibility['SCLC-Y'] = min(sclc_y_score, 100)
    
    return compatibility


def get_therapeutic_targets_by_subtype(subtype: str) -> str:
    """Get therapeutic targets based on Rudin et al. 2019 findings"""
    
    targets = {
        "SCLC-A": """DLL3-targeted therapies (Rovalpituzumab tesirine)
• BCL-2 inhibitors (Venetoclax) - Neuroendocrine dependency
• Aurora kinase inhibitors (Alisertib)
• Standard platinum-based chemotherapy (highly sensitive)
• PARP inhibitors (synthetic lethality with RB1 loss)""",
        
        "SCLC-N": """Aurora kinase inhibitors (Alisertib) - NEUROD1 pathway
• Platinum-based chemotherapy (chemosensitive)
• PARP inhibitors (DNA repair pathway)
• Investigational: NEUROD1-targeted approaches
• Consider immunotherapy combinations""",
        
        "SCLC-P": """PARP inhibitors (Olaparib) - Preferred for POU2F3 subtype
• Immunotherapy (PD-1/PD-L1 inhibitors) - Inflammatory features
• Experimental: Tuft cell-specific targeting
• Platinum chemotherapy (may be less effective)
• Clinical trials for novel agents""",
        
        "SCLC-Y": """Immunotherapy (Nivolumab, Pembrolizumab) - Inflammatory subtype
• YAP1 pathway inhibitors (Verteporfin) - Under investigation
• FAK inhibitors (Defactinib)
• SRC family kinase inhibitors
• Targeted therapy over chemotherapy (chemoresistant)"""
    }
    
    return targets.get(subtype, "Standard platinum-based chemotherapy")


# ============================================
# LOAD ML MODEL
# ============================================

class SubtypeMLWrapper:
    """Wrapper for subtype ML model"""
    
    def __init__(self):
        model_path = _PROJECT_ROOT / 'models' / 'subtype_classifier.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        print(f"Loading subtype model from: {model_path}")
        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.imputer = self.model_data['imputer']
        self.feature_names = self.model_data['feature_names']
        self.classes = self.model_data['classes']
    
    def predict(self, biomarker_data: dict) -> dict:
        """Make ML prediction with all 14 features"""
        
        # Map UI field names to model feature names
        feature_mapping = {
            'ascl1': 'ascl1_expression',
            'neurod1': 'neurod1_expression',
            'pou2f3': 'pou2f3_expression',
            'yap1': 'yap1_expression',
            'tp53': 'tp53_expression',
            'rb1': 'rb1_expression',
            'myc': 'myc_expression',
            'mycl': 'mycl_expression',
            'mycn': 'mycn_expression',
            'dll3': 'dll3_expression',
            'bcl2': 'bcl2_expression',
            'notch1': 'notch1_expression',
            'myc_family_score': 'myc_family_score',
            'tp53_rb1_dual_loss': 'tp53_rb1_dual_loss',
            'mean_expression': 'mean_expression',
            'stddev_expression': 'stddev_expression',
            'genes_expressed': 'genes_expressed'
        }
        
        # Extract features in model's expected order
        feature_values = []
        for feat in self.feature_names:
            # Find matching UI key
            ui_key = None
            for ui_field, model_field in feature_mapping.items():
                if model_field.lower() == feat.lower():
                    ui_key = ui_field
                    break
            
            if ui_key and ui_key in biomarker_data:
                value = biomarker_data[ui_key]
            else:
                value = np.nan
            
            feature_values.append(value)
        
        X = pd.DataFrame([feature_values], columns=self.feature_names)
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        prediction = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0]
        
        return {
            'predicted_subtype': str(prediction),
            'confidence': float(max(proba)),
            'probabilities': {str(cls): float(prob) for cls, prob in zip(self.classes, proba)}
        }


subtype_ml_model = None

def get_subtype_model():
    """Lazy load subtype model"""
    global subtype_ml_model
    if subtype_ml_model is None:
        subtype_ml_model = SubtypeMLWrapper()
    return subtype_ml_model


# ============================================
# LANGGRAPH NODES
# ============================================

def ml_subtype_prediction_node(state: SubtypeAgentState) -> SubtypeAgentState:
    """Node 1: ML Baseline Prediction"""
    print("\n🔹 Node 1: ML Subtype Prediction")
    
    model = get_subtype_model()
    ml_result = model.predict(state['biomarker_data'])
    state['ml_prediction'] = ml_result
    
    msg = f"ML predicted subtype: {ml_result['predicted_subtype']} (Confidence: {ml_result['confidence']:.1%})"
    state['messages'].append(HumanMessage(content=msg))
    
    print(f"  ✓ Predicted: {ml_result['predicted_subtype']} ({ml_result['confidence']:.1%})")
    
    return state


def biomarker_investigation_node(state: SubtypeAgentState) -> SubtypeAgentState:
    """Node 2: Investigate biomarker patterns"""
    print("\n🔹 Node 2: Biomarker Investigation")
    
    biomarkers = state['biomarker_data']
    ml_pred = state['ml_prediction']
    
    # Tool 1: Analyze transcription factors
    print("  🔬 Analyzing transcription factors...")
    tf_analysis = analyze_transcription_factors(
        biomarkers['ascl1'],
        biomarkers['neurod1'],
        biomarkers['pou2f3'],
        biomarkers['yap1']
    )
    state['biomarker_analysis'] = tf_analysis
    
    # Tool 2: Search for similar biomarker profiles in Snowflake
    print("  🔍 Searching Snowflake for similar biomarker profiles...")
    similar_profiles = search_similar_biomarker_profiles(
        biomarkers['ascl1'],
        biomarkers['neurod1'],
        biomarkers['pou2f3'],
        biomarkers['yap1'],
        ml_pred['predicted_subtype']
    )
    state['biomarker_analysis']['similar_profiles'] = similar_profiles
    print("  ✓ Found similar samples")
    
    # LLM investigates the biomarker pattern
    print("  🧠 LLM investigating biomarker pattern...")
    
    llm = get_llm(model_name="llama-4-scout", temperature=0.3)
    
    investigation_prompt = f"""You are a molecular pathologist analyzing SCLC biomarkers.

BIOMARKER PROFILE:
- ASCL1: {biomarkers['ascl1']:.2f}
- NEUROD1: {biomarkers['neurod1']:.2f}
- POU2F3: {biomarkers['pou2f3']:.2f}
- YAP1: {biomarkers['yap1']:.2f}
- TP53: {biomarkers['tp53']:.2f}
- RB1: {biomarkers['rb1']:.2f}
- MYC: {biomarkers['myc']:.2f}

ANALYSIS RESULTS:
- Dominant transcription factor: {tf_analysis['dominant_tf']} ({tf_analysis['dominant_value']:.2f})
- Dominance ratio: {tf_analysis['dominance_ratio']:.2f}x over others
- Classification: {tf_analysis['classification']}

SIMILAR SAMPLES IN DATABASE:
{similar_profiles}

Investigate this biomarker pattern in 4-5 sentences:
1. Which transcription factor dominates and what does this mean?
2. Are these neuroendocrine (NE) or non-NE markers?
3. What do the similar samples tell us about this classification?
4. What do the tumor suppressors (TP53, RB1) and MYC tell us?

Be specific and reference both the biomarker values and database evidence."""
    
    response = llm.invoke(investigation_prompt)
    investigation = response.content
    
    state['messages'].append(AIMessage(content=f"Biomarker Investigation:\n{investigation}"))
    
    print("  ✓ Investigation complete")
    
    return state


def hypothesis_testing_node(state: SubtypeAgentState) -> SubtypeAgentState:
    """Node 3: Test each subtype hypothesis"""
    print("\n🔹 Node 3: Hypothesis Testing")
    
    biomarkers = state['biomarker_data']
    ml_pred = state['ml_prediction']
    
    # Calculate compatibility scores
    print("  🧪 Testing subtype hypotheses...")
    compatibility = calculate_subtype_compatibility(biomarkers)
    state['hypothesis_testing'] = compatibility
    
    # LLM evaluates each hypothesis
    print("  🧠 LLM evaluating hypotheses...")
    
    llm = get_llm(model_name="llama-4-scout", temperature=0.3)
    
    hypothesis_prompt = f"""Evaluate why this patient matches or doesn't match each SCLC subtype.

BIOMARKERS:
ASCL1: {biomarkers['ascl1']:.2f}, NEUROD1: {biomarkers['neurod1']:.2f}, 
POU2F3: {biomarkers['pou2f3']:.2f}, YAP1: {biomarkers['yap1']:.2f}

ML PREDICTION: {ml_pred['predicted_subtype']} ({ml_pred['confidence']:.1%} confidence)

COMPATIBILITY SCORES:
{chr(10).join([f'  - {k}: {v}% match' for k, v in compatibility.items()])}

For each subtype, explain in 2-3 sentences:

**SCLC-A (ASCL1-high):** Expected if ASCL1 > 10 and dominates
Evidence: [Your analysis]

**SCLC-N (NEUROD1-high):** Expected if NEUROD1 > 10 and dominates  
Evidence: [Your analysis]

**SCLC-P (POU2F3-high):** Expected if POU2F3 > 10 and dominates
Evidence: [Your analysis]

**SCLC-Y (YAP1-high):** Expected if YAP1 > 10 and dominates
Evidence: [Your analysis]

**CONCLUSION:** Which subtype best fits and why?"""
    
    response = llm.invoke(hypothesis_prompt)
    hypothesis_analysis = response.content
    
    state['messages'].append(AIMessage(content=f"Differential Diagnosis:\n{hypothesis_analysis}"))
    
    print("  ✓ Hypothesis testing complete")
    
    return state


def clinical_synthesis_node(state: SubtypeAgentState) -> SubtypeAgentState:
    """Node 4: Generate comprehensive clinical report"""
    print("\n🔹 Node 4: Clinical Synthesis")
    
    biomarkers = state['biomarker_data']
    ml_pred = state['ml_prediction']
    tf_analysis = state['biomarker_analysis']
    compatibility = state['hypothesis_testing']
    
    # Get therapeutic targets
    therapeutic_targets = get_therapeutic_targets_by_subtype(ml_pred['predicted_subtype'])
    
    print("  🧠 LLM generating clinical report...")
    
    llm = get_llm(model_name="llama-4-scout", temperature=0.2)
    
    synthesis_prompt = f"""Generate a comprehensive SCLC subtype classification report.

PATIENT BIOMARKER PROFILE:
- ASCL1: {biomarkers['ascl1']:.2f}
- NEUROD1: {biomarkers['neurod1']:.2f}
- POU2F3: {biomarkers['pou2f3']:.2f}
- YAP1: {biomarkers['yap1']:.2f}
- TP53: {biomarkers['tp53']:.2f}
- RB1: {biomarkers['rb1']:.2f}
- MYC: {biomarkers['myc']:.2f}

ML PREDICTION:
Subtype: {ml_pred['predicted_subtype']}
Confidence: {ml_pred['confidence']:.1%}

BIOMARKER ANALYSIS:
Dominant TF: {tf_analysis['dominant_tf']} ({tf_analysis['dominant_value']:.2f})
Classification: {tf_analysis['classification']}
Dominance Ratio: {tf_analysis['dominance_ratio']:.2f}x

COMPATIBILITY SCORES:
{chr(10).join([f'{k}: {v}%' for k, v in compatibility.items()])}

THERAPEUTIC TARGETS:
{therapeutic_targets}

Generate a structured clinical report with these sections:

**1. MOLECULAR SUBTYPE DIAGNOSIS**
[Subtype, confidence, reference to Rudin et al. 2019]

**2. BIOMARKER EVIDENCE**
[Why these biomarkers indicate this subtype]
[Explain dominant transcription factor]
[Reference specific patterns from Rudin paper]

**3. WHY NOT OTHER SUBTYPES**
[Brief explanation eliminating other 3 subtypes]

**4. THERAPEUTIC IMPLICATIONS**
[Targeted therapies for this subtype]
[Expected treatment response]
[Clinical trial opportunities]

**5. BIOLOGICAL CHARACTERISTICS**
[NE vs Non-NE features]
[Tumor suppressor status]
[Proliferation markers]

**6. CLINICAL RECOMMENDATIONS**
[Immediate next steps]
[Monitoring strategy]

Be concise, cite Rudin et al. 2019 where relevant, use clinical language."""
    
    response = llm.invoke(synthesis_prompt)
    clinical_report = response.content
    
    state['messages'].append(AIMessage(content=clinical_report))
    state['final_diagnosis'] = ml_pred['predicted_subtype']
    state['next_step'] = 'END'
    
    print("  ✓ Clinical report generated")
    
    return state


# ============================================
# BUILD GRAPH
# ============================================

def create_subtype_agent_graph():
    """Create the subtype classification agent graph"""
    
    workflow = StateGraph(SubtypeAgentState)
    
    # Add nodes
    workflow.add_node("ml_prediction", ml_subtype_prediction_node)
    workflow.add_node("investigation", biomarker_investigation_node)
    workflow.add_node("hypothesis_testing", hypothesis_testing_node)
    workflow.add_node("clinical_synthesis", clinical_synthesis_node)
    
    # Define flow
    workflow.set_entry_point("ml_prediction")
    workflow.add_edge("ml_prediction", "investigation")
    workflow.add_edge("investigation", "hypothesis_testing")
    workflow.add_edge("hypothesis_testing", "clinical_synthesis")
    workflow.add_edge("clinical_synthesis", END)
    
    return workflow.compile()


# ============================================
# MAIN
# ============================================

def main():
    """Test the agentic subtype classifier"""
    
    print("="*70)
    print("🧬 OncoDetect-AI: Agentic SCLC Subtype Classifier")
    print("   Based on Rudin et al. 2019 (PMC6538259)")
    print("="*70)
    
    # Create graph
    print("\nBuilding agent graph...")
    app = create_subtype_agent_graph()
    print("✓ Agent graph built with 4 nodes")
    
    # Test with SCLC-A example
    print("\n" + "="*70)
    print("TEST: SCLC-A (Classic Neuroendocrine)")
    print("="*70)
    
    biomarker_data = {
        'ascl1': 12.5,
        'neurod1': 5.0,
        'pou2f3': 4.0,
        'yap1': 5.5,
        'tp53': 11.0,
        'rb1': 4.5,
        'myc': 8.0,
        'mycl': 9.5,
        'mycn': 6.0
    }
    
    initial_state = {
        'messages': [],
        'biomarker_data': biomarker_data,
        'ml_prediction': {},
        'biomarker_analysis': {},
        'hypothesis_testing': {},
        'final_diagnosis': '',
        'next_step': 'start'
    }
    
    print("\n🔄 Executing agent workflow...\n")
    
    final_state = app.invoke(initial_state)
    
    print("\n" + "="*70)
    print("📋 FINAL CLINICAL REPORT")
    print("="*70)
    
    if final_state['messages']:
        final_report = final_state['messages'][-1].content
        print(final_report)
    
    print("\n" + "="*70)
    print("✅ Agentic Subtype Analysis Complete!")
    print("="*70)
    print(f"Final Diagnosis: {final_state['final_diagnosis']}")
    print(f"Workflow executed {len(final_state['messages'])} analysis steps")


if __name__ == "__main__":
    main()