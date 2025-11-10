"""
OncoDetect-AI: LangGraph Agentic Risk Score Agent
Fixed: Clean output and no duplicate executions
"""

import pandas as pd
import numpy as np
import joblib
from typing import TypedDict, Annotated, Sequence
from datetime import datetime
import operator
import os
import sys
from pathlib import Path

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Project imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.connections.snowflake_connector import SnowflakeConnector
from src.agents.llm import get_llm


# ============================================
# DEFINE AGENT STATE
# ============================================

class AgentState(TypedDict):
    """State that gets passed between nodes"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    patient_data: dict
    ml_prediction: dict
    similar_patients: str
    next_step: str


# ============================================
# TOOLS
# ============================================

def search_similar_patients(age: float, tmb: float, mutations: int) -> str:
    """Search Snowflake for similar patients"""
    try:
        sql_query = f"""
        SELECT 
            sample_id, age, tmb, mutation_count,
            target_survival_months, target_category
        FROM MART_RISK_FEATURES_CLINICAL
        WHERE age BETWEEN {age-10} AND {age+10}
          AND tmb BETWEEN {tmb-5} AND {tmb+5}
          AND mutation_count BETWEEN {mutations-100} AND {mutations+100}
        ORDER BY ABS(age - {age}) + ABS(tmb - {tmb})
        LIMIT 8
        """
        
        with SnowflakeConnector() as sf:
            df = sf.execute_query(sql_query)
        
        if len(df) == 0:
            return "No similar patients found."
        
        # Convert numpy types to Python types for clean display
        category_counts = df['TARGET_CATEGORY'].value_counts().to_dict()
        category_counts = {k: int(v) for k, v in category_counts.items()}
        
        summary = f"Found {len(df)} similar patients:\n"
        summary += f"Average survival: {df['TARGET_SURVIVAL_MONTHS'].mean():.1f} months\n"
        summary += f"Distribution: {category_counts}\n"
        
        return summary
        
    except Exception as e:
        return f"Error: {str(e)}"


def get_treatment_protocol(risk_level: str) -> str:
    """Get treatment protocols"""
    protocols = {
        "VERY_HIGH": "Aggressive platinum doublet + immunotherapy. Consider clinical trials. Bi-weekly monitoring.",
        "HIGH": "Aggressive platinum doublet + immunotherapy. Consider clinical trials. Bi-weekly monitoring.",
        "MODERATE": "Standard platinum + etoposide. Monthly monitoring.",
        "LOW": "Standard chemotherapy. Quarterly follow-up."
    }
    return protocols.get(risk_level, "Standard treatment")


# ============================================
# LOAD ML MODEL
# ============================================

class MLModelWrapper:
    """Wrapper for the ML model"""
    
    def __init__(self):
        model_path = _PROJECT_ROOT / 'models' / 'risk_score_agent_binary_fixed.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        print(f"Loading model from: {model_path}")
        self.model_data = joblib.load(model_path)
        self.ml_model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.imputer = self.model_data['imputer']
        self.selected_features = self.model_data['selected_features']
    
    def predict(self, patient_data: dict) -> dict:
        """Make ML prediction"""
        feature_values = []
        for feat in self.selected_features:
            value = patient_data.get(feat)
            if value is None:
                value = 0 if feat.startswith('is_') else np.nan
            feature_values.append(value)
        
        X = pd.DataFrame([feature_values], columns=self.selected_features)
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        prediction = self.ml_model.predict(X_scaled)[0]
        proba = self.ml_model.predict_proba(X_scaled)[0]
        
        risk_score = proba[1] * 100
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = 'VERY_HIGH'
        elif risk_score >= 50:
            risk_level = 'HIGH'
        elif risk_score >= 30:
            risk_level = 'MODERATE'
        else:
            risk_level = 'LOW'
        
        return {
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'confidence': float(max(proba))
        }


# Only create ml_model when needed, not at module import
ml_model = None

def get_ml_model():
    """Lazy load the ML model"""
    global ml_model
    if ml_model is None:
        ml_model = MLModelWrapper()
    return ml_model


# ============================================
# GRAPH NODES
# ============================================

def ml_prediction_node(state: AgentState) -> AgentState:
    """Node 1: Run ML prediction"""
    print("\n🔹 Node: ML Prediction")
    
    model = get_ml_model()
    ml_result = model.predict(state['patient_data'])
    state['ml_prediction'] = ml_result
    
    message = HumanMessage(content=f"ML model predicted risk score of {ml_result['risk_score']:.1f}/100 ({ml_result['risk_level']} RISK)")
    state['messages'].append(message)
    
    print(f"  ✓ Risk Score: {ml_result['risk_score']:.1f}/100")
    
    return state


def reasoning_node(state: AgentState) -> AgentState:
    """Node 2: LLM analyzes and calls tools"""
    print("\n🔹 Node: LLM Reasoning & Tool Usage")
    
    patient = state['patient_data']
    ml = state['ml_prediction']
    
    # Call tool: Search similar patients
    print("  🔧 Calling tool: search_similar_patients")
    similar_patients = search_similar_patients(
        age=patient['age'],
        tmb=patient['tmb'],
        mutations=patient['mutation_count']
    )
    state['similar_patients'] = similar_patients
    print(f"  ✓ Found similar patients")
    
    # Call tool: Get treatment protocol
    print("  🔧 Calling tool: get_treatment_protocol")
    treatment = get_treatment_protocol(ml['risk_level'])
    
    # LLM analyzes
    print("  🧠 LLM analyzing data...")
    
    llm = get_llm(model_name="llama-4-scout", temperature=0.3)
    
    analysis_prompt = f"""Analyze this SCLC patient's risk assessment.

PATIENT: Age {patient['age']}, {'Male' if patient['is_male']==1 else 'Female'}, {patient['mutation_count']} mutations, TMB {patient['tmb']}

ML PREDICTION: {ml['risk_score']:.1f}/100 ({ml['risk_level']} RISK)

SIMILAR PATIENTS DATA:
{similar_patients}

Explain in 3-4 sentences:
1. Why this patient has this risk level
2. What the similar patient data tells us
3. Key clinical implications
"""
    
    analysis_response = llm.invoke(analysis_prompt)
    analysis = analysis_response.content
    
    state['messages'].append(AIMessage(content=f"Analysis: {analysis}"))
    state['messages'].append(AIMessage(content=f"Treatment Protocol: {treatment}"))
    
    print("  ✓ LLM analysis complete")
    
    return state


def synthesis_node(state: AgentState) -> AgentState:
    """Node 3: Synthesize final report"""
    print("\n🔹 Node: Synthesizing Report")
    
    llm = get_llm(model_name="llama-4-scout", temperature=0.3)
    
    patient = state['patient_data']
    ml = state['ml_prediction']
    
    prompt = f"""Generate a comprehensive clinical report for this SCLC patient.

PATIENT PROFILE:
- Age: {patient['age']} years
- Sex: {'Male' if patient['is_male']==1 else 'Female'}
- Mutations: {patient['mutation_count']}
- TMB: {patient['tmb']}

ML PREDICTION:
- Risk Score: {ml['risk_score']:.1f}/100
- Risk Level: {ml['risk_level']} RISK
- Confidence: {ml['confidence']:.1%}

Previous analysis from messages:
{chr(10).join([str(m.content) for m in state['messages'][-3:]])}

Generate a structured clinical report with:
1. RISK ASSESSMENT SUMMARY
2. KEY RISK FACTORS
3. EVIDENCE FROM SIMILAR PATIENTS (if available)
4. TREATMENT RECOMMENDATIONS
5. PROGNOSIS

Be concise, clinical, and actionable.
"""
    
    final_report_response = llm.invoke(prompt)
    final_report = final_report_response.content
    
    state['messages'].append(AIMessage(content=final_report))
    state['next_step'] = 'END'
    
    print("  ✓ Final report generated")
    
    return state


# ============================================
# BUILD GRAPH
# ============================================

def create_agent_graph():
    """Create the LangGraph workflow"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("ml_prediction", ml_prediction_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("synthesis", synthesis_node)
    
    # Define edges (linear flow, no loops!)
    workflow.set_entry_point("ml_prediction")
    workflow.add_edge("ml_prediction", "reasoning")
    workflow.add_edge("reasoning", "synthesis")
    workflow.add_edge("synthesis", END)
    
    return workflow.compile()


# ============================================
# MAIN
# ============================================

def main():
    """Run the LangGraph agentic system"""
    
    print("="*70)
    print("🤖 OncoDetect-AI: LangGraph Agentic Risk Score Agent")
    print("="*70)
    print("\nBuilding agent graph...")
    
    app = create_agent_graph()
    print("✓ Agent graph built with 3 nodes")
    
    print("\n" + "="*70)
    print("TEST: HIGH RISK PATIENT")
    print("="*70)
    
    patient_data = {
        'age': 78,
        'is_male': 1,
        'is_former_smoker': 0,
        'mutation_count': 580,
        'tmb': 21.5,
        'is_tmb_high': 1,
        'is_tmb_intermediate': 0,
        'is_tmb_low': 0,
        'smoker_x_tmb': 0.0
    }
    
    initial_state = {
        'messages': [],
        'patient_data': patient_data,
        'ml_prediction': {},
        'similar_patients': '',
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
    print("✅ LangGraph Workflow Complete!")
    print("="*70)
    print(f"\nWorkflow executed {len(final_state['messages'])} messages")
    print(f"Final ML Prediction: {final_state['ml_prediction']['risk_score']:.1f}/100")


if __name__ == "__main__":
    main()