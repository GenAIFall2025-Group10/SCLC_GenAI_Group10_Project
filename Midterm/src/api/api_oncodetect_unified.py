"""
OncoDetect-AI: Unified API with Integrated Agentic Analysis
Single API for both Risk Score (ML + Agentic) and Subtype Classification
Port: 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime
import uuid
import asyncio
from contextlib import asynccontextmanager
import logging
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
prediction_results = {}
risk_agent_graph = None
subtype_agent_graph = None
risk_ml_model = None
subtype_model_manager = None

# ============================================
# IMPORT COMPONENTS
# ============================================

try:
    # Import risk agentic components
    from src.agents.agentic_risk_score_agent import (
        create_agent_graph as create_risk_graph,
        AgentState as RiskAgentState
    )
    logger.info("✓ Risk agentic components imported")
    RISK_AGENTIC_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠ Risk agentic components not available: {e}")
    RISK_AGENTIC_AVAILABLE = False

try:
    # Import subtype agentic components
    from src.agents.agentic_subtype_classifier import (
        create_subtype_agent_graph,
        SubtypeAgentState
    )
    logger.info("✓ Subtype agentic components imported")
    SUBTYPE_AGENTIC_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠ Subtype agentic components not available: {e}")
    SUBTYPE_AGENTIC_AVAILABLE = False

try:
    # Import Snowflake connector
    from src.connections.snowflake_connector import SnowflakeConnector
    logger.info("✓ Snowflake connector imported")
    SNOWFLAKE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠ Snowflake not available: {e}")
    SNOWFLAKE_AVAILABLE = False


# ============================================
# MODEL MANAGERS
# ============================================

class RiskScoreModelManager:
    """Manages Risk Score ML Model"""
    
    def __init__(self):
        self.model_path = _PROJECT_ROOT / 'models' / 'risk_score_agent_binary_fixed.pkl'
        self.model = None
        self.scaler = None
        self.imputer = None
        self.selected_features = None
        self.model_data = None
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            logger.info(f"Loading Risk Score model from: {self.model_path}")
            self.model_data = joblib.load(self.model_path)
            
            self.model = self.model_data['model']
            self.scaler = self.model_data['scaler']
            self.imputer = self.model_data['imputer']
            self.selected_features = self.model_data['selected_features']
            
            logger.info(f"✓ Risk Score model loaded: {self.model_data.get('version', 'unknown')}")
            logger.info(f"✓ Features: {self.selected_features}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Risk Score model: {e}")
            return False
    
    def predict(self, patient_data: dict) -> dict:
        """Make prediction with fixed model"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Extract features in correct order
        feature_values = []
        for feat in self.selected_features:
            value = patient_data.get(feat)
            if value is None:
                value = 0 if feat.startswith('is_') else np.nan
            feature_values.append(value)
        
        # Create DataFrame and process
        X = pd.DataFrame([feature_values], columns=self.selected_features)
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0]
        
        # FIXED MODEL: Use proba[1] for HIGH_RISK (class 1)
        # Training validation showed this is correct:
        # - High risk patients get 86.7% on class 1
        # - Low risk patients get 6.7% on class 1
        risk_score = proba[1] * 100
        
        # Thresholds
        if risk_score >= 70:
            risk_level = 'VERY_HIGH_RISK'
        elif risk_score >= 50:
            risk_level = 'HIGH_RISK'
        elif risk_score >= 30:
            risk_level = 'MODERATE_RISK'
        else:
            risk_level = 'LOW_RISK'
        
        return {
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'confidence': float(max(proba)),
            'raw_prediction': int(prediction),
            'raw_probabilities': {
                'prob_class_0_LOW': float(proba[0]),
                'prob_class_1_HIGH': float(proba[1])
            }
        }


class SubtypeModelManager:
    """Manages Subtype Classification Model"""
    
    def __init__(self):
        self.model_path = _PROJECT_ROOT / 'models' / 'subtype_classifier.pkl'
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        self.classes = None
        self.model_data = None
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            logger.info(f"Loading Subtype model from: {self.model_path}")
            self.model_data = joblib.load(self.model_path)
            
            self.model = self.model_data['model']
            self.scaler = self.model_data['scaler']
            self.imputer = self.model_data['imputer']
            self.feature_names = self.model_data['feature_names']
            self.classes = self.model_data['classes']
            
            logger.info(f"✓ Subtype model loaded: {self.model_data.get('version', 'unknown')}")
            logger.info(f"✓ Classes: {self.classes}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Subtype model: {e}")
            return False
    
    def predict(self, biomarker_data: dict) -> dict:
        """Make prediction with all 14 features"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
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
            # Find the matching key in biomarker_data
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
        
        # Create DataFrame and process
        X = pd.DataFrame([feature_values], columns=self.feature_names)
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0]
        
        return {
            'predicted_subtype': str(prediction),
            'confidence': float(max(proba)),
            'probabilities': {str(cls): float(prob) for cls, prob in zip(self.classes, proba)}
        }


# ============================================
# FASTAPI INITIALIZATION
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup"""
    global risk_agent_graph, subtype_agent_graph, risk_ml_model, subtype_model_manager
    
    logger.info("="*70)
    logger.info("🚀 Initializing OncoDetect-AI Unified API")
    logger.info("="*70)
    
    # Load Risk Score ML Model
    try:
        risk_ml_model = RiskScoreModelManager()
        logger.info("✓ Risk Score ML Model loaded")
    except Exception as e:
        logger.error(f"✗ Failed to load Risk Score model: {e}")
        risk_ml_model = None
    
    # Load Subtype Classifier Model
    try:
        subtype_model_manager = SubtypeModelManager()
        logger.info("✓ Subtype Classifier Model loaded")
    except Exception as e:
        logger.error(f"✗ Failed to load Subtype model: {e}")
        subtype_model_manager = None
    
    # Initialize Risk Agentic Graph
    try:
        if RISK_AGENTIC_AVAILABLE:
            risk_agent_graph = create_risk_graph()
            logger.info("✓ Risk Score Agentic Agent initialized")
        else:
            logger.warning("⚠ Risk agentic components not available")
    except Exception as e:
        logger.error(f"✗ Failed to initialize risk agentic agent: {e}")
        risk_agent_graph = None
    
    # Initialize Subtype Agentic Graph
    try:
        if SUBTYPE_AGENTIC_AVAILABLE:
            subtype_agent_graph = create_subtype_agent_graph()
            logger.info("✓ Subtype Agentic Agent initialized")
        else:
            logger.warning("⚠ Subtype agentic components not available")
    except Exception as e:
        logger.error(f"✗ Failed to initialize subtype agentic agent: {e}")
        subtype_agent_graph = None
    
    logger.info("="*70)
    logger.info(f"Risk ML Model: {'✓' if risk_ml_model else '✗'}")
    logger.info(f"Risk Agent: {'✓' if risk_agent_graph else '✗'}")
    logger.info(f"Subtype ML Model: {'✓' if subtype_model_manager else '✗'}")
    logger.info(f"Subtype Agent: {'✓' if subtype_agent_graph else '✗'}")
    logger.info("="*70)
    
    yield
    
    logger.info("Shutting down OncoDetect-AI API...")

# Initialize FastAPI app
app = FastAPI(
    title="OncoDetect-AI Unified API",
    description="Risk Prediction & Subtype Classification with Agentic Analysis",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# PYDANTIC MODELS
# ============================================

class RiskPatientData(BaseModel):
    """Input model for risk prediction - Matches fixed model features"""
    age: float = Field(..., ge=0, le=120, description="Patient age in years")
    mutation_count: int = Field(..., ge=0, description="Number of mutations")
    tmb: float = Field(..., ge=0, description="Tumor Mutation Burden")
    is_male: int = Field(..., ge=0, le=1, description="1 for male, 0 for female")
    is_former_smoker: int = Field(0, ge=0, le=1, description="1 if former smoker")
    is_tmb_high: int = Field(0, ge=0, le=1, description="1 if TMB >= 20")
    is_tmb_intermediate: int = Field(0, ge=0, le=1, description="1 if 10 <= TMB < 20")
    is_tmb_low: int = Field(0, ge=0, le=1, description="1 if TMB < 10")
    smoker_x_tmb: float = Field(0.0, description="Interaction: former_smoker * TMB")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 65,
                "mutation_count": 450,
                "tmb": 15.5,
                "is_male": 1,
                "is_former_smoker": 1,
                "is_tmb_high": 0,
                "is_tmb_intermediate": 1,
                "is_tmb_low": 0,
                "smoker_x_tmb": 15.5
            }
        }


class SubtypeBiomarkerData(BaseModel):
    """Input model for subtype classification - All 14 features"""
    ascl1: float = Field(..., ge=0, le=20, description="ASCL1 expression")
    neurod1: float = Field(..., ge=0, le=20, description="NEUROD1 expression")
    pou2f3: float = Field(..., ge=0, le=20, description="POU2F3 expression")
    yap1: float = Field(..., ge=0, le=20, description="YAP1 expression")
    tp53: float = Field(..., ge=0, le=20, description="TP53 expression")
    rb1: float = Field(..., ge=0, le=20, description="RB1 expression")
    myc: float = Field(..., ge=0, le=20, description="MYC expression")
    mycl: float = Field(..., ge=0, le=20, description="MYCL expression")
    mycn: float = Field(..., ge=0, le=20, description="MYCN expression")
    dll3: float = Field(7.0, ge=0, le=20, description="DLL3 expression")
    bcl2: float = Field(8.0, ge=0, le=20, description="BCL2 expression")
    notch1: float = Field(6.5, ge=0, le=20, description="NOTCH1 expression")
    myc_family_score: float = Field(..., ge=0, description="Sum of MYC family")
    tp53_rb1_dual_loss: int = Field(0, ge=0, le=1, description="Dual tumor suppressor loss")
    mean_expression: float = Field(..., ge=0, description="Mean expression across markers")
    stddev_expression: float = Field(..., ge=0, description="Std dev of expression")
    genes_expressed: int = Field(..., ge=0, description="Count of expressed genes")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ascl1": 12.5, "neurod1": 5.0, "pou2f3": 4.0, "yap1": 5.5,
                "tp53": 11.0, "rb1": 4.5, "myc": 8.0, "mycl": 9.5, "mycn": 6.0,
                "dll3": 7.0, "bcl2": 8.0, "notch1": 6.5,
                "myc_family_score": 23.5, "tp53_rb1_dual_loss": 0,
                "mean_expression": 7.8, "stddev_expression": 2.5, "genes_expressed": 7
            }
        }


class RiskPredictionResponse(BaseModel):
    """Response for risk prediction"""
    prediction_id: str
    risk_score: float
    risk_level: str
    confidence: float
    timestamp: str


class FullAgenticAnalysisRequest(BaseModel):
    """Request for full agentic analysis"""
    patient_data: RiskPatientData
    include_similar_patients: bool = True
    include_treatment_protocol: bool = True


class FullAgenticAnalysisResponse(BaseModel):
    """Response for full agentic analysis"""
    analysis_id: str
    patient_data: Dict[str, Any]
    ml_prediction: Dict[str, Any]
    similar_patients: Optional[str]
    clinical_report: str
    messages: List[str]
    timestamp: str
    processing_time_seconds: float


class SubtypeAgenticAnalysisRequest(BaseModel):
    """Request for full agentic subtype analysis"""
    biomarker_data: SubtypeBiomarkerData


class SubtypeAgenticAnalysisResponse(BaseModel):
    """Response for full agentic subtype analysis"""
    analysis_id: str
    biomarker_data: Dict[str, Any]
    ml_prediction: Dict[str, Any]
    biomarker_analysis: Dict[str, Any]
    hypothesis_testing: Dict[str, Any]
    clinical_report: str
    messages: List[str]
    timestamp: str
    processing_time_seconds: float


class SubtypePredictionResponse(BaseModel):
    """Response for subtype prediction"""
    prediction_id: str
    predicted_subtype: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: str


# ============================================
# ROOT & HEALTH ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "OncoDetect-AI Unified API",
        "version": "2.0.0",
        "status": "running",
        "capabilities": {
            "risk_ml": risk_ml_model is not None,
            "risk_agentic": risk_agent_graph is not None,
            "subtype": subtype_model_manager is not None
        },
        "endpoints": {
            "risk": {
                "quick": "/risk/predict",
                "agentic": "/risk/analyze",
                "batch": "/risk/batch",
                "info": "/risk/info"
            },
            "subtype": {
                "predict": "/subtype/predict",
                "batch": "/subtype/batch",
                "info": "/subtype/info"
            },
            "general": {
                "health": "/health",
                "docs": "/docs"
            }
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "risk_ml_loaded": risk_ml_model is not None,
        "risk_agent_ready": risk_agent_graph is not None,
        "subtype_loaded": subtype_model_manager is not None,
        "snowflake_available": SNOWFLAKE_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }


# ============================================
# RISK SCORE ENDPOINTS
# ============================================

@app.post("/risk/predict", response_model=RiskPredictionResponse)
async def predict_risk_quick(patient: RiskPatientData):
    """Quick ML-only risk prediction"""
    try:
        if risk_ml_model is None:
            raise HTTPException(
                status_code=503, 
                detail="Risk Score model not loaded"
            )
        
        logger.info("Quick risk prediction requested")
        
        patient_dict = patient.dict()
        prediction = risk_ml_model.predict(patient_dict)
        
        prediction_id = str(uuid.uuid4())
        
        response = RiskPredictionResponse(
            prediction_id=prediction_id,
            risk_score=prediction['risk_score'],
            risk_level=prediction['risk_level'],
            confidence=prediction['confidence'],
            timestamp=datetime.now().isoformat()
        )
        
        prediction_results[prediction_id] = response.dict()
        
        logger.info(f"✓ Quick prediction: {prediction['risk_level']} ({prediction['risk_score']:.1f})")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in quick risk prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/risk/analyze", response_model=FullAgenticAnalysisResponse)
async def analyze_risk_agentic(request: FullAgenticAnalysisRequest):
    """Full agentic analysis with LLM reasoning"""
    try:
        if risk_agent_graph is None:
            raise HTTPException(
                status_code=503,
                detail="Agentic agent not initialized. Possible causes: 1) LLM environment variables not set (OPENAI_API_KEY, OPENAI_BASE_URL), 2) Snowflake connection issue, 3) LangGraph components not installed"
            )
        
        logger.info("Full agentic analysis requested")
        start_time = datetime.now()
        
        patient_dict = request.patient_data.dict()
        
        # Prepare initial state for agent
        initial_state = {
            'messages': [],
            'patient_data': patient_dict,
            'ml_prediction': {},
            'similar_patients': '',
            'next_step': 'start'
        }
        
        logger.info("Starting LangGraph workflow...")
        
        try:
            # Run the agent graph (in thread to avoid blocking)
            final_state = await asyncio.to_thread(risk_agent_graph.invoke, initial_state)
        except Exception as agent_error:
            logger.error(f"Agent execution error: {agent_error}")
            # If agent fails, provide a fallback with just ML prediction
            if risk_ml_model:
                logger.info("Falling back to ML-only prediction")
                ml_prediction = risk_ml_model.predict(patient_dict)
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return FullAgenticAnalysisResponse(
                    analysis_id=str(uuid.uuid4()),
                    patient_data=patient_dict,
                    ml_prediction=ml_prediction,
                    similar_patients="Agent analysis unavailable - using ML prediction only",
                    clinical_report=f"""RISK ASSESSMENT (ML Only - Agent Unavailable)

Risk Score: {ml_prediction['risk_score']:.1f}/100
Risk Level: {ml_prediction['risk_level']}
Confidence: {ml_prediction['confidence']:.1%}

Note: Full AI analysis with LLM reasoning is currently unavailable.
Error: {str(agent_error)}

Please check:
1. LLM API credentials are set (OPENAI_API_KEY)
2. LLM endpoint is accessible (OPENAI_BASE_URL)
3. Snowflake connection is working
""",
                    messages=["ML prediction completed", str(agent_error)],
                    timestamp=datetime.now().isoformat(),
                    processing_time_seconds=processing_time
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Agent failed and no ML fallback available: {str(agent_error)}"
                )
        
        # Extract results - get only the LAST clinical report
        analysis_id = str(uuid.uuid4())
        clinical_report = ""
        messages_content = []
        
        # Collect all messages but filter to avoid duplicates
        seen_messages = set()
        for msg in final_state.get('messages', []):
            content = str(msg.content)
            # Skip exact duplicates
            if content not in seen_messages:
                messages_content.append(content)
                seen_messages.add(content)
            
            # The LAST message with "RISK ASSESSMENT" is the final report
            if "RISK ASSESSMENT" in content or "Clinical Report" in content:
                clinical_report = content
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = FullAgenticAnalysisResponse(
            analysis_id=analysis_id,
            patient_data=patient_dict,
            ml_prediction=final_state.get('ml_prediction', {}),
            similar_patients=final_state.get('similar_patients', ''),
            clinical_report=clinical_report,
            messages=messages_content,
            timestamp=datetime.now().isoformat(),
            processing_time_seconds=processing_time
        )
        
        prediction_results[analysis_id] = response.dict()
        
        logger.info(f"✓ Agentic analysis complete: {analysis_id} ({processing_time:.1f}s)")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in agentic analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agentic analysis failed: {str(e)}")


@app.post("/risk/batch")
async def predict_risk_batch(patients: List[RiskPatientData]):
    """Batch risk predictions (ML only)"""
    try:
        if risk_ml_model is None:
            raise HTTPException(status_code=503, detail="Risk Score model not loaded")
        
        logger.info(f"Batch prediction for {len(patients)} patients")
        
        results = []
        for patient in patients:
            patient_dict = patient.dict()
            prediction = risk_ml_model.predict(patient_dict)
            results.append({
                "patient_data": patient_dict,
                "risk_score": prediction['risk_score'],
                "risk_level": prediction['risk_level'],
                "confidence": prediction['confidence']
            })
        
        return {
            "count": len(results),
            "predictions": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/risk/info")
async def get_risk_model_info():
    """Get Risk Score model information"""
    if risk_ml_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_data = risk_ml_model.model_data
    
    feature_importance = []
    if 'feature_importance' in model_data:
        imp_df = model_data['feature_importance']
        feature_importance = imp_df.head(10).to_dict('records')
    
    # Get model classes to understand prediction mapping
    classes = risk_ml_model.model.classes_.tolist() if hasattr(risk_ml_model.model, 'classes_') else []
    
    return {
        "version": model_data.get('version', 'unknown'),
        "model_type": model_data.get('model_type', 'binary_classification'),
        "trained_at": model_data.get('trained_at', 'unknown'),
        "feature_count": len(risk_ml_model.selected_features),
        "selected_features": risk_ml_model.selected_features,
        "top_features": feature_importance,
        "classes": classes,
        "class_mapping": f"Class 0 = {classes[0] if len(classes) > 0 else 'unknown'}, Class 1 = {classes[1] if len(classes) > 1 else 'unknown'}"
    }


@app.post("/risk/debug")
async def debug_risk_prediction(patient: RiskPatientData):
    """Debug endpoint to see raw model output"""
    if risk_ml_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        patient_dict = patient.dict()
        
        # Extract features manually to debug
        feature_values = []
        for feat in risk_ml_model.selected_features:
            value = patient_dict.get(feat)
            if value is None:
                value = 0 if feat.startswith('is_') else np.nan
            feature_values.append(value)
        
        X = pd.DataFrame([feature_values], columns=risk_ml_model.selected_features)
        X_imputed = risk_ml_model.imputer.transform(X)
        X_scaled = risk_ml_model.scaler.transform(X_imputed)
        
        # Get raw predictions
        prediction = risk_ml_model.model.predict(X_scaled)[0]
        proba = risk_ml_model.model.predict_proba(X_scaled)[0]
        
        # Get result using our predict method
        result = risk_ml_model.predict(patient_dict)
        
        return {
            "input_features": patient_dict,
            "selected_features": risk_ml_model.selected_features,
            "feature_values_used": dict(zip(risk_ml_model.selected_features, feature_values)),
            "model_classes": risk_ml_model.model.classes_.tolist(),
            "raw_prediction_class": int(prediction),
            "raw_probabilities": {
                "class_0_LOW_RISK": float(proba[0]),
                "class_1_HIGH_RISK": float(proba[1])
            },
            "our_calculation": {
                "risk_score": result['risk_score'],
                "risk_level": result['risk_level'],
                "explanation": f"Using proba[1] = {proba[1]:.3f} * 100 = {proba[1]*100:.1f}"
            },
            "expected_behavior": {
                "low_risk_patient": "Should have class_1_HIGH_RISK < 0.40 (score < 40)",
                "high_risk_patient": "Should have class_1_HIGH_RISK > 0.60 (score > 60)"
            }
        }
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


# ============================================
# SUBTYPE CLASSIFICATION ENDPOINTS
# ============================================

@app.post("/subtype/analyze", response_model=SubtypeAgenticAnalysisResponse)
async def analyze_subtype_agentic(request: SubtypeAgenticAnalysisRequest):
    """Full agentic subtype analysis with biomarker investigation"""
    try:
        if subtype_agent_graph is None:
            raise HTTPException(
                status_code=503,
                detail="Subtype agentic agent not initialized. Check LLM environment variables."
            )
        
        logger.info("Full agentic subtype analysis requested")
        start_time = datetime.now()
        
        biomarker_dict = request.biomarker_data.dict()
        
        # Prepare initial state
        initial_state = {
            'messages': [],
            'biomarker_data': biomarker_dict,
            'ml_prediction': {},
            'biomarker_analysis': {},
            'hypothesis_testing': {},
            'final_diagnosis': '',
            'next_step': 'start'
        }
        
        logger.info("Starting Subtype Agent LangGraph workflow...")
        
        try:
            # Run the agent graph
            final_state = await asyncio.to_thread(subtype_agent_graph.invoke, initial_state)
        except Exception as agent_error:
            logger.error(f"Subtype agent execution error: {agent_error}")
            # Fallback to ML only
            if subtype_model_manager:
                logger.info("Falling back to ML-only subtype prediction")
                ml_prediction = subtype_model_manager.predict(biomarker_dict)
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return SubtypeAgenticAnalysisResponse(
                    analysis_id=str(uuid.uuid4()),
                    biomarker_data=biomarker_dict,
                    ml_prediction=ml_prediction,
                    biomarker_analysis={},
                    hypothesis_testing={},
                    clinical_report=f"""SUBTYPE CLASSIFICATION (ML Only - Agent Unavailable)

Predicted Subtype: {ml_prediction['predicted_subtype']}
Confidence: {ml_prediction['confidence']:.1%}

Note: Full AI analysis unavailable. Error: {str(agent_error)}
""",
                    messages=["ML prediction completed", str(agent_error)],
                    timestamp=datetime.now().isoformat(),
                    processing_time_seconds=processing_time
                )
            else:
                raise HTTPException(status_code=500, detail=str(agent_error))
        
        # Extract results
        analysis_id = str(uuid.uuid4())
        clinical_report = ""
        messages_content = []
        
        # Filter duplicate messages
        seen_messages = set()
        for msg in final_state.get('messages', []):
            content = str(msg.content)
            if content not in seen_messages:
                messages_content.append(content)
                seen_messages.add(content)
            
            # Last message with report structure
            if "MOLECULAR SUBTYPE DIAGNOSIS" in content or "BIOMARKER EVIDENCE" in content:
                clinical_report = content
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = SubtypeAgenticAnalysisResponse(
            analysis_id=analysis_id,
            biomarker_data=biomarker_dict,
            ml_prediction=final_state.get('ml_prediction', {}),
            biomarker_analysis=final_state.get('biomarker_analysis', {}),
            hypothesis_testing=final_state.get('hypothesis_testing', {}),
            clinical_report=clinical_report,
            messages=messages_content,
            timestamp=datetime.now().isoformat(),
            processing_time_seconds=processing_time
        )
        
        prediction_results[analysis_id] = response.dict()
        
        logger.info(f"✓ Subtype agentic analysis complete: {analysis_id} ({processing_time:.1f}s)")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in subtype agentic analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/subtype/predict", response_model=SubtypePredictionResponse)
async def predict_subtype(biomarkers: SubtypeBiomarkerData):
    """Predict SCLC subtype"""
    try:
        if subtype_model_manager is None:
            raise HTTPException(status_code=503, detail="Subtype model not loaded")
        
        logger.info("Subtype prediction requested")
        
        biomarker_dict = biomarkers.dict()
        prediction = subtype_model_manager.predict(biomarker_dict)
        
        prediction_id = str(uuid.uuid4())
        
        response = SubtypePredictionResponse(
            prediction_id=prediction_id,
            predicted_subtype=prediction['predicted_subtype'],
            confidence=prediction['confidence'],
            probabilities=prediction['probabilities'],
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"✓ Subtype prediction: {prediction['predicted_subtype']}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in subtype prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/subtype/batch")
async def predict_subtype_batch(biomarkers_list: List[SubtypeBiomarkerData]):
    """Batch subtype predictions"""
    try:
        if subtype_model_manager is None:
            raise HTTPException(status_code=503, detail="Subtype model not loaded")
        
        logger.info(f"Batch subtype prediction for {len(biomarkers_list)} samples")
        
        results = []
        for biomarkers in biomarkers_list:
            biomarker_dict = biomarkers.dict()
            prediction = subtype_model_manager.predict(biomarker_dict)
            results.append({
                "biomarker_data": biomarker_dict,
                "predicted_subtype": prediction['predicted_subtype'],
                "confidence": prediction['confidence'],
                "probabilities": prediction['probabilities']
            })
        
        return {
            "count": len(results),
            "predictions": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in batch subtype prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/subtype/info")
async def get_subtype_model_info():
    """Get Subtype model information"""
    if subtype_model_manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_data = subtype_model_manager.model_data
    
    feature_importance = []
    if 'feature_importance' in model_data:
        imp_df = model_data['feature_importance']
        feature_importance = imp_df.head(10).to_dict('records')
    
    return {
        "version": model_data.get('version', 'unknown'),
        "model_type": model_data.get('model_type', 'multiclass_classification'),
        "trained_at": model_data.get('trained_at', 'unknown'),
        "feature_count": len(subtype_model_manager.feature_names),
        "features": subtype_model_manager.feature_names,
        "classes": subtype_model_manager.classes,
        "top_features": feature_importance
    }


# ============================================
# RESULTS MANAGEMENT
# ============================================

@app.get("/results/{result_id}")
async def get_result(result_id: str):
    """Retrieve a previous prediction result"""
    if result_id not in prediction_results:
        raise HTTPException(status_code=404, detail="Result not found")
    return prediction_results[result_id]


@app.get("/results")
async def list_results(limit: int = 10):
    """List recent prediction results"""
    recent_ids = list(prediction_results.keys())[-limit:]
    recent_results = []
    
    for rid in reversed(recent_ids):
        result = prediction_results[rid]
        recent_results.append({
            "id": rid,
            "timestamp": result.get("timestamp"),
            "type": "risk" if "risk_level" in result else "subtype"
        })
    
    return {
        "total": len(prediction_results),
        "results": recent_results
    }


# ============================================
# RUN THE APP
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 Starting OncoDetect-AI Unified API")
    print("="*70)
    print(f"Project Root: {_PROJECT_ROOT}")
    print("="*70 + "\n")
    
    uvicorn.run(
        "api_oncodetect_unified:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )