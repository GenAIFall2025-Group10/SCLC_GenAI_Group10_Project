-- ========================================
-- ML PREDICTION FUNCTIONS
-- ========================================
-- Purpose: Machine learning inference functions for risk and subtype prediction

USE DATABASE ONCODETECT_DB;
USE WAREHOUSE SCLC_WH;

-- ========================================
-- RISK SCORE PREDICTION FUNCTION
-- ========================================
CREATE OR REPLACE FUNCTION PREDICT_RISK_SCORE(
    age FLOAT,
    is_male FLOAT,
    is_former_smoker FLOAT,
    mutation_count FLOAT,
    tmb FLOAT,
    is_tmb_high FLOAT,
    is_tmb_intermediate FLOAT,
    is_tmb_low FLOAT,
    smoker_x_tmb FLOAT
)
RETURNS OBJECT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.9'
PACKAGES = ('scikit-learn', 'pandas', 'numpy', 'joblib')
IMPORTS = ('@ML_MODELS_STAGE/risk_score_agent_binary_fixed.pkl')
HANDLER = 'predict_risk'
AS
$$
import joblib
import pandas as pd
import numpy as np
import sys

def predict_risk(age, is_male, is_former_smoker, mutation_count, tmb, 
                 is_tmb_high, is_tmb_intermediate, is_tmb_low, smoker_x_tmb):
    
    # Load model from Snowflake stage
    import_dir = sys._xoptions.get("snowflake_import_directory")
    model_data = joblib.load(import_dir + 'risk_score_agent_binary_fixed.pkl')
    
    # Extract model components
    model = model_data['model']
    imputer = model_data['imputer']
    scaler = model_data['scaler']
    selected_features = model_data['selected_features']
    
    # Create feature array - now with ALL 9 features
    features = [age, mutation_count, tmb, is_male, is_former_smoker, 
                is_tmb_high, is_tmb_intermediate, is_tmb_low, smoker_x_tmb]
    
    # Create DataFrame
    X = pd.DataFrame([features], columns=selected_features)
    
    # Preprocess
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    
    # Calculate risk score (0-100)
    risk_score = proba[1] * 100
    confidence = float(proba[prediction])
    
    # Determine risk level
    if risk_score >= 80:
        risk_level = "VERY_HIGH_RISK"
    elif risk_score >= 60:
        risk_level = "HIGH_RISK"
    elif risk_score >= 40:
        risk_level = "MODERATE_RISK"
    else:
        risk_level = "LOW_RISK"
    
    return {
        'risk_score': float(risk_score),
        'risk_level': risk_level,
        'confidence': float(confidence),
        'prediction': int(prediction)
    }
$$;

-- ========================================
-- SUBTYPE PREDICTION FUNCTION
-- ========================================
CREATE OR REPLACE FUNCTION PREDICT_SUBTYPE(
    ascl1_expression FLOAT,
    neurod1_expression FLOAT,
    pou2f3_expression FLOAT,
    yap1_expression FLOAT,
    tp53_expression FLOAT,
    rb1_expression FLOAT,
    myc_family_score FLOAT,
    tp53_rb1_dual_loss FLOAT,
    dll3_expression FLOAT,
    bcl2_expression FLOAT,
    notch1_expression FLOAT,
    mean_expression FLOAT,
    stddev_expression FLOAT,
    genes_expressed FLOAT
)
RETURNS OBJECT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.9'
PACKAGES = ('scikit-learn', 'pandas', 'numpy', 'joblib')
IMPORTS = ('@ML_MODELS_STAGE/subtype_classifier.pkl')
HANDLER = 'predict_subtype'
AS
$$
import joblib
import pandas as pd
import numpy as np
import sys

def predict_subtype(ascl1_expression, neurod1_expression, pou2f3_expression,
                   yap1_expression, tp53_expression, rb1_expression,
                   myc_family_score, tp53_rb1_dual_loss, dll3_expression,
                   bcl2_expression, notch1_expression, mean_expression,
                   stddev_expression, genes_expressed):
    
    # Load model from Snowflake stage
    import_dir = sys._xoptions.get("snowflake_import_directory")
    model_data = joblib.load(import_dir + 'subtype_classifier.pkl')
    
    # Extract components
    model = model_data['model']
    scaler = model_data['scaler']
    imputer = model_data['imputer']
    
    # Feature names in EXACT training order
    feature_names = [
        'ascl1_expression',
        'neurod1_expression',
        'pou2f3_expression',
        'yap1_expression',
        'tp53_expression',
        'rb1_expression',
        'myc_family_score',
        'tp53_rb1_dual_loss',
        'dll3_expression',
        'bcl2_expression',
        'notch1_expression',
        'mean_expression',
        'stddev_expression',
        'genes_expressed'
    ]
    
    # Create feature array in EXACT order
    features = [
        ascl1_expression,
        neurod1_expression,
        pou2f3_expression,
        yap1_expression,
        tp53_expression,
        rb1_expression,
        myc_family_score,
        tp53_rb1_dual_loss,
        dll3_expression,
        bcl2_expression,
        notch1_expression,
        mean_expression,
        stddev_expression,
        genes_expressed
    ]
    
    # Create DataFrame
    X = pd.DataFrame([features], columns=feature_names)
    
    # Preprocess (impute then scale)
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    
    # Get classes
    classes = model.classes_
    
    # Find confidence
    pred_idx = list(classes).index(prediction)
    confidence = float(proba[pred_idx])
    
    # All probabilities
    all_probabilities = {}
    for i, class_name in enumerate(classes):
        all_probabilities[class_name] = float(proba[i])
    
    return {
        'predicted_subtype': prediction,
        'confidence': confidence,
        'probabilities': all_probabilities
    }
$$;