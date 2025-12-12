-- models/mart_risk_features_clinical.sql
-- Clinical features for Risk Score Agent (survival prediction)

{{ config(materialized='table') }}

WITH clinical AS (
    SELECT * FROM {{ ref('stg_clinical') }}
)

SELECT
    -- IDs
    sample_id,
    patient_id,
    study_id,
    
    -- TARGETS (for ML - survival prediction)
    overall_survival_months AS target_survival_months,
    survival_event AS target_event,
    survival_category AS target_category,
    
    -- DEMOGRAPHIC FEATURES
    diagnosis_age AS age,
    age_group,
    CASE WHEN sex = 'Male' THEN 1 ELSE 0 END AS is_male,
    
    -- SMOKING FEATURES (critical for SCLC)
    smoking_status,
    CASE WHEN smoking_status = 'CURRENT' THEN 1 ELSE 0 END AS is_current_smoker,
    CASE WHEN smoking_status = 'FORMER' THEN 1 ELSE 0 END AS is_former_smoker,
    CASE WHEN smoking_status = 'NEVER' THEN 1 ELSE 0 END AS is_never_smoker,
    
    -- GENOMIC FEATURES (mutations & TMB)
    mutation_count,
    tmb,
    tmb_category,
    CASE WHEN tmb_category = 'TMB_HIGH' THEN 1 ELSE 0 END AS is_tmb_high,
    CASE WHEN tmb_category = 'TMB_INTERMEDIATE' THEN 1 ELSE 0 END AS is_tmb_intermediate,
    CASE WHEN tmb_category = 'TMB_LOW' THEN 1 ELSE 0 END AS is_tmb_low,
    
    -- STAGING FEATURES
    stage_group,
    stage_at_diagnosis,
    CASE WHEN stage_group = 'STAGE_I' THEN 1 ELSE 0 END AS is_stage_i,
    CASE WHEN stage_group = 'STAGE_II' THEN 1 ELSE 0 END AS is_stage_ii,
    CASE WHEN stage_group = 'STAGE_III' THEN 1 ELSE 0 END AS is_stage_iii,
    CASE WHEN stage_group = 'STAGE_IV' THEN 1 ELSE 0 END AS is_stage_iv,
    CASE WHEN stage_group IN ('STAGE_III', 'STAGE_IV') THEN 1 ELSE 0 END AS is_advanced_stage,
    
    -- DERIVED RISK FEATURES
    -- Age-stage interaction
    diagnosis_age * CASE WHEN stage_group IN ('STAGE_III', 'STAGE_IV') THEN 1 ELSE 0 END AS age_x_advanced_stage,
    
    -- Smoking-TMB interaction
    CASE WHEN smoking_status = 'CURRENT' THEN 1 ELSE 0 END * tmb AS smoker_x_tmb,
    
    -- High risk flag
    CASE 
        WHEN stage_group IN ('STAGE_III', 'STAGE_IV')
             AND tmb < 6
             AND diagnosis_age > 65
        THEN 1 ELSE 0 
    END AS high_risk_flag,
    
    -- Metadata
    'CLINICAL_COHORT' AS cohort_type,
    CURRENT_TIMESTAMP() AS created_at

FROM clinical
WHERE overall_survival_months IS NOT NULL