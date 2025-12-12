-- models/stg_clinical.sql
-- Clean and prepare clinical data

{{ config(materialized='view') }}

SELECT
    -- IDs
    UPPER(TRIM(sample_id)) AS sample_id,
    UPPER(TRIM(patient_id)) AS patient_id,
    study_id,
    
    -- Demographics
    diagnosis_age,
    sex,
    CASE 
        WHEN diagnosis_age < 50 THEN 'YOUNG'
        WHEN diagnosis_age BETWEEN 50 AND 65 THEN 'MIDDLE'
        WHEN diagnosis_age BETWEEN 65 AND 75 THEN 'OLDER'
        WHEN diagnosis_age > 75 THEN 'ELDERLY'
    END AS age_group,
    
    -- Smoking
    CASE
        WHEN smoker LIKE '%Current%' OR smoking_history LIKE '%Current%' THEN 'CURRENT'
        WHEN smoker LIKE '%Former%' OR smoking_history LIKE '%Former%' THEN 'FORMER'
        WHEN smoker LIKE '%Never%' OR smoking_history LIKE '%Never%' THEN 'NEVER'
        ELSE 'UNKNOWN'
    END AS smoking_status,
    
    -- Genomics
    COALESCE(mutation_count, 0) AS mutation_count,
    COALESCE(tmb_nonsynonymous, 0) AS tmb,
    CASE
        WHEN tmb_nonsynonymous < 6 THEN 'TMB_LOW'
        WHEN tmb_nonsynonymous BETWEEN 6 AND 20 THEN 'TMB_INTERMEDIATE'
        WHEN tmb_nonsynonymous > 20 THEN 'TMB_HIGH'
        ELSE 'UNKNOWN'
    END AS tmb_category,
    
    -- Staging
    stage_at_diagnosis,
    CASE
        WHEN stage_at_diagnosis LIKE 'I%' THEN 'STAGE_I'
        WHEN stage_at_diagnosis LIKE 'II%' THEN 'STAGE_II'
        WHEN stage_at_diagnosis LIKE 'III%' THEN 'STAGE_III'
        WHEN stage_at_diagnosis LIKE 'IV%' THEN 'STAGE_IV'
        ELSE 'UNKNOWN'
    END AS stage_group,
    
    -- Survival
    overall_survival_months,
    CASE 
        WHEN overall_survival_status LIKE '%DECEASED%' THEN 1
        WHEN overall_survival_status LIKE '%LIVING%' THEN 0
    END AS survival_event,
    CASE
        WHEN overall_survival_months < 12 THEN 'SHORT'
        WHEN overall_survival_months BETWEEN 12 AND 36 THEN 'MEDIUM'
        WHEN overall_survival_months > 36 THEN 'LONG'
    END AS survival_category

FROM {{ source('raw_data', 'clinical_data') }}
WHERE sample_id IS NOT NULL