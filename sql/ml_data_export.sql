-- ========================================
-- ML DATA EXPORT QUERIES
-- ========================================
-- Purpose: Export formatted data for ML training and analysis


USE DATABASE ONCODETECT_DB;
USE WAREHOUSE SCLC_WH;

-- ========================================
-- EXPORT: CLINICAL FEATURES FOR ML TRAINING
-- ========================================
SELECT 
    sample_id,
    
    -- TARGET VARIABLES (what we're predicting)
    target_survival_months,
    target_event,
    target_category,
    
    -- FEATURES (input to model)
    age,
    is_male,
    is_current_smoker,
    is_former_smoker,
    is_never_smoker,
    mutation_count,
    tmb,
    is_tmb_high,
    is_tmb_intermediate,
    is_tmb_low,
    is_stage_i,
    is_stage_ii,
    is_stage_iii,
    is_stage_iv,
    is_advanced_stage,
    age_x_advanced_stage,
    smoker_x_tmb,
    high_risk_flag
    
FROM DBT_STAGING.MART_RISK_FEATURES_CLINICAL
ORDER BY sample_id;

-- ========================================
-- EXPORT: GENOMIC FEATURES FOR ML TRAINING
-- ========================================
SELECT *
FROM DBT_STAGING.MART_RISK_FEATURES_GENOMIC
ORDER BY sample_id;

-- ========================================
-- EXPORT: COMBINED FEATURES
-- ========================================
SELECT *
FROM DBT_STAGING.MART_RISK_FEATURES_COMBINED
ORDER BY sample_id;

-- ========================================
-- EXPORT: DRUG SENSITIVITY DATA
-- ========================================
SELECT *
FROM DBT_STAGING.MART_DRUG_SENSITIVITY
ORDER BY cell_line, nsc;

-- ========================================
-- EXPORT: TREATMENT RECOMMENDATIONS
-- ========================================
SELECT 
    sample_id,
    sclc_subtype,
    drug_rank,
    drug_name,
    recommendation_score,
    recommendation_priority,
    treatment_rationale
FROM DBT_STAGING.MART_TREATMENT_RECOMMENDATIONS
ORDER BY sample_id, drug_rank;

-- ========================================
-- EXPORT: GENE EXPRESSION DATA
-- ========================================
SELECT * 
FROM DBT_STAGING.STG_GENE_EXPRESSION
ORDER BY sample_id, gene_symbol;

-- ========================================
-- EXPORT: CLINICAL DATA (RAW)
-- ========================================
SELECT *
FROM RAW_DATA.CLINICAL_DATA
ORDER BY sample_id;

-- ========================================
-- EXPORT: PUBMED PAPERS METADATA
-- ========================================
SELECT * 
FROM UNSTRUCTURED_DATA.PUBMED_PAPERS_METADATA
ORDER BY paper_id;