-- ========================================
-- DBT STAGING SCHEMA - VIEWS & TABLES
-- ========================================
-- Purpose: DBT staging layer views and table exploration

USE DATABASE ONCODETECT_DB;
USE WAREHOUSE SCLC_WH;
USE SCHEMA DBT_STAGING;

-- ========================================
-- LIST ALL VIEWS IN DBT_STAGING
-- ========================================
SHOW VIEWS IN DBT_STAGING;

-- ========================================
-- LIST ALL TABLES IN DBT_STAGING
-- ========================================
SHOW TABLES IN DBT_STAGING;

-- ========================================
-- STAGING TABLE: STG_CLINICAL
-- ========================================
-- Preview clinical staging data
SELECT *
FROM DBT_STAGING.STG_CLINICAL
LIMIT 10;

-- Check sample IDs
SELECT DISTINCT sample_id
FROM DBT_STAGING.STG_CLINICAL
ORDER BY sample_id
LIMIT 20;

-- ========================================
-- STAGING TABLE: STG_GENE_EXPRESSION
-- ========================================
-- Preview gene expression staging data
SELECT *
FROM DBT_STAGING.STG_GENE_EXPRESSION
LIMIT 20;

-- Check available genes
SELECT DISTINCT gene_symbol
FROM DBT_STAGING.STG_GENE_EXPRESSION
ORDER BY gene_symbol;

-- Check sample IDs in gene expression
SELECT DISTINCT sample_id
FROM DBT_STAGING.STG_GENE_EXPRESSION
ORDER BY sample_id
LIMIT 20;

-- Check biomarker genes availability
SELECT DISTINCT gene_symbol
FROM DBT_STAGING.STG_GENE_EXPRESSION
WHERE gene_symbol IN (
    'ASCL1', 'NEUROD1', 'POU2F3', 'YAP1',
    'TP53', 'RB1', 'MYC', 'MYCL', 'MYCN'
)
ORDER BY gene_symbol;

-- Check gene expression patterns
SELECT DISTINCT gene_symbol
FROM DBT_STAGING.STG_GENE_EXPRESSION
WHERE gene_symbol LIKE 'ASCL%'
   OR gene_symbol LIKE 'NEURO%'
   OR gene_symbol LIKE 'POU2%'
   OR gene_symbol LIKE 'YAP%'
   OR gene_symbol LIKE 'TP5%'
   OR gene_symbol LIKE 'RB%'
   OR gene_symbol LIKE 'MYC%'
ORDER BY gene_symbol;

-- Sample count per biomarker gene
SELECT 
    gene_symbol,
    COUNT(DISTINCT sample_id) AS sample_count,
    AVG(expression_value) AS avg_expression
FROM DBT_STAGING.STG_GENE_EXPRESSION
WHERE gene_symbol IN ('ASCL1', 'NEUROD1', 'POU2F3', 'YAP1')
GROUP BY gene_symbol;

-- ========================================
-- MART TABLE: MART_RISK_FEATURES_CLINICAL
-- ========================================
-- Preview clinical risk features
SELECT *
FROM DBT_STAGING.MART_RISK_FEATURES_CLINICAL
LIMIT 10;

-- Count total samples
SELECT COUNT(*) AS total_samples
FROM DBT_STAGING.MART_RISK_FEATURES_CLINICAL;

-- Check survival distribution
SELECT 
    target_category,
    COUNT(*) AS sample_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS percentage
FROM DBT_STAGING.MART_RISK_FEATURES_CLINICAL
GROUP BY target_category
ORDER BY sample_count DESC;

-- Sex distribution
SELECT 
    is_male,
    COUNT(*) AS patient_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS percentage
FROM DBT_STAGING.MART_RISK_FEATURES_CLINICAL
GROUP BY is_male
ORDER BY is_male;

-- Sex distribution by survival
SELECT 
    CASE WHEN is_male = 1 THEN 'Male' ELSE 'Female' END AS sex,
    target_category,
    COUNT(*) AS count
FROM DBT_STAGING.MART_RISK_FEATURES_CLINICAL
GROUP BY is_male, target_category
ORDER BY is_male, target_category;

-- ========================================
-- MART TABLE: MART_RISK_FEATURES_GENOMIC
-- ========================================
-- Preview genomic risk features
SELECT *
FROM DBT_STAGING.MART_RISK_FEATURES_GENOMIC
LIMIT 10;

-- Count total samples
SELECT COUNT(*) AS total_samples
FROM DBT_STAGING.MART_RISK_FEATURES_GENOMIC;

-- Check SCLC subtype distribution
SELECT 
    sclc_subtype,
    COUNT(*) AS sample_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS percentage
FROM DBT_STAGING.MART_RISK_FEATURES_GENOMIC
GROUP BY sclc_subtype
ORDER BY sample_count DESC;

-- Check genomic sample IDs
SELECT DISTINCT sample_id
FROM DBT_STAGING.MART_RISK_FEATURES_GENOMIC
ORDER BY sample_id
LIMIT 20;

-- ========================================
-- MART TABLE: MART_RISK_FEATURES_COMBINED
-- ========================================
-- Preview combined features
SELECT *
FROM DBT_STAGING.MART_RISK_FEATURES_COMBINED
LIMIT 10;

-- Check genomic data coverage
SELECT 
    has_genomic_data,
    data_source,
    COUNT(*) AS patient_count
FROM DBT_STAGING.MART_RISK_FEATURES_COMBINED
GROUP BY has_genomic_data, data_source;

-- ========================================
-- MART TABLE: MART_DRUG_SENSITIVITY
-- ========================================
-- Preview drug sensitivity data
SELECT *
FROM DBT_STAGING.MART_DRUG_SENSITIVITY
LIMIT 10;

SELECT COUNT(*) AS total_records
FROM DBT_STAGING.MART_DRUG_SENSITIVITY;

-- ========================================
-- MART TABLE: MART_DRUG_SUBTYPE_PROFILE
-- ========================================
-- Preview drug-subtype profiles
SELECT *
FROM DBT_STAGING.MART_DRUG_SUBTYPE_PROFILE
LIMIT 10;

SELECT COUNT(*) AS total_profiles
FROM DBT_STAGING.MART_DRUG_SUBTYPE_PROFILE;

-- ========================================
-- MART TABLE: MART_TREATMENT_RECOMMENDATIONS
-- ========================================
-- Preview treatment recommendations
SELECT *
FROM DBT_STAGING.MART_TREATMENT_RECOMMENDATIONS
LIMIT 10;

-- Top 3 drug recommendations per sample
SELECT 
    sample_id,
    sclc_subtype,
    drug_rank,
    drug_name,
    recommendation_score,
    recommendation_priority,
    treatment_rationale
FROM DBT_STAGING.MART_TREATMENT_RECOMMENDATIONS
WHERE drug_rank <= 3
ORDER BY sample_id, drug_rank
LIMIT 30;

-- All treatment recommendations
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
-- MART TABLE: MART_DOSE_RESPONSE_SUMMARY
-- ========================================
-- Preview dose response summary
SELECT *
FROM DBT_STAGING.MART_DOSE_RESPONSE_SUMMARY
LIMIT 10;

SELECT COUNT(*) AS total_responses
FROM DBT_STAGING.MART_DOSE_RESPONSE_SUMMARY;

-- ========================================
-- DATA QUALITY CHECKS
-- ========================================

-- Check samples in clinical but NOT in gene expression
SELECT COUNT(DISTINCT c.sample_id) AS clinical_only_samples
FROM DBT_STAGING.STG_CLINICAL c
LEFT JOIN DBT_STAGING.STG_GENE_EXPRESSION g
    ON c.sample_id = g.sample_id
WHERE g.sample_id IS NULL;

-- Check if clinical sample_id contains gene expression patterns
SELECT DISTINCT c.sample_id AS clinical_id
FROM DBT_STAGING.STG_CLINICAL c
WHERE c.sample_id LIKE '%11A%'
   OR c.sample_id LIKE '%12A%'
   OR c.sample_id LIKE '%B08%'
   OR c.sample_id LIKE '%B10%'
LIMIT 10;

-- ========================================
-- COMPREHENSIVE SUMMARY
-- ========================================
-- Summary of all mart tables
SELECT 'Clinical Features' AS table_name, COUNT(*) AS row_count 
FROM DBT_STAGING.MART_RISK_FEATURES_CLINICAL
UNION ALL
SELECT 'Genomic Features', COUNT(*) 
FROM DBT_STAGING.MART_RISK_FEATURES_GENOMIC
UNION ALL
SELECT 'Combined Features', COUNT(*) 
FROM DBT_STAGING.MART_RISK_FEATURES_COMBINED
UNION ALL
SELECT 'Drug Sensitivity', COUNT(*) 
FROM DBT_STAGING.MART_DRUG_SENSITIVITY
UNION ALL
SELECT 'Drug-Subtype Profile', COUNT(*) 
FROM DBT_STAGING.MART_DRUG_SUBTYPE_PROFILE
UNION ALL
SELECT 'Treatment Recommendations', COUNT(*) 
FROM DBT_STAGING.MART_TREATMENT_RECOMMENDATIONS
UNION ALL
SELECT 'Dose Response Summary', COUNT(*) 
FROM DBT_STAGING.MART_DOSE_RESPONSE_SUMMARY;