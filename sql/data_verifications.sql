-- ========================================
-- DATA VERIFICATION & VALIDATION QUERIES
-- ========================================
-- Purpose: Quality checks and data exploration for OncoDetect-AI


USE DATABASE ONCODETECT_DB;
USE WAREHOUSE SCLC_WH;

-- ========================================
-- TABLE ROW COUNTS
-- ========================================
SELECT 'clinical_data' AS table_name, COUNT(*) AS row_count 
FROM RAW_DATA.CLINICAL_DATA
UNION ALL
SELECT 'gene_expression', COUNT(*) 
FROM RAW_DATA.GENE_EXPRESSION
UNION ALL
SELECT 'compounds', COUNT(*) 
FROM RAW_DATA.COMPOUNDS
UNION ALL
SELECT 'cell_lines', COUNT(*) 
FROM RAW_DATA.CELL_LINES
UNION ALL
SELECT 'experiments', COUNT(*) 
FROM RAW_DATA.EXPERIMENTS
UNION ALL
SELECT 'dose_response', COUNT(*) 
FROM RAW_DATA.DOSE_RESPONSE;

-- ========================================
-- MART TABLES SUMMARY
-- ========================================
SELECT 'Clinical Features' AS table_name, COUNT(*) AS row_count 
FROM DBT_STAGING.MART_RISK_FEATURES_CLINICAL
UNION ALL
SELECT 'Genomic Features', COUNT(*) 
FROM DBT_STAGING.MART_RISK_FEATURES_GENOMIC
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

-- ========================================
-- SURVIVAL TARGET DISTRIBUTION
-- ========================================
SELECT 
    target_category,
    COUNT(*) AS sample_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS percentage
FROM DBT_STAGING.MART_RISK_FEATURES_CLINICAL
GROUP BY target_category
ORDER BY sample_count DESC;

-- ========================================
-- SCLC SUBTYPE DISTRIBUTION
-- ========================================
SELECT 
    sclc_subtype,
    COUNT(*) AS sample_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS percentage
FROM DBT_STAGING.MART_RISK_FEATURES_GENOMIC
GROUP BY sclc_subtype
ORDER BY sample_count DESC;

-- ========================================
-- SEX DISTRIBUTION
-- ========================================
SELECT 
    CASE WHEN is_male = 1 THEN 'Male' ELSE 'Female' END AS sex,
    COUNT(*) AS patient_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS percentage
FROM DBT_STAGING.MART_RISK_FEATURES_CLINICAL
GROUP BY is_male
ORDER BY is_male;

-- ========================================
-- BIOMARKER GENE AVAILABILITY
-- ========================================
SELECT DISTINCT gene_symbol
FROM DBT_STAGING.STG_GENE_EXPRESSION
WHERE gene_symbol IN (
    'ASCL1', 'NEUROD1', 'POU2F3', 'YAP1',
    'TP53', 'RB1', 'MYC', 'MYCL', 'MYCN'
)
ORDER BY gene_symbol;

-- ========================================
-- SAMPLE ID FORMAT CHECK
-- ========================================
-- Clinical sample IDs
SELECT DISTINCT sample_id AS clinical_sample_id
FROM DBT_STAGING.STG_CLINICAL
ORDER BY sample_id
LIMIT 20;

-- Gene expression sample IDs
SELECT DISTINCT sample_id AS gene_sample_id
FROM DBT_STAGING.STG_GENE_EXPRESSION
ORDER BY sample_id
LIMIT 20;

-- ========================================
-- GENOMIC DATA COVERAGE
-- ========================================
SELECT 
    has_genomic_data,
    data_source,
    COUNT(*) AS patient_count
FROM DBT_STAGING.MART_RISK_FEATURES_COMBINED
GROUP BY has_genomic_data, data_source;

-- ========================================
-- CELL LINE PANEL DISTRIBUTION
-- ========================================
SELECT 
    panel, 
    COUNT(*) AS cell_line_count
FROM RAW_DATA.CELL_LINES
GROUP BY panel
ORDER BY cell_line_count DESC;

-- ========================================
-- TREATMENT RECOMMENDATIONS PREVIEW
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
WHERE drug_rank <= 3
ORDER BY sample_id, drug_rank
LIMIT 10;

-- ========================================
-- TEXT EMBEDDINGS AVAILABILITY
-- ========================================
SELECT COUNT(*) AS total_chunks
FROM UNSTRUCTURED_DATA.TEXT_EMBEDDINGS_TABLE;

SELECT * 
FROM UNSTRUCTURED_DATA.TEXT_EMBEDDINGS_TABLE
LIMIT 5;

-- ========================================
-- AVAILABLE VIEWS AND TABLES
-- ========================================
SHOW VIEWS IN DBT_STAGING;
SHOW TABLES IN DBT_STAGING;