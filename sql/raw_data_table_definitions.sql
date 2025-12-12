-- ========================================
-- TABLE DEFINITIONS
-- ========================================
-- Purpose: Raw data table schemas for OncoDetect-AI

USE DATABASE ONCODETECT_DB;
USE WAREHOUSE SCLC_WH;
USE SCHEMA RAW_DATA;

-- ========================================
-- CLINICAL DATA TABLE
-- ========================================
CREATE OR REPLACE TABLE CLINICAL_DATA (
    study_id VARCHAR(100),
    patient_id VARCHAR(100),
    sample_id VARCHAR(100),
    oncotree_code VARCHAR(50),
    mutation_count NUMBER,
    tmb_nonsynonymous FLOAT,
    overall_survival_months FLOAT,
    overall_survival_status VARCHAR(50),
    sex VARCHAR(20),
    diagnosis_age NUMBER,
    smoker VARCHAR(100),
    smoking_history VARCHAR(500),
    stage_at_diagnosis VARCHAR(50),
    t_stage VARCHAR(20),
    n_stage VARCHAR(20),
    m_stage VARCHAR(20)
);

-- ========================================
-- GENE EXPRESSION TABLE
-- ========================================
CREATE OR REPLACE TABLE GENE_EXPRESSION (
    gene_symbol VARCHAR(100),
    sample_id VARCHAR(100),
    expression_value FLOAT
);

-- ========================================
-- COMPOUNDS TABLE
-- ========================================
CREATE OR REPLACE TABLE COMPOUNDS (
    nsc VARCHAR(100),
    cas VARCHAR(100),
    drug_name VARCHAR(500)
);

-- ========================================
-- CELL LINES TABLE
-- ========================================
CREATE OR REPLACE TABLE CELL_LINES (
    cell_line VARCHAR(100),
    panel VARCHAR(100)
);

-- ========================================
-- EXPERIMENTS TABLE
-- ========================================
CREATE OR REPLACE TABLE EXPERIMENTS (
    experiment_id VARCHAR(100),
    nsc VARCHAR(100),
    highest_tested_concentration VARCHAR(100),
    cell_line VARCHAR(100),
    log_ec50 VARCHAR(100),
    response_zero VARCHAR(100),
    response_inflection VARCHAR(100),
    hill_slope VARCHAR(100),
    log_ic50 VARCHAR(100),
    flag_ic50 VARCHAR(50)
);

-- ========================================
-- DOSE RESPONSE TABLE
-- ========================================
CREATE OR REPLACE TABLE DOSE_RESPONSE (
    experiment_id NUMBER,
    concentration FLOAT,
    mean_pct_ctrl FLOAT,
    stddev_pct_ctrl FLOAT,
    count_pct_ctrl NUMBER
);


