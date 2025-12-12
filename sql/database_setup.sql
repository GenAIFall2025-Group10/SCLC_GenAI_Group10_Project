
-- ========================================
-- DATABASE & SCHEMA SETUP
-- ========================================
-- Initial database and schema creation for OncoDetect-AI

USE DATABASE ONCODETECT_DB;
USE WAREHOUSE SCLC_WH;

-- Create Schema
CREATE SCHEMA IF NOT EXISTS RAW_DATA;

-- ========================================
-- STAGE SETUP
-- ========================================

-- Create stage to store ML models
CREATE OR REPLACE STAGE ML_MODELS_STAGE
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Storage for OncoDetect-AI ML models';

-- Verify stage created
SHOW STAGES LIKE 'ML_MODELS_STAGE';

-- List files in stage
LIST @ML_MODELS_STAGE;