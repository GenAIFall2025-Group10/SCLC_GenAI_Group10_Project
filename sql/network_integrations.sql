-- ========================================
-- NETWORK RULES & EXTERNAL INTEGRATIONS
-- ========================================
-- Purpose: External access setup for Google Maps and other APIs


USE DATABASE ONCODETECT_DB;
USE SCHEMA PUBLIC;

-- ========================================
-- GOOGLE MAPS NETWORK RULE
-- ========================================
-- Switch to ACCOUNTADMIN to create network rules
USE ROLE ACCOUNTADMIN;

-- Create network rule to allow Google Maps access
CREATE OR REPLACE NETWORK RULE google_maps_network_rule
    MODE = EGRESS
    TYPE = HOST_PORT
    VALUE_LIST = (
        'www.google.com:443',
        'maps.google.com:443',
        'maps.googleapis.com:443',
        'maps.gstatic.com:443',
        'google.com:443'
    );

-- Verify network rule was created
SHOW NETWORK RULES LIKE '%google%';

-- Get detailed information about the network rule
SELECT SYSTEM$GET_NETWORK_RULE_DETAILS('ONCODETECT_DB.PUBLIC.GOOGLE_MAPS_NETWORK_RULE');

-- ========================================
-- EXTERNAL ACCESS INTEGRATION
-- ========================================
-- Create external access integration
CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION google_maps_access_integration
    ALLOWED_NETWORK_RULES = (google_maps_network_rule)
    ENABLED = TRUE;

-- Verify integration was created
SHOW INTEGRATIONS;

-- Describe the integration
DESC INTEGRATION GOOGLE_MAPS_ACCESS_INTEGRATION;

-- ========================================
-- GRANT PERMISSIONS
-- ========================================
-- Grant usage to TRAINING_ROLE
GRANT USAGE ON INTEGRATION google_maps_access_integration TO ROLE TRAINING_ROLE;

-- Verify grants
SHOW GRANTS ON INTEGRATION GOOGLE_MAPS_ACCESS_INTEGRATION;

-- ========================================
-- UPDATE STREAMLIT APP (if applicable)
-- ========================================
-- Switch back to TRAINING_ROLE
USE ROLE TRAINING_ROLE;

-- Update Streamlit app with external access
-- ALTER STREAMLIT ONCODETECT_AI_SCLC_Predicton
--     SET EXTERNAL_ACCESS_INTEGRATIONS = (GOOGLE_MAPS_ACCESS_INTEGRATION);

-- ========================================
-- VERIFICATION QUERIES
-- ========================================
-- Check network rules
SHOW NETWORK RULES;

-- Describe network rule configuration
SHOW NETWORK RULES LIKE 'GOOGLE_MAPS_NETWORK_RULE' IN DATABASE ONCODETECT_DB;

-- Check Streamlit apps
-- SHOW STREAMLITS;