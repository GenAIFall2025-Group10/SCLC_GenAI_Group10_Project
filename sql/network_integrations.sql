-- ============================================================================
-- NETWORK RULES & EXTERNAL ACCESS INTEGRATIONS
-- Database: ONCODETECT_DB
-- Schema: PUBLIC
-- ============================================================================
-- Purpose: Configure external access for Google Maps, arXiv, Confluent Cloud,
--          and SERP API integrations. These integrations enable Snowflake
--          functions and Streamlit apps to make external API calls.
-- ============================================================================

USE DATABASE ONCODETECT_DB;
USE SCHEMA PUBLIC;

-- Switch to ACCOUNTADMIN role (required for creating network rules and integrations)
USE ROLE ACCOUNTADMIN;

-- ============================================================================
-- INTEGRATION 1: GOOGLE MAPS API
-- Purpose: Enable access to Google Maps services for geocoding, location
--          search, and mapping functionality
-- ============================================================================

-- Step 1: Create network rule for Google Maps domains
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

-- Step 2: Create external access integration
CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION google_maps_access_integration
    ALLOWED_NETWORK_RULES = (google_maps_network_rule)
    ENABLED = TRUE;

-- Step 3: Grant usage to TRAINING_ROLE
GRANT USAGE ON INTEGRATION google_maps_access_integration TO ROLE TRAINING_ROLE;

-- ============================================================================
-- INTEGRATION 2: ARXIV API
-- Purpose: Enable access to arXiv.org for fetching research papers and
--          academic publications
-- ============================================================================

-- Step 1: Create network rule for arXiv domains
CREATE OR REPLACE NETWORK RULE arxiv_network_rule
    MODE = EGRESS
    TYPE = HOST_PORT
    VALUE_LIST = (
        'export.arxiv.org:443',
        'export.arxiv.org:80'
    );

-- Step 2: Create external access integration
CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION arxiv_access_integration
    ALLOWED_NETWORK_RULES = (arxiv_network_rule)
    ENABLED = TRUE;

-- Step 3: Grant usage to TRAINING_ROLE
GRANT USAGE ON INTEGRATION arxiv_access_integration TO ROLE TRAINING_ROLE;

-- ============================================================================
-- INTEGRATION 3: CONFLUENT CLOUD (KAFKA)
-- Purpose: Enable access to Confluent Cloud REST API for Kafka streaming
--          data integration and management
-- ============================================================================

-- Step 1: Create network rule for Confluent Cloud
-- NOTE: Replace 'YOUR_CONFLUENT_CLOUD_REST_API_URL' with your actual 
--       Confluent Cloud endpoint (e.g., pkc-12345.us-east-1.aws.confluent.cloud:443)
CREATE OR REPLACE NETWORK RULE confluent_network_rule
    MODE = EGRESS
    TYPE = HOST_PORT
    VALUE_LIST = ('YOUR_CONFLUENT_CLOUD_REST_API_URL:443');

-- Step 2: Create external access integration
CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION confluent_access_integration
    ALLOWED_NETWORK_RULES = (confluent_network_rule)
    ENABLED = TRUE;

-- Step 3: Grant usage to TRAINING_ROLE
GRANT USAGE ON INTEGRATION confluent_access_integration TO ROLE TRAINING_ROLE;

-- ============================================================================
-- INTEGRATION 4: SERP API (WEB SEARCH)
-- Purpose: Enable web search capabilities through SerpAPI for research and
--          data enrichment functionality
-- ============================================================================

-- Step 1: Create network rule for SERP API domains
CREATE OR REPLACE NETWORK RULE serp_web_search_rule
    MODE = EGRESS
    TYPE = HOST_PORT
    VALUE_LIST = (
        'serpapi.com:443',
        'www.serpapi.com:443'
    );

-- Step 2: Create external access integration
CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION serp_web_search_integration
    ALLOWED_NETWORK_RULES = (serp_web_search_rule)
    ENABLED = TRUE;

-- Step 3: Grant usage to TRAINING_ROLE
GRANT USAGE ON INTEGRATION serp_web_search_integration TO ROLE TRAINING_ROLE;

-- Step 4: Grant database and schema access for SERP configuration
GRANT USAGE ON DATABASE ONCODETECT_DB TO ROLE TRAINING_ROLE;
GRANT USAGE ON SCHEMA ONCODETECT_DB.USER_MGMT TO ROLE TRAINING_ROLE;
GRANT SELECT ON TABLE ONCODETECT_DB.USER_MGMT.SERP_CONFIG TO ROLE TRAINING_ROLE;

-- ============================================================================
-- VERIFICATION & VALIDATION
-- Execute these statements to verify all integrations are properly configured
-- ============================================================================

-- ============================================================================
-- 1. VERIFY NETWORK RULES EXIST
-- ============================================================================
SHOW NETWORK RULES;
-- Expected: google_maps_network_rule, arxiv_network_rule, 
--           confluent_network_rule, serp_web_search_rule

-- Get detailed information about each network rule
SELECT SYSTEM$GET_NETWORK_RULE_DETAILS('ONCODETECT_DB.PUBLIC.GOOGLE_MAPS_NETWORK_RULE');
SELECT SYSTEM$GET_NETWORK_RULE_DETAILS('ONCODETECT_DB.PUBLIC.ARXIV_NETWORK_RULE');
SELECT SYSTEM$GET_NETWORK_RULE_DETAILS('ONCODETECT_DB.PUBLIC.CONFLUENT_NETWORK_RULE');
SELECT SYSTEM$GET_NETWORK_RULE_DETAILS('ONCODETECT_DB.PUBLIC.SERP_WEB_SEARCH_RULE');

-- ============================================================================
-- 2. VERIFY INTEGRATIONS EXIST AND ARE ENABLED
-- ============================================================================
SHOW INTEGRATIONS;
-- Expected: All four integrations should be listed

-- Describe each integration to verify configuration
DESC INTEGRATION google_maps_access_integration;
DESC INTEGRATION arxiv_access_integration;
DESC INTEGRATION confluent_access_integration;
DESC INTEGRATION serp_web_search_integration;
-- Expected: ENABLED = TRUE for all integrations

-- ============================================================================
-- 3. VERIFY GRANTS TO TRAINING_ROLE
-- ============================================================================

-- Check grants on each integration
SHOW GRANTS ON INTEGRATION google_maps_access_integration;
SHOW GRANTS ON INTEGRATION arxiv_access_integration;
SHOW GRANTS ON INTEGRATION confluent_access_integration;
SHOW GRANTS ON INTEGRATION serp_web_search_integration;
-- Expected: TRAINING_ROLE has USAGE privilege on all integrations

-- Check all grants to TRAINING_ROLE
SHOW GRANTS TO ROLE TRAINING_ROLE;
-- Expected: Should include USAGE on all four integrations, plus database/schema/table grants