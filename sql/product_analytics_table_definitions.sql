-- ============================================================================
-- Database: ONCODETECT_DB
-- Schema: PRODUCT_ANALYTICS
-- ============================================================================

-- ============================================================================
-- TABLE 1: SCLC_USER_AUTHENTICATION
-- Purpose: Stores user authentication records and related metadata for SCLC
--          (Small Cell Lung Cancer) application users. Uses VARIANT data 
--          types for flexible schema to accommodate various authentication 
--          methods and metadata structures.
-- Use Cases:
--   - Track user login events and authentication attempts
--   - Store authentication metadata (timestamps, IP addresses, methods)
--   - Audit user access patterns
-- ============================================================================
CREATE OR REPLACE TABLE ONCODETECT_DB.PRODUCT_ANALYTICS.SCLC_USER_AUTHENTICATION (
    RECORD_METADATA VARIANT,
    RECORD_CONTENT VARIANT
);

-- ============================================================================
-- TABLE 2: TABLE_METADATA
-- Purpose: Maintains a searchable catalog of database tables with semantic
--          embeddings for AI-powered search and discovery. Stores table 
--          descriptions and column information with vector embeddings for
--          natural language queries.
-- Use Cases:
--   - Enable semantic search across database schema
--   - Power AI chatbots for data discovery
--   - Document data catalog and data dictionary
--   - Support automated data governance
--   - Facilitate vector similarity search for table recommendations
-- ============================================================================
CREATE OR REPLACE TABLE ONCODETECT_DB.PRODUCT_ANALYTICS.TABLE_METADATA (
    TABLE_NAME VARCHAR(16777216),
    TABLE_SCHEMA VARCHAR(16777216),
    FULL_TABLE_PATH VARCHAR(16777216),
    DESCRIPTION VARCHAR(16777216),
    COLUMNS_INFO VARCHAR(16777216),
    METADATA_EMBEDDING VECTOR(FLOAT, 768)
);

-- ============================================================================
-- TABLE 3: USER_FEEDBACK
-- Purpose: Captures user feedback and satisfaction ratings for the product.
--          Includes auto-incrementing primary key and timestamps for tracking
--          feedback trends over time.
-- Use Cases:
--   - Collect user satisfaction metrics
--   - Analyze product feedback and sentiment
--   - Track user experience improvements
--   - Generate reports on user engagement
--   - Monitor feedback trends by user and time period
-- ============================================================================
CREATE OR REPLACE TABLE ONCODETECT_DB.PRODUCT_ANALYTICS.USER_FEEDBACK (
    FEEDBACK_ID NUMBER(38,0) NOT NULL AUTOINCREMENT START 1 INCREMENT 1 NOORDER,
    USER_ID NUMBER(38,0) NOT NULL,
    USERNAME VARCHAR(100) NOT NULL,
    FEEDBACK_TEXT VARCHAR(50) NOT NULL,
    RATING NUMBER(38,0) NOT NULL,
    FEEDBACK_DATE TIMESTAMP_NTZ(9) DEFAULT CURRENT_TIMESTAMP(),
    CREATED_AT TIMESTAMP_NTZ(9) DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (FEEDBACK_ID)
);

-- ============================================================================
-- TABLE 4: RAG_EVALUATION_LOGS
-- Purpose: Acts as the central observability log for the RAG (Retrieval-Augmented
--          Generation) pipeline. It captures every runtime interaction, storing
--          the full trace from user question to generated answer, alongside the
--          specific contexts retrieved. Crucially, it stores computed quality metrics
--          (Faithfulness, Relevancy) to quantify model performance and hallucination rates.
-- Use Cases:
--   - Monitor RAG pipeline performance and latency in real-time
--   - Detect and debug hallucinations (Low Faithfulness Score)
--   - Analyze retrieval quality by inspecting retrieved contexts vs. answers
--   - Track specific metrics like Correctness and Relevancy over time
--   - Identify "drifting" queries where model performance is degrading
--   - A/B test different LLM models or prompt strategies using score comparison
-- ============================================================================
CREATE OR REPLACE TABLE ONCODETECT_DB.PRODUCT_ANALYTICS.RAG_EVALUATION_LOGS (
    LOG_ID VARCHAR(50),
    TIMESTAMP TIMESTAMP_NTZ(9),
    USER_ID VARCHAR(50),
    QUESTION VARCHAR(16777216),
    GENERATED_ANSWER VARCHAR(16777216),
    RETRIEVED_CONTEXTS ARRAY,
    FAITHFULNESS_SCORE FLOAT,
    RELEVANCY_SCORE FLOAT,
    CONTEXT_QUALITY_SCORE FLOAT,
    OVERALL_SCORE FLOAT,
    CORRECTNESS_SCORE FLOAT
);

-- ============================================================================
-- TABLE 5: GOLDEN_DATASET
-- Purpose: Serves as the "Ground Truth" reference library for automated model
--          evaluation. It stores verified question-answer pairs that define
--          ideal behavior. The inclusion of vector embeddings allows the system
--          to semantically match incoming user queries against this dataset to
--          automatically grade answers or provide few-shot examples.
-- Use Cases:
--   - Run regression testing before deploying new RAG pipeline versions
--   - Calculate "Correctness" scores by comparing generated answers to Ground Truth
--   - Enable semantic search to find similar historical questions
--   - Bootstrapping few-shot prompting context for the LLM
--   - Establish a baseline benchmark for model accuracy
-- ============================================================================
CREATE OR REPLACE TABLE ONCODETECT_DB.PRODUCT_ANALYTICS.GOLDEN_DATASET (
    GOLDEN_ID VARCHAR(50) DEFAULT UUID_STRING(),
    QUESTION VARCHAR(16777216),
    GROUND_TRUTH_ANSWER VARCHAR(16777216),
    QUESTION_EMBEDDING VECTOR(FLOAT, 768)
);

-- ============================================================================
-- VIEW 1: USER_ACQUISITION_EVENTS
-- Purpose: Transforms semi-structured authentication data into a structured
--          view for analyzing user acquisition and interaction events. Extracts
--          key fields from VARIANT columns and adds Kafka ingestion timestamps
--          for data pipeline monitoring.
-- Use Cases:
--   - Track user acquisition funnel and conversion events
--   - Analyze button clicks and user interactions
--   - Monitor authentication status and event types
--   - Distinguish between different user types and sources
--   - Calculate data latency between app events and ingestion
--   - Support real-time analytics on user behavior
-- Source Table: SCLC_USER_AUTHENTICATION
-- ============================================================================
CREATE OR REPLACE VIEW ONCODETECT_DB.PRODUCT_ANALYTICS.USER_ACQUISITION_EVENTS(
    EVENT_ID,
    USER_ID,
    SESSION_ID,
    EVENT_TYPE,
    BUTTON_NAME,
    STATUS,
    USER_TYPE,
    SOURCE,
    APP_TIMESTAMP,
    KAFKA_INGEST_TIMESTAMP
) AS
SELECT 
    RECORD_CONTENT:event_id::STRING AS event_id,
    RECORD_CONTENT:user_id::STRING AS user_id,
    RECORD_CONTENT:session_id::STRING AS session_id,
    RECORD_CONTENT:event_type::STRING AS event_type,
    RECORD_CONTENT:button_name::STRING AS button_name,
    RECORD_CONTENT:status::STRING AS status,
    RECORD_CONTENT:user_type::STRING AS user_type,
    RECORD_CONTENT:source::STRING AS source,
    RECORD_CONTENT:timestamp::TIMESTAMP AS app_timestamp,
    TO_TIMESTAMP_NTZ(RECORD_METADATA:CreateTime::INT / 1000) AS kafka_ingest_timestamp
FROM ONCODETECT_DB.PRODUCT_ANALYTICS.SCLC_USER_AUTHENTICATION;

-- ============================================================================
-- Notes:
-- - All tables are in the ONCODETECT_DB database under PRODUCT_ANALYTICS schema
-- - SCLC_USER_AUTHENTICATION uses flexible VARIANT types for semi-structured data
-- - TABLE_METADATA includes 768-dimensional vector embeddings (typical for models
--   like sentence-transformers or OpenAI embeddings)
-- - USER_FEEDBACK has constraints and auto-increment for data integrity
-- ============================================================================
