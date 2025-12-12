-- ============================================================================
-- Database: ONCODETECT_DB
-- Schema: UNSTRUCTURED_DATA
-- ============================================================================

-- ============================================================================
-- TABLE 1: IMAGES_PROCESSED_PAPERS
-- Purpose: Tracks which research papers have completed image extraction and
--          processing. Serves as a processing checkpoint to prevent duplicate
--          image extraction operations.
-- Use Cases:
--   - Monitor image extraction pipeline progress
--   - Identify papers pending image processing
--   - Track total images extracted per paper
--   - Prevent reprocessing of already-extracted images
--   - Generate processing statistics and metrics
-- ============================================================================
CREATE OR REPLACE TABLE ONCODETECT_DB.UNSTRUCTURED_DATA.IMAGES_PROCESSED_PAPERS (
    PAPER_ID VARCHAR(16777216) NOT NULL,
    PROCESSED_AT TIMESTAMP_NTZ(9) DEFAULT CURRENT_TIMESTAMP(),
    IMAGES_COUNT NUMBER(38,0),
    PRIMARY KEY (PAPER_ID)
);

-- ============================================================================
-- TABLE 2: IMAGE_EMBEDDINGS_TABLE
-- Purpose: Stores vector embeddings for images extracted from research papers,
--          enabling semantic search and similarity matching across scientific
--          figures, charts, and diagrams. Uses 1024-dimensional embeddings
--          (typical for vision models like CLIP).
-- Use Cases:
--   - Semantic search for similar figures across papers
--   - Find related visualizations and diagrams
--   - Image-based research paper recommendations
--   - Duplicate or similar figure detection
--   - Visual analysis and clustering of scientific imagery
--   - Multimodal search (text-to-image queries)
-- ============================================================================
CREATE OR REPLACE TABLE ONCODETECT_DB.UNSTRUCTURED_DATA.IMAGE_EMBEDDINGS_TABLE (
    PAPER_ID VARCHAR(16777216),
    PAGE_NUMBER NUMBER(38,0),
    IMAGE_INDEX NUMBER(38,0),
    S3_URL VARCHAR(16777216),
    IMAGE_HASH VARCHAR(16777216),
    WIDTH NUMBER(38,0),
    HEIGHT NUMBER(38,0),
    FORMAT VARCHAR(16777216),
    CREATED_AT TIMESTAMP_NTZ(9),
    EMBEDDING VECTOR(FLOAT, 1024)
);

-- ============================================================================
-- TABLE 3: PARSED_IMAGES_METADATA
-- Purpose: Master metadata catalog for all images extracted from research
--          papers. Uses unique image hash constraint to prevent duplicate
--          storage and tracks physical image properties and S3 locations.
-- Use Cases:
--   - Central registry of extracted images
--   - Deduplicate identical images across papers
--   - Track image storage locations in S3
--   - Audit image extraction processes
--   - Reference lookup for image properties
--   - Support image retrieval and display
-- ============================================================================
CREATE OR REPLACE TABLE ONCODETECT_DB.UNSTRUCTURED_DATA.PARSED_IMAGES_METADATA (
    ID NUMBER(38,0) NOT NULL AUTOINCREMENT START 1 INCREMENT 1 NOORDER,
    PAPER_ID VARCHAR(255),
    PAGE_NUMBER NUMBER(38,0),
    IMAGE_INDEX NUMBER(38,0),
    IMAGE_HASH VARCHAR(64) NOT NULL,
    S3_URL VARCHAR(500),
    WIDTH NUMBER(38,0),
    HEIGHT NUMBER(38,0),
    FORMAT VARCHAR(10),
    EXTRACTED_AT TIMESTAMP_NTZ(9) DEFAULT CURRENT_TIMESTAMP(),
    UNIQUE (IMAGE_HASH),
    PRIMARY KEY (ID)
);

-- ============================================================================
-- TABLE 4: RESEARCH_NOTES
-- Purpose: Stores user research sessions and chat histories when interacting
--          with the research assistant. Maintains conversation context and
--          allows users to save and revisit research queries.
-- Use Cases:
--   - Save and organize research sessions
--   - Track user interactions with AI assistant
--   - Resume previous research conversations
--   - Analyze user research patterns and behaviors
--   - Generate usage metrics and engagement statistics
--   - Support collaborative research note-taking
-- Foreign Keys:
--   - USER_ID references USERS table in USER_MGMT schema
-- ============================================================================
CREATE OR REPLACE TABLE ONCODETECT_DB.UNSTRUCTURED_DATA.RESEARCH_NOTES (
    NOTE_ID NUMBER(38,0) NOT NULL AUTOINCREMENT START 1 INCREMENT 1 NOORDER,
    USER_ID NUMBER(38,0) NOT NULL,
    USERNAME VARCHAR(50),
    NOTE_TITLE VARCHAR(500),
    CHAT_HISTORY VARIANT,
    TOTAL_QUERIES NUMBER(38,0),
    CREATED_AT TIMESTAMP_NTZ(9) DEFAULT CURRENT_TIMESTAMP(),
    LAST_VIEWED TIMESTAMP_NTZ(9),
    PRIMARY KEY (NOTE_ID),
    FOREIGN KEY (USER_ID) REFERENCES ONCODETECT_DB.USER_MGMT.USERS(USER_ID)
);

-- ============================================================================
-- TABLE 5: PUBMED_PAPERS_METADATA
-- Purpose: Central repository of metadata for PubMed research papers including
--          titles, abstracts, authors, and references to full-text PDFs stored
--          in S3. Serves as the master catalog for the research paper corpus.
-- Use Cases:
--   - Search and browse research paper catalog
--   - Link to full-text PDFs in S3
--   - Track paper citations and references (via DOI)
--   - Support paper recommendation systems
--   - Enable metadata-based filtering and discovery
--   - Integrate with PubMed Central (PMC) identifiers
-- ============================================================================
CREATE OR REPLACE TABLE ONCODETECT_DB.UNSTRUCTURED_DATA.PUBMED_PAPERS_METADATA (
    TITLE VARCHAR(2000),
    ABSTRACT VARCHAR(16777216),
    AUTHORS VARCHAR(5000),
    DOI VARCHAR(200),
    S3_URL VARCHAR(500),
    PMC_ID VARCHAR(50)
);

-- ============================================================================
-- TABLE 6: TEXT_EMBEDDINGS_TABLE
-- Purpose: Stores vector embeddings for text chunks extracted from research
--          papers, enabling semantic search across scientific literature. Uses
--          768-dimensional embeddings (typical for models like sentence-
--          transformers) and tracks chunk boundaries for precise citation.
-- Use Cases:
--   - Semantic search across research paper content
--   - Find relevant passages for user queries
--   - Support Retrieval-Augmented Generation (RAG) workflows
--   - Citation and source attribution
--   - Cross-paper similarity analysis
--   - Research question answering with context
--   - Identify duplicate or overlapping content
-- ============================================================================
CREATE OR REPLACE TABLE ONCODETECT_DB.UNSTRUCTURED_DATA.TEXT_EMBEDDINGS_TABLE (
    ID NUMBER(38,0) NOT NULL AUTOINCREMENT START 1 INCREMENT 1 NOORDER,
    PAPER_ID VARCHAR(255) NOT NULL,
    CHUNK_INDEX NUMBER(38,0) NOT NULL,
    CHUNK_TEXT VARCHAR(16777216) NOT NULL,
    CHUNK_HASH VARCHAR(64) NOT NULL,
    START_CHAR NUMBER(38,0),
    END_CHAR NUMBER(38,0),
    START_PAGE NUMBER(38,0),
    END_PAGE NUMBER(38,0),
    PAGES_SPANNED VARIANT,
    CONTENT_HASH VARCHAR(64),
    CREATED_AT TIMESTAMP_NTZ(9) DEFAULT CURRENT_TIMESTAMP(),
    EMBEDDING VECTOR(FLOAT, 768),
    PRIMARY KEY (ID)
);

-- ============================================================================
-- TABLE 7: TEXT_PROCESSED_PAPERS
-- Purpose: Tracks which research papers have completed text extraction and
--          chunking. Serves as a processing checkpoint to prevent duplicate
--          text processing operations.
-- Use Cases:
--   - Monitor text extraction pipeline progress
--   - Identify papers pending text processing
--   - Track total text chunks extracted per paper
--   - Prevent reprocessing of already-chunked papers
--   - Generate processing statistics and metrics
--   - Coordinate with image processing pipeline
-- ============================================================================
CREATE OR REPLACE TABLE ONCODETECT_DB.UNSTRUCTURED_DATA.TEXT_PROCESSED_PAPERS (
    PAPER_ID VARCHAR(255) NOT NULL,
    PROCESSED_AT TIMESTAMP_NTZ(9) DEFAULT CURRENT_TIMESTAMP(),
    TEXT_CHUNKS_COUNT NUMBER(38,0),
    PRIMARY KEY (PAPER_ID)
);

-- ============================================================================
-- Schema Overview:
-- This schema supports a complete research paper processing and semantic
-- search system with the following workflow:
--
-- 1. Papers are cataloged in PUBMED_PAPERS_METADATA
-- 2. Text is extracted, chunked, and tracked in TEXT_PROCESSED_PAPERS
-- 3. Text embeddings are stored in TEXT_EMBEDDINGS_TABLE for semantic search
-- 4. Images are extracted and tracked in IMAGES_PROCESSED_PAPERS
-- 5. Image metadata is stored in PARSED_IMAGES_METADATA (deduplicated)
-- 6. Image embeddings enable visual search via IMAGE_EMBEDDINGS_TABLE
-- 7. Users interact via RESEARCH_NOTES to save research sessions
--
-- The system enables:
-- - Multimodal search (text + images)
-- - RAG-based question answering
-- - Research paper recommendations
-- - Visual and textual similarity detection
-- ============================================================================