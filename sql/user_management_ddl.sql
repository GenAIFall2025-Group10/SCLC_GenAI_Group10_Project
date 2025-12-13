-- ============================================================================
-- SCHEMA: USER_MGMT
-- Description: User authentication, authorization, and configuration management
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Table: USERS
-- Description: Stores user account information for authentication and authorization
-- Purpose: Manages user credentials, roles, and account status
-- ----------------------------------------------------------------------------
CREATE OR REPLACE TABLE ONCODETECT_DB.USER_MGMT.USERS (
    USER_ID NUMBER(38,0) NOT NULL AUTOINCREMENT START 1 INCREMENT 1 NOORDER,
    FIRST_NAME VARCHAR(100) NOT NULL,
    LAST_NAME VARCHAR(100) NOT NULL,
    DATE_OF_BIRTH DATE NOT NULL,
    EMAIL VARCHAR(255) NOT NULL,
    USERNAME VARCHAR(50) NOT NULL,
    PASSWORD_HASH VARCHAR(255) NOT NULL,
    CREATED_AT TIMESTAMP_NTZ(9) DEFAULT CURRENT_TIMESTAMP(),
    LAST_LOGIN TIMESTAMP_NTZ(9),
    IS_ACTIVE BOOLEAN DEFAULT TRUE,
    USER_ROLE VARCHAR(20) DEFAULT 'user',
    CONSTRAINT PK_USERS PRIMARY KEY (USER_ID),
    CONSTRAINT UK_USERNAME UNIQUE (USERNAME),
    CONSTRAINT UK_EMAIL UNIQUE (EMAIL)
);

-- ----------------------------------------------------------------------------
-- Table: SERP_CONFIG
-- Description: Configuration table for API keys and system settings
-- Purpose: Securely stores API credentials and application configuration
-- ----------------------------------------------------------------------------
CREATE OR REPLACE TABLE ONCODETECT_DB.USER_MGMT.SERP_CONFIG (
    CONFIG_KEY VARCHAR(100) NOT NULL,
    CONFIG_VALUE VARCHAR(500),
    CREATED_AT TIMESTAMP_NTZ(9) DEFAULT CURRENT_TIMESTAMP(),
    UPDATED_AT TIMESTAMP_NTZ(9),
    DESCRIPTION VARCHAR(500),
    CONSTRAINT PK_SERP_CONFIG PRIMARY KEY (CONFIG_KEY)
);

COMMENT ON TABLE ON
