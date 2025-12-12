-- ========================================
-- LLM ANALYSIS FUNCTIONS
-- ========================================
-- Purpose: Snowflake Cortex LLM functions for clinical interpretation


USE DATABASE ONCODETECT_DB;
USE WAREHOUSE SCLC_WH;

-- ========================================
-- RISK ANALYSIS WITH LLM
-- ========================================
CREATE OR REPLACE FUNCTION ANALYZE_RISK_WITH_LLM(
    risk_score FLOAT,
    age FLOAT,
    tmb FLOAT,
    mutations FLOAT,
    is_male INT
)
RETURNS VARCHAR
AS
$$
    SELECT SNOWFLAKE.CORTEX.COMPLETE(
        'llama3.1-405b',
        CONCAT(
            'You are an oncology expert analyzing Small Cell Lung Cancer patients.\n\n',
            'PATIENT DATA:\n',
            '- Risk Score: ', risk_score, '/100\n',
            '- Age: ', age, ' years\n',
            '- Sex: ', CASE WHEN is_male = 1 THEN 'Male' ELSE 'Female' END, '\n',
            '- TMB: ', tmb, ' mutations/Mb\n',
            '- Total Mutations: ', mutations, '\n\n',
            'Provide a concise clinical assessment (4-5 sentences) with:\n',
            '1. Risk interpretation\n',
            '2. Key contributing factors\n',
            '3. Treatment approach recommendation\n',
            '4. Prognosis estimate'
        )
    ) AS analysis
$$;

-- ========================================
-- SUBTYPE ANALYSIS WITH LLM
-- ========================================
CREATE OR REPLACE FUNCTION ANALYZE_SUBTYPE_WITH_LLM(
    predicted_subtype VARCHAR,
    confidence FLOAT,
    ascl1_expression FLOAT,
    neurod1_expression FLOAT,
    pou2f3_expression FLOAT,
    yap1_expression FLOAT,
    tp53_expression FLOAT,
    rb1_expression FLOAT,
    myc_family_score FLOAT,
    ne_score FLOAT,
    non_ne_score FLOAT
)
RETURNS VARCHAR
AS
$$
    SELECT SNOWFLAKE.CORTEX.COMPLETE(
        'llama3.1-405b',
        CONCAT(
            'You are an expert in Small Cell Lung Cancer molecular subtypes (Rudin et al. 2019 classification).\n\n',
            'PREDICTED SUBTYPE: ', predicted_subtype, ' (Confidence: ', ROUND(confidence * 100, 1), '%)\n\n',
            'BIOMARKER EXPRESSION:\n',
            '- ASCL1: ', ascl1_expression, ' (Classic neuroendocrine marker)\n',
            '- NEUROD1: ', neurod1_expression, ' (Alternative neuroendocrine marker)\n',
            '- POU2F3: ', pou2f3_expression, ' (Tuft-like cell marker)\n',
            '- YAP1: ', yap1_expression, ' (Non-neuroendocrine/MYC-driven marker)\n',
            '- TP53: ', tp53_expression, '\n',
            '- RB1: ', rb1_expression, '\n',
            '- MYC Family Score: ', myc_family_score, '\n',
            '- Neuroendocrine Score: ', ne_score, '\n',
            '- Non-NE Score: ', non_ne_score, '\n\n',
            'Provide a structured clinical report with exactly 5 sections:\n\n',
            '1. SUBTYPE INTERPRETATION:\n',
            '[Explain why this ', predicted_subtype, ' classification is appropriate based on the biomarker pattern in 4-5 sentences]\n\n',
            '2. KEY BIOMARKER FINDINGS:\n',
            '[Highlight the dominant markers and their clinical significance in 4-5 sentences]\n\n',
            '3. DIFFERENTIAL DIAGNOSIS:\n',
            '[Explain why the other three subtypes (SCLC-A, SCLC-N, SCLC-P, SCLC-Y) are less likely in 4-5 sentences]\n\n',
            '4. THERAPEUTIC IMPLICATIONS:\n',
            '[Recommend specific treatment approaches for this subtype including targeted therapies in 4-5 sentences]\n\n',
            '5. PROGNOSIS:\n',
            '[Describe expected clinical behavior and outcomes for this subtype in 3-4 sentences]\n\n',
            'Be concise, clinical, and reference Rudin et al. 2019 when appropriate.'
        )
    ) AS analysis
$$;

-- ========================================
-- CANCER CENTER LOCATOR WITH LLM
-- ========================================
CREATE OR REPLACE FUNCTION FIND_CANCER_CENTERS_WITH_LOCATIONS(
    patient_city VARCHAR,
    patient_state VARCHAR,
    risk_level VARCHAR
)
RETURNS VARCHAR
AS
$$
    SELECT SNOWFLAKE.CORTEX.COMPLETE(
        'llama3.1-405b',
        'List 5 real major cancer treatment centers near ' || patient_city || ', ' || patient_state || ' for Small Cell Lung Cancer.

For ' || risk_level || ' patients, prioritize NCI-designated centers with SCLC clinical trials.

For EACH center, provide EXACTLY this format:

**CENTER_NAME**
📍 Full Address: [Street address, City, State ZIP]
🏥 Type: [NCI-Designated / Academic Medical Center / etc]
🎯 SCLC Specialties: [List 2-3 key specialties]
🧪 Clinical Trials: [Yes/No - Active SCLC trials]
💊 Immunotherapy: [Available programs]
📞 Phone: [Main number if known, or write "Contact via website"]
Why Recommended for ' || risk_level || ':
[2-3 sentences explaining why this center is good for this risk level]

---

List 5 centers. Use real center names, real addresses. Be specific and accurate.'
    )
$$;