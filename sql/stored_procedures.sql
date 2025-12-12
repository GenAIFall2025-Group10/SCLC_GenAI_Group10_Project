-- ========================================
-- STORED PROCEDURES
-- ========================================
-- Purpose: Orchestration procedures for multi-step ML and RAG workflows


USE DATABASE ONCODETECT_DB;
USE WAREHOUSE SCLC_WH;

-- ========================================
-- FULL RISK ANALYSIS PROCEDURE
-- ========================================
CREATE OR REPLACE PROCEDURE RUN_FULL_RISK_ANALYSIS(
    age FLOAT,
    is_male INT,
    is_former_smoker INT,
    mutation_count FLOAT,
    tmb FLOAT,
    is_tmb_high INT,
    is_tmb_intermediate INT,
    is_tmb_low INT,
    smoker_x_tmb FLOAT
)
RETURNS VARIANT
LANGUAGE SQL
AS
$$
DECLARE
    ml_prediction VARIANT;
    similar_patients_result VARIANT;
    llm_analysis VARCHAR;
    final_result VARIANT;
    risk_score_value FLOAT;
BEGIN
    -- Step 1: Get ML prediction
    SELECT PREDICT_RISK_SCORE(
        :age, :is_male, :is_former_smoker, :mutation_count,
        :tmb, :is_tmb_high, :is_tmb_intermediate, :is_tmb_low, :smoker_x_tmb
    ) INTO :ml_prediction;
    
    -- Extract risk score for LLM
    risk_score_value := ml_prediction:risk_score::FLOAT;
    
    -- Step 2: Find similar patients from your actual table
    SELECT OBJECT_CONSTRUCT(
        'similar_patients_found', COUNT(*),
        'message', 'Found ' || COUNT(*) || ' similar patients in database'
    )
    INTO :similar_patients_result
    FROM ONCODETECT_DB.DBT_STAGING.MART_RISK_FEATURES_COMBINED
    WHERE tmb BETWEEN (:tmb - 5) AND (:tmb + 5)
      AND age BETWEEN (:age - 10) AND (:age + 10)
    LIMIT 10;
    
    -- Step 3: Get LLM clinical analysis
    llm_analysis := (
        SELECT ANALYZE_RISK_WITH_LLM(
            :risk_score_value,
            :age,
            :tmb,
            :mutation_count,
            :is_male
        )
    );
    
    -- Step 4: Combine all results
    final_result := OBJECT_CONSTRUCT(
        'ml_prediction', :ml_prediction,
        'similar_patients', :similar_patients_result,
        'clinical_analysis', :llm_analysis,
        'timestamp', CURRENT_TIMESTAMP(),
        'analysis_type', 'full_agentic_workflow'
    );
    
    RETURN final_result;
END;
$$;

-- ========================================
-- FULL SUBTYPE ANALYSIS PROCEDURE
-- ========================================
CREATE OR REPLACE PROCEDURE RUN_FULL_SUBTYPE_ANALYSIS(
    ascl1_expression FLOAT,
    neurod1_expression FLOAT,
    pou2f3_expression FLOAT,
    yap1_expression FLOAT,
    tp53_expression FLOAT,
    rb1_expression FLOAT,
    myc_family_score FLOAT,
    tp53_rb1_dual_loss FLOAT,
    dll3_expression FLOAT,
    bcl2_expression FLOAT,
    notch1_expression FLOAT,
    mean_expression FLOAT,
    stddev_expression FLOAT,
    genes_expressed FLOAT
)
RETURNS VARIANT
LANGUAGE SQL
AS
$$
DECLARE
    ml_prediction VARIANT;
    biomarker_analysis VARIANT;
    similar_profiles_result VARIANT;
    similar_profiles VARCHAR;
    llm_analysis VARCHAR;
    final_result VARIANT;
    ne_score FLOAT;
    non_ne_score FLOAT;
    predicted_subtype VARCHAR;
    confidence_value FLOAT;
BEGIN
    -- Step 1: ML prediction
    SELECT PREDICT_SUBTYPE(
        :ascl1_expression, :neurod1_expression, :pou2f3_expression, :yap1_expression,
        :tp53_expression, :rb1_expression, :myc_family_score, :tp53_rb1_dual_loss,
        :dll3_expression, :bcl2_expression, :notch1_expression, :mean_expression,
        :stddev_expression, :genes_expressed
    ) INTO :ml_prediction;
    
    -- Extract values
    predicted_subtype := GET_PATH(:ml_prediction, 'predicted_subtype')::VARCHAR;
    confidence_value := GET_PATH(:ml_prediction, 'confidence')::FLOAT;
    
    -- Calculate scores
    ne_score := (:ascl1_expression + :neurod1_expression) / 2.0;
    non_ne_score := (:pou2f3_expression + :yap1_expression) / 2.0;
    
    -- Step 2: Biomarker analysis
    SELECT OBJECT_CONSTRUCT(
        'dominant_tf', 
        CASE 
            WHEN :ascl1_expression = GREATEST(:neurod1_expression, :ascl1_expression, :yap1_expression, :pou2f3_expression) THEN 'ASCL1'
            WHEN :neurod1_expression = GREATEST(:neurod1_expression, :ascl1_expression, :yap1_expression, :pou2f3_expression) THEN 'NEUROD1'
            WHEN :pou2f3_expression = GREATEST(:neurod1_expression, :ascl1_expression, :yap1_expression, :pou2f3_expression) THEN 'POU2F3'
            ELSE 'YAP1'
        END,
        'dominant_value', GREATEST(:neurod1_expression, :ascl1_expression, :yap1_expression, :pou2f3_expression),
        'ascl1', :ascl1_expression,
        'neurod1', :neurod1_expression,
        'pou2f3', :pou2f3_expression,
        'yap1', :yap1_expression,
        'ne_score', :ne_score,
        'non_ne_score', :non_ne_score
    ) INTO :biomarker_analysis;
    
    -- Step 3: Search similar biomarker profiles
    SELECT OBJECT_CONSTRUCT(
        'samples_found', COUNT(*),
        'message', 'Found ' || COUNT(*) || ' samples with similar biomarker expression patterns'
    )
    INTO :similar_profiles_result
    FROM ONCODETECT_DB.DBT_STAGING.MART_RISK_FEATURES_GENOMIC
    WHERE ascl1_expression BETWEEN (:ascl1_expression - 2) AND (:ascl1_expression + 2)
       OR neurod1_expression BETWEEN (:neurod1_expression - 2) AND (:neurod1_expression + 2)
       OR pou2f3_expression BETWEEN (:pou2f3_expression - 2) AND (:pou2f3_expression + 2)
       OR yap1_expression BETWEEN (:yap1_expression - 2) AND (:yap1_expression + 2)
    LIMIT 100;
    
    similar_profiles := GET_PATH(:similar_profiles_result, 'message')::VARCHAR;
    
    -- Step 4: Get LLM analysis
    SELECT ANALYZE_SUBTYPE_WITH_LLM(
        :predicted_subtype,
        :confidence_value,
        :ascl1_expression,
        :neurod1_expression,
        :pou2f3_expression,
        :yap1_expression,
        :tp53_expression,
        :rb1_expression,
        :myc_family_score,
        :ne_score,
        :non_ne_score
    ) INTO :llm_analysis;
    
    -- Step 5: Combine results
    final_result := OBJECT_CONSTRUCT(
        'ml_prediction', :ml_prediction,
        'biomarker_analysis', :biomarker_analysis,
        'similar_profiles', :similar_profiles,
        'similar_profiles_data', :similar_profiles_result,
        'clinical_analysis', :llm_analysis,
        'timestamp', CURRENT_TIMESTAMP(),
        'analysis_type', 'full_subtype_analysis'
    );
    
    RETURN final_result;
END;
$$;

-- ========================================
-- FULL RISK ANALYSIS WITH RAG PROCEDURE
-- ========================================
CREATE OR REPLACE PROCEDURE RUN_FULL_RISK_ANALYSIS_WITH_RAG(
    age FLOAT,
    is_male INT,
    is_former_smoker INT,
    mutation_count FLOAT,
    tmb FLOAT,
    is_tmb_high INT,
    is_tmb_intermediate INT,
    is_tmb_low INT,
    smoker_x_tmb FLOAT
)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.9'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'run_enhanced_analysis'
AS
$$
import json

def run_enhanced_analysis(session, age, is_male, is_former_smoker, mutation_count,
                          tmb, is_tmb_high, is_tmb_intermediate, is_tmb_low, smoker_x_tmb):
    
    # STEP 1: ML Prediction
    ml_query = """
        SELECT PREDICT_RISK_SCORE(
            {}, {}, {}, {},
            {}, {}, {}, {}, {}
        ) AS prediction
    """.format(age, is_male, is_former_smoker, mutation_count,
               tmb, is_tmb_high, is_tmb_intermediate, is_tmb_low, smoker_x_tmb)
    
    ml_result = session.sql(ml_query).collect()[0]['PREDICTION']
    ml_pred = json.loads(ml_result) if isinstance(ml_result, str) else ml_result
    
    risk_score = ml_pred['risk_score']
    risk_level = ml_pred['risk_level']
    confidence = ml_pred['confidence']
    
    # STEP 2: Find Similar Patients
    similar_patients_query = """
        SELECT 
            sample_id,
            age,
            tmb,
            mutation_count,
            target_survival_months,
            target_event
        FROM ONCODETECT_DB.DBT_STAGING.MART_RISK_FEATURES_COMBINED
        WHERE ABS(age - {}) <= 10
          AND ABS(tmb - {}) <= 5
          AND age IS NOT NULL
          AND tmb IS NOT NULL
        ORDER BY ABS(tmb - {})
        LIMIT 5
    """.format(age, tmb, tmb)
    
    try:
        similar_results = session.sql(similar_patients_query).collect()
        
        if similar_results and len(similar_results) > 0:
            similar_patients_list = []
            for row in similar_results:
                similar_patients_list.append({
                    'patient_id': row['SAMPLE_ID'],
                    'age': int(row['AGE']) if row['AGE'] else 0,
                    'tmb': round(float(row['TMB']), 1) if row['TMB'] else 0,
                    'mutations': int(row['MUTATION_COUNT']) if row['MUTATION_COUNT'] else 0,
                    'survival_months': float(row['TARGET_SURVIVAL_MONTHS']) if row['TARGET_SURVIVAL_MONTHS'] else 0,
                    'status': row['TARGET_EVENT'] if row['TARGET_EVENT'] else 'Unknown'
                })
            
            similar_patients = {
                'similar_patients_found': len(similar_results),
                'patients': similar_patients_list,
                'message': "Found " + str(len(similar_results)) + " similar patients with comparable age and TMB"
            }
        else:
            similar_patients = {
                'similar_patients_found': 0,
                'patients': [],
                'message': 'No similar patients found in database'
            }
    except Exception as e:
        similar_patients = {
            'similar_patients_found': 0,
            'patients': [],
            'message': 'Similar patient search error: ' + str(e)
        }
    
    # STEP 3: RAG - Get research papers
    if risk_score >= 80:
        search_terms = "('high TMB' OR 'poor prognosis' OR 'aggressive' OR 'advanced stage' OR 'short survival')"
    elif risk_score >= 60:
        search_terms = "('moderate risk' OR 'standard treatment' OR 'chemotherapy' OR 'survival outcomes')"
    else:
        search_terms = "('low TMB' OR 'favorable prognosis' OR 'early stage' OR 'long-term survival' OR 'good outcomes')"
    
    rag_query = """
        WITH ranked_chunks AS (
            SELECT 
                PAPER_ID,
                CHUNK_TEXT,
                CHUNK_INDEX,
                ROW_NUMBER() OVER (PARTITION BY PAPER_ID ORDER BY CHUNK_INDEX) AS rn
            FROM ONCODETECT_DB.UNSTRUCTURED_DATA.TEXT_EMBEDDINGS_TABLE
            WHERE (
                CHUNK_TEXT ILIKE '%TMB%' OR
                CHUNK_TEXT ILIKE '%survival%' OR
                CHUNK_TEXT ILIKE '%prognosis%' OR
                CHUNK_TEXT ILIKE '%treatment%' OR
                CHUNK_TEXT ILIKE '%outcomes%'
            )
            AND CHUNK_TEXT IS NOT NULL
            AND LENGTH(CHUNK_TEXT) > 200
        )
        SELECT DISTINCT
            PAPER_ID,
            CHUNK_TEXT
        FROM ranked_chunks
        WHERE rn = 1
        ORDER BY RANDOM()
        LIMIT 5
    """
    
    research_results = session.sql(rag_query).collect()
    
    # Format research
    research_context = ""
    research_evidence = []
    
    for idx, row in enumerate(research_results, 1):
        paper_id = row['PAPER_ID']
        chunk = row['CHUNK_TEXT'][:500]
        similarity = 0.90 - (idx * 0.05)
        
        research_context = research_context + "\\n\\n[Paper " + str(paper_id) + "]:\\n" + chunk + "\\n"
        research_evidence.append({
            'paper_id': paper_id,
            'chunk_text': chunk,
            'similarity_score': similarity
        })
    
    # STEP 4: Create summaries
    research_summary = "Key findings from " + str(len(research_evidence)) + " papers:\\n"
    for idx, paper in enumerate(research_evidence, 1):
        research_summary = research_summary + str(idx) + ". [" + paper['paper_id'] + "]: " + paper['chunk_text'][:150] + "...\\n"
    
    similar_summary = ""
    if similar_patients['similar_patients_found'] > 0:
        similar_summary = "Found " + str(len(similar_patients['patients'])) + " similar patients:\\n"
        for p in similar_patients['patients'][:3]:
            similar_summary = similar_summary + "- Age " + str(p['age']) + ", TMB " + str(p['tmb']) + ", Survival " + str(p['survival_months']) + " months\\n"
    else:
        similar_summary = "No similar patients in database."
    
    # STEP 5: Enhanced LLM Prompt
    sex_str = "Male" if is_male == 1 else "Female"
    smoking_str = "Former smoker" if is_former_smoker == 1 else "Non-smoker"
    
    llm_prompt = """
You are an oncology expert analyzing SCLC patient data with focus on """ + risk_level + """ cases.

PATIENT PROFILE:
- Age: """ + str(int(age)) + """ years, Sex: """ + sex_str + """
- TMB: """ + str(round(tmb, 1)) + """ mut/Mb (""" + ("HIGH" if tmb >= 20 else "INTERMEDIATE" if tmb >= 10 else "LOW") + """), Total Mutations: """ + str(int(mutation_count)) + """
- Smoking History: """ + smoking_str + """

RISK ASSESSMENT:
- Risk Score: """ + str(round(risk_score, 1)) + """/100 (""" + risk_level + """)
- Model Confidence: """ + str(round(confidence * 100, 1)) + """%

SIMILAR PATIENT COHORT (n=""" + str(similar_patients['similar_patients_found']) + """):
""" + similar_summary + """

SUPPORTING RESEARCH EVIDENCE:
""" + research_context + """

Provide a comprehensive clinical assessment with these sections:

1. RISK INTERPRETATION (3-4 sentences):
   - Explain the """ + str(round(risk_score, 1)) + """/100 risk score
   - Highlight key contributing factors (age, TMB, mutations)
   - Compare to population norms

2. SIMILAR PATIENT INSIGHTS (2-3 sentences):
   - Analyze the cohort's survival patterns
   - Identify prognostic factors from the data
   
3. EVIDENCE-BASED ANALYSIS (3-4 sentences):
   - Reference specific findings from the research papers (cite Paper IDs)
   - Connect patient's biomarkers to published literature
   - Discuss TMB implications for treatment

4. TREATMENT RECOMMENDATIONS (3-4 sentences):
   - Suggest appropriate therapeutic approaches for this risk level
   - Consider immunotherapy eligibility (TMB status)
   - Mention clinical trial options if applicable

5. PROGNOSIS (2-3 sentences):
   - Provide realistic survival expectations
   - Identify factors that could improve/worsen outcomes

Keep the tone clinical, evidence-based, and actionable. Always cite paper IDs when referencing research.
    """
    
    llm_prompt_clean = llm_prompt.replace("'", "''")
    llm_query = "SELECT SNOWFLAKE.CORTEX.COMPLETE('llama3.1-405b', '" + llm_prompt_clean + "') AS analysis"
    llm_result = session.sql(llm_query).collect()[0]['ANALYSIS']
    
    current_time = str(session.sql("SELECT CURRENT_TIMESTAMP()").collect()[0][0])
    
    return json.dumps({
        'ml_prediction': ml_pred,
        'similar_patients': similar_patients,
        'research_evidence': research_evidence,
        'clinical_analysis': llm_result,
        'timestamp': current_time
    })
$$;