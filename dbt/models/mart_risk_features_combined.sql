-- models/marts/mart_risk_features_combined.sql
-- Combined clinical + genomic features for enhanced predictions

{{ config(materialized='table') }}

WITH clinical AS (
    SELECT * FROM {{ ref('mart_risk_features_clinical') }}
),

genomic AS (
    SELECT * FROM {{ ref('mart_risk_features_genomic') }}
)

SELECT
    -- Use clinical as base (has survival outcomes)
    c.sample_id,
    c.patient_id,
    c.study_id,
    
    -- TARGETS
    c.target_survival_months,
    c.target_event,
    c.target_category,
    
    -- CLINICAL FEATURES
    c.age,
    c.age_group,
    c.is_male,
    c.smoking_status,
    c.is_current_smoker,
    c.is_former_smoker,
    c.is_never_smoker,
    c.mutation_count,
    c.tmb,
    c.tmb_category,
    c.is_tmb_high,
    c.is_tmb_intermediate,
    c.is_tmb_low,
    c.stage_group,
    c.is_advanced_stage,
    c.age_x_advanced_stage,
    c.smoker_x_tmb,
    c.high_risk_flag,
    
    -- GENOMIC FEATURES (NULL if not available)
    g.sclc_subtype,
    g.is_sclc_a,
    g.is_sclc_n,
    g.is_sclc_p,
    g.is_sclc_y,
    g.ascl1_expression,
    g.neurod1_expression,
    g.pou2f3_expression,
    g.yap1_expression,
    g.tp53_expression,
    g.rb1_expression,
    g.myc_family_score,
    g.tp53_rb1_dual_loss,
    g.dll3_targetable,
    g.bcl2_targetable,
    g.myc_amplified,
    g.mean_expression,
    g.genomic_high_risk_flag,
    
    -- FLAG: Does this patient have genomic data?
    CASE WHEN g.sample_id IS NOT NULL THEN 1 ELSE 0 END AS has_genomic_data,
    
    -- DATA SOURCE
    CASE 
        WHEN g.sample_id IS NOT NULL THEN 'CLINICAL_AND_GENOMIC'
        ELSE 'CLINICAL_ONLY'
    END AS data_source,
    
    CURRENT_TIMESTAMP() AS created_at

FROM clinical c
LEFT JOIN genomic g ON c.sample_id = g.sample_id