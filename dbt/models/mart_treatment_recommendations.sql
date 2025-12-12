-- models/mart_treatment_recommendations.sql
-- Personalized treatment recommendations combining patient features and drug data

{{ config(materialized='table') }}

WITH genomic_features AS (
    SELECT * FROM {{ ref('mart_risk_features_genomic') }}
),

drug_subtype_profile AS (
    SELECT * FROM {{ ref('mart_drug_subtype_profile') }}
),

-- Get top 10 most effective drugs per subtype
top_drugs_per_subtype AS (
    SELECT
        sclc_subtype,
        nsc,
        drug_name,
        avg_ic50_um,
        avg_efficacy_score,
        subtype_effectiveness,
        recommendation_score,
        num_cell_lines_tested,
        ROW_NUMBER() OVER (
            PARTITION BY sclc_subtype 
            ORDER BY recommendation_score DESC, avg_efficacy_score DESC
        ) AS drug_rank
    FROM drug_subtype_profile
    WHERE subtype_effectiveness IN ('HIGHLY_EFFECTIVE', 'EFFECTIVE')
)

SELECT
    -- Patient identifiers
    g.sample_id,
    g.sclc_subtype,
    g.cohort_type,
    
    -- Patient biomarkers
    g.ascl1_expression,
    g.neurod1_expression,
    g.pou2f3_expression,
    g.yap1_expression,
    g.myc_family_score,
    g.tp53_rb1_dual_loss,
    
    -- Therapeutic targets
    g.dll3_targetable,
    g.bcl2_targetable,
    g.myc_amplified,
    g.genomic_high_risk_flag,
    
    -- Drug recommendations
    d.drug_rank,
    d.nsc,
    d.drug_name,
    d.avg_ic50_um,
    d.avg_efficacy_score,
    d.subtype_effectiveness,
    d.recommendation_score,
    d.num_cell_lines_tested,
    
    -- Treatment rationale
    CASE 
        WHEN g.dll3_targetable = 1 AND d.drug_name LIKE '%Rovalpituzumab%' 
            THEN 'DLL3-targeted therapy recommended'
        WHEN g.bcl2_targetable = 1 AND d.drug_name LIKE '%Venetoclax%' 
            THEN 'BCL2 inhibitor therapy recommended'
        WHEN g.myc_amplified = 1 
            THEN 'MYC amplification detected - consider targeted therapy'
        WHEN g.genomic_high_risk_flag = 1 
            THEN 'High genomic risk - aggressive treatment recommended'
        ELSE 'Standard subtype-based treatment'
    END AS treatment_rationale,
    
    -- Priority flag
    CASE
        WHEN d.drug_rank <= 3 AND d.recommendation_score >= 70 THEN 'HIGH_PRIORITY'
        WHEN d.drug_rank <= 5 AND d.recommendation_score >= 60 THEN 'MEDIUM_PRIORITY'
        ELSE 'CONSIDER'
    END AS recommendation_priority,
    
    CURRENT_TIMESTAMP() AS created_at

FROM genomic_features g
INNER JOIN top_drugs_per_subtype d 
    ON g.sclc_subtype = d.sclc_subtype
WHERE d.drug_rank <= 10  -- Top 10 drugs per subtype
ORDER BY g.sample_id, d.drug_rank