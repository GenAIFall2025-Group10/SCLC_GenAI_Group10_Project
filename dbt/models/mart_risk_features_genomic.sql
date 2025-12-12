-- models/mart_risk_features_genomic.sql
-- Genomic features for subtype classification and treatment recommendations

{{ config(materialized='table') }}

WITH biomarkers AS (
    SELECT * FROM {{ ref('core_biomarkers') }}
),

gene_summary AS (
    SELECT
        sample_id,
        COUNT(DISTINCT gene_symbol) AS total_genes_expressed,
        AVG(expression_value) AS mean_gene_expression,
        STDDEV(expression_value) AS stddev_gene_expression,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY expression_value) AS median_gene_expression
    FROM {{ ref('stg_gene_expression') }}
    GROUP BY sample_id
)

SELECT
    -- IDs
    b.sample_id,
    
    -- SCLC SUBTYPE (main classification target)
    b.sclc_subtype,
    CASE WHEN b.sclc_subtype = 'SCLC-A' THEN 1 ELSE 0 END AS is_sclc_a,
    CASE WHEN b.sclc_subtype = 'SCLC-N' THEN 1 ELSE 0 END AS is_sclc_n,
    CASE WHEN b.sclc_subtype = 'SCLC-P' THEN 1 ELSE 0 END AS is_sclc_p,
    CASE WHEN b.sclc_subtype = 'SCLC-Y' THEN 1 ELSE 0 END AS is_sclc_y,
    
    -- BIOMARKER EXPRESSION (key features)
    COALESCE(b.ascl1_expr, 0) AS ascl1_expression,
    COALESCE(b.neurod1_expr, 0) AS neurod1_expression,
    COALESCE(b.pou2f3_expr, 0) AS pou2f3_expression,
    COALESCE(b.yap1_expr, 0) AS yap1_expression,
    COALESCE(b.tp53_expr, 0) AS tp53_expression,
    COALESCE(b.rb1_expr, 0) AS rb1_expression,
    COALESCE(b.myc_expr, 0) AS myc_expression,
    COALESCE(b.mycl_expr, 0) AS mycl_expression,
    COALESCE(b.mycn_expr, 0) AS mycn_expression,
    COALESCE(b.dll3_expr, 0) AS dll3_expression,
    COALESCE(b.bcl2_expr, 0) AS bcl2_expression,
    COALESCE(b.notch1_expr, 0) AS notch1_expression,
    
    -- DERIVED BIOMARKER FEATURES
    b.myc_family_score,
    b.tp53_rb1_dual_loss,
    
    -- Dominant subtype marker expression
    GREATEST(
        COALESCE(b.ascl1_expr, 0),
        COALESCE(b.neurod1_expr, 0),
        COALESCE(b.pou2f3_expr, 0),
        COALESCE(b.yap1_expr, 0)
    ) AS dominant_subtype_marker_expr,
    
    -- Subtype confidence (difference between top and second marker)
    GREATEST(
        COALESCE(b.ascl1_expr, 0),
        COALESCE(b.neurod1_expr, 0),
        COALESCE(b.pou2f3_expr, 0),
        COALESCE(b.yap1_expr, 0)
    ) - (
        CASE 
            WHEN COALESCE(b.ascl1_expr, 0) >= GREATEST(COALESCE(b.neurod1_expr, 0), COALESCE(b.pou2f3_expr, 0), COALESCE(b.yap1_expr, 0))
            THEN GREATEST(COALESCE(b.neurod1_expr, 0), COALESCE(b.pou2f3_expr, 0), COALESCE(b.yap1_expr, 0))
            ELSE 0
        END
    ) AS subtype_confidence_score,
    
    -- THERAPEUTIC TARGET FLAGS
    CASE WHEN COALESCE(b.dll3_expr, 0) >= 5 THEN 1 ELSE 0 END AS dll3_targetable,
    CASE WHEN COALESCE(b.bcl2_expr, 0) >= 5 THEN 1 ELSE 0 END AS bcl2_targetable,
    CASE WHEN b.myc_family_score >= 10 THEN 1 ELSE 0 END AS myc_amplified,
    
    -- GENE EXPRESSION SUMMARY STATISTICS
    COALESCE(g.total_genes_expressed, 0) AS genes_expressed,
    COALESCE(g.mean_gene_expression, 0) AS mean_expression,
    COALESCE(g.stddev_gene_expression, 0) AS stddev_expression,
    COALESCE(g.median_gene_expression, 0) AS median_expression,
    
    -- RISK INDICATORS BASED ON GENOMICS
    CASE 
        WHEN b.tp53_rb1_dual_loss = 1 
             AND b.myc_family_score > 10
        THEN 1 ELSE 0 
    END AS genomic_high_risk_flag,
    
    -- Metadata
    'GENOMIC_COHORT' AS cohort_type,
    CURRENT_TIMESTAMP() AS created_at

FROM biomarkers b
LEFT JOIN gene_summary g ON b.sample_id = g.sample_id
WHERE b.sclc_subtype IS NOT NULL
  AND b.sclc_subtype != 'MIXED'