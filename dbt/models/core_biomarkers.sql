-- models/core_biomarkers.sql
-- Extract SCLC biomarker expression

{{ config(materialized='table') }}

WITH biomarker_genes AS (
    SELECT * FROM {{ ref('stg_gene_expression') }}
    WHERE gene_symbol IN (
        'ASCL1', 'NEUROD1', 'POU2F3', 'YAP1',
        'TP53', 'RB1', 'MYC', 'MYCL', 'MYCN',
        'DLL3', 'BCL2', 'NOTCH1'
    )
),

pivoted AS (
    SELECT
        sample_id,
        MAX(CASE WHEN gene_symbol = 'ASCL1' THEN expression_value END) AS ascl1_expr,
        MAX(CASE WHEN gene_symbol = 'NEUROD1' THEN expression_value END) AS neurod1_expr,
        MAX(CASE WHEN gene_symbol = 'POU2F3' THEN expression_value END) AS pou2f3_expr,
        MAX(CASE WHEN gene_symbol = 'YAP1' THEN expression_value END) AS yap1_expr,
        MAX(CASE WHEN gene_symbol = 'TP53' THEN expression_value END) AS tp53_expr,
        MAX(CASE WHEN gene_symbol = 'RB1' THEN expression_value END) AS rb1_expr,
        MAX(CASE WHEN gene_symbol = 'MYC' THEN expression_value END) AS myc_expr,
        MAX(CASE WHEN gene_symbol = 'MYCL' THEN expression_value END) AS mycl_expr,
        MAX(CASE WHEN gene_symbol = 'MYCN' THEN expression_value END) AS mycn_expr,
        MAX(CASE WHEN gene_symbol = 'DLL3' THEN expression_value END) AS dll3_expr,
        MAX(CASE WHEN gene_symbol = 'BCL2' THEN expression_value END) AS bcl2_expr,
        MAX(CASE WHEN gene_symbol = 'NOTCH1' THEN expression_value END) AS notch1_expr
    FROM biomarker_genes
    GROUP BY sample_id
)

SELECT
    *,
    
    -- SCLC Subtype Classification
    CASE
        WHEN COALESCE(ascl1_expr, 0) > GREATEST(
            COALESCE(neurod1_expr, 0),
            COALESCE(pou2f3_expr, 0),
            COALESCE(yap1_expr, 0)
        ) THEN 'SCLC-A'
        
        WHEN COALESCE(neurod1_expr, 0) > GREATEST(
            COALESCE(ascl1_expr, 0),
            COALESCE(pou2f3_expr, 0),
            COALESCE(yap1_expr, 0)
        ) THEN 'SCLC-N'
        
        WHEN COALESCE(pou2f3_expr, 0) > GREATEST(
            COALESCE(ascl1_expr, 0),
            COALESCE(neurod1_expr, 0),
            COALESCE(yap1_expr, 0)
        ) THEN 'SCLC-P'
        
        WHEN COALESCE(yap1_expr, 0) > GREATEST(
            COALESCE(ascl1_expr, 0),
            COALESCE(neurod1_expr, 0),
            COALESCE(pou2f3_expr, 0)
        ) THEN 'SCLC-Y'
        
        ELSE 'MIXED'
    END AS sclc_subtype,
    
    -- MYC family score
    GREATEST(
        COALESCE(myc_expr, 0),
        COALESCE(mycl_expr, 0),
        COALESCE(mycn_expr, 0)
    ) AS myc_family_score,
    
    -- Dual loss indicator
    CASE
        WHEN COALESCE(tp53_expr, 0) < 5 
             AND COALESCE(rb1_expr, 0) < 5 
        THEN 1 ELSE 0
    END AS tp53_rb1_dual_loss

FROM pivoted