-- models/stg_gene_expression.sql
-- Clean gene expression data

{{ config(materialized='view') }}

SELECT
    UPPER(TRIM(gene_symbol)) AS gene_symbol,
    UPPER(TRIM(sample_id)) AS sample_id,
    TRY_CAST(expression_value AS FLOAT) AS expression_value,
    
    -- Log2 transformation
    CASE 
        WHEN TRY_CAST(expression_value AS FLOAT) > 0 
        THEN LOG(2, TRY_CAST(expression_value AS FLOAT)) 
    END AS log2_expression,
    
    -- Expression level categories
    CASE
        WHEN TRY_CAST(expression_value AS FLOAT) = 0 THEN 'NONE'
        WHEN TRY_CAST(expression_value AS FLOAT) > 0 
             AND TRY_CAST(expression_value AS FLOAT) < 1 THEN 'LOW'
        WHEN TRY_CAST(expression_value AS FLOAT) BETWEEN 1 AND 10 THEN 'MEDIUM'
        WHEN TRY_CAST(expression_value AS FLOAT) > 10 THEN 'HIGH'
    END AS expression_level

FROM {{ source('raw_data', 'gene_expression') }}
WHERE expression_value IS NOT NULL
  AND gene_symbol IS NOT NULL
  AND sample_id IS NOT NULL
  AND TRY_CAST(expression_value AS FLOAT) IS NOT NULL