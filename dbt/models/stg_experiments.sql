-- models/stg_experiments.sql
-- Clean and prepare drug screening experiments

{{ config(materialized='view') }}

SELECT
    TRY_CAST(experiment_id AS INTEGER) AS experiment_id,
    TRY_CAST(nsc AS INTEGER) AS nsc,
    UPPER(TRIM(cell_line)) AS cell_line,
    TRY_CAST(highest_tested_concentration AS FLOAT) AS highest_tested_concentration,
    TRY_CAST(log_ec50 AS FLOAT) AS log_ec50,
    TRY_CAST(response_zero AS FLOAT) AS response_zero,
    TRY_CAST(response_inflection AS FLOAT) AS response_inflection,
    TRY_CAST(hill_slope AS FLOAT) AS hill_slope,
    TRY_CAST(log_ic50 AS FLOAT) AS log_ic50,
    TRIM(flag_ic50) AS flag_ic50,
    
    -- Calculate IC50 in micromolar (from log scale)
    POWER(10, TRY_CAST(log_ic50 AS FLOAT)) AS ic50_um,
    
    -- Sensitivity classification based on IC50
    CASE
        WHEN TRY_CAST(log_ic50 AS FLOAT) < -6 THEN 'HIGHLY_SENSITIVE'
        WHEN TRY_CAST(log_ic50 AS FLOAT) BETWEEN -6 AND -5 THEN 'SENSITIVE'
        WHEN TRY_CAST(log_ic50 AS FLOAT) BETWEEN -5 AND -4 THEN 'MODERATELY_RESISTANT'
        WHEN TRY_CAST(log_ic50 AS FLOAT) > -4 THEN 'RESISTANT'
        ELSE 'UNKNOWN'
    END AS sensitivity_category

FROM {{ source('raw_data', 'experiments') }}
WHERE experiment_id IS NOT NULL
  AND nsc IS NOT NULL
  AND cell_line IS NOT NULL

  