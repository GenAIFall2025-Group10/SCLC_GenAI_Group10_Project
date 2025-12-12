-- models/mart_drug_sensitivity.sql
-- Drug sensitivity profiles for treatment recommendations

{{ config(materialized='table') }}

WITH experiments AS (
    SELECT * FROM {{ ref('stg_experiments') }}
),

compounds AS (
    SELECT 
        TRY_CAST(nsc AS INTEGER) AS nsc,
        TRIM(cas) AS cas,
        TRIM(drug_name) AS drug_name
    FROM {{ source('raw_data', 'compounds') }}
),

cell_lines AS (
    SELECT 
        UPPER(TRIM(cell_line)) AS cell_line,
        TRIM(panel) AS panel
    FROM {{ source('raw_data', 'cell_lines') }}
),

combined AS (
    SELECT
        e.experiment_id,
        e.nsc,
        c.drug_name,
        c.cas,
        e.cell_line,
        cl.panel,
        e.ic50_um,
        e.log_ic50,
        e.log_ec50,
        e.sensitivity_category,
        e.hill_slope,
        e.response_zero,
        e.response_inflection
    FROM experiments e
    LEFT JOIN compounds c ON e.nsc = c.nsc
    LEFT JOIN cell_lines cl ON e.cell_line = cl.cell_line
    WHERE e.ic50_um IS NOT NULL
)

SELECT
    *,
    
    -- Drug efficacy score (lower IC50 = more effective)
    CASE
        WHEN ic50_um < 0.000001 THEN 100  -- Highly effective
        WHEN ic50_um < 0.00001 THEN 80
        WHEN ic50_um < 0.0001 THEN 60
        WHEN ic50_um < 0.001 THEN 40
        ELSE 20  -- Less effective
    END AS efficacy_score,
    
    -- Response magnitude (difference between max and min response)
    ABS(response_inflection - response_zero) AS response_magnitude,
    
    CURRENT_TIMESTAMP() AS created_at

FROM combined