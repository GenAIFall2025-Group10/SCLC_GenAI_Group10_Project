-- models/stg_dose_response.sql
-- Clean and prepare dose-response curve data

{{ config(materialized='view') }}

SELECT
    TRY_CAST(experiment_id AS INTEGER) AS experiment_id,
    TRY_CAST(concentration AS FLOAT) AS concentration,
    TRY_CAST(mean_pct_ctrl AS FLOAT) AS mean_pct_ctrl,
    TRY_CAST(stddev_pct_ctrl AS FLOAT) AS stddev_pct_ctrl,
    TRY_CAST(count_pct_ctrl AS INTEGER) AS count_pct_ctrl,
    
    -- Log transformation of concentration
    CASE 
        WHEN TRY_CAST(concentration AS FLOAT) > 0 
        THEN LOG(10, TRY_CAST(concentration AS FLOAT))
    END AS log_concentration,
    
    -- Calculate coefficient of variation (CV) for data quality
    CASE
        WHEN TRY_CAST(mean_pct_ctrl AS FLOAT) != 0
        THEN ABS(TRY_CAST(stddev_pct_ctrl AS FLOAT) / TRY_CAST(mean_pct_ctrl AS FLOAT)) * 100
    END AS coefficient_variation,
    
    -- Data quality flag
    CASE
        WHEN TRY_CAST(stddev_pct_ctrl AS FLOAT) IS NULL THEN 'NO_STDDEV'
        WHEN ABS(TRY_CAST(stddev_pct_ctrl AS FLOAT) / NULLIF(TRY_CAST(mean_pct_ctrl AS FLOAT), 0)) * 100 > 30 
            THEN 'HIGH_VARIABILITY'
        WHEN ABS(TRY_CAST(stddev_pct_ctrl AS FLOAT) / NULLIF(TRY_CAST(mean_pct_ctrl AS FLOAT), 0)) * 100 > 15 
            THEN 'MODERATE_VARIABILITY'
        ELSE 'LOW_VARIABILITY'
    END AS data_quality_flag,
    
    -- Response classification
    CASE
        WHEN TRY_CAST(mean_pct_ctrl AS FLOAT) < 20 THEN 'STRONG_INHIBITION'
        WHEN TRY_CAST(mean_pct_ctrl AS FLOAT) BETWEEN 20 AND 50 THEN 'MODERATE_INHIBITION'
        WHEN TRY_CAST(mean_pct_ctrl AS FLOAT) BETWEEN 50 AND 80 THEN 'WEAK_INHIBITION'
        WHEN TRY_CAST(mean_pct_ctrl AS FLOAT) > 80 THEN 'MINIMAL_EFFECT'
        ELSE 'UNKNOWN'
    END AS response_category

FROM {{ source('raw_data', 'dose_response') }}
WHERE experiment_id IS NOT NULL
  AND concentration IS NOT NULL
  AND mean_pct_ctrl IS NOT NULL