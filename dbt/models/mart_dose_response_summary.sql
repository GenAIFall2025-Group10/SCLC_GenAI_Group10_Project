-- models/mart_dose_response_summary.sql
-- Aggregate dose-response data with quality metrics

{{ config(materialized='table') }}

WITH dose_response AS (
    SELECT * FROM {{ ref('stg_dose_response') }}
),

experiments AS (
    SELECT * FROM {{ ref('stg_experiments') }}
),

dose_response_summary AS (
    SELECT
        dr.experiment_id,
        
        -- Curve characteristics
        COUNT(*) AS num_dose_points,
        MIN(dr.concentration) AS min_concentration,
        MAX(dr.concentration) AS max_concentration,
        MIN(dr.mean_pct_ctrl) AS min_response,
        MAX(dr.mean_pct_ctrl) AS max_response,
        AVG(dr.mean_pct_ctrl) AS avg_response,
        
        -- Response range (dynamic range of the assay)
        MAX(dr.mean_pct_ctrl) - MIN(dr.mean_pct_ctrl) AS response_range,
        
        -- Data quality metrics
        AVG(dr.coefficient_variation) AS avg_cv,
        COUNT(CASE WHEN dr.data_quality_flag = 'LOW_VARIABILITY' THEN 1 END) AS low_variability_points,
        COUNT(CASE WHEN dr.data_quality_flag = 'HIGH_VARIABILITY' THEN 1 END) AS high_variability_points,
        
        -- Response categories distribution
        COUNT(CASE WHEN dr.response_category = 'STRONG_INHIBITION' THEN 1 END) AS strong_inhibition_points,
        COUNT(CASE WHEN dr.response_category = 'MODERATE_INHIBITION' THEN 1 END) AS moderate_inhibition_points,
        COUNT(CASE WHEN dr.response_category = 'WEAK_INHIBITION' THEN 1 END) AS weak_inhibition_points,
        COUNT(CASE WHEN dr.response_category = 'MINIMAL_EFFECT' THEN 1 END) AS minimal_effect_points
        
    FROM dose_response dr
    GROUP BY dr.experiment_id
)

SELECT
    drs.*,
    e.nsc,
    e.cell_line,
    e.ic50_um,
    e.log_ic50,
    e.sensitivity_category,
    
    -- Overall curve quality score (0-100)
    ROUND(
        (
            (drs.low_variability_points::FLOAT / NULLIF(drs.num_dose_points, 0) * 40) + -- 40% weight
            (CASE WHEN drs.response_range > 50 THEN 30 ELSE drs.response_range * 0.6 END) + -- 30% weight
            (CASE WHEN drs.num_dose_points >= 8 THEN 30 
                  WHEN drs.num_dose_points >= 5 THEN 20 
                  ELSE 10 END) -- 30% weight
        ),
        2
    ) AS curve_quality_score,
    
    -- Curve quality classification
    CASE
        WHEN drs.high_variability_points::FLOAT / NULLIF(drs.num_dose_points, 0) > 0.3 
            THEN 'POOR_QUALITY'
        WHEN drs.response_range < 30 
            THEN 'LOW_DYNAMIC_RANGE'
        WHEN drs.num_dose_points < 5 
            THEN 'INSUFFICIENT_POINTS'
        WHEN drs.low_variability_points::FLOAT / NULLIF(drs.num_dose_points, 0) > 0.7 
             AND drs.response_range > 50 
            THEN 'HIGH_QUALITY'
        ELSE 'ACCEPTABLE_QUALITY'
    END AS curve_quality_classification,
    
    -- Recommendation based on curve quality
    CASE
        WHEN drs.high_variability_points::FLOAT / NULLIF(drs.num_dose_points, 0) > 0.3 
            THEN 'USE_WITH_CAUTION'
        WHEN drs.response_range < 30 
            THEN 'VERIFY_ACTIVITY'
        ELSE 'RELIABLE'
    END AS data_reliability,
    
    CURRENT_TIMESTAMP() AS created_at

FROM dose_response_summary drs
LEFT JOIN experiments e ON drs.experiment_id = e.experiment_id