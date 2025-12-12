-- models/mart_drug_subtype_profile.sql
-- Link drug sensitivities to SCLC subtypes for personalized recommendations

{{ config(materialized='table') }}

WITH drug_sensitivity AS (
    SELECT * FROM {{ ref('mart_drug_sensitivity') }}
),

-- Map cell lines to SCLC subtypes based on known classifications
-- Note: This is a simplified mapping. In production, you'd have actual cell line subtype data
cell_line_subtypes AS (
    SELECT 
        UPPER(TRIM(cell_line)) AS cell_line,
        TRIM(panel) AS panel,
        -- Assign subtypes based on typical SCLC cell line characteristics
        CASE
            WHEN cell_line IN ('H69', 'H82', 'H187', 'DMS273', 'H209', 'DMS-273', 'DMS 273') THEN 'SCLC-A'
            WHEN cell_line IN ('H1048', 'H1963', 'H2171') THEN 'SCLC-N'
            WHEN cell_line IN ('H1339', 'H1450') THEN 'SCLC-P'
            WHEN cell_line IN ('H446', 'H1672') THEN 'SCLC-Y'
            ELSE 'UNKNOWN'
        END AS inferred_subtype
    FROM {{ source('raw_data', 'cell_lines') }}
    WHERE panel = 'Small Cell Lung Cancer'  -- Changed from 'SCLC'
),

drug_subtype_aggregated AS (
    SELECT
        ds.nsc,
        ds.drug_name,
        cls.inferred_subtype AS sclc_subtype,
        COUNT(DISTINCT ds.cell_line) AS num_cell_lines_tested,
        AVG(ds.ic50_um) AS avg_ic50_um,
        AVG(ds.log_ic50) AS avg_log_ic50,
        MIN(ds.ic50_um) AS min_ic50_um,
        MAX(ds.ic50_um) AS max_ic50_um,
        AVG(ds.efficacy_score) AS avg_efficacy_score,
        AVG(ds.response_magnitude) AS avg_response_magnitude,
        
        -- Count sensitivity categories
        SUM(CASE WHEN ds.sensitivity_category = 'HIGHLY_SENSITIVE' THEN 1 ELSE 0 END) AS highly_sensitive_count,
        SUM(CASE WHEN ds.sensitivity_category = 'SENSITIVE' THEN 1 ELSE 0 END) AS sensitive_count,
        SUM(CASE WHEN ds.sensitivity_category = 'MODERATELY_RESISTANT' THEN 1 ELSE 0 END) AS moderate_resistant_count,
        SUM(CASE WHEN ds.sensitivity_category = 'RESISTANT' THEN 1 ELSE 0 END) AS resistant_count
        
    FROM drug_sensitivity ds
    INNER JOIN cell_line_subtypes cls ON ds.cell_line = cls.cell_line
    WHERE cls.inferred_subtype != 'UNKNOWN'
    GROUP BY ds.nsc, ds.drug_name, cls.inferred_subtype
)

SELECT
    *,
    
    -- Overall subtype sensitivity rating
    CASE
        WHEN highly_sensitive_count >= num_cell_lines_tested * 0.5 THEN 'HIGHLY_EFFECTIVE'
        WHEN (highly_sensitive_count + sensitive_count) >= num_cell_lines_tested * 0.5 THEN 'EFFECTIVE'
        WHEN moderate_resistant_count >= num_cell_lines_tested * 0.5 THEN 'MODERATELY_EFFECTIVE'
        ELSE 'LOW_EFFECTIVENESS'
    END AS subtype_effectiveness,
    
    -- Response consistency (lower variance = more consistent)
    CASE
        WHEN (max_ic50_um - min_ic50_um) < 0.00001 THEN 'HIGHLY_CONSISTENT'
        WHEN (max_ic50_um - min_ic50_um) < 0.0001 THEN 'CONSISTENT'
        ELSE 'VARIABLE'
    END AS response_consistency,
    
    -- Recommendation score (0-100)
    ROUND(
        (avg_efficacy_score * 0.6) + 
        (avg_response_magnitude * 0.4),
        2
    ) AS recommendation_score,
    
    CURRENT_TIMESTAMP() AS created_at

FROM drug_subtype_aggregated
WHERE num_cell_lines_tested >= 1  -- At least 1 cell line tested
ORDER BY sclc_subtype, avg_efficacy_score DESC