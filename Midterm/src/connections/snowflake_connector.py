"""
OncoDetect-AI: Snowflake Connection Module
Handles all Snowflake database connections
"""

import snowflake.connector
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SnowflakeConnector:
    """Manages Snowflake database connections"""
    
    def __init__(self, config=None):
        """
        Initialize Snowflake connector
        
        Args:
            config: Optional dict with Snowflake credentials
                   If None, loads from environment variables
        """
        self.config = config or self._load_config_from_env()
        self.conn = None
    
    def _load_config_from_env(self):
        """Load Snowflake configuration from .env file"""
        return {
            'user': os.getenv('SNOWFLAKE_USER'),
            'password': os.getenv('SNOWFLAKE_PASSWORD'),
            'account': os.getenv('SNOWFLAKE_ACCOUNT'),
            'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE', 'SCLC_WH'),
            'database': os.getenv('SNOWFLAKE_DATABASE', 'ONCODETECT_DB'),
            'schema': os.getenv('SNOWFLAKE_SCHEMA', 'DBT_STAGING')
        }
    
    def connect(self):
        """Establish connection to Snowflake"""
        print("Connecting to Snowflake...")
        
        try:
            self.conn = snowflake.connector.connect(
                user=self.config['user'],
                password=self.config['password'],
                account=self.config['account'],
                warehouse=self.config['warehouse'],
                database=self.config['database'],
                schema=self.config['schema']
            )
            print(f"✓ Connected to Snowflake successfully!")
            print(f"  Database: {self.config['database']}")
            print(f"  Schema: {self.config['schema']}")
            return self.conn
            
        except Exception as e:
            print(f"❌ Error connecting to Snowflake: {e}")
            raise
    
    def execute_query(self, query):
        """Execute SQL query and return results as DataFrame"""
        if not self.conn:
            self.connect()
        
        try:
            df = pd.read_sql(query, self.conn)
            return df
        except Exception as e:
            print(f"❌ Error executing query: {e}")
            raise
    
    def close(self):
        """Close Snowflake connection"""
        if self.conn:
            self.conn.close()
            print("✓ Snowflake connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# ============================================
# PREDEFINED QUERIES FOR ONCODETECT-AI
# ============================================

class OncoDetectQueries:
    """Common SQL queries for OncoDetect-AI"""
    
    @staticmethod
    def get_clinical_features():
        """Query to get clinical features for Risk Score Agent"""
        return """
        SELECT 
            sample_id,
            target_survival_months,
            target_event,
            target_category,
            age,
            is_male,
            is_current_smoker,
            is_former_smoker,
            is_never_smoker,
            mutation_count,
            tmb,
            is_tmb_high,
            is_tmb_intermediate,
            is_tmb_low,
            is_stage_i,
            is_stage_ii,
            is_stage_iii,
            is_stage_iv,
            is_advanced_stage,
            age_x_advanced_stage,
            smoker_x_tmb,
            high_risk_flag
        FROM MART_RISK_FEATURES_CLINICAL
        ORDER BY sample_id
        """
    
    @staticmethod
    def get_genomic_features():
        """Query to get genomic features"""
        return """
        SELECT *
        FROM MART_RISK_FEATURES_GENOMIC
        ORDER BY sample_id
        """
    
    @staticmethod
    def get_drug_sensitivity(cell_line=None):
        """Query to get drug sensitivity data"""
        base_query = """
        SELECT *
        FROM MART_DRUG_SENSITIVITY
        """
        if cell_line:
            return base_query + f" WHERE cell_line = '{cell_line}'"
        return base_query
    
    @staticmethod
    def get_treatment_recommendations(sample_id=None, sclc_subtype=None):
        """Query to get treatment recommendations"""
        query = """
        SELECT *
        FROM MART_TREATMENT_RECOMMENDATIONS
        WHERE 1=1
        """
        if sample_id:
            query += f" AND sample_id = '{sample_id}'"
        if sclc_subtype:
            query += f" AND sclc_subtype = '{sclc_subtype}'"
        
        query += " ORDER BY sample_id, drug_rank"
        return query
    
    @staticmethod
    def get_patient_profile(sample_id):
        """Get complete patient profile"""
        return f"""
        SELECT 
            c.*,
            g.sclc_subtype,
            g.ascl1_expression,
            g.neurod1_expression,
            g.pou2f3_expression,
            g.yap1_expression
        FROM MART_RISK_FEATURES_CLINICAL c
        LEFT JOIN MART_RISK_FEATURES_GENOMIC g
            ON c.sample_id = g.sample_id
        WHERE c.sample_id = '{sample_id}'
        """


# ============================================
# HELPER FUNCTIONS
# ============================================

def load_clinical_data():
    """Quick helper to load clinical features"""
    with SnowflakeConnector() as sf:
        df = sf.execute_query(OncoDetectQueries.get_clinical_features())
    return df


def load_genomic_data():
    """Quick helper to load genomic features"""
    with SnowflakeConnector() as sf:
        df = sf.execute_query(OncoDetectQueries.get_genomic_features())
    return df


def load_treatment_recommendations(sclc_subtype=None):
    """Quick helper to load treatment recommendations"""
    with SnowflakeConnector() as sf:
        df = sf.execute_query(
            OncoDetectQueries.get_treatment_recommendations(sclc_subtype=sclc_subtype)
        )
    return df


# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Snowflake Connection")
    print("=" * 60)
    
    # Test connection
    with SnowflakeConnector() as sf:
        # Test query
        df = sf.execute_query("SELECT COUNT(*) as row_count FROM MART_RISK_FEATURES_CLINICAL")
        print(f"\n✓ Clinical features count: {df['ROW_COUNT'].iloc[0]}")
        
        # Load clinical data
        df_clinical = sf.execute_query(OncoDetectQueries.get_clinical_features())
        print(f"✓ Loaded {len(df_clinical)} clinical samples")
        print(f"\nFirst 5 samples:")
        print(df_clinical.head())
    
    print("\n" + "=" * 60)
    print("✅ Connection test successful!")
    print("=" * 60)