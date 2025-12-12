from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.models import Variable
from datetime import datetime, timedelta
import boto3
import io
import json
import hashlib
import os
import fitz  # PyMuPDF
from PIL import Image
import snowflake.connector

# Configuration from environment variables
# AWS Configuration
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-2')
S3_BUCKET_PAPERS = os.getenv('S3_BUCKET_NAME', 'oncodetect-ai-researchpapers')

# Snowflake Configuration
SNOWFLAKE_ACCOUNT = os.getenv('SNOWFLAKE_ACCOUNT')
SNOWFLAKE_USER = os.getenv('SNOWFLAKE_USER')
SNOWFLAKE_PASSWORD = os.getenv('SNOWFLAKE_PASSWORD')
SNOWFLAKE_DATABASE = os.getenv('SNOWFLAKE_DATABASE')
SNOWFLAKE_SCHEMA = os.getenv('SNOWFLAKE_SCHEMA')
SNOWFLAKE_WAREHOUSE = os.getenv('SNOWFLAKE_WAREHOUSE')
SNOWFLAKE_ROLE = os.getenv('SNOWFLAKE_ROLE')

# Snowflake table names
SNOWFLAKE_PAPERS_TABLE = os.getenv('SNOWFLAKE_TABLE', 'PUBMED_PAPERS_METADATA')
SNOWFLAKE_TEXT_PROCESSED_TABLE = os.getenv('SNOWFLAKE_TEXT_PROCESSED_PAPERS_TABLE', 'TEXT_PROCESSED_PAPERS')
SNOWFLAKE_TEXT_EMBEDDINGS_TABLE = os.getenv('SNOWFLAKE_TEXT_EMBEDDINGS_TABLE', 'TEXT_EMBEDDINGS_TABLE')
SNOWFLAKE_PARSED_TEXT_TABLE = os.getenv('SNOWFLAKE_PARSED_TEXT_METADATA_TABLE', 'PARSED_TEXT_METADATA')

# Processing configuration
BATCH_SIZE = 25  # Papers per batch
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))
CORTEX_TEXT_MODEL = os.getenv('CORTEX_TEXT_MODEL', 'snowflake-arctic-embed-m')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=24),
}

dag = DAG(
    'text_embeddings_pipeline',
    default_args=default_args,
    description='Process SCLC papers: parse text, chunk, embed, and store in Snowflake',
    schedule=None,
    catchup=False,
    tags=['sclc', 'text-processing', 'embeddings'],
)


def get_snowflake_connection():
    """Create and return a Snowflake connection."""
    return snowflake.connector.connect(
        account=SNOWFLAKE_ACCOUNT,
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA,
        warehouse=SNOWFLAKE_WAREHOUSE,
        role=SNOWFLAKE_ROLE
    )


def get_s3_client():
    """Create and return an S3 client."""
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION
    )


def get_unprocessed_papers(**context):
    """Step 1: Query Snowflake to find papers that haven't been processed yet."""
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    
    try:
        # Check if processed table exists
        check_table_query = f"""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = '{SNOWFLAKE_SCHEMA}' 
        AND TABLE_NAME = '{SNOWFLAKE_TEXT_PROCESSED_TABLE}'
        """
        
        cursor.execute(check_table_query)
        table_exists = cursor.fetchone()[0] > 0
        
        if table_exists:
            query = f"""
            SELECT p.PMC_ID, p.S3_URL, p.DOI, p.TITLE
            FROM {SNOWFLAKE_PAPERS_TABLE} p
            LEFT JOIN {SNOWFLAKE_TEXT_PROCESSED_TABLE} pp ON p.PMC_ID = pp.PAPER_ID
            WHERE pp.PAPER_ID IS NULL
            """
        else:
            query = f"""
            SELECT PMC_ID, S3_URL, DOI, TITLE
            FROM {SNOWFLAKE_PAPERS_TABLE}
            """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        unprocessed_papers = [
            {'paper_id': row[0], 's3_url': row[1], 'doi': row[2], 'title': row[3]}
            for row in results
        ]
        
        # Limit to batch size
        unprocessed_papers = unprocessed_papers[:BATCH_SIZE]
        print(f"Found {len(unprocessed_papers)} unprocessed papers")
        
        context['task_instance'].xcom_push(key='unprocessed_papers', value=unprocessed_papers)
        return len(unprocessed_papers)
        
    finally:
        cursor.close()
        conn.close()


def parse_and_chunk(**context):
    """Step 2: Parse PDFs and chunk text, store chunks in Snowflake regular table."""
    papers = context['task_instance'].xcom_pull(task_ids='get_unprocessed_papers', key='unprocessed_papers')
    s3_client = get_s3_client()
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    
    try:
        # Create unique table name using run_id
        run_id = context['run_id'].replace('-', '_').replace(':', '_').replace('+', '_').replace('.', '_')
        table_name = f"TEMP_CHUNKS_{run_id}"
        
        print(f"Creating chunks table: {table_name}")
        
        # Create regular table (not TEMPORARY) so it persists across sessions
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            PAPER_ID VARCHAR,
            CHUNK_INDEX INTEGER,
            CHUNK_TEXT TEXT,
            CHUNK_HASH VARCHAR,
            START_CHAR INTEGER,
            END_CHAR INTEGER,
            START_PAGE INTEGER,
            END_PAGE INTEGER,
            PAGES_SPANNED_JSON TEXT,
            CONTENT_HASH VARCHAR
        )
        """)
        
        paper_chunk_counts = {}
        
        for paper in papers:
            try:
                # Download PDF from S3
                s3_key = paper['s3_url'].replace(f"s3://{S3_BUCKET_PAPERS}/", "")
                response = s3_client.get_object(Bucket=S3_BUCKET_PAPERS, Key=s3_key)
                pdf_content = response['Body'].read()
                
                # Parse PDF with PyMuPDF
                doc = fitz.open(stream=pdf_content, filetype="pdf")
                full_text = ""
                page_texts = []
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    page_text = page.get_text()
                    full_text += page_text + "\n"
                    page_texts.append({
                        'page_number': page_num + 1,
                        'text': page_text,
                        'char_count': len(page_text)
                    })
                
                doc.close()
                
                # Create content hash
                content_hash = hashlib.sha256(full_text.encode()).hexdigest()
                
                print(f"✅ Parsed {paper['paper_id']}: {len(full_text)} chars, {len(page_texts)} pages")
                
                # Build character-to-page mapping
                char_to_page = []
                for page_info in page_texts:
                    page_text = page_info['text']
                    page_num = page_info['page_number']
                    for _ in range(len(page_text) + 1):
                        char_to_page.append(page_num)
                
                # Create chunks
                start = 0
                chunk_idx = 0
                
                while start < len(full_text):
                    end = min(start + CHUNK_SIZE, len(full_text))
                    chunk_text = full_text[start:end]
                    
                    # Determine page numbers for this chunk
                    start_page = char_to_page[start] if start < len(char_to_page) else char_to_page[-1]
                    end_page = char_to_page[end - 1] if end - 1 < len(char_to_page) else char_to_page[-1]
                    pages_in_chunk = sorted(set(char_to_page[start:end])) if end <= len(char_to_page) else [start_page]
                    
                    chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
                    
                    # Insert chunk into Snowflake table
                    cursor.execute(f"""
                        INSERT INTO {table_name} 
                        (PAPER_ID, CHUNK_INDEX, CHUNK_TEXT, CHUNK_HASH, START_CHAR, END_CHAR, START_PAGE, END_PAGE, PAGES_SPANNED_JSON, CONTENT_HASH)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        paper['paper_id'],
                        chunk_idx,
                        chunk_text,
                        chunk_hash,
                        start,
                        end,
                        start_page,
                        end_page,
                        json.dumps(pages_in_chunk),
                        content_hash
                    ))
                    
                    start += CHUNK_SIZE - CHUNK_OVERLAP
                    chunk_idx += 1
                
                paper_chunk_counts[paper['paper_id']] = chunk_idx
                print(f"✅ Created {chunk_idx} chunks for {paper['paper_id']}")
                
            except Exception as e:
                print(f"❌ Error parsing {paper['paper_id']}: {str(e)}")
                import traceback
                print(traceback.format_exc())
                continue
        
        conn.commit()
        
        # Push table name and paper chunk counts to XCom (very small!)
        context['task_instance'].xcom_push(key='chunks_table_name', value=table_name)
        context['task_instance'].xcom_push(key='paper_chunk_counts', value=paper_chunk_counts)
        
        total_chunks = sum(paper_chunk_counts.values())
        print(f"✅ Total chunks stored in {table_name}: {total_chunks}")
        
        return total_chunks
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise
    finally:
        cursor.close()
        conn.close()


def create_and_insert_embeddings(**context):
    """Step 3: Read chunks from Snowflake, create embeddings, and insert."""
    # Get table name and paper counts from previous task
    table_name = context['task_instance'].xcom_pull(task_ids='parse_and_chunk', key='chunks_table_name')
    paper_chunk_counts = context['task_instance'].xcom_pull(task_ids='parse_and_chunk', key='paper_chunk_counts')
    
    print(f"Reading chunks from table: {table_name}")
    
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    
    try:
        # Create temporary staging table for embeddings
        cursor.execute("""
        CREATE TEMPORARY TABLE IF NOT EXISTS TEMP_TEXT_EMBEDDINGS_STAGE (
            PAPER_ID VARCHAR,
            CHUNK_INDEX INTEGER,
            CHUNK_TEXT TEXT,
            CHUNK_HASH VARCHAR,
            EMBEDDING_JSON TEXT,
            START_CHAR INTEGER,
            END_CHAR INTEGER,
            START_PAGE INTEGER,
            END_PAGE INTEGER,
            PAGES_SPANNED_JSON TEXT,
            CONTENT_HASH VARCHAR
        )
        """)
        
        total_embeddings = 0
        
        # Process each paper's chunks
        for paper_id, chunk_count in paper_chunk_counts.items():
            print(f"Processing {chunk_count} chunks for {paper_id}")
            
            # Read chunks for this paper from the regular table
            cursor.execute(f"""
                SELECT PAPER_ID, CHUNK_INDEX, CHUNK_TEXT, CHUNK_HASH, START_CHAR, END_CHAR, 
                       START_PAGE, END_PAGE, PAGES_SPANNED_JSON, CONTENT_HASH
                FROM {table_name}
                WHERE PAPER_ID = %s
                ORDER BY CHUNK_INDEX
            """, (paper_id,))
            
            chunks = cursor.fetchall()
            
            for i, chunk_row in enumerate(chunks):
                try:
                    paper_id, chunk_index, chunk_text, chunk_hash, start_char, end_char, start_page, end_page, pages_json, content_hash = chunk_row
                    
                    # Escape text for SQL
                    escaped_text = chunk_text.replace("'", "''")
                    
                    # Create embedding using Snowflake Cortex
                    embedding_query = f"SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768('{CORTEX_TEXT_MODEL}', $${escaped_text}$$)"
                    cursor.execute(embedding_query)
                    result = cursor.fetchone()
                    
                    # Parse embedding
                    embedding = json.loads(result[0]) if isinstance(result[0], str) else result[0]
                    
                    # Insert into staging table
                    cursor.execute("""
                        INSERT INTO TEMP_TEXT_EMBEDDINGS_STAGE 
                        (PAPER_ID, CHUNK_INDEX, CHUNK_TEXT, CHUNK_HASH, EMBEDDING_JSON, 
                         START_CHAR, END_CHAR, START_PAGE, END_PAGE, PAGES_SPANNED_JSON, CONTENT_HASH)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        paper_id,
                        chunk_index,
                        chunk_text,
                        chunk_hash,
                        json.dumps(embedding),
                        start_char,
                        end_char,
                        start_page,
                        end_page,
                        pages_json,
                        content_hash
                    ))
                    
                    total_embeddings += 1
                    
                    if (i + 1) % 50 == 0:
                        print(f"  Created embeddings for {i + 1}/{len(chunks)} chunks")
                    
                except Exception as e:
                    print(f"❌ Error creating embedding for chunk {chunk_index}: {str(e)}")
                    continue
            
            print(f"✅ Completed {paper_id}: {len(chunks)} embeddings created")
        
        conn.commit()
        print(f"Staged {total_embeddings} embeddings")
        
        # Copy from staging to final table with VECTOR conversion
        print("Copying embeddings to final table...")
        
        copy_query = f"""
        INSERT INTO {SNOWFLAKE_TEXT_EMBEDDINGS_TABLE} 
        (paper_id, chunk_index, chunk_text, chunk_hash, embedding, 
         start_char, end_char, start_page, end_page, pages_spanned, content_hash, created_at)
        SELECT 
            PAPER_ID,
            CHUNK_INDEX,
            CHUNK_TEXT,
            CHUNK_HASH,
            PARSE_JSON(EMBEDDING_JSON)::VECTOR(FLOAT, 768),
            START_CHAR,
            END_CHAR,
            START_PAGE,
            END_PAGE,
            PARSE_JSON(PAGES_SPANNED_JSON),
            CONTENT_HASH,
            CURRENT_TIMESTAMP()
        FROM TEMP_TEXT_EMBEDDINGS_STAGE
        """
        
        cursor.execute(copy_query)
        rows_inserted = cursor.rowcount
        conn.commit()
        
        print(f"✅ Successfully inserted {rows_inserted} text embeddings into {SNOWFLAKE_TEXT_EMBEDDINGS_TABLE}")
        
        # Clean up the chunks table
        print(f"🧹 Cleaning up chunks table: {table_name}")
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()
        print(f"✅ Cleanup complete")
        
        return rows_inserted
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise
    finally:
        cursor.close()
        conn.close()


def mark_papers_as_processed(**context):
    """Step 4: Mark papers as processed in PROCESSED_PAPERS table."""
    papers = context['task_instance'].xcom_pull(task_ids='get_unprocessed_papers', key='unprocessed_papers')
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    
    try:
        processed_count = 0
        
        for paper in papers:
            paper_id = paper['paper_id']
            
            # Check if this paper has text embeddings
            cursor.execute(
                f"SELECT COUNT(*) FROM {SNOWFLAKE_TEXT_EMBEDDINGS_TABLE} WHERE paper_id = %s",
                (paper_id,)
            )
            text_count = cursor.fetchone()[0]
            
            # Only mark as processed if it has text embeddings
            if text_count > 0:
                # Check if already marked as processed
                cursor.execute(
                    f"SELECT COUNT(*) FROM {SNOWFLAKE_TEXT_PROCESSED_TABLE} WHERE paper_id = %s",
                    (paper_id,)
                )
                already_processed = cursor.fetchone()[0] > 0
                
                if not already_processed:
                    cursor.execute(
                        f"""
                        INSERT INTO {SNOWFLAKE_TEXT_PROCESSED_TABLE} 
                        (paper_id, processed_at, text_chunks_count)
                        VALUES (%s, CURRENT_TIMESTAMP(), %s)
                        """,
                        (paper_id, text_count)
                    )
                    processed_count += 1
                    print(f"✅ Marked {paper_id} as processed ({text_count} text chunks)")
                else:
                    print(f"⏭️  {paper_id} already marked as processed")
        
        conn.commit()
        print(f"✅ Successfully marked {processed_count} papers as processed")
        return processed_count
        
    except Exception as e:
        print(f"❌ Error marking papers as processed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise
    finally:
        cursor.close()
        conn.close()


def check_for_more_papers(**context):
    """Step 5: Check if more unprocessed papers exist and decide next action."""
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    
    try:
        # Check if processed table exists
        check_table_query = f"""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = '{SNOWFLAKE_SCHEMA}' 
        AND TABLE_NAME = '{SNOWFLAKE_TEXT_PROCESSED_TABLE}'
        """
        
        cursor.execute(check_table_query)
        table_exists = cursor.fetchone()[0] > 0
        
        if table_exists:
            query = f"""
            SELECT COUNT(*) 
            FROM {SNOWFLAKE_PAPERS_TABLE} p
            LEFT JOIN {SNOWFLAKE_TEXT_PROCESSED_TABLE} pp ON p.PMC_ID = pp.PAPER_ID
            WHERE pp.PAPER_ID IS NULL
            """
        else:
            query = f"SELECT COUNT(*) FROM {SNOWFLAKE_PAPERS_TABLE}"
        
        cursor.execute(query)
        unprocessed_count = cursor.fetchone()[0]
        
        print(f"📊 Found {unprocessed_count} unprocessed papers remaining")
        
        # Safety check: Get iteration count from Airflow Variables
        try:
            iteration_count = int(Variable.get("text_processing_iterations", default_var="0"))
        except:
            iteration_count = 0
        
        # Safety limit: Stop after 200 iterations (200 * 25 = 5000 papers max)
        MAX_ITERATIONS = 200
        
        if iteration_count >= MAX_ITERATIONS:
            print(f"⚠️  Reached maximum iterations ({MAX_ITERATIONS}). Stopping for safety.")
            return 'end_processing'
        
        # Check manual stop flag
        try:
            manual_stop = Variable.get("text_processing_stop", default_var="false").lower() == "true"
        except:
            manual_stop = False
        
        if manual_stop:
            print("🛑 Manual stop flag detected. Stopping processing.")
            return 'end_processing'
        
        # Increment iteration counter - MUST BE STRING!
        Variable.set("text_processing_iterations", str(iteration_count + 1))
        
        if unprocessed_count > 0:
            print(f"🔄 More papers to process. Triggering next run (iteration {iteration_count + 1})")
            return 'trigger_next_run'
        else:
            print("🎉 All papers processed! Resetting iteration counter.")
            Variable.set("text_processing_iterations", "0")
            return 'end_processing'
        
    finally:
        cursor.close()
        conn.close()


# Task definitions
task_get_papers = PythonOperator(
    task_id='get_unprocessed_papers',
    python_callable=get_unprocessed_papers,
    dag=dag,
)

task_parse_and_chunk = PythonOperator(
    task_id='parse_and_chunk',
    python_callable=parse_and_chunk,
    dag=dag,
)

task_create_and_insert = PythonOperator(
    task_id='create_and_insert_embeddings',
    python_callable=create_and_insert_embeddings,
    dag=dag,
)

task_mark_processed = PythonOperator(
    task_id='mark_papers_as_processed',
    python_callable=mark_papers_as_processed,
    dag=dag,
)

task_check_more = BranchPythonOperator(
    task_id='check_for_more_papers',
    python_callable=check_for_more_papers,
    dag=dag,
)

task_trigger_next = TriggerDagRunOperator(
    task_id='trigger_next_run',
    trigger_dag_id='text_embeddings_pipeline',
    wait_for_completion=False,
    dag=dag,
)

task_end = EmptyOperator(
    task_id='end_processing',
    dag=dag,
)

# Task dependencies - Fixed temp table pipeline
task_get_papers >> task_parse_and_chunk >> task_create_and_insert >> task_mark_processed >> task_check_more
task_check_more >> [task_trigger_next, task_end]