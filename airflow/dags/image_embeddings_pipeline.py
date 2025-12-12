from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.python import BranchPythonOperator
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
S3_BUCKET_IMAGES = os.getenv('S3_IMAGES_BUCKET', 'oncodetectai-researchpaper-images')
S3_IMAGES_PREFIX = os.getenv('S3_IMAGES_PREFIX', 'images')

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
SNOWFLAKE_IMAGE_EMBEDDINGS_TABLE = os.getenv('SNOWFLAKE_IMAGE_EMBEDDINGS_TABLE', 'IMAGE_EMBEDDINGS_TABLE')
SNOWFLAKE_PARSED_IMAGES_TABLE = os.getenv('SNOWFLAKE_PARSED_IMAGES_METADATA_TABLE', 'PARSED_IMAGES_METADATA')
SNOWFLAKE_IMAGES_PROCESSED_TABLE = os.getenv('SNOWFLAKE_IMAGES_PROCESSED_TABLE', 'IMAGES_PROCESSED_PAPERS')

# Processing configuration
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 25))
CORTEX_IMAGE_MODEL = os.getenv('CORTEX_IMAGE_MODEL', 'voyage-multimodal-3')

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
    'image_embeddings_pipeline',
    default_args=default_args,
    description='Extract images from PDFs, create embeddings, and store in Snowflake',
    schedule=None,
    catchup=False,
    tags=['images', 'embeddings', 'processing'],
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


def get_papers_for_image_processing(**context):
    """Step 1: Query Snowflake to find papers that need image processing."""
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    
    try:
        # Check if images_processed_papers table exists
        check_table_query = f"""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = '{SNOWFLAKE_SCHEMA}' 
        AND TABLE_NAME = '{SNOWFLAKE_IMAGES_PROCESSED_TABLE}'
        """
        
        cursor.execute(check_table_query)
        table_exists = cursor.fetchone()[0] > 0
        
        if table_exists:
            # Get papers that haven't been processed for images yet
            query = f"""
            SELECT p.PMC_ID, p.S3_URL, p.DOI, p.TITLE
            FROM {SNOWFLAKE_PAPERS_TABLE} p
            LEFT JOIN {SNOWFLAKE_IMAGES_PROCESSED_TABLE} ip ON p.PMC_ID = ip.PAPER_ID
            WHERE ip.PAPER_ID IS NULL
            LIMIT {BATCH_SIZE}
            """
        else:
            # Table doesn't exist, get all papers
            query = f"""
            SELECT PMC_ID, S3_URL, DOI, TITLE
            FROM {SNOWFLAKE_PAPERS_TABLE}
            LIMIT {BATCH_SIZE}
            """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        papers_to_process = [
            {'paper_id': row[0], 's3_url': row[1], 'doi': row[2], 'title': row[3]}
            for row in results
        ]
        
        print(f"📄 Found {len(papers_to_process)} papers needing image processing")
        
        context['task_instance'].xcom_push(key='papers_to_process', value=papers_to_process)
        return len(papers_to_process)
        
    finally:
        cursor.close()
        conn.close()


def extract_images_from_pdfs(**context):
    """Step 2: Extract images from PDFs and upload to S3."""
    papers = context['task_instance'].xcom_pull(task_ids='get_papers_for_image_processing', key='papers_to_process')
    s3_client = get_s3_client()
    
    all_images = []
    
    for paper in papers:
        try:
            # Download PDF from S3
            s3_key = paper['s3_url'].replace(f"s3://{S3_BUCKET_PAPERS}/", "")
            response = s3_client.get_object(Bucket=S3_BUCKET_PAPERS, Key=s3_key)
            pdf_content = response['Body'].read()
            
            # Open PDF and extract images
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        image_hash = hashlib.sha256(image_bytes).hexdigest()
                        
                        # Get image dimensions
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        width, height = pil_image.size
                        
                        # Upload to S3
                        image_key = f"{S3_IMAGES_PREFIX}/{paper['paper_id']}/page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                        s3_client.put_object(
                            Bucket=S3_BUCKET_IMAGES,
                            Key=image_key,
                            Body=image_bytes,
                            ContentType=f'image/{image_ext}'
                        )
                        
                        image_s3_url = f"s3://{S3_BUCKET_IMAGES}/{image_key}"
                        
                        all_images.append({
                            'paper_id': paper['paper_id'],
                            'page_number': page_num + 1,
                            'image_index': img_index + 1,
                            's3_url': image_s3_url,
                            's3_key': image_key,
                            'image_hash': image_hash,
                            'width': width,
                            'height': height,
                            'format': image_ext
                        })
                        
                    except Exception as e:
                        print(f"❌ Error extracting image on page {page_num + 1}, index {img_index + 1}: {str(e)}")
                        continue
            
            doc.close()
            
            paper_image_count = len([img for img in all_images if img['paper_id'] == paper['paper_id']])
            if paper_image_count > 0:
                print(f"✅ Extracted {paper_image_count} images from {paper['paper_id']}")
            else:
                print(f"ℹ️  No images found in {paper['paper_id']}")
            
        except Exception as e:
            print(f"❌ Error processing {paper['paper_id']}: {str(e)}")
            continue
    
    context['task_instance'].xcom_push(key='extracted_images', value=all_images)
    print(f"🖼️  Total images extracted: {len(all_images)}")
    return len(all_images)


def create_image_embeddings(**context):
    """Step 3: Create embeddings for all images using AI_EMBED - Voyage Multimodal model."""
    images = context['task_instance'].xcom_pull(task_ids='extract_images_from_pdfs', key='extracted_images')
    
    if not images:
        print("⏭️  No images to process - skipping embedding creation")
        context['task_instance'].xcom_push(key='images_with_embeddings', value=[])
        return 0
    
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    
    images_with_embeddings = []
    
    try:
        for i, image in enumerate(images):
            try:
                # Use AI_EMBED with Voyage multimodal model
                # Build scoped file URL for the image in S3 stage
                embedding_query = f"""
                SELECT AI_EMBED(
                    '{CORTEX_IMAGE_MODEL}',
                    BUILD_SCOPED_FILE_URL(@image_stage, '{image['s3_key']}')
                )
                """
                
                cursor.execute(embedding_query)
                result = cursor.fetchone()
                
                # Parse embedding
                embedding = json.loads(result[0]) if isinstance(result[0], str) else result[0]
                
                # Add embedding to image
                image['embedding'] = embedding
                images_with_embeddings.append(image)
                
                if (i + 1) % 20 == 0:
                    print(f"Created embeddings for {i + 1}/{len(images)} images")
                
            except Exception as e:
                print(f"❌ Error creating embedding for image {i} (paper: {image['paper_id']}, page: {image['page_number']}): {str(e)}")
                continue
        
        context['task_instance'].xcom_push(key='images_with_embeddings', value=images_with_embeddings)
        print(f"✅ Successfully created {len(images_with_embeddings)} image embeddings")
        return len(images_with_embeddings)
        
    finally:
        cursor.close()
        conn.close()


def insert_image_embeddings(**context):
    """Step 4: Insert image embeddings into Snowflake using staging table."""
    images = context['task_instance'].xcom_pull(task_ids='create_image_embeddings', key='images_with_embeddings')
    
    if not images:
        print("⏭️  No image embeddings to insert")
        return 0
    
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    
    try:
        # Create temporary staging table
        cursor.execute("""
        CREATE TEMPORARY TABLE IF NOT EXISTS TEMP_IMAGE_EMBEDDINGS_STAGE (
            PAPER_ID VARCHAR,
            PAGE_NUMBER INTEGER,
            IMAGE_INDEX INTEGER,
            S3_URL VARCHAR,
            IMAGE_HASH VARCHAR,
            EMBEDDING_JSON TEXT,
            WIDTH INTEGER,
            HEIGHT INTEGER,
            FORMAT VARCHAR
        )
        """)
        
        # Insert images into staging table
        for image in images:
            cursor.execute(
                """
                INSERT INTO TEMP_IMAGE_EMBEDDINGS_STAGE
                (PAPER_ID, PAGE_NUMBER, IMAGE_INDEX, S3_URL, IMAGE_HASH, EMBEDDING_JSON, WIDTH, HEIGHT, FORMAT)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    image['paper_id'],
                    image['page_number'],
                    image['image_index'],
                    image['s3_url'],
                    image['image_hash'],
                    json.dumps(image['embedding']),
                    image['width'],
                    image['height'],
                    image['format']
                )
            )
        
        print(f"Staged {len(images)} images")
        
        # Copy from staging to final table with VECTOR conversion (1024 dimensions for Voyage Multimodal!)
        copy_query = f"""
        INSERT INTO {SNOWFLAKE_IMAGE_EMBEDDINGS_TABLE}
        (paper_id, page_number, image_index, s3_url, image_hash, embedding, width, height, format, created_at)
        SELECT 
            PAPER_ID,
            PAGE_NUMBER,
            IMAGE_INDEX,
            S3_URL,
            IMAGE_HASH,
            PARSE_JSON(EMBEDDING_JSON)::VECTOR(FLOAT, 1024),
            WIDTH,
            HEIGHT,
            FORMAT,
            CURRENT_TIMESTAMP()
        FROM TEMP_IMAGE_EMBEDDINGS_STAGE
        """
        
        cursor.execute(copy_query)
        rows_inserted = cursor.rowcount
        conn.commit()
        
        print(f"✅ Successfully inserted {rows_inserted} image embeddings into {SNOWFLAKE_IMAGE_EMBEDDINGS_TABLE}")
        
        # Insert metadata for tracking
        for image in images:
            cursor.execute(
                f"""
                INSERT INTO {SNOWFLAKE_PARSED_IMAGES_TABLE}
                (paper_id, page_number, image_index, image_hash, s3_url, width, height, format, extracted_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP())
                """,
                (
                    image['paper_id'],
                    image['page_number'],
                    image['image_index'],
                    image['image_hash'],
                    image['s3_url'],
                    image['width'],
                    image['height'],
                    image['format']
                )
            )
        
        conn.commit()
        print(f"✅ Inserted metadata for {len(images)} images")
        
        return rows_inserted
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise
    finally:
        cursor.close()
        conn.close()


def mark_papers_as_image_processed(**context):
    """Step 5: Mark papers as image-processed in the tracking table."""
    papers = context['task_instance'].xcom_pull(task_ids='get_papers_for_image_processing', key='papers_to_process')
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    
    try:
        processed_count = 0
        
        for paper in papers:
            paper_id = paper['paper_id']
            
            # Check how many images were processed for this paper
            cursor.execute(
                f"SELECT COUNT(*) FROM {SNOWFLAKE_IMAGE_EMBEDDINGS_TABLE} WHERE paper_id = %s",
                (paper_id,)
            )
            image_count = cursor.fetchone()[0]
            
            # Mark as processed (even if 0 images, to avoid reprocessing)
            cursor.execute(
                f"""
                INSERT INTO {SNOWFLAKE_IMAGES_PROCESSED_TABLE} 
                (paper_id, processed_at, images_count)
                VALUES (%s, CURRENT_TIMESTAMP(), %s)
                """,
                (paper_id, image_count)
            )
            processed_count += 1
        
        conn.commit()
        print(f"✅ Marked {processed_count} papers as image-processed")
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
    """Step 6: Check if more papers need processing."""
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    
    try:
        # Check if there are more unprocessed papers
        query = f"""
        SELECT COUNT(*) 
        FROM {SNOWFLAKE_PAPERS_TABLE} p
        LEFT JOIN {SNOWFLAKE_IMAGES_PROCESSED_TABLE} ip ON p.PMC_ID = ip.PAPER_ID
        WHERE ip.PAPER_ID IS NULL
        """
        
        cursor.execute(query)
        remaining_papers = cursor.fetchone()[0]
        
        print(f"📊 Remaining unprocessed papers: {remaining_papers}")
        
        if remaining_papers > 0:
            print(f"🔄 More papers to process - {remaining_papers} remaining")
            return 'trigger_next_batch'  # Branch to trigger task
        else:
            print("✅ All papers processed! No more batches needed.")
            return 'end_pipeline'  # Branch to end task
        
    except Exception as e:
        print(f"❌ Error checking for remaining papers: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise
    finally:
        cursor.close()
        conn.close()


def end_pipeline(**context):
    """Final task when all processing is complete."""
    print("🎉 Image processing pipeline completed successfully!")
    print("All papers have been processed.")
    return True


# Task definitions
task_get_papers = PythonOperator(
    task_id='get_papers_for_image_processing',
    python_callable=get_papers_for_image_processing,
    dag=dag,
)

task_extract_images = PythonOperator(
    task_id='extract_images_from_pdfs',
    python_callable=extract_images_from_pdfs,
    dag=dag,
)

task_create_image_embeddings = PythonOperator(
    task_id='create_image_embeddings',
    python_callable=create_image_embeddings,
    dag=dag,
)

task_insert_image_embeddings = PythonOperator(
    task_id='insert_image_embeddings',
    python_callable=insert_image_embeddings,
    dag=dag,
)

task_mark_processed = PythonOperator(
    task_id='mark_papers_as_image_processed',
    python_callable=mark_papers_as_image_processed,
    dag=dag,
)

task_check_more = BranchPythonOperator(
    task_id='check_for_more_papers',
    python_callable=check_for_more_papers,
    dag=dag,
)

task_trigger_next = TriggerDagRunOperator(
    task_id='trigger_next_batch',
    trigger_dag_id='image_embeddings_pipeline',
    wait_for_completion=False,
    dag=dag,
)

task_end = PythonOperator(
    task_id='end_pipeline',
    python_callable=end_pipeline,
    dag=dag,
)

# Task dependencies - Linear pipeline with conditional branching
task_get_papers >> task_extract_images >> task_create_image_embeddings >> task_insert_image_embeddings >> task_mark_processed >> task_check_more
task_check_more >> [task_trigger_next, task_end]