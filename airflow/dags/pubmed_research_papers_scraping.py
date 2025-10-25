from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import logging
from xml.etree import ElementTree as ET
import re
import os
import json
import snowflake.connector

# ============================================
# CONFIGURATION - Read from environment variables
# ============================================

# S3 Configuration
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'your-bucket-name')
S3_PREFIX = os.getenv('S3_PREFIX', 'pubmed_papers/')

# Snowflake Configuration
SNOWFLAKE_ACCOUNT = os.getenv('SNOWFLAKE_ACCOUNT', 'your_account')
SNOWFLAKE_USER = os.getenv('SNOWFLAKE_USER', 'your_user')
SNOWFLAKE_PASSWORD = os.getenv('SNOWFLAKE_PASSWORD', '')  # PAT token goes here as password
SNOWFLAKE_DATABASE = os.getenv('SNOWFLAKE_DATABASE', 'your_database')
SNOWFLAKE_SCHEMA = os.getenv('SNOWFLAKE_SCHEMA', 'your_schema')
SNOWFLAKE_WAREHOUSE = os.getenv('SNOWFLAKE_WAREHOUSE', 'your_warehouse')
SNOWFLAKE_ROLE = os.getenv('SNOWFLAKE_ROLE', 'your_role')
SNOWFLAKE_TABLE = os.getenv('SNOWFLAKE_TABLE', 'pubmed_papers')

# Search Configuration
SEARCH_TERMS = ["Small Cell Lung Cancer", "SCLC"]
MAX_PAPERS = 1000

# ============================================
# NEW TASK: FETCH PMC IDs
# ============================================

def fetch_pmc_ids_for_search(**context):
    """
    Fetch PMC IDs for papers related to Small Cell Lung Cancer (SCLC)
    Returns the most recent 1000 papers
    """
    print("=" * 100)
    print("FETCH_PMC_IDS FUNCTION STARTED")
    print("=" * 100)
    
    logging.info("=" * 100)
    logging.info("STARTING FETCH_PMC_IDS TASK")
    logging.info(f"Search terms: {SEARCH_TERMS}")
    logging.info(f"Max papers to fetch: {MAX_PAPERS}")
    logging.info("=" * 100)
    
    # Build search query for PubMed
    # Using OR to capture all variations
    search_query = ' OR '.join([f'"{term}"[Title/Abstract]' for term in SEARCH_TERMS])
    
    logging.info(f"PubMed search query: {search_query}")
    print(f"Search query: {search_query}")
    
    all_pmc_ids = []
    
    try:
        # Step 1: Search PubMed to get PubMed IDs
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        esearch_params = {
            'db': 'pubmed',
            'term': search_query,
            'retmax': MAX_PAPERS,
            'retmode': 'json',
            'sort': 'relevance',  # Most recent first
            'usehistory': 'y'
        }
        
        logging.info(f"Searching PubMed for papers...")
        print(f"Searching PubMed...")
        
        response = requests.get(esearch_url, params=esearch_params, timeout=30)
        response.raise_for_status()
        search_data = response.json()
        
        if 'esearchresult' not in search_data:
            logging.error("Invalid response from PubMed search")
            return []
        
        pubmed_ids = search_data['esearchresult'].get('idlist', [])
        count = search_data['esearchresult'].get('count', 0)
        
        logging.info(f"Found {count} total papers matching search criteria")
        logging.info(f"Retrieved {len(pubmed_ids)} PubMed IDs")
        print(f"Found {len(pubmed_ids)} PubMed IDs")
        
        if not pubmed_ids:
            logging.warning("No papers found matching search criteria")
            return []
        
        # Step 2: Convert PubMed IDs to PMC IDs (in batches)
        # PMC has full-text articles, not all PubMed articles are in PMC
        batch_size = 200  # Process 200 IDs at a time
        
        for i in range(0, len(pubmed_ids), batch_size):
            batch = pubmed_ids[i:i+batch_size]
            batch_str = ','.join(batch)
            
            logging.info(f"Converting batch {i//batch_size + 1} ({len(batch)} IDs) to PMC IDs...")
            print(f"Converting batch {i//batch_size + 1}...")
            
            # Use ID Converter API to get PMC IDs
            idconv_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
            idconv_params = {
                'ids': batch_str,
                'format': 'json'
            }
            
            try:
                conv_response = requests.get(idconv_url, params=idconv_params, timeout=30)
                conv_response.raise_for_status()
                conv_data = conv_response.json()
                
                if 'records' in conv_data:
                    for record in conv_data['records']:
                        pmc_id = record.get('pmcid')
                        if pmc_id:
                            all_pmc_ids.append(pmc_id)
                
                logging.info(f"Batch {i//batch_size + 1}: Found {len([r for r in conv_data.get('records', []) if r.get('pmcid')])} PMC IDs")
                
            except Exception as e:
                logging.error(f"Error converting batch {i//batch_size + 1}: {str(e)}")
                continue
            
            # Be nice to NCBI servers
            import time
            time.sleep(0.5)
        
        # Remove duplicates and limit to MAX_PAPERS
        all_pmc_ids = list(dict.fromkeys(all_pmc_ids))[:MAX_PAPERS]
        
        logging.info("=" * 100)
        logging.info(f"SEARCH COMPLETE")
        logging.info(f"Total unique PMC IDs found: {len(all_pmc_ids)}")
        logging.info(f"Sample PMC IDs: {all_pmc_ids[:5]}")
        logging.info("=" * 100)
        
        print("=" * 100)
        print(f"SEARCH COMPLETE: Found {len(all_pmc_ids)} PMC IDs")
        print(f"Sample: {all_pmc_ids[:5]}")
        print("=" * 100)
        
        # Push PMC IDs to XCom for next task
        context['ti'].xcom_push(key='pmc_ids', value=all_pmc_ids)
        
        return len(all_pmc_ids)
        
    except Exception as e:
        error_msg = f"CRITICAL ERROR in fetch_pmc_ids_for_search: {str(e)}"
        logging.error(error_msg)
        logging.exception("Full traceback:")
        print(error_msg)
        raise

# ============================================
# SNOWFLAKE CONNECTION
# ============================================

def get_snowflake_connection():
    """
    Create Snowflake connection using password (PAT token as password)
    Returns a Snowflake connection object
    """
    import snowflake.connector
    
    try:
        # Log connection attempt details (without sensitive info)
        logging.info(f"Attempting Snowflake connection:")
        logging.info(f"  Account: {SNOWFLAKE_ACCOUNT}")
        logging.info(f"  User: {SNOWFLAKE_USER}")
        logging.info(f"  Database: {SNOWFLAKE_DATABASE}")
        logging.info(f"  Schema: {SNOWFLAKE_SCHEMA}")
        logging.info(f"  Warehouse: {SNOWFLAKE_WAREHOUSE}")
        logging.info(f"  Authentication: Password (PAT token)")
        
        if not SNOWFLAKE_PASSWORD:
            raise ValueError("SNOWFLAKE_PASSWORD (PAT token) is required but not set")
        
        connection_params = {
            'account': SNOWFLAKE_ACCOUNT,
            'user': SNOWFLAKE_USER,
            'password': SNOWFLAKE_PASSWORD,
            'warehouse': SNOWFLAKE_WAREHOUSE,
            'database': SNOWFLAKE_DATABASE,
            'schema': SNOWFLAKE_SCHEMA,
        }
        
        # Add role if specified
        if SNOWFLAKE_ROLE:
            connection_params['role'] = SNOWFLAKE_ROLE
        
        conn = snowflake.connector.connect(**connection_params)
        logging.info("✅ Successfully created Snowflake connection")
        return conn
    except Exception as e:
        logging.error(f"❌ Failed to create Snowflake connection: {str(e)}")
        logging.error(f"Connection parameters used (password hidden):")
        safe_params = {k: v for k, v in connection_params.items() if k != 'password'}
        logging.error(f"  {safe_params}")
        raise

# ============================================
# DUPLICATE CHECKING FUNCTIONS
# ============================================

def check_existing_papers_in_snowflake(pmc_ids):
    """
    Check which PMC IDs already exist in Snowflake
    Returns a set of PMC IDs that are already processed
    """
    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        # Create a temporary table to hold PMC IDs for efficient lookup
        pmc_ids_str = "','".join(pmc_ids)
        
        query = f"""
        SELECT DISTINCT PMC_ID 
        FROM {SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.{SNOWFLAKE_TABLE}
        WHERE PMC_ID IN ('{pmc_ids_str}')
        """
        
        logging.info(f"Checking for existing papers in Snowflake...")
        cursor.execute(query)
        result = cursor.fetchall()
        
        existing_ids = {row[0] for row in result if row}
        logging.info(f"Found {len(existing_ids)} papers already in Snowflake")
        
        cursor.close()
        conn.close()
        
        return existing_ids
        
    except Exception as e:
        logging.warning(f"Error checking Snowflake for existing papers: {str(e)}")
        logging.warning("Proceeding without duplicate check from Snowflake")
        return set()


def check_existing_papers_in_s3(pmc_ids):
    """
    Check which PMC IDs already have PDFs in S3
    Returns a set of PMC IDs that already have PDFs
    """
    try:
        s3_hook = S3Hook(aws_conn_id='aws_default')
        existing_ids = set()
        
        logging.info(f"Checking for existing PDFs in S3...")
        
        for pmc_id in pmc_ids:
            pdf_key = f"{S3_PREFIX}{pmc_id}.pdf"
            if s3_hook.check_for_key(key=pdf_key, bucket_name=S3_BUCKET_NAME):
                existing_ids.add(pmc_id)
        
        logging.info(f"Found {len(existing_ids)} PDFs already in S3")
        
        return existing_ids
        
    except Exception as e:
        logging.warning(f"Error checking S3 for existing PDFs: {str(e)}")
        logging.warning("Proceeding without duplicate check from S3")
        return set()


def filter_already_processed_papers(pmc_ids):
    """
    Filter out papers that are already in S3 (since that's the expensive operation)
    Returns list of PMC IDs that need to be processed
    """
    logging.info("=" * 100)
    logging.info("CHECKING FOR DUPLICATE PAPERS")
    logging.info("=" * 100)
    
    print("=" * 100)
    print("DUPLICATE CHECK STARTING")
    print("=" * 100)
    
    # Check S3 first (most important - avoid re-downloading PDFs)
    existing_in_s3 = check_existing_papers_in_s3(pmc_ids)
    
    # Check Snowflake for informational purposes
    existing_in_snowflake = check_existing_papers_in_snowflake(pmc_ids)
    
    # Skip papers that are already in S3 (regardless of Snowflake status)
    # If PDF is in S3 but not in Snowflake, the Snowflake task will handle it
    papers_to_process = [pmc_id for pmc_id in pmc_ids if pmc_id not in existing_in_s3]
    
    logging.info("=" * 100)
    logging.info(f"DUPLICATE CHECK COMPLETE")
    logging.info(f"Total PMC IDs: {len(pmc_ids)}")
    logging.info(f"Already in S3: {len(existing_in_s3)}")
    logging.info(f"Already in Snowflake: {len(existing_in_snowflake)}")
    logging.info(f"Papers to scrape and upload: {len(papers_to_process)}")
    logging.info("=" * 100)
    
    print("=" * 100)
    print(f"DUPLICATE CHECK RESULTS:")
    print(f"  Total papers: {len(pmc_ids)}")
    print(f"  Already in S3: {len(existing_in_s3)}")
    print(f"  To scrape/upload: {len(papers_to_process)}")
    print("=" * 100)
    
    if existing_in_s3:
        logging.info(f"Skipping {len(existing_in_s3)} papers already in S3")
        logging.info(f"Sample skipped: {list(existing_in_s3)[:5]}")
    
    return papers_to_process

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_pubmed_id_from_pmc(pmc_id):
    """Convert PMC ID to PubMed ID using the ID converter API"""
    url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmc_id}&format=json"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        if 'records' in data and len(data['records']) > 0:
            return data['records'][0].get('pmid')
    except Exception as e:
        logging.warning(f"Could not convert {pmc_id} to PMID: {str(e)}")
    return None


def fetch_pubmed_metadata(pubmed_id):
    """Fetch metadata from PubMed using the E-utilities API"""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        'db': 'pubmed',
        'id': pubmed_id,
        'retmode': 'xml',
        'rettype': 'abstract'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logging.error(f"Error fetching metadata for PMID {pubmed_id}: {str(e)}")
        return None


def parse_pubmed_xml(xml_content):
    """Parse PubMed XML response and extract relevant information"""
    try:
        root = ET.fromstring(xml_content)
        article = root.find('.//PubmedArticle')
        
        if article is None:
            return None
        
        # Extract title
        title_elem = article.find('.//ArticleTitle')
        title = title_elem.text if title_elem is not None else "No title"
        
        # Extract abstract
        abstract_texts = []
        for abstract_elem in article.findall('.//AbstractText'):
            if abstract_elem.text:
                abstract_texts.append(abstract_elem.text)
        abstract = " ".join(abstract_texts) if abstract_texts else "No abstract available"
        
        # Extract authors
        authors = []
        for author in article.findall('.//Author'):
            lastname = author.find('LastName')
            forename = author.find('ForeName')
            if lastname is not None and forename is not None:
                authors.append(f"{forename.text} {lastname.text}")
        authors_str = ", ".join(authors) if authors else "No authors listed"
        
        # Extract DOI
        doi = None
        for article_id in article.findall('.//ArticleId'):
            if article_id.get('IdType') == 'doi':
                doi = article_id.text
                break
        
        return {
            'title': title,
            'abstract': abstract,
            'authors': authors_str,
            'doi': doi if doi else "No DOI available"
        }
    except ET.ParseError as e:
        logging.error(f"Error parsing XML: {str(e)}")
        return None


def get_pdf_url_from_pmc_page(pmc_id):
    """Scrape the PMC page to find the PDF download link"""
    pmc_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/"
    
    # Add headers to avoid being blocked
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.ncbi.nlm.nih.gov/',
    }
    
    try:
        response = requests.get(pmc_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Method 1: Look for citation_pdf_url meta tag (most reliable)
        article_meta = soup.find('meta', {'name': 'citation_pdf_url'})
        if article_meta and article_meta.get('content'):
            pdf_url = article_meta.get('content')
            logging.info(f"Found PDF URL via meta tag for {pmc_id}: {pdf_url}")
            return pdf_url
        
        # Method 2: Look for direct PDF link
        pdf_link = soup.find('a', {'class': 'int-view'})  # PMC's PDF link class
        if not pdf_link:
            pdf_link = soup.find('a', href=re.compile(r'/pdf/'))
        
        if pdf_link and pdf_link.get('href'):
            pdf_path = pdf_link.get('href')
            if pdf_path.startswith('/'):
                pdf_url = f"https://pmc.ncbi.nlm.nih.gov{pdf_path}"
            else:
                pdf_url = pdf_path
            
            logging.info(f"Found PDF URL via link for {pmc_id}: {pdf_url}")
            return pdf_url
        
        # Method 3: Construct standard PMC PDF URL pattern
        # Many PMC articles follow this pattern
        standard_pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/"
        logging.info(f"Trying standard PDF pattern for {pmc_id}: {standard_pdf_url}")
        
        # Test if the standard URL works
        test_response = requests.head(standard_pdf_url, headers=headers, timeout=10, allow_redirects=True)
        if test_response.status_code == 200:
            return standard_pdf_url
        
        logging.warning(f"Could not find PDF link for {pmc_id}")
        return None
        
    except requests.RequestException as e:
        logging.error(f"Error fetching PMC page for {pmc_id}: {str(e)}")
        return None


def download_pdf(pdf_url):
    """Download PDF from the given URL"""
    
    # Add headers to avoid being blocked
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/pdf,application/octet-stream,*/*',
        'Referer': 'https://pmc.ncbi.nlm.nih.gov/',
    }
    
    try:
        response = requests.get(pdf_url, headers=headers, timeout=60)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        
        if 'application/pdf' in content_type or 'application/octet-stream' in content_type:
            logging.info(f"Successfully downloaded PDF from {pdf_url} ({len(response.content)} bytes)")
            return response.content
        else:
            logging.warning(f"URL did not return PDF content (got: {content_type}): {pdf_url}")
            return None
            
    except requests.RequestException as e:
        logging.error(f"Error downloading PDF from {pdf_url}: {str(e)}")
        return None
    
def get_pdf_from_pmc_ftp(pmc_id):
    """
    Try to download PDF from PMC's Open Access FTP server
    This is the most reliable method for open-access articles
    """
    import time
    
    # Remove "PMC" prefix if present
    pmc_number = pmc_id.replace('PMC', '')
    
    # PMC organizes files by number ranges
    prefix_2digit = pmc_number[:2] if len(pmc_number) >= 2 else pmc_number
    
    # Try multiple FTP URL patterns
    ftp_patterns = [
        # Pattern 1: Standard OA path with full naming
        f"https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/{prefix_2digit}/{pmc_number}/{pmc_id}.PMC{pmc_number}.pdf",
        # Pattern 2: Alternative naming
        f"https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/{prefix_2digit}/{pmc_number}/PMC{pmc_number}.pdf",
        # Pattern 3: Just PMC ID
        f"https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/{prefix_2digit}/{pmc_number}/{pmc_id}.pdf",
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0)',
    }
    
    for idx, ftp_url in enumerate(ftp_patterns, 1):
        try:
            logging.info(f"FTP attempt {idx}/{len(ftp_patterns)}: {ftp_url}")
            response = requests.get(ftp_url, headers=headers, timeout=60, allow_redirects=True)
            
            if response.status_code == 200 and len(response.content) > 1000:  # PDFs are at least 1KB
                # Verify it's actually a PDF by checking magic bytes
                if response.content[:4] == b'%PDF':
                    logging.info(f"✅ Successfully downloaded PDF from FTP for {pmc_id} ({len(response.content)} bytes)")
                    return response.content
                else:
                    logging.debug(f"Response was not a PDF (magic bytes: {response.content[:4]})")
            else:
                logging.debug(f"FTP attempt {idx} failed: status={response.status_code}, size={len(response.content)}")
                
        except requests.RequestException as e:
            logging.debug(f"FTP attempt {idx} error: {str(e)}")
            continue
        
        # Small delay between attempts
        if idx < len(ftp_patterns):
            time.sleep(0.5)
    
    logging.warning(f"All FTP patterns failed for {pmc_id}")
    return None

def get_pdf_from_europepmc(pmc_id):
    """
    Download PDF from Europe PMC (mirror of PMC with better API access)
    """
    pmc_number = pmc_id.replace('PMC', '')
    
    # Europe PMC provides direct PDF access
    epmc_url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC{pmc_number}&blobtype=pdf"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/pdf',
    }
    
    try:
        logging.info(f"Europe PMC attempt: {epmc_url}")
        # Reduced timeout to 2 minutes instead of default 60
        response = requests.get(epmc_url, headers=headers, timeout=120, allow_redirects=True)
        
        if response.status_code == 200 and len(response.content) > 1000:
            # Verify it's actually a PDF
            if response.content[:4] == b'%PDF':
                logging.info(f"✅ Successfully downloaded PDF from Europe PMC for {pmc_id} ({len(response.content)} bytes)")
                return response.content
            else:
                logging.debug(f"Europe PMC returned non-PDF content (size: {len(response.content)})")
        else:
            logging.debug(f"Europe PMC failed: status={response.status_code}")
            
    except requests.Timeout:
        logging.warning(f"Europe PMC timeout for {pmc_id} after 2 minutes")
    except requests.RequestException as e:
        logging.error(f"Europe PMC error for {pmc_id}: {str(e)}")
    
    return None

# ============================================
# COMBINED SCRAPE AND UPLOAD TASK
# ============================================

def scrape_and_upload_papers(**context):
    """
    Combined function to scrape papers from PMC and upload PDFs to S3
    Returns results for papers that were successfully uploaded
    """
    
    print("=" * 100)
    print("SCRAPE_AND_UPLOAD_PAPERS FUNCTION STARTED")
    print("=" * 100)
    
    logging.info("=" * 100)
    logging.info("STARTING SCRAPE_AND_UPLOAD_PAPERS TASK")
    logging.info("=" * 100)
    
    # Get PMC IDs from previous task
    ti = context['ti']
    all_pmc_ids = ti.xcom_pull(key='pmc_ids', task_ids='fetch_pmc_ids')
    
    # Log configuration
    logging.info(f"Configuration at runtime:")
    logging.info(f"Total PMC_IDS fetched: {len(all_pmc_ids) if all_pmc_ids else 0}")
    logging.info(f"S3 Bucket: {S3_BUCKET_NAME}")
    logging.info(f"Snowflake: {SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.{SNOWFLAKE_TABLE}")
    
    print(f"Total PMC_IDS fetched: {len(all_pmc_ids) if all_pmc_ids else 0} papers")
    
    if not all_pmc_ids:
        error_msg = "ERROR: PMC_IDS is empty!"
        logging.error(error_msg)
        print(error_msg)
        return 0
    
    # Filter out already processed papers
    PMC_IDS = filter_already_processed_papers(all_pmc_ids)
    
    if not PMC_IDS:
        logging.info("All papers have already been processed. Nothing to do.")
        print("All papers already processed!")
        # Still return empty results for Snowflake task
        ti.xcom_push(key='successful_papers', value=[])
        ti.xcom_push(key='failed_pmc_ids', value=[])
        return 0
    
    logging.info(f"Processing {len(PMC_IDS)} new papers")
    print(f"Processing {len(PMC_IDS)} new papers")
    
    # Initialize S3 hook for immediate uploads
    try:
        s3_hook = S3Hook(aws_conn_id='aws_default')
        logging.info("S3 Hook created successfully")
    except Exception as e:
        logging.error(f"Failed to create S3 hook: {str(e)}")
        raise
    
    successful_papers = []
    failed_pmc_ids = []
    
    try:
        for idx, pmc_id in enumerate(PMC_IDS, 1):
            try:
                logging.info("-" * 80)
                logging.info(f"Processing {idx}/{len(PMC_IDS)}: {pmc_id}")
                print(f"Processing {idx}/{len(PMC_IDS)}: {pmc_id}")
                logging.info("-" * 80)
                
                # Get PubMed ID for metadata
                pubmed_id = get_pubmed_id_from_pmc(pmc_id)
                logging.info(f"PubMed ID obtained: {pubmed_id}")
                
                metadata = None
                if pubmed_id:
                    # Fetch metadata from PubMed
                    xml_content = fetch_pubmed_metadata(pubmed_id)
                    if xml_content:
                        metadata = parse_pubmed_xml(xml_content)
                
                # If metadata fetch failed, use placeholder values
                if not metadata or not isinstance(metadata, dict):
                    logging.warning(f"Metadata is None or invalid for {pmc_id}, using placeholders")
                    metadata = {
                        'title': f"Paper {pmc_id}",
                        'abstract': "Metadata not available",
                        'authors': "Unknown",
                        'doi': "Not available"
                    }
                else:
                    logging.info(f"Metadata fetched - Title: {metadata.get('title', 'No title')[:50]}...")
                
                # Try multiple methods to get PDF
                pdf_url = None
                pdf_content = None
                
                # Method 1: Try scraping PMC page
                logging.info(f"Method 1: Trying PMC web scraping for {pmc_id}")
                pdf_url = get_pdf_url_from_pmc_page(pmc_id)
                if pdf_url:
                    logging.info(f"Found PDF URL: {pdf_url}")
                    pdf_content = download_pdf(pdf_url)
                    if pdf_content:
                        logging.info(f"✅ PDF downloaded via web scraping: {len(pdf_content)} bytes")
                
                # Method 2: Try FTP if web scraping failed
                if not pdf_content:
                    logging.info(f"Method 2: Trying FTP download for {pmc_id}")
                    pdf_content = get_pdf_from_pmc_ftp(pmc_id)
                    if pdf_content:
                        pdf_url = f"FTP download"
                        logging.info(f"✅ PDF downloaded via FTP: {len(pdf_content)} bytes")
                
                # Method 3: Try Europe PMC as last resort
                if not pdf_content:
                    logging.info(f"Method 3: Trying Europe PMC for {pmc_id}")
                    pdf_content = get_pdf_from_europepmc(pmc_id)
                    if pdf_content:
                        pdf_url = f"Europe PMC download"
                        logging.info(f"✅ PDF downloaded via Europe PMC: {len(pdf_content)} bytes")
                
                # Upload PDF to S3 if we got it
                s3_url = None
                if pdf_content:
                    try:
                        pdf_key = f"{S3_PREFIX}{pmc_id}.pdf"
                        s3_hook.load_bytes(
                            bytes_data=pdf_content,
                            key=pdf_key,
                            bucket_name=S3_BUCKET_NAME,
                            replace=True
                        )
                        s3_url = f"s3://{S3_BUCKET_NAME}/{pdf_key}"
                        logging.info(f"✅ Uploaded to S3: {s3_url}")
                        print(f"✅ SUCCESS: {pmc_id}")
                        
                        # Add to successful papers (only if uploaded to S3)
                        successful_papers.append({
                            'pmc_id': pmc_id,
                            'pubmed_id': pubmed_id,
                            'metadata': metadata,
                            's3_url': s3_url,
                            'pdf_source': pdf_url
                        })
                        
                    except Exception as upload_error:
                        logging.error(f"❌ Failed to upload {pmc_id} to S3: {str(upload_error)}")
                        print(f"❌ S3 UPLOAD FAILED: {pmc_id}")
                        failed_pmc_ids.append(pmc_id)
                else:
                    logging.warning(f"❌ Failed to download PDF for {pmc_id}")
                    print(f"❌ PDF DOWNLOAD FAILED: {pmc_id}")
                    failed_pmc_ids.append(pmc_id)
                
                logging.info(f"Completed processing {pmc_id}")
                
                # Add delay between papers to be nice to servers
                if idx < len(PMC_IDS):
                    import time
                    time.sleep(1)  # Reduced from 2 to 1 second
                    logging.info("Waiting 1 second before next paper...")
                
            except Exception as paper_error:
                logging.error(f"❌ Error processing paper {pmc_id}: {str(paper_error)}")
                logging.exception(f"Full traceback for {pmc_id}:")
                print(f"❌ EXCEPTION: {pmc_id} - {str(paper_error)}")
                failed_pmc_ids.append(pmc_id)
                # Continue with next paper instead of failing entire task
                continue
        
        # Summary
        logging.info("=" * 100)
        logging.info(f"SCRAPE AND UPLOAD COMPLETE")
        logging.info(f"Total papers attempted: {len(PMC_IDS)}")
        logging.info(f"Successfully uploaded to S3: {len(successful_papers)}")
        logging.info(f"Failed papers: {len(failed_pmc_ids)}")
        logging.info("=" * 100)
        
        if failed_pmc_ids:
            logging.warning("=" * 100)
            logging.warning(f"FAILED PMC IDs ({len(failed_pmc_ids)} total):")
            logging.warning(f"{failed_pmc_ids}")
            logging.warning("=" * 100)
        
        print("=" * 100)
        print(f"SUMMARY:")
        print(f"  Attempted: {len(PMC_IDS)}")
        print(f"  Successful: {len(successful_papers)}")
        print(f"  Failed: {len(failed_pmc_ids)}")
        if failed_pmc_ids:
            print(f"  Failed IDs: {failed_pmc_ids[:10]}..." if len(failed_pmc_ids) > 10 else f"  Failed IDs: {failed_pmc_ids}")
        print("=" * 100)
        
        # Push results to XCom
        # Only push papers that were successfully uploaded to S3
        logging.info(f"Pushing {len(successful_papers)} successful papers to XCom for Snowflake")
        ti.xcom_push(key='successful_papers', value=successful_papers)
        ti.xcom_push(key='failed_pmc_ids', value=failed_pmc_ids)
        
        return len(successful_papers)
        
    except Exception as e:
        error_msg = f"CRITICAL ERROR in scrape_and_upload_papers: {str(e)}"
        logging.error(error_msg)
        logging.exception("Full traceback:")
        print(error_msg)
        raise

# ============================================
# SNOWFLAKE INSERT TASK
# ============================================

def insert_to_snowflake(**context):
    """
    Insert metadata and S3 URLs into Snowflake table
    Handles both newly uploaded papers AND papers that were in S3 but missing from Snowflake
    """
    
    print("=" * 100)
    print("INSERT_TO_SNOWFLAKE FUNCTION STARTED")
    print("=" * 100)
    
    logging.info("=" * 100)
    logging.info("STARTING INSERT_TO_SNOWFLAKE TASK")
    logging.info("=" * 100)
    
    try:
        ti = context['ti']
        successful_papers = ti.xcom_pull(key='successful_papers', task_ids='scrape_and_upload')
        failed_pmc_ids = ti.xcom_pull(key='failed_pmc_ids', task_ids='scrape_and_upload')
        
        logging.info(f"Received from previous task:")
        logging.info(f"  Successful papers (newly uploaded): {len(successful_papers) if successful_papers else 0}")
        logging.info(f"  Failed PMC IDs: {len(failed_pmc_ids) if failed_pmc_ids else 0}")
        
        if failed_pmc_ids:
            logging.warning("=" * 100)
            logging.warning(f"REMINDER: {len(failed_pmc_ids)} papers failed in scraping/upload:")
            logging.warning(f"{failed_pmc_ids}")
            logging.warning("=" * 100)
            print(f"⚠️  WARNING: {len(failed_pmc_ids)} papers were not uploaded to S3")
        
        # Get all PMC IDs from fetch task to check for S3-only papers
        all_pmc_ids = ti.xcom_pull(key='pmc_ids', task_ids='fetch_pmc_ids')
        
        # Check which papers are in S3 but not in Snowflake
        logging.info("Checking for papers in S3 but missing from Snowflake...")
        existing_in_s3 = check_existing_papers_in_s3(all_pmc_ids)
        existing_in_snowflake = check_existing_papers_in_snowflake(all_pmc_ids)
        
        # Papers in S3 but not in Snowflake (from previous failed runs)
        s3_only_papers = existing_in_s3 - existing_in_snowflake
        
        if s3_only_papers:
            logging.info(f"Found {len(s3_only_papers)} papers in S3 but missing from Snowflake")
            logging.info(f"Will fetch metadata and insert these papers")
            print(f"📝 Found {len(s3_only_papers)} papers in S3 that need metadata insertion")
            
            # Fetch metadata for S3-only papers
            for pmc_id in s3_only_papers:
                logging.info(f"Fetching metadata for S3-only paper: {pmc_id}")
                
                pubmed_id = get_pubmed_id_from_pmc(pmc_id)
                metadata = None
                
                if pubmed_id:
                    xml_content = fetch_pubmed_metadata(pubmed_id)
                    if xml_content:
                        metadata = parse_pubmed_xml(xml_content)
                
                if not metadata or not isinstance(metadata, dict):
                    metadata = {
                        'title': f"Paper {pmc_id}",
                        'abstract': "Metadata not available",
                        'authors': "Unknown",
                        'doi': "Not available"
                    }
                
                s3_url = f"s3://{S3_BUCKET_NAME}/{S3_PREFIX}{pmc_id}.pdf"
                
                successful_papers.append({
                    'pmc_id': pmc_id,
                    'pubmed_id': pubmed_id,
                    'metadata': metadata,
                    's3_url': s3_url,
                    'pdf_source': 'Already in S3'
                })
        
        print(f"Papers to insert: {len(successful_papers) if successful_papers else 0}")
        
        if not successful_papers:
            logging.warning("No papers to insert into Snowflake")
            print("Nothing to insert - all papers already in Snowflake")
            return 0
        
        logging.info(f"Snowflake Configuration:")
        logging.info(f"Account: {SNOWFLAKE_ACCOUNT}")
        logging.info(f"User: {SNOWFLAKE_USER}")
        logging.info(f"Database: {SNOWFLAKE_DATABASE}")
        logging.info(f"Schema: {SNOWFLAKE_SCHEMA}")
        logging.info(f"Target table: {SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.{SNOWFLAKE_TABLE}")
        
        # Create Snowflake connection using PAT token as password
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        logging.info("Snowflake connection established successfully")
        
        # INSERT query with PMC_ID for duplicate checking
        insert_query = f"""
        INSERT INTO {SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.{SNOWFLAKE_TABLE} 
        (PMC_ID, Title, Abstract, Authors, DOI, S3_URL)
        VALUES (%(pmc_id)s, %(title)s, %(abstract)s, %(authors)s, %(doi)s, %(s3_url)s)
        """
        
        logging.info(f"Insert query template: {insert_query}")
        
        inserted_count = 0
        insert_failed_ids = []
        
        for paper in successful_papers:
            metadata = paper['metadata']
            s3_url = paper.get('s3_url')
            pmc_id = paper['pmc_id']
            
            try:
                cursor.execute(
                    insert_query,
                    {
                        'pmc_id': pmc_id,
                        'title': metadata['title'],
                        'abstract': metadata['abstract'],
                        'authors': metadata['authors'],
                        'doi': metadata['doi'],
                        's3_url': s3_url
                    }
                )
                logging.info(f"✅ Inserted {pmc_id} into Snowflake")
                inserted_count += 1
                
            except Exception as e:
                logging.error(f"❌ Error inserting {pmc_id}: {str(e)}")
                print(f"❌ Insert failed for {pmc_id}: {str(e)}")
                insert_failed_ids.append(pmc_id)
        
        # Commit all insertions
        conn.commit()
        logging.info("All insertions committed successfully")
        
        cursor.close()
        conn.close()
        
        logging.info("=" * 100)
        logging.info(f"SNOWFLAKE INSERT COMPLETE")
        logging.info(f"Successfully inserted: {inserted_count}/{len(successful_papers)}")
        if insert_failed_ids:
            logging.warning(f"Failed to insert: {len(insert_failed_ids)} papers")
            logging.warning(f"Failed insert IDs: {insert_failed_ids}")
        logging.info("=" * 100)
        
        print("=" * 100)
        print(f"SNOWFLAKE COMPLETE:")
        print(f"  Inserted: {inserted_count}/{len(successful_papers)}")
        if insert_failed_ids:
            print(f"  ❌ Insert failures: {insert_failed_ids}")
        print("=" * 100)
        
        return inserted_count
        
    except Exception as e:
        error_msg = f"CRITICAL ERROR in insert_to_snowflake: {str(e)}"
        logging.error(error_msg)
        logging.exception("Full traceback:")
        print(error_msg)
        raise

# ============================================
# DAG DEFINITION
# ============================================

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='pmc_paper_scraper_to_snowflake',
    default_args=default_args,
    description='Search and scrape SCLC papers from PMC, upload PDFs to S3, and store metadata in Snowflake',
    schedule=None,
    catchup=False,
    tags=['pmc', 'scraping', 's3', 'snowflake', 'sclc'],
) as dag:
    
    # Task 1: Fetch PMC IDs based on search
    fetch_ids_task = PythonOperator(
        task_id='fetch_pmc_ids',
        python_callable=fetch_pmc_ids_for_search,
    )
    
    # Task 2: Scrape papers and upload to S3 (combined task)
    scrape_and_upload_task = PythonOperator(
        task_id='scrape_and_upload',
        python_callable=scrape_and_upload_papers,
        execution_timeout=timedelta(hours=6),  # Allow up to 6 hours for scraping
    )
    
    # Task 3: Insert metadata to Snowflake (only for successful uploads)
    snowflake_task = PythonOperator(
        task_id='insert_to_snowflake',
        python_callable=insert_to_snowflake,
    )
    
    # Updated task dependencies (3 tasks instead of 4)
    fetch_ids_task >> scrape_and_upload_task >> snowflake_task