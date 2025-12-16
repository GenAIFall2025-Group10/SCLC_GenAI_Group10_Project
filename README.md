# ONCODETECT-AI 🔬

> An AI-powered platform for Small Cell Lung Cancer (SCLC) research, risk prediction, and molecular subtype classification using multimodal RAG and multi-agent architecture.

[![Watch Demo](https://img.shields.io/badge/YouTube-Demo-red?style=for-the-badge&logo=youtube)](https://youtu.be/dcm3u0esiF4)
[![Project Tracker](https://img.shields.io/badge/Project-Tracker-blue?style=for-the-badge&logo=notion)](https://github.com/GenAIFall2025-Group10/SCLC_GenAI_Group10_Project/blob/main/documentation/OncoDetectAI_Project_Tracker.pdf)
[![Codelabs](https://img.shields.io/badge/View-Codelabs-orange?style=for-the-badge&logo=google)](https://codelabs-preview.appspot.com/?file_id=1fpYOL5b-wnK3izmBTfenKjzkmqsbFCd8KYCp2MHNu5k/edit?tab=t.0#8)

---

## 📋 Project Overview

ONCODETECT-AI is an advanced generative AI platform designed to revolutionize Small Cell Lung Cancer (SCLC) research and clinical decision-making. By leveraging multimodal RAG (Retrieval-Augmented Generation), multi-agent architecture, and state-of-the-art machine learning models, the platform provides comprehensive solutions for oncology research, risk assessment, and molecular subtype classification.

### 🎯 Key Features

- **Intelligent Research Assistant**: Multi-agent chatbot powered by RAG, Arxiv, and web search agents with confidence-based routing
- **Risk Prediction Engine**: AI-driven risk assessment using demographic, clinical, and genomic data
- **Molecular Subtype Classification**: Automated SCLC subtype classification (SCLC-N, SCLC-P, SCLC-A, SCLC-Y) with therapeutic implications
- **Multimodal Data Processing**: Unified pipeline for text and image embeddings from research papers
- **Product Analytics Dashboard**: Real-time insights using Kafka streaming and modern BI architecture
- **LLM Evaluation Framework**: Comprehensive metrics for correctness, relevance, faithfulness, and context

### 💡 Impact

- Accelerates SCLC research by providing instant access to 530+ curated research papers
- Enables evidence-based clinical decision-making through AI-powered analysis
- Reduces diagnostic time with automated molecular subtype classification
- Provides personalized risk assessments and treatment center recommendations
- Facilitates data-driven product improvements through comprehensive analytics

---

## 🔍 Project Description

ONCODETECT-AI is built on a sophisticated technical architecture that combines modern data engineering, machine learning, and cloud-native technologies to deliver a comprehensive SCLC research and clinical decision support platform.

### Technical Architecture

#### 1. **Unstructured Data Pipeline (Apache Airflow)**

The platform employs three orchestrated Airflow DAGs to process research literature:

**DAG 1 - Data Ingestion**
- Fetches 530+ SCLC-related research papers from PubMed
- Stores PDFs in Amazon S3 with versioning
- Inserts metadata into Snowflake for queryability

**DAG 2 - Text Embedding Pipeline**
- Parses text content from PDFs using advanced document processing
- Chunks text into semantic segments for optimal embedding
- Generates embeddings using `snowflake-arctic-embed-m` model
- Stores embeddings in Snowflake vector tables for similarity search

**DAG 3 - Image Embedding Pipeline**
- Extracts figures, charts, and diagrams from research PDFs
- Stores images with metadata in dedicated S3 bucket
- Creates multimodal embeddings using `voyage-multimodal-3` model
- Enables visual similarity search across research imagery

#### 2. **Structured Data Modeling (DBT)**

- Ingests raw cell line and gene datasets into Snowflake
- Applies transformation logic through DBT models
- Creates staging tables for cohort statistics and patient insights
- Enables efficient analytical queries for AI analysis features

#### 3. **Application Layer (Streamlit in Snowflake)**

**User Authentication & Authorization**
- Secure signup/login with credential storage in Snowflake
- Role-based access control (User vs Admin views)

**Research Assistant (Multimodal RAG System)**
- **Multi-Agent Architecture**: Orchestrates 7 specialized agents
  - **RAG Agent**: Queries text and image embeddings using cosine similarity
  - **Arxiv Agent**: Fetches external research papers via Arxiv API
  - **Web Search Agent**: Performs real-time searches using SerpAPI
  - **Confidence-Based Routing**:
    - Confidence > 60%: RAG Agent only
    - Confidence 50-60%: RAG + Arxiv Agents
    - Confidence < 50%: Arxiv + Web Search Agents
- **Features**: Response translation, summarization, research note saving
- **Validation**: Query-response pairs stored in `rag_validation_table`

**Risk Prediction Module**
- **Quick Prediction**: Instant risk scoring based on user-provided demographics, clinical history, and genomic features
- **Geographic Integration**: High-risk users receive nearby cancer center recommendations via Google Maps API
- **AI Analysis**: Comprehensive report generation including:
  - Cohort statistics from DBT-modeled structured data
  - Risk interpretation with confidence scores
  - Similar patient insights using vector similarity
  - Evidence-based treatment recommendations
  - Prognosis analysis from research embeddings
- **LLM Model**: Powered by Cortex Llama 3.1-405B

**Molecular Subtype Classifier**
- Accepts transcription factors, tumor suppressors, markers, MYC family genes
- Classifies into 4 SCLC subtypes: SCLC-N, SCLC-P, SCLC-A, SCLC-Y
- Provides therapeutic implications for each subtype
- Generates comprehensive clinical assessment reports
- Downloadable PDF reports with AI-generated insights
- **LLM Model**: Powered by Cortex Llama 3.1-405B

#### 4. **Product Analytics (Modern BI Architecture)**

**Event Streaming Pipeline**
- Kafka producers in Streamlit app emit user interaction events
- Events stream to Confluent Cloud for reliable message handling
- Sink connector returns processed events to Snowflake

**Analytics Capabilities**
- Text-to-SQL using Snowflake Cortex for natural language querying
- Real-time visualization of feature usage patterns
- User retention and engagement metrics
- Feedback analysis and sentiment tracking

**LLM Evaluation Framework**
- Metrics: Correctness, Relevance, Faithfulness, Context Precision
- Compares LLM responses against golden dataset ground truth
- Uses Mistral-Large for evaluation scoring
- Continuous model performance monitoring

### Agent Architecture

The platform implements 7 specialized agents:

1. **Subtype Classifier Agent**: Analyzes molecular markers for SCLC subtype determination
2. **Interpreter Agent**: Translates complex medical terminology and research findings
3. **Risk Prediction Agent**: Calculates personalized risk scores using ML models
4. **Arxiv Agent**: Retrieves relevant research papers from Arxiv repository
5. **Web Search Agent**: Performs real-time web searches via SerpAPI
6. **Multimodal RAG Agent**: Queries text and image embeddings for contextual responses
7. **Map Agent**: Provides geographic information for treatment center recommendations

---

## 🏗️ Architecture Diagram


<img width="1559" height="912" alt="Architecture_Diagram" src="https://github.com/user-attachments/assets/3c8a595f-55f0-496c-aa2f-d3e8cecb80f3" />


## ✍️ Attestation

**WE ATTEST THAT WE HAVEN'T USED ANY OTHER STUDENTS' WORK IN OUR ASSIGNMENT AND ABIDE BY THE POLICIES LISTED IN THE STUDENT HANDBOOK**

### 👥 Contribution

| Contributor | Contribution Percentage |
|------------|------------------------|
| Pranali Chipkar | 33% |
| Aditi Deshmukh | 33% |
| Siddharth Pawar | 33% |

## 🛠️ Tech Stack

| Technology | Purpose | Details |
|-----------|---------|---------|
| **Snowflake** | Data Warehouse & ML Platform | Central data repository, vector storage, Cortex AI services |
| **Snowflake Cortex AI** | LLM & Embedding Models | Hosts Llama 3.1-405B, Mistral-Large, Arctic-Embed-M models |
| **Apache Airflow** | Workflow Orchestration | Manages 3 DAGs for data ingestion and embedding pipelines |
| **DBT (Data Build Tool)** | Data Transformation | Models and transforms structured cell line and gene data |
| **Streamlit** | Frontend Application | Built-in Snowflake for seamless data integration |
| **Python** | Development | Core application logic, agent orchestration, data processing |
| **Amazon S3** | Object Storage | Stores PDFs and images with metadata |
| **Apache Kafka** | Event Streaming | Real-time event collection via Confluent Cloud |
| **Confluent Cloud** | Managed Kafka Service | Handles event streaming and sink connectors |
| **SQL** | Database Queries | Data retrieval and transformation logic |
| **Arctic-Embed-M** | Text Embeddings | Snowflake's embedding model for semantic search |
| **Voyage-Multimodal-3** | Image Embeddings | Creates vector representations of research images |
| **Llama 3.1-405B** | Large Language Model | Powers AI analysis, risk interpretation, subtype classification |
| **Mistral-Large** | Evaluation Model | Assesses LLM response quality against ground truth |

---

## 🚀 Installation & Setup

### Prerequisites
- Docker and Docker Compose installed
- AWS Account with S3 access
- Snowflake Account
- Confluent Cloud Account

---

### 1. Airflow Setup

**Install Apache Airflow using Docker:**
```bash
# Clone the repository
git clone https://github.com/your-repo/oncodetect-ai.git
cd oncodetect-ai

# Initialize Airflow with Docker
docker-compose up airflow-init

# Start Airflow services
docker-compose up -d
```

**Configure Environment Variables:**

Create a `.env` file in the root directory with the following variables:
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=your_region
S3_BUCKET_PAPERS=your_bucket_nme
S3_BUCKET_IMAGES=your_bucket_nme
S3_IMAGES_PREFIX=prefix/
S3_PREFIX=prefix/

# Snowflake Configuration
SNOWFLAKE_ACCOUNT=your_snowflake_account
SNOWFLAKE_USER=your_snowflake_user
SNOWFLAKE_PASSWORD=your_snowflake_password
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_ROLE=your_role

# Snowflake Table Names
SNOWFLAKE_PAPERS_TABLE=your_table_name
SNOWFLAKE_TEXT_PROCESSED_TABLE=your_table_name
SNOWFLAKE_TEXT_EMBEDDINGS_TABLE=your_table_name
SNOWFLAKE_PARSED_TEXT_TABLE=your_table_name
SNOWFLAKE_IMAGE_EMBEDDINGS_TABLE=your_table_name
SNOWFLAKE_PARSED_IMAGES_TABLE=your_table_name
SNOWFLAKE_IMAGES_PROCESSED_TABLE=your_table_name

# Processing Configuration
BATCH_SIZE=25
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
CORTEX_TEXT_MODEL=snowflake-arctic-embed-m
CORTEX_IMAGE_MODEL=voyage-multimodal-3
```

---

### 2. AWS S3 Setup

**Create S3 Buckets:**
```bash
# Create bucket for research paper PDFs

# Create bucket for extracted images

# Enable versioning on both buckets

```
---

### 3. Snowflake Setup

**Create Database Objects:**
```sql
-- Execute all DDL scripts from the SQL folder

-- Execute all table creation scripts from /sql folder

```

**Store API Keys:**
```sql
-- Insert API keys into CONFIG table
INSERT INTO ONCODETECT_AI.USER_MANAGEMENT.CONFIG (key, value)
VALUES 
  ('GOOGLE_MAPS_API_KEY', 'your_google_maps_api_key'),
  ('SERPAPI_KEY', 'your_serpapi_key'),
  ('ARXIV_API_KEY', 'your_arxiv_api_key');
```

**Create Network Integrations:**
```sql
-- Create external access integrations for APIs
CREATE OR REPLACE NETWORK RULE google_maps_network_rule
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = ('maps.googleapis.com');

CREATE OR REPLACE NETWORK RULE serpapi_network_rule
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = ('serpapi.com');

```

---

### 4. Confluent Cloud & Kafka Setup

**Create Kafka Topic:**
```bash
# Login to Confluent Cloud
confluent login

# Create topic for application events
confluent kafka topic create oncodetect-ai-events \
  --cluster <your-cluster-id> \
  --partitions 1 \
  --config retention.ms=604800000
```

**Define Event Structure:**
```json
{
  "event_id": "abc-123-def-456-ghi-789",
  "event_type": "user_action_type",
  "timestamp": "YYYY-MM-DDTHH:MM:SS.000Z",
  "user_type": "user_type_value",
  "user_id": "user_id_value",
  "session_id": "session_id_value",
  "button_name": "button_name_value",
  "status": "status_value"
}
```

**Snowflake Sink Configuration (`snowflake-sink-config.json`):**
```json
{
  "name": "oncodetect-ai-snowflake-sink",
  "config": {
    "connector.class": "SnowflakeSink",
    "topics": "oncodetect-ai-events",
    "snowflake.url.name": "your_account.snowflakecomputing.com",
    "snowflake.user.name": "your_user",
    "snowflake.private.key": "your_private_key",
    "snowflake.database.name": "ONCODETECT_AI",
    "snowflake.schema.name": "ANALYTICS",
    "buffer.count.records": "10000",
    "buffer.flush.time": "60",
    "buffer.size.bytes": "5000000"
  }
}
```

---

### 7. Run Initial Data Pipeline
```bash
# Trigger Airflow DAGs
# DAG 1: Fetch research papers from PubMed
airflow dags trigger pubmed_ingestion_dag

# DAG 2: Generate text embeddings
airflow dags trigger text_embedding_dag

# DAG 3: Generate image embeddings
airflow dags trigger image_embedding_dag
```

---

## 📚 References

### API Documentation
- [SerpAPI Documentation](https://serpapi.com/search-api) - Web Search API
- [Google Maps API Documentation](https://developers.google.com/maps/documentation) - Geographic Services
- [PubMed API](https://www.ncbi.nlm.nih.gov/home/develop/api/) - Medical Literature Access

### AI & ML Platforms
- [Snowflake Cortex AI Documentation](https://docs.snowflake.com/en/user-guide/snowflake-cortex/overview) - LLM & Embedding Models
- [Claude AI by Anthropic](https://www.anthropic.com/claude) - AI Assistant

### Data Engineering Tools
- [DBT Documentation](https://docs.getdbt.com/docs/introduction) - Data Transformation
- [Apache Airflow Setup Tutorial](https://www.youtube.com/results?search_query=apache+airflow+tutorial) - Workflow Orchestration

### Additional Resources
- [Confluent Kafka Documentation](https://docs.confluent.io/) - Event Streaming
- [Streamlit Documentation](https://docs.streamlit.io/) - Application Framework
- [Amazon S3 Documentation](https://docs.aws.amazon.com/s3/) - Object Storage

---

## 📄 License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---
