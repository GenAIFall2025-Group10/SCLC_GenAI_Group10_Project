# ONCODETECT-AI 🔬

> An AI-powered platform for Small Cell Lung Cancer (SCLC) research, risk prediction, and molecular subtype classification using multimodal RAG and multi-agent architecture.

[![Watch Demo](https://img.shields.io/badge/YouTube-Demo-red?style=for-the-badge&logo=youtube)](YOUR_YOUTUBE_LINK_HERE)
[![Project Tracker](https://img.shields.io/badge/Project-Tracker-blue?style=for-the-badge&logo=notion)](https://github.com/GenAIFall2025-Group10/SCLC_GenAI_Group10_Project/blob/main/documentation/OncoDetectAI_Project_Tracker.pdf)

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

---

## 📑 Table of Contents

- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [Data Pipeline Architecture](#-data-pipeline-architecture)
- [Agent System](#-agent-system)
- [Application Features](#-application-features)
- [Product Analytics](#-product-analytics)
- [Evaluation Metrics](#-evaluation-metrics)
- [API Documentation](#-api-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---

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
| **SerpAPI** | Web Search | Enables real-time web search capabilities |
| **Arxiv API** | Research Paper Retrieval | Fetches external academic papers |
| **Google Maps API** | Geographic Services | Provides cancer treatment center recommendations |
| **PubMed API** | Medical Literature | Source for SCLC research papers |

---

<!-- Add remaining sections here based on your table of contents -->

## 📄 License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---
