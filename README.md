# RAG Ingestion with Vertex AI Pipelines

This repository contains an implementation for building a Retrieval-Augmented Generation (RAG) system using Google Cloud Vertex AI. It demonstrates how to scrape data, build a hybrid search index (Sparse BM25 + Dense Gemini Embeddings), and perform RAG queries.

## Architecture
The system implements a complex ingestion and retrieval architecture designed to handle hybrid search.

The workflow consists of four main logical stages:

1. **Data Collection** ([scrape/](scrape/)): Crawls a website (Bulbapedia) to gather raw HTML documents and stores them in Google Cloud Storage (GCS).
2. **Infrastructure** ([terraform/](terraform/)): Provisions the necessary Vertex AI Vector Search resources (Index and Endpoint) using Terraform.
3. **BM25 Index Generation** ([bm25_corpus_index/](bm25_corpus_index/)): A dedicated pipeline that scans the entire corpus to calculate global statistics (TF-IDF) and generates a BM25 index.
4. **Ingestion Pipeline** ([ingestion_pipeline/](ingestion_pipeline/)): A Kubeflow Pipeline (KFP) that chunks content, generates hybrid embeddings, and updates the Vertex AI Vector Search Index.
5. **Query** ([query/](query/)): A sample application that performs hybrid search, re-ranks results, and generates answers using Gemini.

## Detailed Pipeline Workflows

### 1. BM25 Corpus Indexing (Training Phase)
* Before individual documents can be embedded sparsely, a global index must be calculated.
* **Tokenization**: The entire corpus is tokenized, removing stopwords (e.g., "a", "an", "the").
* **Global Statistics**: The pipeline calculates Inverse Document Frequency (IDF) and average document length across all chunks.
* **Output**: The resulting index is written to GCS (becoming read-only) and registered in Vertex AI metadata for discovery.
* **Drift Management**: If the chunking strategy changes or the corpus is significantly updated, this index must be recalculated to avoid drift.

### 2. Ingestion Logic (Inference Phase)
The ingestion flow transforms raw HTML into searchable vectors.

* **Conversion**: HTML documents are converted to Markdown.
* **Chunking Strategy**:
    * Level 1: Split by Markdown headers (#, ##, ###) using `MarkdownHeaderTextSplitter`.
    * Level 2: Recursive character split with 500 character chunks and 100 character overlap.
      *Note: Tables and sentences are handled specifically by the recursive splitter.*
* **Hybrid Embedding**:
    * Sparse: Uses the pre-calculated BM25 index to generate sparse vectors for each chunk.
    * Dense: Uses `gemini-embedding-001` (or similar) to generate dense vectors.
* **Storage**: The schema merges id, embedding (dense), and sparse_embedding before upserting into Vertex Vector Search.

### 3. Query Execution
The query process utilizes Reciprocal Rank Fusion (RRF) to combine results.
* **Input**: User query string.
* **Embedding**: Generates both sparse (BM25) and dense (Gemini) embeddings for the query.
* **Hybrid Search**: Finds nearest neighbors for both vector types.
* **Ranking**: Applies RRF to merge and rank the Top-K results.
* **Generation**: Constructs a prompt with the context and sends it to the LLM for the final answer.

## Repository Structure
* `scrape/`: Scrapy spider to download Pokemon data from Bulbapedia.
* `terraform/`: Terraform configuration to create the Vertex AI Vector Search Index and Endpoint.
* `bm25_corpus_index/`: Scripts and KFP pipeline to train and save a BM25 encoder model.
* `ingestion_pipeline/`: The core KFP pipeline for chunking, embedding, and indexing documents.
* `query/`: Scripts to demonstrate hybrid search and answer generation.

## Prerequisites
* **Google Cloud Project**: With billing enabled.
* **APIs Enabled**: Vertex AI API, Cloud Storage API.

**Tools:**
* Python 3.11+
* Google Cloud SDK (gcloud)
* Terraform
* pip

## Getting Started
Follow these steps to set up the complete system.

### 1. Data Scraping
Navigate to the `scrape/` directory and run the spider to populate your GCS bucket with HTML files.

```bash
cd scrape
pip install -r requirements.txt
# Edit scrape_bulbapedia.py to set your GCS bucket name
python scrape_bulbapedia.py
```

### 2. Infrastructure Setup
Provision the Vector Search resources.

*Note: The deployment of the index to the endpoint can take approximately 30 minutes.*

```bash
cd terraform
# Update variables.tf or create a terraform.tfvars file with your project details
terraform init
terraform apply
```
Be sure to note down the `index_id` and `endpoint_id` output by Terraform.

### 3. Build BM25 Index
Train the BM25 encoder on your scraped data. This establishes the "Global Statistics" required for sparse embedding.

```bash
cd bm25_corpus_index
pip install -r requirements.txt
# Submit the training job to Vertex AI Pipelines
python submit_pipeline.py --project-id YOUR_PROJECT_ID --gcs-uri gs://YOUR_BUCKET/pokemon/bulbapedia/html
```

### 4. Run Ingestion Pipeline
Process the documents and populate the Vector Search Index. This step converts HTML to Markdown, chunks specifically by header and character count, and merges sparse/dense vectors.

```bash
cd ingestion_pipeline
pip install -r requirements.txt
# Submit the ingestion job
python submit_pipeline.py \
    --project YOUR_PROJECT_ID \
    --gcs-input-uri gs://YOUR_BUCKET/pokemon/bulbapedia/html \
    --index-name YOUR_INDEX_Resource_Name
```

### 5. Querying
Perform a RAG query to test the system.

```bash
cd query
pip install -r requirements.txt
# Update main.py with your Project ID, Index Endpoint ID, and Deployed Index ID
python main.py
```

## Future Roadmap & Advanced RAG
The following features are being considered for future iterations of this pipeline:

* **PDF & Image Support**: Using models like ColPali to encode PDF pages as image patches for better retrieval of complex documents.
* **Late Interaction Models**: Investigating ColBERT for improved recall.
* **Agentic RAG**: Adding "Agentic Memory" to utilize context from past conversations.
* **Smart Query Rewriting**: Automatically rewriting unclear user questions before retrieval.
* **Re-ranking**: Implementing a Cross-Encoder to filter unrelated chunks before the generation step.

## Key Technologies
* **Google Cloud Vertex AI**: Vector Search (Hybrid), Pipelines, and Gemini Models.
* **Kubeflow Pipelines (KFP)**: Orchestration of ML workflows.
* **Sparse Embedding**: BM25 / SPLADE.
* **Dense Embedding**: Gemini-embedding-001 / BERT.
* **Langchain**: Used for MarkdownHeaderTextSplitter and RecursiveCharacterTextSplitter.
* **Terraform**: Infrastructure as Code.