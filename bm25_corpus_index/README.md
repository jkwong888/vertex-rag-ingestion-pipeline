# BM25 Corpus Indexer

This project provides a pipeline to train a BM25 encoder model from HTML documents stored in a Google Cloud Storage (GCS) bucket. It uses [`pinecone-text`](https://github.com/pinecone-io/pinecone-text) for efficient sparse vector generation and `kfp` (Kubeflow Pipelines) for orchestration.

## Features

-   **Ingestion**: Iterates over HTML files in a specified GCS bucket prefix.
-   **Preprocessing**: Converts HTML content to Markdown for cleaner text extraction.
-   **Indexing**: Tokenizes text and trains a BM25 encoder (calculating IDFs) using `pinecone-text`.
-   **Output**: Saves encoder artifacts (BM25 parameters) and document ID mappings compatible with `pinecone-text` loading.

## Directory Structure

-   `pipeline.py`: Defines the KFP pipeline and the core `build_bm25_index` component logic.
-   `build_index.py`: Utility to run the pipeline **locally** using the KFP `SubprocessRunner`.
-   `submit_pipeline.py`: Utility to submit the pipeline to **Vertex AI Pipelines**.
-   `requirements.txt`: Python dependencies.

## Prerequisites

-   Python 3.11+
-   Google Cloud Project with Vertex AI API enabled.
-   Google Cloud Storage bucket containing source HTML files.
-   Local authentication to Google Cloud (for local runs or submission).

## Setup

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Authenticate**
    Ensure you have application-default credentials set up:
    ```bash
    gcloud auth application-default login
    ```

## Usage

### 1. Local Execution
To build the index on your local machine (useful for testing or small datasets):

```bash
python build_index.py <GCS_URI> [--output <OUTPUT_DIR>]
```

**Example:**
```bash
python build_index.py gs://my-corpus-bucket/docs/ --output ./my_local_index
```

The artifacts will be saved in the specified output directory (default: `./local_bm25_index_output`).

### 2. Vertex AI Pipelines (Cloud)
To submit the job to Google Cloud Vertex AI for scalable execution:

```bash
python submit_pipeline.py \
    --project-id <YOUR_PROJECT_ID> \
    --region <REGION> \
    --gcs-uri <GCS_INPUT_URI> \
    --pipeline-root <GCS_STAGING_URI> \
    --pipeline-name "bm25-index-generation"
```

**Parameters:**
-   `--project-id`: GCP Project ID.
-   `--region`: Vertex AI region (e.g., `us-central1`).
-   `--gcs-uri`: Source GCS URI containing HTML files (e.g., `gs://my-bucket/data/`).
-   `--pipeline-root`: GCS URI for storing pipeline artifacts/staging (e.g., `gs://my-bucket/pipeline-root/`).
-   `--pipeline-name`: Name of the pipeline job.
-   `--service-account`: (Optional) Service Account email to run the pipeline.
-   `--cron-schedule`: (Optional) Cron string for scheduling (e.g., `"0 0 * * 1"` for weekly).
-   `--schedule-only`: (Optional) If true, creates/updates the schedule without running the job immediately.
-   `--disable-caching`: (Optional) Set to true to disable execution caching.

## Output Artifacts

The pipeline generates the following files in the output directory (or GCS artifact path):

-   `bm25_params.json`: The BM25 parameters (IDF values, avg doc length, etc.) required for encoding.
-   `doc_ids.json`: A list mapping the training corpus chunks to their source IDs (GCS URIs).

### Loading the Encoder
You can load the trained BM25 encoder in Python using `pinecone-text`:

```python
from pinecone_text.sparse import BM25Encoder
import json

# Load the BM25 parameters
bm25 = BM25Encoder.load("path/to/bm25_params.json")

# Encode a query
query_embedding = bm25.encode_queries("search query")

# Encode a document
doc_embedding = bm25.encode_documents("some document text")
```
