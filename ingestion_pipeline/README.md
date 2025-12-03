# RAG Ingestion Pipeline

This project implements a Retrieval-Augmented Generation (RAG) ingestion pipeline using Kubeflow Pipelines (KFP) and Google Cloud Vertex AI. It automates the process of ingesting documents, generating hybrid embeddings (sparse BM25 + dense Gemini), and updating a Vertex AI Vector Search Index.

## Overview

The pipeline performs the following steps:
1.  **GCS Input:** Lists documents from a specified Google Cloud Storage (GCS) bucket URI.
2.  **Retrieve BM25 Index:** Fetches the URI of a pre-built BM25 index artifact.
3.  **Chunking:** Splits the documents into smaller chunks.
4.  **Sparse Embeddings:** Generates sparse embeddings using the BM25 index.
5.  **Dense Embeddings:** Generates dense embeddings using the Gemini embedding model (`gemini-embedding-001`).
6.  **Merge:** Combines sparse and dense embeddings into a single dataset.
7.  **Update Index:** Updates a Vertex AI Vector Search Index with the new embeddings.

## Prerequisites

-   Python 3.11+
-   Google Cloud SDK (`gcloud`) installed and authenticated.
-   Access to a Google Cloud Project with Vertex AI API enabled.
-   An existing Vertex AI Vector Search Index.
-   A pre-built BM25 index artifact in Vertex AI Metadata (created by a separate process).

## Installation

1.  **Clone the repository** (if applicable).

2.  **Create and activate a virtual environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Run Locally (for development/testing)

You can run the pipeline components locally using the KFP SubprocessRunner. This is useful for debugging without submitting a job to Vertex AI.

```bash
python ingest.py \
  --gcs-input-uri "gs://your-bucket/path/to/documents/" \
  --index-name "your-index-name" \
  --project "your-project-id" \
  --region "us-central1" \
  --bm25-artifact-name "bm25_index" \
  --embedding-model "gemini-embedding-001"
```

### 2. Submit to Vertex AI Pipelines

To run the pipeline as a managed job on Vertex AI:

```bash
python submit_pipeline.py \
  --project "your-project-id" \
  --location "us-central1" \
  --pipeline-root "gs://your-bucket/pipeline_root" \
  --gcs-input-uri "gs://your-bucket/path/to/documents/" \
  --index-name "your-index-name" \
  --bm25-artifact-name "bm25_index" \
  --embedding-model "gemini-embedding-001"
```

**Note:** You can optionally use the `--no-cache` flag to disable execution caching.

## Pipeline Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--project` | Google Cloud Project ID. | Required (except for local run where it defaults to empty string but might fail if needed) |
| `--location` | Google Cloud Region. | `us-central1` |
| `--gcs-input-uri` | GCS URI containing input documents. | Required |
| `--index-name` | Name of the Vertex AI Vector Search Index to update. | Required |
| `--pipeline-root` | GCS URI for storing pipeline artifacts (only for `submit_pipeline.py`). | `os.getenv("PIPELINE_ROOT")` |
| `--bm25-artifact-name` | Display name of the BM25 index artifact in Vertex Metadata. | `bm25_index` |
| `--embedding-model` | Name of the Gemini embedding model to use. | `gemini-embedding-001` |

## Components

The pipeline logic is modularized into components located in the `components/` directory:

-   `components/gcs.py`: Utilities for GCS operations and artifact retrieval.
-   `components/chunk.py`: Document chunking logic.
-   `components/embedding.py`: Sparse and dense embedding generation.
-   `components/update_index.py`: Vertex AI Vector Search Index update logic.
