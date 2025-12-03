import argparse
from kfp import local
from pipeline import rag_ingestion_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG ingestion pipeline locally using KFP SubprocessRunner.")
    parser.add_argument("--gcs-input-uri", required=True, help="GCS URI for input documents")
    parser.add_argument("--bm25-artifact-name", default="bm25_index", help="BM25 artifact display name")
    parser.add_argument("--embedding-model", default="gemini-embedding-001", help="Embedding model name")
    parser.add_argument("--project", default="", help="GCP Project ID")
    parser.add_argument("--location", default="us-central1", help="GCP Location")
    parser.add_argument("--index-name", required=True, help="Vertex AI Vector Search Index Name")

    args = parser.parse_args()

    # Initialize the local runner
    # This allows running KFP components locally as subprocesses
    local.init(runner=local.SubprocessRunner())

    # Run the pipeline
    rag_ingestion_pipeline(
        gcs_input_uri=args.gcs_input_uri,
        index_name=args.index_name,
        bm25_artifact_name=args.bm25_artifact_name,
        embedding_model=args.embedding_model,
        project=args.project,
        location=args.location
    )