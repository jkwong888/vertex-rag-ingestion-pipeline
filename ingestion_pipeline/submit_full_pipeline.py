import argparse
import os
import sys
from google.cloud import aiplatform
from kfp.compiler import Compiler

# Add parent directory to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from full_pipeline import rag_full_ingestion_pipeline

PIPELINE_FILE_NAME = "rag_full_ingestion_pipeline.yaml"

def submit_pipeline(
    project: str,
    location: str,
    pipeline_root: str,
    gcs_html_uri: str,
    gcs_chunks_uri: str,
    index_name: str,
    embedding_model: str,
    enable_caching: bool
):
    aiplatform.init(project=project, location=location)

    print(f"Compiling pipeline to {PIPELINE_FILE_NAME}...")
    Compiler().compile(
        pipeline_func=rag_full_ingestion_pipeline,
        package_path=PIPELINE_FILE_NAME
    )

    parameter_values = {
        "gcs_html_uri": gcs_html_uri,
        "gcs_chunks_uri": gcs_chunks_uri,
        "index_name": index_name,
        "embedding_model": embedding_model,
        "project": project,
        "location": location
    }

    print(f"Submitting PipelineJob to Vertex AI in {location}...")
    job = aiplatform.PipelineJob(
        display_name=f"rag-full-ingestion-{index_name}",
        template_path=PIPELINE_FILE_NAME,
        pipeline_root=pipeline_root,
        parameter_values=parameter_values,
        enable_caching=enable_caching
    )

    job.submit()
    print(f"Pipeline job submitted. Resource name: {job.resource_name}")

    # Clean up pipeline file locally
    if os.path.exists(PIPELINE_FILE_NAME):
        os.remove(PIPELINE_FILE_NAME)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit Full RAG Ingestion Pipeline to Vertex AI")

    parser.add_argument("--project", required=True, help="Google Cloud Project ID")
    parser.add_argument("--region", default="us-central1", help="Google Cloud Region")
    parser.add_argument("--pipeline-root", required=True, help="GCS URI for pipeline root/artifacts")
    parser.add_argument("--gcs-html-uri", required=True, help="GCS URI containing source HTML files")
    parser.add_argument("--gcs-chunks-uri", required=True, help="GCS URI where chunks will be stored")
    parser.add_argument("--index-name", required=True, help="Display name for the Vector Search index")
    parser.add_argument("--embedding-model", default="gemini-embedding-001", help="Embedding model to use")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")

    args = parser.parse_args()

    submit_pipeline(
        project=args.project,
        location=args.region,
        pipeline_root=args.pipeline_root,
        gcs_html_uri=args.gcs_html_uri,
        gcs_chunks_uri=args.gcs_chunks_uri,
        index_name=args.index_name,
        embedding_model=args.embedding_model,
        enable_caching=not args.no_cache
    )
