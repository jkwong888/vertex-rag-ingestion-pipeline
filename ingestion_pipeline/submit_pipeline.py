import argparse
import os
from google.cloud import aiplatform
from kfp.compiler import Compiler
from pipeline import rag_ingestion_pipeline

PIPELINE_FILE_NAME = "rag_ingestion_pipeline.yaml"

def submit_pipeline(
    project: str,
    location: str,
    pipeline_root: str,
    gcs_input_uri: str,
    index_name: str,
    bm25_artifact_name: str,
    embedding_model: str,
    enable_caching: bool
):
    aiplatform.init(project=project, location=location)

    Compiler().compile(
        pipeline_func=rag_ingestion_pipeline,
        package_path=PIPELINE_FILE_NAME
    )

    parameter_values = {
        "gcs_input_uri": gcs_input_uri,
        "index_name": index_name,
        "bm25_artifact_name": bm25_artifact_name,
        "embedding_model": embedding_model,
        "project": project,
        "location": location
    }

    job = aiplatform.PipelineJob(
        display_name="rag-ingestion-pipeline",
        template_path=PIPELINE_FILE_NAME,
        pipeline_root=pipeline_root,
        parameter_values=parameter_values,
        enable_caching=enable_caching
    )

    job.submit()
    print(f"Pipeline job submitted. Resource name: {job.resource_name}")

    # Clean up pipeline file
    if os.path.exists(PIPELINE_FILE_NAME):
        os.remove(PIPELINE_FILE_NAME)
        print(f"Deleted {PIPELINE_FILE_NAME}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit RAG Ingestion Pipeline to Vertex AI")

    parser.add_argument(
        "--pipeline-root",
        default=os.getenv("PIPELINE_ROOT"),
        help="Pipeline root directory",
    )
    parser.add_argument("--project", required=True, help="Google Cloud Project ID")
    parser.add_argument("--region", default="us-central1", help="Google Cloud Region")
    parser.add_argument("--gcs-input-uri", required=True, help="GCS URI for input documents")
    parser.add_argument("--index-name", required=True, help="Vertex AI Vector Search Index Name")
    parser.add_argument("--bm25-artifact-name", default="bm25_index", help="Display name of the BM25 index artifact")
    parser.add_argument("--embedding-model", default="gemini-embedding-001", help="Embedding model to use")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")

    args = parser.parse_args()

    submit_pipeline(
        project=args.project,
        location=args.region,
        pipeline_root=args.pipeline_root,
        gcs_input_uri=args.gcs_input_uri,
        index_name=args.index_name,
        bm25_artifact_name=args.bm25_artifact_name,
        embedding_model=args.embedding_model,
        enable_caching=not args.no_cache
    )
