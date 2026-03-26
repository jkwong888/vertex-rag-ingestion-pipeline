import argparse
import os
from google.cloud import aiplatform
from kfp.compiler import Compiler
from pipeline import chunk_pipeline

PIPELINE_FILE_NAME = "chunk_pipeline.yaml"

def submit_pipeline(
    project: str,
    location: str,
    pipeline_root: str,
    gcs_html_uri: str,
    gcs_chunks_uri: str,
    enable_caching: bool
):
    aiplatform.init(project=project, location=location)

    Compiler().compile(
        pipeline_func=chunk_pipeline,
        package_path=PIPELINE_FILE_NAME
    )

    parameter_values = {
        "gcs_html_uri": gcs_html_uri,
        "gcs_chunks_uri": gcs_chunks_uri,
    }

    job = aiplatform.PipelineJob(
        display_name="unified-chunking-pipeline",
        template_path=PIPELINE_FILE_NAME,
        pipeline_root=pipeline_root,
        parameter_values=parameter_values,
        enable_caching=enable_caching
    )

    job.submit()
    print(f"Pipeline job submitted. Resource name: {job.resource_name}")

    if os.path.exists(PIPELINE_FILE_NAME):
        os.remove(PIPELINE_FILE_NAME)
        print(f"Deleted {PIPELINE_FILE_NAME}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit Unified Chunking Pipeline to Vertex AI")

    parser.add_argument(
        "--pipeline-root",
        default=os.getenv("PIPELINE_ROOT"),
        help="Pipeline root directory",
        required=True
    )
    parser.add_argument("--project", required=True, help="Google Cloud Project ID")
    parser.add_argument("--region", default="us-central1", help="Google Cloud Region")
    parser.add_argument("--gcs-html-uri", required=True, help="GCS URI for raw HTML files")
    parser.add_argument("--version", required=True, help="Version (e.g., v1)")
    parser.add_argument("--gcs-chunks-root-uri", required=True, help="GCS URI for chunks root (e.g., gs://bucket/pokemon/bulbapedia/chunks/)")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")

    args = parser.parse_args()

    gcs_chunks_uri = os.path.join(args.gcs_chunks_root_uri, args.version)

    submit_pipeline(
        project=args.project,
        location=args.region,
        pipeline_root=args.pipeline_root,
        gcs_html_uri=args.gcs_html_uri,
        gcs_chunks_uri=gcs_chunks_uri,
        enable_caching=not args.no_cache
    )
