import argparse
import os
from google.cloud import aiplatform
from kfp.compiler import Compiler
from pipeline import scrape_pipeline

PIPELINE_FILE_NAME = "scrape_pipeline.yaml"

def submit_pipeline(
    project: str,
    location: str,
    pipeline_root: str,
    gcs_bucket_name: str,
    gcs_bucket_path: str,
    enable_caching: bool
):
    aiplatform.init(project=project, location=location)

    Compiler().compile(
        pipeline_func=scrape_pipeline,
        package_path=PIPELINE_FILE_NAME
    )

    parameter_values = {
        "gcs_bucket_name": gcs_bucket_name,
        "gcs_bucket_path": gcs_bucket_path,
    }

    job = aiplatform.PipelineJob(
        display_name="scrape-bulbapedia-pipeline",
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
    parser = argparse.ArgumentParser(description="Submit Scrape Pipeline to Vertex AI")

    parser.add_argument(
        "--pipeline-root",
        default=os.getenv("PIPELINE_ROOT"),
        help="Pipeline root directory",
        required=True
    )
    parser.add_argument("--project", required=True, help="Google Cloud Project ID")
    parser.add_argument("--region", default="us-central1", help="Google Cloud Region")
    parser.add_argument("--gcs-bucket-name", default="jkwng-vertex-experiments", help="GCS Bucket Name")
    parser.add_argument("--gcs-bucket-path", default="pokemon/bulbapedia/html", help="GCS Bucket Path")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")

    args = parser.parse_args()

    submit_pipeline(
        project=args.project,
        location=args.region,
        pipeline_root=args.pipeline_root,
        gcs_bucket_name=args.gcs_bucket_name,
        gcs_bucket_path=args.gcs_bucket_path,
        enable_caching=not args.no_cache
    )
