import argparse
import os
import tempfile
from urllib.parse import urlparse
# google.cloud.storage might not be directly needed in build_index.py anymore
# if LocalRunner handles all artifact persistence.
# from google.cloud import storage 
from kfp import local
from pipeline import pipeline

def main():
    parser = argparse.ArgumentParser(description="Run BM25 index pipeline locally using KFP SubprocessRunner.")
    parser.add_argument("gcs_uri", help="GCS URI (e.g., gs://my-bucket/path/to/docs/)")
    parser.add_argument("--output", default="./local_bm25_index_output", help="Local output directory for the index artifact.")

    args = parser.parse_args()

    print(f"Running pipeline locally with GCS URI: {args.gcs_uri}")
    print(f"Artifacts will be stored in local directory: {args.output}")

    # Create LocalRunner instance. The output_path_prefix specifies where artifacts will be stored locally.
    local.init(runner=local.SubprocessRunner(), pipeline_root=args.output)

    # Run the pipeline function. Arguments are passed as a dictionary.
    pipeline_task = pipeline(
        gcs_uri=args.gcs_uri
    )
    print("Pipeline run complete. Check the specified output directory for artifacts.")

if __name__ == "__main__":
    main()
