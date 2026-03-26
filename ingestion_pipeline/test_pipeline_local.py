import sys
import os
import argparse
from kfp import local
from kfp import dsl

# Set up local execution runner
local.init(runner=local.SubprocessRunner())

# Add parent directory to sys.path to allow cross-directory imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from full_pipeline import rag_full_ingestion_pipeline
from update_pipeline import rag_update_ingestion_pipeline

def main():
    parser = argparse.ArgumentParser(description="Test Ingestion Pipelines Locally")
    parser.add_argument("--pipeline", choices=["full", "update"], required=True, help="Which pipeline to run locally")
    parser.add_argument("--gcs-html-uri", required=True, help="GCS URI for small subset of test HTML files")
    parser.add_argument("--gcs-chunks-uri", required=True, help="GCS URI for output test chunks")
    parser.add_argument("--index-name", required=True, help="Existing test index display name")
    parser.add_argument("--project", required=True, help="GCP Project ID")
    
    args = parser.parse_args()
    
    print(f"Running {args.pipeline} pipeline locally using SubprocessRunner...")
    print(f"Inputs: HTML={args.gcs_html_uri}, Chunks={args.gcs_chunks_uri}")
    
    if args.pipeline == "full":
        rag_full_ingestion_pipeline(
            gcs_html_uri=args.gcs_html_uri,
            gcs_chunks_uri=args.gcs_chunks_uri,
            index_name=args.index_name,
            project=args.project
        )
    else:
        rag_update_ingestion_pipeline(
            gcs_html_uri=args.gcs_html_uri,
            gcs_chunks_uri=args.gcs_chunks_uri,
            index_name=args.index_name,
            project=args.project
        )
        
    print("Local pipeline execution completed.")

if __name__ == "__main__":
    main()
