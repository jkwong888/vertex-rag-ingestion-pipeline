# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import sys

from pipeline import pipeline
from google.cloud import aiplatform
from kfp import compiler

PIPELINE_FILE_NAME = "bm25_index_pipeline.yaml"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for pipeline configuration."""

    parser = argparse.ArgumentParser(description="Pipeline configuration")
    parser.add_argument(
        "--project-id", default=os.getenv("PROJECT_ID"), help="GCP Project ID"
    )
    parser.add_argument(
        "--region", default=os.getenv("REGION"), help="Vertex AI Pipelines region"
    )
    parser.add_argument(
        "--chunk-version",
        required=True,
        help="Chunk version (e.g., v1)",
    )
    parser.add_argument(
        "--gcs-chunks-root-uri",
        required=True,
        help="GCS URI for chunks root (e.g., gs://bucket/pokemon/bulbapedia/chunks/)",
    )
    parser.add_argument(
        "--service-account",
        default=os.getenv("SERVICE_ACCOUNT"),
        help="Service account",
    )
    parser.add_argument(
        "--pipeline-root",
        default=os.getenv("PIPELINE_ROOT"),
        help="Pipeline root directory",
    )
    parser.add_argument(
        "--pipeline-name", default=os.getenv("PIPELINE_NAME"), help="Pipeline name"
    )
    parser.add_argument(
        "--disable-caching",
        type=bool,
        default=os.getenv("DISABLE_CACHING", "false").lower() == "true",
        help="Enable pipeline caching",
    )
    parser.add_argument(
        "--cron-schedule",
        default=os.getenv("CRON_SCHEDULE", None),
        help="Cron schedule",
    )
    parser.add_argument(
        "--schedule-only",
        type=bool,
        default=os.getenv("SCHEDULE_ONLY", "false").lower() == "true",
        help="Schedule only (do not submit)",
    )
    parsed_args = parser.parse_args()

    # Validate required parameters
    missing_params = []
    required_params = {
        "project_id": parsed_args.project_id,
        "region": parsed_args.region,
        "pipeline_root": parsed_args.pipeline_root,
    }

    for param_name, param_value in required_params.items():
        if param_value is None:
            missing_params.append(param_name)

    if missing_params:
        logging.error("Error: The following required parameters are missing:")
        for param in missing_params:
            logging.error(f"  - {param}")
        logging.error(
            "\nPlease provide these parameters either through environment variables or command line arguments."
        )
        sys.exit(1)

    return parsed_args


def main():
    """Main entry point for pipeline submission."""

    args = parse_args()

    # Initialize AI Platform
    aiplatform.init(project=args.project_id, location=args.region)

    # Compile the pipeline
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=PIPELINE_FILE_NAME,
    )

    gcs_chunks_uri = os.path.join(args.gcs_chunks_root_uri, args.chunk_version)

    # Set parameter values
    parameter_values = {
        "gcs_chunks_uri": gcs_chunks_uri,
    }

    # Pipeline Display Name
    display_name = args.pipeline_name or f"bm25-index-generation-pipeline-{args.chunk_version}"

    # If cron schedule is provided, create a scheduled job
    if args.cron_schedule:
        pipeline_job = aiplatform.PipelineJob(
            display_name=display_name,
            template_path=PIPELINE_FILE_NAME,
            pipeline_root=args.pipeline_root,
            parameter_values=parameter_values,
            enable_caching=not args.disable_caching,
        )

        job = aiplatform.PipelineJob.create_schedule(
            display_name=display_name,
            pipeline_job=pipeline_job,
            schedule=args.cron_schedule,
        )
        logging.info(f"Scheduled pipeline job created: {job.resource_name}")

        if args.schedule_only:
            return

    # Create and submit the PipelineJob
    pipeline_job = aiplatform.PipelineJob(
        display_name=display_name,
        template_path=PIPELINE_FILE_NAME,
        pipeline_root=args.pipeline_root,
        parameter_values=parameter_values,
        enable_caching=not args.disable_caching,
    )

    pipeline_job.submit(service_account=args.service_account)
    logging.info(f"Pipeline job submitted: {pipeline_job.resource_name}")


if __name__ == "__main__":
    main()
