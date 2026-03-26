from kfp import dsl
from typing import List

@dsl.component(
    base_image='python:3.11',
    packages_to_install=['google-cloud-storage']
)
def get_gcs_blobs(gcs_input_uri: str) -> List[str]:
    """
    Lists all non-directory blobs in a GCS path.

    Args:
        gcs_input_uri: GCS URI (gs://bucket/prefix) to search.

    Returns:
        List of absolute GCS URIs for all found blobs.
    """
    from google.cloud import storage
    from urllib.parse import urlparse
    from typing import Tuple

    def parse_gcs_uri(uri: str) -> Tuple[str, str]:
        """Parses GCS URI into bucket name and prefix."""
        parsed = urlparse(uri)
        if parsed.scheme != "gs":
            raise ValueError(f"URI {uri} must start with gs://")
        return parsed.netloc, parsed.path.lstrip("/")

    target_bucket_name, target_prefix = parse_gcs_uri(gcs_input_uri)
    storage_client = storage.Client()
    bucket = storage_client.bucket(target_bucket_name)
    blob_iterator = bucket.list_blobs(prefix=target_prefix)
    
    blob_uris = []
    for blob in blob_iterator:
        if not blob.name.endswith('/'):
             blob_uris.append(f"gs://{target_bucket_name}/{blob.name}")
    
    return blob_uris

@dsl.component(
    base_image='python:3.11',
    packages_to_install=['google-cloud-aiplatform']
)
def get_bm25_index_artifact(
    artifact_display_name: str, 
    project: str, 
    location: str,
    bm25_index: dsl.Output[dsl.Artifact]
):
    """
    Retrieves the most recent BM25 index artifact by its display name and outputs it as an artifact.

    Args:
        artifact_display_name: The display name used for the BM25 artifact in Vertex AI.
        project: Google Cloud Project ID.
        location: Google Cloud region.
        bm25_index: Output artifact referencing the found BM25 index.
    """
    from google.cloud import aiplatform
    import google.auth
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    effective_project = project
    if not effective_project:
         _, inferred_project = google.auth.default()
         effective_project = inferred_project
    
    try:
        aiplatform.init(project=effective_project, location=location)
        matching_artifacts = aiplatform.Artifact.list(filter=f'display_name="{artifact_display_name}"')
        
        if not matching_artifacts:
            logger.warning(f"No artifact found with display name: {artifact_display_name}")
            return

        matching_artifacts.sort(key=lambda artifact: artifact.create_time, reverse=True)
        latest_artifact = matching_artifacts[0]
        
        logger.info(f"Found latest artifact: {latest_artifact.resource_name} at {latest_artifact.uri}")
        bm25_index.uri = latest_artifact.uri
    except Exception as error:
        logger.error(f"Error searching for BM25 artifact: {error}")

@dsl.component(
    base_image='python:3.11',
    packages_to_install=['google-cloud-storage']
)
def download_chunks_dataset(
    gcs_chunks_uri: str,
    output_dataset: dsl.Output[dsl.Artifact]
):
    """
    Downloads and combines all JSONL chunk files from a GCS directory into a single local artifact.

    Args:
        gcs_chunks_uri: GCS URI (gs://bucket/prefix) containing JSONL chunk files.
        output_dataset: Output artifact path where combined data will be written.
    """
    import os
    import logging
    from typing import Tuple
    from urllib.parse import urlparse
    from google.cloud import storage

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def parse_gcs_uri(uri: str) -> Tuple[str, str]:
        """Parses GCS URI into bucket name and prefix."""
        parsed = urlparse(uri)
        if parsed.scheme != "gs":
            raise ValueError(f"URI {uri} must start with gs://")
        return parsed.netloc, parsed.path.lstrip("/")

    source_bucket_name, source_prefix = parse_gcs_uri(gcs_chunks_uri)
    storage_client = storage.Client()
    source_bucket = storage_client.bucket(source_bucket_name)
    blobs_to_download = source_bucket.list_blobs(prefix=source_prefix)
    
    total_combined_lines = 0
    with open(output_dataset.path, 'w', encoding='utf-8') as combined_file:
        for blob in blobs_to_download:
            # Only process JSONL files
            if not blob.name.endswith('.jsonl'):
                continue
            
            logger.info(f"Downloading and appending: {blob.name}")
            blob_content = blob.download_as_text(encoding="utf-8")
            
            for line in blob_content.splitlines():
                sanitized_line = line.strip()
                if sanitized_line:
                    combined_file.write(sanitized_line + '\n')
                    total_combined_lines += 1
    
    logger.info(f"Successfully combined {total_combined_lines} chunks into local path: {output_dataset.path}")
