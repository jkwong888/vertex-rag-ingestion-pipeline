from kfp import dsl
from kfp.dsl import Input, Artifact

@dsl.component(
    base_image='python:3.11',
    packages_to_install=['google-cloud-aiplatform', 'google-cloud-storage']
)
def update_batch_index(
    project: str,
    location: str,
    index_name: str,
    data_gcs_artifact: Input[Artifact],
    gcs_input_uri: str,
    is_complete_overwrite: bool = False
) -> str:
    """
    Updates a Vertex AI Matching Engine (Vector Search) index with new embeddings.

    Args:
        project: Google Cloud Project ID.
        location: Google Cloud region.
        index_name: Display name of the index to update.
        data_gcs_artifact: Input artifact containing the merged embeddings.
        gcs_input_uri: Target GCS URI for staging local data if artifact is local.
        is_complete_overwrite: If true, completely overwrite the index.

    Returns:
        The resource name of the updated index.
    """
    from google.cloud import aiplatform, storage
    import logging
    import os
    import uuid
    from urllib.parse import urlparse

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def upload_to_staging(local_directory_path: str, staging_uri: str) -> str:
        """Uploads local files to a GCS staging location and returns the GCS URI."""
        parsed_uri = urlparse(staging_uri)
        if parsed_uri.scheme != "gs":
             raise ValueError("gcs_input_uri must start with gs://")
        
        target_bucket_name = parsed_uri.netloc
        storage_client = storage.Client(project=project)
        target_bucket = storage_client.bucket(target_bucket_name)
        
        unique_session_id = str(uuid.uuid4())
        gcs_staging_prefix = f"staging/{unique_session_id}"
        
        logger.info(f"Uploading local artifact from {local_directory_path} to gs://{target_bucket_name}/{gcs_staging_prefix}")

        for root, _, files in os.walk(local_directory_path):
            for filename in files:
                full_local_path = os.path.join(root, filename)
                destination_blob_name = f"{gcs_staging_prefix}/{filename}"
                
                blob = target_bucket.blob(destination_blob_name)
                blob.upload_from_filename(full_local_path)
                logger.debug(f"Uploaded {filename} to {destination_blob_name}")
        
        return f"gs://{target_bucket_name}/{gcs_staging_prefix}"

    aiplatform.init(project=project, location=location)
    
    # Determine the data URI (use artifact URI or upload local path to GCS)
    source_data_uri = data_gcs_artifact.uri
    if not source_data_uri.startswith("gs://"):
        logger.info("Artifact URI is local. Initiating GCS upload...")
        source_data_uri = upload_to_staging(data_gcs_artifact.path, gcs_input_uri)

    # Locate the target Vector Search index
    logger.info(f"Searching for index with display name: {index_name}")
    matching_indexes = aiplatform.MatchingEngineIndex.list(filter=f'display_name="{index_name}"')
    
    if not matching_indexes:
        raise RuntimeError(f"Index '{index_name}' not found in project {project} ({location}).")
    
    target_index = matching_indexes[0]
    logger.info(f"Found index: {target_index.resource_name}")

    # Trigger the asynchronous update operation
    logger.info(f"Updating index with data from: {source_data_uri}")
    update_operation = target_index.update_embeddings(
        contents_delta_uri=source_data_uri,
        is_complete_overwrite=is_complete_overwrite
    )
    
    logger.info(f"Update operation started: {update_operation.name}")
    update_operation.wait()
    logger.info("Index update completed successfully.")
    
    return target_index.resource_name
