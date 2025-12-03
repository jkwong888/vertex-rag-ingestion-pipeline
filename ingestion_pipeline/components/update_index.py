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
    gcs_input_uri: str
) -> str:
    from google.cloud import aiplatform, storage
    import logging
    import os
    import uuid
    from urllib.parse import urlparse

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    aiplatform.init(project=project, location=location)
    
    # Determine the URI to use
    data_uri = data_gcs_artifact.uri
    
    if not data_uri.startswith("gs://"):
        logger.info(f"Artifact URI {data_uri} is local. Uploading to GCS...")
        
        parsed = urlparse(gcs_input_uri)
        if parsed.scheme != "gs":
             raise ValueError("gcs_input_uri must start with gs://")
        bucket_name = parsed.netloc
        
        storage_client = storage.Client(project=project)
        bucket = storage_client.bucket(bucket_name)
        
        # Upload all files in the directory
        local_path = data_gcs_artifact.path
        upload_session_id = uuid.uuid4()

        # Walk through the directory and upload every file
        for root, _, files in os.walk(local_path):
            for filename in files:
                local_file_path = os.path.join(root, filename)
                blob_name = f"staging/{upload_session_id}/{filename}"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(local_file_path)
                logger.info(f"Uploaded {local_file_path} to gs://{bucket.name}/{blob_name}")
        data_uri = f"gs://{bucket.name}/staging/{upload_session_id}"

    # Find the index
    indexes = aiplatform.MatchingEngineIndex.list(filter=f'display_name="{index_name}"')
    if not indexes:
        raise RuntimeError(f"Index {index_name} not found.")
    
    index = indexes[0]
    logger.info(f"Found index: {index.resource_name}")

    logger.info(f"Updating index with data from: {data_uri}")
    # Update the index
    operation = index.update_embeddings(
        contents_delta_uri=data_uri,
        is_complete_overwrite=False
    )
    
    logger.info(f"Update operation started: {operation.name}")
    operation.wait()
    logger.info("Update operation completed.")
    
    return index.resource_name
