from kfp import dsl
from typing import List

@dsl.component(
    base_image='python:3.11',
    packages_to_install=['google-cloud-storage']
)
def get_gcs_blobs(gcs_input_uri: str) -> List[str]:
    from google.cloud import storage
    from urllib.parse import urlparse
    
    def parse_gcs_uri(uri: str):
        parsed = urlparse(uri)
        if parsed.scheme != "gs":
            raise ValueError("URI must start with gs://")
        bucket_name = parsed.netloc
        prefix = parsed.path.lstrip("/")
        return bucket_name, prefix

    bucket_name, prefix = parse_gcs_uri(gcs_input_uri)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    uris = []
    for blob in blobs:
        if not blob.name.endswith('/'):
             uris.append(f"gs://{bucket_name}/{blob.name}")
    return uris

@dsl.component(
    base_image='python:3.11',
    packages_to_install=['google-cloud-aiplatform']
)
def get_bm25_index_uri(artifact_display_name: str, project: str, location: str) -> str:
    from google.cloud import aiplatform
    import google.auth
    
    if not project:
         credentials, project_id = google.auth.default()
         project = project_id
    
    try:
        aiplatform.init(project=project, location=location)
        artifacts = aiplatform.Artifact.list(filter=f'display_name="{artifact_display_name}"')
        
        if not artifacts:
            return ""

        artifacts.sort(key=lambda x: x.create_time, reverse=True)
        return artifacts[0].uri
    except Exception:
        return ""
