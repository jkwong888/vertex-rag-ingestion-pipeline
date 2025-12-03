from kfp import dsl
from kfp.dsl import Input, Output, Artifact

@dsl.component(
    base_image='python:3.11',
    packages_to_install=[
        'google-cloud-storage',
        'pinecone-text',
        'numpy'
    ]
)
def generate_sparse_embeddings(
    bm25_index_uri: str,
    chunks_dataset: Input[Artifact],
    output_dataset: Output[Artifact]
):
    import json
    import logging
    import os
    import tempfile
    from urllib.parse import urlparse
    from google.cloud import storage
    from pinecone_text.sparse import BM25Encoder

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def parse_gcs_uri(uri: str):
        parsed = urlparse(uri)
        return parsed.netloc, parsed.path.lstrip("/")

    def download_gcs_directory(gcs_uri: str, local_dir: str):
        bucket_name, prefix = parse_gcs_uri(gcs_uri)
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            if blob.name.endswith('/'): continue
            rel_path = os.path.relpath(blob.name, prefix)
            local_file_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            blob.download_to_filename(local_file_path)

    def load_bm25_retriever(index_gcs_uri: str) -> BM25Encoder:
        if not index_gcs_uri: return None
        local_dir = tempfile.mkdtemp()
        try:
            download_gcs_directory(index_gcs_uri, local_dir)
            index_path = os.path.join(local_dir, "bm25_params.json")
            if not os.path.exists(index_path):
                 for root, _, files in os.walk(local_dir):
                     if "bm25_params.json" in files:
                         index_path = os.path.join(root, "bm25_params.json")
                         break
            if not os.path.exists(index_path): return None
            retriever = BM25Encoder()
            retriever.load(index_path)
            return retriever
        except Exception as e:
            logger.error(f"Error loading BM25: {e}")
            return None

    retriever = load_bm25_retriever(bm25_index_uri)
    
    with open(chunks_dataset.path, 'r') as infile, open(output_dataset.path, 'w') as outfile:
        for line in infile:
            chunk = json.loads(line)
            sparse_embedding = {}
            if retriever:
                sparse_embedding = retriever.encode_documents(chunk['chunk'])
            
            output = {
                "id": chunk['id'],
                "sparse_embedding": sparse_embedding
            }
            outfile.write(json.dumps(output) + '\n')

@dsl.component(
    base_image='python:3.11',
    packages_to_install=[
        'google-genai',
        'google-auth'
    ]
)
def generate_dense_embeddings(
    embedding_model: str,
    project: str,
    location: str,
    chunks_dataset: Input[Artifact],
    output_dataset: Output[Artifact]
):
    import json
    import logging
    import google.auth
    from google import genai
    from google.genai import types

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not project:
         _, project_id = google.auth.default()
         project = project_id
    
    try:
        client = genai.Client(
            vertexai=True, 
            project=project, 
            location="global", 
            http_options=types.HttpOptions(
                timeout=30000,
                retry_options=types.HttpRetryOptions(
                    attempts=5,
                )

            )
        )
    except Exception as e:
        logger.error(f"Failed to init GenAI: {e}")
        client = None

    with open(chunks_dataset.path, 'r') as infile, open(output_dataset.path, 'w') as outfile:
        for line in infile:
            chunk = json.loads(line)
            dense_embedding = []
            if client:
                try:
                    result = client.models.embed_content(
                        model=embedding_model,
                        contents=chunk['chunk'],
                        config=types.EmbedContentConfig(output_dimensionality=768)
                    )
                    dense_embedding = result.embeddings[0].values
                except Exception as e:
                    logger.error(f"Embedding failed for {chunk['id']}: {e}")
            
            output = {
                "id": chunk['id'],
                "embedding": dense_embedding
            }
            outfile.write(json.dumps(output) + '\n')
