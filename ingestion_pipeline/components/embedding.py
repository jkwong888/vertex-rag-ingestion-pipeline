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
    bm25_index: Input[Artifact],
    chunks_dataset: Input[Artifact],
    output_dataset: Output[Artifact]
):
    """
    Generates sparse embeddings for document chunks using a pre-trained BM25 index.

    Args:
        bm25_index: Input artifact containing the BM25 index parameters.
        chunks_dataset: Input artifact containing document chunks in JSONL format.
        output_dataset: Output artifact where sparse embeddings will be saved in JSONL format.
    """
    import json
    import logging
    import os
    import tempfile
    from typing import Tuple, Optional
    from urllib.parse import urlparse
    from google.cloud import storage
    from pinecone_text.sparse import BM25Encoder

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def parse_gcs_uri(uri: str) -> Tuple[str, str]:
        """Parses a GCS URI into bucket name and prefix."""
        parsed = urlparse(uri)
        return parsed.netloc, parsed.path.lstrip("/")

    def download_gcs_directory(gcs_uri: str, local_dir: str) -> None:
        """Downloads all blobs from a GCS directory to a local directory."""
        bucket_name, prefix = parse_gcs_uri(gcs_uri)
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            if blob.name.endswith('/'):
                continue
            relative_path = os.path.relpath(blob.name, prefix)
            local_file_path = os.path.join(local_dir, relative_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            blob.download_to_filename(local_file_path)

    def load_bm25_retriever(index_artifact: Artifact) -> Optional[BM25Encoder]:
        """Downloads and loads the BM25 encoder from an artifact path or GCS URI."""
        index_gcs_uri = index_artifact.uri
        if not index_gcs_uri:
            return None
        
        local_temp_dir = tempfile.mkdtemp()
        try:
            # If the artifact is already downloaded locally (common in local test/orchestration)
            if index_artifact.path and os.path.isdir(index_artifact.path):
                local_temp_dir = index_artifact.path
            else:
                download_gcs_directory(index_gcs_uri, local_temp_dir)
            
            # Find the parameter file
            params_filename = "bm25_params.json"
            index_path = os.path.join(local_temp_dir, params_filename)
            if not os.path.exists(index_path):
                 for root, _, files in os.walk(local_temp_dir):
                     if params_filename in files:
                         index_path = os.path.join(root, params_filename)
                         break
            
            if not os.path.exists(index_path):
                logger.warning(f"{params_filename} not found in {index_gcs_uri}")
                return None
                
            encoder = BM25Encoder()
            encoder.load(index_path)
            logger.info(f"Loaded BM25 encoder from {index_path}")
            return encoder
        except Exception as error:
            logger.error(f"Error loading BM25 from {index_gcs_uri}: {error}")
            return None

    retriever = load_bm25_retriever(bm25_index)
    
    with open(chunks_dataset.path, 'r', encoding='utf-8') as input_file, \
         open(output_dataset.path, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            chunk_data = json.loads(line)
            sparse_vector = {}
            
            if retriever:
                sparse_vector = retriever.encode_documents(chunk_data['chunk'])
            
            result = {
                "id": chunk_data['id'],
                "sparse_embedding": sparse_vector
            }
            output_file.write(json.dumps(result) + '\n')

@dsl.component(
    base_image='python:3.11',
    packages_to_install=[
        'google-genai',
        'google-auth',
        'sentence-transformers',
        'torch'
    ]
)
def generate_dense_embeddings(
    embedding_model: str,
    project: str,
    location: str,
    chunks_dataset: Input[Artifact],
    output_dataset: Output[Artifact]
):
    """
    Generates dense embeddings for document chunks using either Vertex AI or a local model.

    Args:
        embedding_model: Name of the model to use (e.g., 'text-embedding-004' or 'local').
        project: Google Cloud Project ID.
        location: Google Cloud region.
        chunks_dataset: Input artifact containing document chunks in JSONL format.
        output_dataset: Output artifact where dense embeddings will be saved in JSONL format.
    """
    import json
    import logging
    from typing import Callable, List, Optional
    import google.auth
    from google import genai
    from google.genai import types

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def get_vertex_embed_fn(model_id: str, project_id: str) -> Optional[Callable[[str], List[float]]]:
        """Initializes and returns a function for Vertex AI embeddings."""
        if not project_id:
             _, inferred_project = google.auth.default()
             project_id = inferred_project
        
        try:
            client = genai.Client(
                vertexai=True, 
                project=project_id, 
                location="global", 
                http_options=types.HttpOptions(
                    timeout=30000,
                    retry_options=types.HttpRetryOptions(attempts=5)
                )
            )
            
            def embed_text(text: str) -> List[float]:
                response = client.models.embed_content(
                    model=model_id,
                    contents=text,
                    config=types.EmbedContentConfig(output_dimensionality=768)
                )
                return response.embeddings[0].values
            
            logger.info(f"Initialized Vertex AI model: {model_id}")
            return embed_text
        except Exception as error:
            logger.error(f"Failed to initialize GenAI client: {error}")
            return None

    def get_local_embed_fn(model_name: str) -> Optional[Callable[[str], List[float]]]:
        """Initializes and returns a function for local SentenceTransformer embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            actual_model = "all-MiniLM-L6-v2" if model_name == "local" else model_name
            
            logger.info(f"Loading local model: {actual_model}")
            model = SentenceTransformer(actual_model)
            
            def embed_text(text: str) -> List[float]:
                return model.encode(text).tolist()
                
            return embed_text
        except Exception as error:
            logger.error(f"Failed to load local model {model_name}: {error}")
            return None

    # Determine which embedding function to use
    is_vertex_managed = any(
        m in embedding_model 
        for m in ["gemini", "text-embedding", "text-multilingual-embedding"]
    )
    
    if is_vertex_managed:
        embed_fn = get_vertex_embed_fn(embedding_model, project)
    else:
        embed_fn = get_local_embed_fn(embedding_model)

    # Process chunks
    with open(chunks_dataset.path, 'r', encoding='utf-8') as input_file, \
         open(output_dataset.path, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            chunk_data = json.loads(line)
            dense_vector = []
            
            if embed_fn:
                try:
                    dense_vector = embed_fn(chunk_data['chunk'])
                except Exception as error:
                    logger.error(f"Embedding failed for chunk {chunk_data['id']}: {error}")
            
            result = {
                "id": chunk_data['id'],
                "embedding": dense_vector
            }
            output_file.write(json.dumps(result) + '\n')
