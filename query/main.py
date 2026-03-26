import argparse
import os
import tempfile
import json
from urllib.parse import urlparse
from typing import List, Dict, Any

from google.cloud import aiplatform
from google.cloud import storage
from google import genai
from google.genai import types
from pinecone_text.sparse import BM25Encoder

def parse_gcs_uri(uri):
    """Parses a GCS URI into bucket name and prefix."""
    if not uri.startswith("gs://"):
        raise ValueError("URI must start with gs://")
    
    parsed = urlparse(uri)
    bucket_name = parsed.netloc
    prefix = parsed.path.lstrip("/")
    return bucket_name, prefix


def download_blob(bucket_name: str, source_blob_name: str, destination_file_name: str):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded storage object {source_blob_name} from bucket {bucket_name} to {destination_file_name}.")

def load_bm25_encoder(gcs_uri: str) -> BM25Encoder:
    """Loads the BM25 encoder from a JSON file in GCS."""
    print(f"Loading BM25 encoder from {gcs_uri}...")
    gcs_bucket, gcs_path = parse_gcs_uri(gcs_uri)

    file_path = "bm25_params.json"

    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        download_blob(gcs_bucket, f"{gcs_path}/{file_path}", temp_file.name)
        temp_file_path = temp_file.name

    try:
        bm25_encoder = BM25Encoder()
        bm25_encoder.load(temp_file_path)
        print("BM25 encoder loaded successfully.")
        return bm25_encoder
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def get_dense_embedding(text: str, client: genai.Client, model_name: str = "gemini-embedding-001") -> List[float]:
    """Generates a dense embedding using the Google GenAI SDK (Vertex AI backend)."""
    result = client.models.embed_content(
        model=model_name,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT", 
            title="Query",
            output_dimensionality=768,
        )
    )
    return result.embeddings[0].values

def get_sparse_embedding(text: str, bm25_encoder: BM25Encoder) -> Dict[str, List[Any]]:
    """Generates a sparse embedding using Pinecone BM25 encoder."""
    # Pinecone text returns {'indices': [...], 'values': [...]}
    sparse_vec = bm25_encoder.encode_queries(text)
    return sparse_vec

def hybrid_search(
    index_endpoint_name: str,
    deployed_index_id: str,
    query_text: str,
    bm25_encoder: BM25Encoder,
    project_id: str,
    location: str,
    client: genai.Client,
    dense_model_name: str = "gemini-embedding-001",
    neighbor_count: int = 10,
    rrf_ranking_alpha: float = 0.5,
):
    """
    Performs a hybrid search on Vertex AI Vector Search Index.
    """
    # Initialize aiplatform for Vector Search interaction
    aiplatform.init(project=project_id, location=location)

    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name)

    # 1. Generate Dense Embedding
    print(f"Generating dense embedding for query: '{query_text}'")
    dense_embedding = get_dense_embedding(query_text, client, dense_model_name)

    # 2. Generate Sparse Embedding
    print("Generating sparse embedding...")
    sparse_embedding_dict = get_sparse_embedding(query_text, bm25_encoder)
    
    from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import HybridQuery
    import requests
    import google.auth
    import google.auth.transport.requests

    # Create HybridQuery
    # Note: 'rrf_ranking_alpha' is a parameter to control the merge (0.5 is a good default).
    # We keep the HybridQuery object for clarity/reference or just construct the dict directly.
    # But for REST, we construct the dict manually.

    print("Executing Hybrid Search on Vertex AI Index Endpoint (REST API)...")
    
    # Get the public endpoint domain
    public_endpoint_domain = index_endpoint.public_endpoint_domain_name
    if not public_endpoint_domain:
        # Fallback or error if no public endpoint
        print("No public endpoint found. Trying private endpoint...")
        # If private endpoint is needed, one might use index_endpoint.private_endpoint_domain_name
        # But for now let's assume public or fail.
        # Note: The SDK might handle this logic, but here we are manual.
        # Let's assume public for this user request context.
        pass

    # Construct URL
    # https://[endpoint-domain]/v1/projects/[project]/locations/[location]/indexEndpoints/[endpoint-id]/deployedIndexes/[deployed-index-id]:findNeighbors
    # Actually the format is: https://[endpoint-domain]/v1/projects/[project]/locations/[location]/indexEndpoints/[endpoint-id]:findNeighbors
    # And deployedIndexId is in the body.
    
    # We need the index_endpoint resource ID.
    # index_endpoint.resource_name gives projects/.../indexEndpoints/...
    # We can parse it or just use the index_endpoint_name if it is the ID.
    # The index_endpoint_name passed to the function might be the ID.
    
    api_endpoint = f"https://{public_endpoint_domain}/v1/{index_endpoint.resource_name}:findNeighbors"

    # Get Credentials
    credentials, _ = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    token = credentials.token

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # Construct Payload
    payload = {
        "deployedIndexId": deployed_index_id,
        "queries": [
            {
                "datapoint": {
                    "featureVector": dense_embedding,
                    "sparseEmbedding": {
                        "values": sparse_embedding_dict['values'],
                        "dimensions": sparse_embedding_dict['indices']
                    }
                },
                "neighborCount": neighbor_count,
                "rrfRankingAlpha": rrf_ranking_alpha,
            }
        ],
        "returnFullDatapoint": True
    }

    print(payload)

    response = requests.post(api_endpoint, headers=headers, json=payload)
    response.raise_for_status()
    response_json = response.json()

    # Parse Response
    # Response structure: {"nearestNeighbors": [{"id": "query_0", "neighbors": [...]}]}
    
    # TODO: note this is a hacky code for the following reasons:
    # - Vertex Vector Search is not designed to hold the actual data, just the vectors - but we stored the chunk in the embedding metadata
    # - the embedding_metadata field is not returned by default (we need to pass "returnFullDatapoint")
    # - and the SDK MatchNeighbor object does not deserialize the embedding metadata either, only the REST API does
    nearest_neighbors = response_json.get("nearestNeighbors", [])
    if nearest_neighbors:
        neighbors_list = nearest_neighbors[0].get("neighbors", [])
        print(f"Found {len(neighbors_list)} neighbors.")
        for idx, neighbor in enumerate(neighbors_list):
            datapoint = neighbor.get("datapoint", {})
            neighbor_id = datapoint.get("datapointId")
            distance = neighbor.get("distance")
            sparse_distance = neighbor.get("sparse_distance")
            chunk = datapoint.get("embeddingMetadata", {}).get("chunk", {})
            print(json.dumps(neighbor))
            print(f"Rank {idx+1}: ID={neighbor_id}, Dense Distance={distance}")
        return neighbors_list # Return list of dicts instead of objects
    else:
        print("No neighbors found.")
        return []

def generate_answer(query: str, context_chunks: List[str], client: genai.Client, model_name: str = "gemini-2.5-flash") -> str:
    """Generates an answer using Gemini Flash based on the retrieved context."""
    print(f"Generating answer using {model_name}...")
    
    context_text = "\n".join([f"Context {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])
    
    prompt = f"""You are a helpful assistant. Answer the user's question using only the provided context.

Context:
{context_text}

Question:
{query}

Answer:
"""

    print(f"Prompt:\n{prompt}")
    
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=0,  # Use `0` to turn off thinking
            )
        ),
    )
    return response.text
def get_bm25_index_artifact_uri(project_id: str, location: str) -> str:
    """Fetches the URI of the BM25 index artifact from Vertex AI Metadata."""
    aiplatform.init(project=project_id, location=location)

    artifacts = aiplatform.Artifact.list(filter='display_name="bm25_index" AND state="LIVE"')

    if not artifacts:
        raise RuntimeError("BM25 index artifact not found.")

    # Assuming there's only one live artifact with this display name
    return artifacts[0].uri

def main():
    project_id = "jkwng-vertex-playground"
    location = "us-central1"
    index_endpoint_name = "4455874225655250944"
    deployed_index_id = "bulbapedia_index"
    bm25_index_uri = get_bm25_index_artifact_uri(project_id, location)
    
    bm25_encoder = load_bm25_encoder(bm25_index_uri)

    query = "What flying type Pokemon can be found in Kanto region?"

    neighbor_count = 25
    rrf_ranking_alpha = 0.5
    
    client = genai.Client(vertexai=True, project=project_id, location="global")

    neighbors = hybrid_search(
        index_endpoint_name=index_endpoint_name,
        deployed_index_id=deployed_index_id,
        query_text=query,
        bm25_encoder=bm25_encoder,
        project_id=project_id,
        location=location,
        client=client,
        neighbor_count=neighbor_count,
        rrf_ranking_alpha=rrf_ranking_alpha
    )

    context_chunks = []
    blob_cache = {}
    storage_client = storage.Client()
    
    for neighbor in neighbors:
        datapoint = neighbor.get("datapoint", {})
        metadata = datapoint.get("embeddingMetadata", {})
        
        chunk_uri = metadata.get("chunk_gcs_uri")
        chunk_line_offset = metadata.get("chunk_line_offset")
        
        if chunk_uri and chunk_line_offset is not None:
            if chunk_uri not in blob_cache:
                if chunk_uri.startswith("gs://"):
                    parsed = urlparse(chunk_uri)
                    bucket = storage_client.bucket(parsed.netloc)
                    blob = bucket.blob(parsed.path.lstrip("/"))
                    content = blob.download_as_text(encoding="utf-8")
                    blob_cache[chunk_uri] = content.splitlines()
                else:
                    # Assume local path (for local test runner)
                    if os.path.exists(chunk_uri):
                        with open(chunk_uri, 'r', encoding='utf-8') as f:
                            blob_cache[chunk_uri] = f.readlines()
                    else:
                        print(f"Warning: Chunk URI {chunk_uri} not found locally or via GCS.")
                        continue
                
            lines = blob_cache[chunk_uri]
            offset_idx = int(chunk_line_offset)
            if offset_idx < len(lines):
                line = lines[offset_idx]
                chunk_data = json.loads(line)
                chunk_text = chunk_data.get("chunk")
                if chunk_text:
                    context_chunks.append(chunk_text)

    if context_chunks:
        answer = generate_answer(query, context_chunks, client, model_name="gemini-2.5-flash")
        print("-" * 80)
        print("Generated Answer:")
        print(answer)
        print("-" * 80)

if __name__ == "__main__":
    main()
