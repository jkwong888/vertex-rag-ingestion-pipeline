from kfp import dsl
from kfp.dsl import Input, Output, Artifact

@dsl.component(
    base_image='python:3.11'
)
def merge_embeddings(
    chunks_dataset: Input[Artifact],
    sparse_embeddings_dataset: Input[Artifact],
    dense_embeddings_dataset: Input[Artifact],
    output_dataset: Output[Artifact]
):
    """
    Merges original document chunks with their corresponding sparse and dense embeddings.

    Args:
        chunks_dataset: Input artifact containing original document chunks.
        sparse_embeddings_dataset: Input artifact containing sparse embeddings.
        dense_embeddings_dataset: Input artifact containing dense embeddings.
        output_dataset: Output artifact where merged data will be saved in JSONL format.
    """
    import json
    import logging
    import os
    from typing import Dict, Any, List

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def load_sparse_embeddings(file_path: str) -> Dict[str, Dict[str, Any]]:
        """Loads sparse embeddings from a JSONL file into a mapping by ID."""
        sparse_map = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                chunk_id = item.get('id')
                if not chunk_id:
                    continue
                
                sparse_embedding = item.get('sparse_embedding', {})
                # Normalize schema: change "indices" to "dimensions" if present
                if 'indices' in sparse_embedding:
                    sparse_embedding['dimensions'] = sparse_embedding.pop('indices')
                
                sparse_map[chunk_id] = sparse_embedding
        return sparse_map

    def load_dense_embeddings(file_path: str) -> Dict[str, List[float]]:
        """Loads dense embeddings from a JSONL file into a mapping by ID."""
        dense_map = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                chunk_id = item.get('id')
                if chunk_id:
                    dense_map[chunk_id] = item.get('embedding', [])
        return dense_map

    def prepare_output_path(base_path: str) -> str:
        """Ensures output directory exists and returns the full output file path."""
        if not base_path.startswith("gs://"):
            os.makedirs(base_path, exist_ok=True)
        
        if base_path.endswith(".json") or base_path.endswith(".jsonl"):
            return base_path
        return os.path.join(base_path, "batch.json")

    # Load all embeddings into memory for fast lookup
    logger.info("Loading sparse embeddings...")
    sparse_embeddings_by_id = load_sparse_embeddings(sparse_embeddings_dataset.path)
    
    logger.info("Loading dense embeddings...")
    dense_embeddings_by_id = load_dense_embeddings(dense_embeddings_dataset.path)
    
    output_file_path = prepare_output_path(output_dataset.path)
    logger.info(f"Merging data into {output_file_path}...")

    non_metadata_keys = {"id", "embedding", "sparse_embedding", "restricts", "numeric_restricts", "chunk"}

    with open(chunks_dataset.path, 'r', encoding='utf-8') as input_file, \
         open(output_file_path, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            chunk_data = json.loads(line)
            chunk_id = chunk_data['id']
            
            merged_record = {
                "id": chunk_id,
                "restricts": [],
                "numeric_restricts": [],
                "embedding_metadata": {},
            }

            # Attach embeddings if available
            if chunk_id in sparse_embeddings_by_id:
                merged_record['sparse_embedding'] = sparse_embeddings_by_id[chunk_id]
            
            if chunk_id in dense_embeddings_by_id:
                merged_record['embedding'] = dense_embeddings_by_id[chunk_id]
            
            # Vertex AI Vector Search requirement:
            # If a sparse_embedding object is present, it MUST contain at least one value.
            # We skip the entire record if the sparse embedding is missing or empty.
            sparse_obj = merged_record.get('sparse_embedding')
            if not sparse_obj or not sparse_obj.get('values'):
                logger.warning(f"Skipping record {chunk_id}: missing or empty sparse embedding.")
                continue

            # Map all other fields to embedding_metadata
            for key, value in chunk_data.items():
                if key not in non_metadata_keys:
                    merged_record["embedding_metadata"][key] = value

            output_file.write(json.dumps(merged_record) + '\n')
    
    logger.info("Merge operation completed successfully.")
