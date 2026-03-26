import sys
import os
from kfp import dsl

# Add parent directory to sys.path to allow cross-directory imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chunking.components.chunk_op import chunk_documents_op
from ingestion_pipeline.components.gcs import download_chunks_dataset, get_bm25_index_artifact
from ingestion_pipeline.components.embedding import generate_sparse_embeddings, generate_dense_embeddings
from ingestion_pipeline.components.merge import merge_embeddings
from ingestion_pipeline.components.update_index import update_batch_index

@dsl.pipeline(
    name='rag-update-ingestion-pipeline',
    description='Incremental index update pipeline: chunking, embedding, and partial index update.'
)
def rag_update_ingestion_pipeline(
    gcs_html_uri: str,
    gcs_chunks_uri: str, # Still used by update_batch_index for staging
    index_name: str,
    bm25_artifact_name: str = "bm25_index",
    embedding_model: str = "gemini-embedding-001",
    project: str = "",
    location: str = "us-central1",
    chunking_strategy_version: str = "v1"
):
    # 1. Chunking (for new HTML files)
    chunking_op = chunk_documents_op(
        gcs_html_uri=gcs_html_uri,
        chunking_strategy_version=chunking_strategy_version
    )

    # 2. Retrieve Existing BM25 Index
    bm25_artifact_op = get_bm25_index_artifact(
        artifact_display_name=bm25_artifact_name,
        project=project,
        location=location
    )

    # 3. Sparse Embedding Generation
    sparse_op = generate_sparse_embeddings(
        bm25_index=bm25_artifact_op.outputs["bm25_index"],
        chunks_dataset=chunking_op.outputs["output_chunks"]
    )

    # 4. Dense Embedding Generation
    dense_op = generate_dense_embeddings(
        embedding_model=embedding_model,
        project=project,
        location=location,
        chunks_dataset=chunking_op.outputs["output_chunks"]
    )

    # 5. Merge Embeddings
    merge_op = merge_embeddings(
        chunks_dataset=chunking_op.outputs["output_chunks"],
        sparse_embeddings_dataset=sparse_op.outputs["output_dataset"],
        dense_embeddings_dataset=dense_op.outputs["output_dataset"]
    )

    # 7. Update Batch Index (Incremental)
    update_op = update_batch_index(
        project=project,
        location=location,
        index_name=index_name,
        data_gcs_artifact=merge_op.outputs["output_dataset"],
        gcs_input_uri=gcs_chunks_uri,
        is_complete_overwrite=False
    )

if __name__ == "__main__":
    from kfp.compiler import Compiler
    Compiler().compile(rag_update_ingestion_pipeline, 'rag_update_ingestion_pipeline.yaml')
