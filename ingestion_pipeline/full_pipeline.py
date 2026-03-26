import sys
import os
from kfp import dsl

# Add parent directory to sys.path to allow cross-directory imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chunking.components.chunk_op import chunk_documents_op
from bm25_corpus_index.components.bm25_op import build_bm25_index
from ingestion_pipeline.components.gcs import download_chunks_dataset
from ingestion_pipeline.components.embedding import generate_sparse_embeddings, generate_dense_embeddings
from ingestion_pipeline.components.merge import merge_embeddings
from ingestion_pipeline.components.update_index import update_batch_index
from ingestion_pipeline.components.vector_search import create_vector_search_index

@dsl.pipeline(
    name='rag-full-ingestion-pipeline',
    description='Full index regeneration pipeline: chunking, bm25, embedding, and full index overwrite.'
)
def rag_full_ingestion_pipeline(
    gcs_html_uri: str,
    gcs_chunks_uri: str, # Still used by update_batch_index for staging
    index_name: str,
    embedding_model: str = "gemini-embedding-001",
    project: str = "",
    location: str = "us-central1",
    previous_chunks_uri: str = "",
    chunking_strategy_version: str = "v1"
):
    # 0. Ensure Vector Search index exists
    create_index_op = create_vector_search_index(
        project=project,
        location=location,
        index_name=index_name,
        dimensions=768 # Default for gemini-embedding-001
    )

    # 1. Chunking
    chunking_op = chunk_documents_op(
        gcs_html_uri=gcs_html_uri,
        previous_chunks_uri=previous_chunks_uri,
        chunking_strategy_version=chunking_strategy_version
    )

    # 2. BM25 Index Build
    bm25_op = build_bm25_index(
        chunks_dataset=chunking_op.outputs["output_chunks"]
    )
    bm25_op.after(chunking_op)

    # 3. Sparse Embedding Generation
    sparse_op = generate_sparse_embeddings(
        bm25_index=bm25_op.outputs["bm25_index"],
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

    # 7. Update Batch Index (Full Overwrite)
    update_op = update_batch_index(
        project=project,
        location=location,
        index_name=index_name,
        data_gcs_artifact=merge_op.outputs["output_dataset"],
        gcs_input_uri=gcs_chunks_uri,
        is_complete_overwrite=True
    )
    update_op.after(create_index_op)


if __name__ == "__main__":
    from kfp.compiler import Compiler
    Compiler().compile(rag_full_ingestion_pipeline, 'rag_full_ingestion_pipeline.yaml')
