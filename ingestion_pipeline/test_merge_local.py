import sys
import os
from kfp import local
from kfp import dsl

# Set up local execution runner
local.init(runner=local.SubprocessRunner())

# Add parent directory to sys.path to allow cross-directory imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chunking.components.chunk_op import chunk_documents_op
from bm25_corpus_index.components.bm25_op import build_bm25_index
from ingestion_pipeline.components.embedding import generate_sparse_embeddings, generate_dense_embeddings
from ingestion_pipeline.components.merge import merge_embeddings

@dsl.pipeline(
    name='rag-test-merge-pipeline',
    description='Local test pipeline that stops after merging embeddings.'
)
def rag_test_merge_pipeline(
    gcs_html_uri: str,
    embedding_model: str = "gemini-embedding-001",
    project: str = "",
    location: str = "us-central1",
    chunking_strategy_version: str = "v1"
):
    # 1. Chunking
    chunking_op = chunk_documents_op(
        gcs_html_uri=gcs_html_uri,
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

if __name__ == "__main__":
    # Configuration for the test run
    TEST_GCS_HTML_URI = "gs://jkwng-vertex-experiments/pokemon/bulbapedia/test-subset/html"
    PROJECT_ID = "jkwng-vertex-playground" 
    LOCATION = "us-central1"

    print(f"Starting local pipeline run...")
    print(f"Input HTML: {TEST_GCS_HTML_URI}")

    try:
        rag_test_merge_pipeline(
            gcs_html_uri=TEST_GCS_HTML_URI,
            project=PROJECT_ID,
            location=LOCATION
        )
        print("\nLocal pipeline execution successful.")
    except Exception as e:
        print(f"\nLocal pipeline execution failed: {e}")
        sys.exit(1)
