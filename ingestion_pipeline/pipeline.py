from kfp import dsl

from components.gcs import get_gcs_blobs, get_bm25_index_uri
from components.chunk import chunk_documents, merge_embeddings
from components.embedding import generate_sparse_embeddings, generate_dense_embeddings
from components.update_index import update_batch_index

@dsl.pipeline(name='rag-ingestion-pipeline')
def rag_ingestion_pipeline(
    gcs_input_uri: str,
    index_name: str,
    bm25_artifact_name: str = "bm25_index",
    embedding_model: str = "gemini-embedding-001",
    project: str = "",
    location: str = "us-central1"
):
    uris_op = get_gcs_blobs(gcs_input_uri=gcs_input_uri)
    
    bm25_uri_op = get_bm25_index_uri(
        artifact_display_name=bm25_artifact_name,
        project=project,
        location=location
    )
    
    chunk_op = chunk_documents(uris=uris_op.output)
    
    sparse_op = generate_sparse_embeddings(
        bm25_index_uri=bm25_uri_op.output,
        chunks_dataset=chunk_op.outputs["output_dataset"]
    )
    
    dense_op = generate_dense_embeddings(
        embedding_model=embedding_model,
        project=project,
        location=location,
        chunks_dataset=chunk_op.outputs["output_dataset"]
    )
    
    merge_op = merge_embeddings(
        chunks_dataset=chunk_op.outputs["output_dataset"],
        sparse_embeddings_dataset=sparse_op.outputs["output_dataset"],
        dense_embeddings_dataset=dense_op.outputs["output_dataset"]
    )
    
    update_op = update_batch_index(
        project=project,
        location=location,
        index_name=index_name,
        data_gcs_artifact=merge_op.outputs["output_dataset"],
        gcs_input_uri=gcs_input_uri
    )
    update_op.after(merge_op)

if __name__ == "__main__":
    from kfp.compiler import Compiler
    Compiler().compile(rag_ingestion_pipeline, 'rag_ingestion_pipeline.yaml')