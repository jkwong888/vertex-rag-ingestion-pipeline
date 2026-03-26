from kfp import dsl
from kfp import compiler
from components.chunk_op import chunk_documents_op

@dsl.pipeline(name="unified-chunking-pipeline")
def chunk_pipeline(
    gcs_html_uri: str,
    gcs_chunks_uri: str,
):
    chunk_documents_op(
        gcs_html_uri=gcs_html_uri,
        gcs_chunks_uri=gcs_chunks_uri
    )

if __name__ == "__main__":
    compiler.Compiler().compile(chunk_pipeline, "chunk_pipeline.yaml")
