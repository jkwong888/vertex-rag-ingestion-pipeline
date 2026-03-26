from kfp import dsl
from kfp import compiler
import logging
from components.bm25_op import build_bm25_index

@dsl.pipeline(
    name="bm25-index-generation-pipeline",
    description="A pipeline to build BM25 index from pre-chunked data in GCS."
)
def pipeline(
    gcs_chunks_uri: str,
):
    build_bm25_index(
        gcs_chunks_uri=gcs_chunks_uri
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="bm25_index_pipeline.yaml"
    )
    logging.info("Pipeline compiled to bm25_index_pipeline.yaml")
