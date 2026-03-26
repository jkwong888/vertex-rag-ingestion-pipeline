from kfp import dsl
from kfp import compiler
import logging
from components.scrape_op import scrape_bulbapedia_op

@dsl.pipeline(name="scrape-bulbapedia-pipeline")
def scrape_pipeline(
    gcs_bucket_name: str = "jkwng-vertex-experiments",
    gcs_bucket_path: str = "pokemon/bulbapedia/html",
):
    scrape_bulbapedia_op(
        gcs_bucket_name=gcs_bucket_name,
        gcs_bucket_path=gcs_bucket_path
    )

if __name__ == "__main__":
    compiler.Compiler().compile(scrape_pipeline, "scrape_pipeline.yaml")
