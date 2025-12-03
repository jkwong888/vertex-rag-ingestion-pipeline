from kfp import dsl
from kfp import compiler
from google.cloud import aiplatform
import logging

@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "google-cloud-storage",
        "markdownify",
        "langchain",
        "langchain-text-splitters"
    ]
)
def chunk_documents(
    gcs_uri: str,
    chunked_corpus: dsl.Output[dsl.Dataset],
):
    import json
    import os
    import logging
    from urllib.parse import urlparse
    from google.cloud import storage
    from markdownify import markdownify as md
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

    logging.getLogger().setLevel(logging.INFO)

    def parse_gcs_uri(uri):
        """Parses a GCS URI into bucket name and prefix."""
        if not uri.startswith("gs://"):
            raise ValueError("URI must start with gs://")
        
        parsed = urlparse(uri)
        bucket_name = parsed.netloc
        prefix = parsed.path.lstrip("/")
        return bucket_name, prefix

    def get_gcs_blobs_iterator(bucket_name, prefix):
        """Returns a bucket object and an iterator over blobs."""
        logging.info(f"Connecting to GCS bucket: {bucket_name}...")
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        logging.info(f"Listing blobs in {bucket_name}/{prefix}...")
        blobs = bucket.list_blobs(prefix=prefix)
        return bucket, blobs

    def convert_to_markdown(html_contents):
        """Converts a list of HTML strings to Markdown."""
        return [md(html) for html in html_contents]

    # Main logic
    try:
        bucket_name, prefix = parse_gcs_uri(gcs_uri)
    except ValueError as e:
        logging.error(f"Error: {e}")
        return

    # 1. Setup Output
    if not os.path.exists(chunked_corpus.path):
        os.makedirs(chunked_corpus.path)

    buffer = []
    file_index = 0
    CHUNK_LIMIT = 100000
    
    # 2. Iterate and Process in Batches
    try:
        bucket, blobs = get_gcs_blobs_iterator(bucket_name, prefix)
    except Exception as e:
         logging.error(f"Error accessing GCS: {e}")
         return

    batch_size = 100
    total_files = 0
    total_text_chunks = 0
    total_md_chunks = 0

    while True:
        batch_blobs = []
        for _ in range(batch_size):
            try:
                batch_blobs.append(next(blobs))
            except StopIteration:
                break
        if not batch_blobs:
            break
            
        batch_html = []
        batch_ids = []
        
        for blob in batch_blobs:
            if blob.name.endswith(".html") or blob.name.endswith(".htm"):
                try:
                    content = blob.download_as_text()
                    batch_html.append(content)
                    batch_ids.append(f"gs://{bucket_name}/{blob.name}")
                except Exception as e:
                    logging.warning(f"Failed to download {blob.name}: {e}")
        
        if not batch_html:
            continue

        # Convert
        batch_markdown = convert_to_markdown(batch_html)
        
        # Chunking
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        
        for md_text, doc_id in zip(batch_markdown, batch_ids):
            # First split by header
            header_splits = markdown_splitter.split_text(md_text)
            # Then split by characters
            final_splits = text_splitter.split_documents(header_splits)

            total_md_chunks += len(header_splits)
            total_text_chunks += len(final_splits)
           
            for i, split in enumerate(final_splits):
                buffer.append({
                    "text": split.page_content,
                    "id": f"{doc_id}_chunk_{i}"
                })
        
        # Write buffer if limit reached
        while len(buffer) >= CHUNK_LIMIT:
            to_write = buffer[:CHUNK_LIMIT]
            buffer = buffer[CHUNK_LIMIT:]
            
            file_name = f"{file_index:05d}.jsonl"
            file_path = os.path.join(chunked_corpus.path, file_name)
            with open(file_path, 'w') as f:
                for item in to_write:
                    f.write(json.dumps(item) + "\n")
            logging.info(f"Wrote {len(to_write)} chunks to {file_path}...")
            file_index += 1
        
        del batch_html
        del batch_markdown

        total_files += len(batch_blobs)
        logging.info(f"Processed {total_files} files, ({total_md_chunks} markdown chunks, {total_text_chunks} text chunks) ...")

    # Write remaining buffer
    if buffer:
        file_name = f"{file_index:05d}.jsonl"
        file_path = os.path.join(chunked_corpus.path, file_name)
        logging.info(f"Writing {len(buffer)} chunks to {file_path}...")
        with open(file_path, 'w') as f:
            for item in buffer:
                f.write(json.dumps(item) + "\n")
    elif file_index == 0:
         logging.warning("No HTML files found or no chunks generated.")


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pinecone-text",
    ]
)
def build_bm25_index(
    chunked_corpus: dsl.Input[dsl.Dataset],
    bm25_index: dsl.Output[dsl.Model],
):
    import os
    import json
    import logging
    from pinecone_text.sparse import BM25Encoder

    logging.getLogger().setLevel(logging.INFO)

    logging.info(f"Loading chunked data from {chunked_corpus.path}...")
    all_texts = []
    all_ids = []
    
    if os.path.isdir(chunked_corpus.path):
        for root, dirs, files in os.walk(chunked_corpus.path):
            for filename in sorted(files):
                if filename.endswith(".jsonl"):
                    file_path = os.path.join(root, filename)
                    logging.info(f"Reading {file_path}...")
                    with open(file_path, "r") as f:
                        for line in f:
                            if line.strip():
                                try:
                                    item = json.loads(line)
                                    all_texts.append(item["text"])
                                    all_ids.append(item["id"])
                                except json.JSONDecodeError:
                                    logging.warning(f"Skipping invalid line in {file_path}")
    else:
         logging.error(f"Error: {chunked_corpus.path} is not a directory.")
         # Create empty list to avoid crash in next step, or return.
         # Proceeding will likely fail at fit() if empty, but cleaner than crash here.

    logging.info("Building BM25 model...")
    
    # Initialize BM25Encoder with default configuration
    bm25 = BM25Encoder.default()
    
    # Fit the model on the corpus
    logging.info(f"Fitting BM25 model on {len(all_texts)} text chunks...")
    bm25.fit(all_texts)

    output_dir = bm25_index.path
    logging.info(f"Saving model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the BM25 parameters
    bm25.dump(os.path.join(output_dir, "bm25_params.json"))
    
    # Save the IDs mapping (useful for downstream tasks to map indices back to IDs)
    with open(os.path.join(output_dir, "doc_ids.json"), "w") as f:
        json.dump(all_ids, f)
        
    logging.info("Model build complete.")


@dsl.pipeline(
    name="bm25-index-generation-pipeline",
    description="A pipeline to build BM25 index from HTML files in GCS."
)
def pipeline(
    gcs_uri: str,
):
    chunk_task = chunk_documents(
        gcs_uri=gcs_uri
    )
    
    build_task = build_bm25_index(
        chunked_corpus=chunk_task.outputs['chunked_corpus']
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="bm25_index_pipeline.yaml"
    )
    logging.info("Pipeline compiled to bm25_index_pipeline.yaml")
