from kfp import dsl
from kfp.dsl import Output, Artifact, Input
from typing import List

@dsl.component(
    base_image='python:3.11',
    packages_to_install=[
        'google-cloud-storage',
        'markdownify',
        'langchain-text-splitters',
        'langchain'
    ]
)
def chunk_documents(
    uris: List[str],
    output_dataset: Output[Artifact]
):
    import json
    import logging
    import os
    from urllib.parse import urlparse
    from google.cloud import storage
    import markdownify
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def parse_gcs_uri(uri: str):
        parsed = urlparse(uri)
        if parsed.scheme != "gs":
            raise ValueError("URI must start with gs://")
        bucket_name = parsed.netloc
        prefix = parsed.path.lstrip("/")
        return bucket_name, prefix

    def chunk_markdown(markdown_text: str, source_file: str, original_url: str = None):
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        header_splits = header_splitter.split_text(markdown_text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        final_chunks = text_splitter.split_documents(header_splits)
        
        chunk_dicts = []
        for i, chunk in enumerate(final_chunks):
            headers = []
            for key in ["Header 1", "Header 2", "Header 3"]:
                if key in chunk.metadata:
                    headers.append(chunk.metadata[key])
            
            chunk_id = f"{source_file}_chunk_{i}"
            
            chunk_data = {
                "id": chunk_id,
                "original_url": original_url,
                "headers": headers,
                "chunk": chunk.page_content,
                "source_file": source_file
            }
            chunk_dicts.append(chunk_data)
        return chunk_dicts

    storage_client = storage.Client()
    total_chunks = 0
    
    with open(output_dataset.path, 'w', encoding='utf-8') as f:
        for uri in uris:
            try:
                bucket_name, blob_name = parse_gcs_uri(uri)
                if not (blob_name.endswith(".html") or blob_name.endswith(".htm")):
                    continue
                
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.get_blob(blob_name)
                if not blob:
                    continue

                content = blob.download_as_text(encoding="utf-8")
                original_url = blob.metadata.get("original_url") if blob.metadata else None
                
                markdown_content = markdownify.markdownify(content, heading_style="atx")
                chunks = chunk_markdown(markdown_content, blob.name, original_url)
                
                for chunk in chunks:
                    f.write(json.dumps(chunk) + '\n')
                    total_chunks += 1
                
                logger.info(f"Processed {blob.name} with {len(chunks)} chunks.")
            except Exception as e:
                logger.error(f"Failed to process {uri}: {e}")
    
    output_dataset.metadata["total_chunks"] = total_chunks

@dsl.component(
    base_image='python:3.11'
)
def merge_embeddings(
    chunks_dataset: Input[Artifact],
    sparse_embeddings_dataset: Input[Artifact],
    dense_embeddings_dataset: Input[Artifact],
    output_dataset: Output[Artifact]
):
    import json
    import logging
    import os

    logging.basicConfig(level=logging.INFO)

    sparse_map = {}
    with open(sparse_embeddings_dataset.path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # change property "indices" to "dimensions" in the sparse embeddingto match the schema
            item['sparse_embedding']['dimensions'] = item['sparse_embedding']['indices']
            del item['sparse_embedding']['indices']
            sparse_map[item['id']] = item['sparse_embedding']
            
    dense_map = {}
    with open(dense_embeddings_dataset.path, 'r') as f:
        for line in f:
            item = json.loads(line)
            dense_map[item['id']] = item['embedding']
    
    # handle local filepath - we want to treat the path as a directory and place the file inside
    if not output_dataset.path.startswith("gs://"):
        os.makedirs(output_dataset.path, exist_ok=True)

    # TODO: write out files 10000 chunk at a time?
    with open(chunks_dataset.path, 'r') as infile, open(f"{output_dataset.path}/batch.json", 'w') as outfile:
        for line in infile:
            chunk = json.loads(line)
            chunk_id = chunk['id']
            
            merged = {
                "id": chunk_id,
                "restricts": [],
                "numeric_restricts": [],
                "embedding_metadata": {}
            }

            if chunk_id in sparse_map:
                merged['sparse_embedding'] = sparse_map[chunk_id]
            if chunk_id in dense_map:
                merged['embedding'] = dense_map[chunk_id]
            # add all columns that aren't "id", "embedding" and "sparse_embedding" to a new key "embdding_metadata"
            for key in chunk.keys():
                if key not in ["id", "embedding", "sparse_embedding", "restricts", "numeric_restricts"]:
                    merged[f"embedding_metadata"][key] = chunk[key]

            # empty "restricts" and "numeric_restricts" keys
            outfile.write(json.dumps(merged) + '\n')
