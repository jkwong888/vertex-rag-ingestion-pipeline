from kfp import dsl

@dsl.component(
    base_image='python:3.11',
    packages_to_install=[
        'google-cloud-storage',
        'markdownify',
        'langchain-text-splitters',
        'langchain',
        'beautifulsoup4'
    ]
)
def chunk_documents_op(
    gcs_html_uri: str,
    output_chunks: dsl.Output[dsl.Dataset],
    previous_chunks_uri: str = "",
    chunking_strategy_version: str = "v1"
):
    """
    Chunks HTML documents stored in GCS into smaller Markdown segments and outputs a dataset.
    If previous_chunks_uri is provided, skips chunking and simply copies those chunks to the output.

    Args:
        gcs_html_uri: The GCS URI (gs://bucket/prefix) containing HTML files.
        output_chunks: The output dataset containing all chunk data.
        previous_chunks_uri: If provided, skips chunking and uses these existing chunks.
        chunking_strategy_version: The version string for the current chunking strategy.
    """
    import json
    import logging
    import os
    from typing import List, Dict, Tuple, Any
    from urllib.parse import urlparse
    from google.cloud import storage
    import markdownify
    from bs4 import BeautifulSoup
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def parse_gcs_uri(uri: str) -> Tuple[str, str]:
        """Parses a GCS URI into bucket name and prefix."""
        parsed = urlparse(uri)
        if parsed.scheme != "gs":
            raise ValueError(f"URI {uri} must start with gs://")
        bucket_name = parsed.netloc
        prefix = parsed.path.lstrip("/")
        return bucket_name, prefix

    # --- Skip chunking if previous_chunks_uri is provided ---
    if previous_chunks_uri:
        logger.info(f"previous_chunks_uri provided: {previous_chunks_uri}. Skipping HTML chunking.")
        source_bucket_name, source_prefix = parse_gcs_uri(previous_chunks_uri)
        storage_client = storage.Client()
        source_bucket = storage_client.bucket(source_bucket_name)
        blobs_to_download = source_bucket.list_blobs(prefix=source_prefix)
        
        total_copied = 0
        with open(output_chunks.path, 'w', encoding='utf-8') as combined_output:
            for blob in blobs_to_download:
                if not blob.name.endswith('.jsonl'):
                    continue
                
                logger.info(f"Copying existing chunks from {blob.name}")
                blob_content = blob.download_as_text(encoding="utf-8")
                
                for line in blob_content.splitlines():
                    sanitized_line = line.strip()
                    if sanitized_line:
                        try:
                            chunk_data = json.loads(sanitized_line)
                            # Update pointers to point to the new dataset artifact
                            chunk_data['chunk_gcs_uri'] = output_chunks.uri
                            chunk_data['chunk_line_offset'] = total_copied
                            combined_output.write(json.dumps(chunk_data) + '\n')
                            total_copied += 1
                        except Exception as e:
                            logger.error(f"Failed to parse chunk JSON: {e}")
        
        logger.info(f"Copied {total_copied} previous chunks to {output_chunks.uri}. Exiting component.")
        return
    # ---------------------------------------------------------

    def clean_html(html_content: str) -> str:
        """
        Removes unwanted HTML elements and extracts the main content.
        Focused on Bulbapedia's structure. Preserves factual tables.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Focus on the main content area if it exists
        main_content = soup.find(id='mw-content-text')
        if main_content:
            # Use the content inside the parser output div to avoid some outer wrappers
            parser_output = main_content.find(class_='mw-parser-output')
            if parser_output:
                soup = parser_output
            else:
                soup = main_content

        # Remove elements that are definitely noise
        tags_to_remove = ["script", "style"]
        classes_to_remove = ["mw-editsection", "toc", "noprint", "navbox", "p-lang", "p-tb", "footer", "catlinks"]
        ids_to_remove = ["mw-navigation", "siteSub", "contentSub", "jump-to-nav"]

        for tag in soup(tags_to_remove):
            tag.decompose()
            
        for cls in classes_to_remove:
            for element in soup.find_all(class_=cls):
                element.decompose()

        for element_id in ids_to_remove:
            element = soup.find(id=element_id)
            if element:
                element.decompose()

        # Specific table filtering: Remove purely navigation tables, keep factual ones
        for table in soup.find_all('table'):
            if table.get('class') and any(c in table.get('class') for c in ['navbox', 'vertical-navbox']):
                table.decompose()

        # Optional: remove specific sections that might be less relevant
        for header in soup.find_all(['h2', 'h3']):
            header_text = header.get_text().lower()
            if any(x in header_text for x in ['trivia', 'external links', 'references']):
                for sibling in header.find_next_siblings():
                    if sibling.name in ['h2', 'h3']:
                        break
                    sibling.decompose()
                header.decompose()

        return str(soup)

    def chunk_markdown(
        markdown_text: str, 
        source_file: str, 
        original_url: str, 
        scrape_time: str
    ) -> List[Dict[str, Any]]:
        """Splits markdown text into chunks based on headers and character count."""
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
        
        chunk_results = []
        for i, chunk in enumerate(final_chunks):
            headers = [
                chunk.metadata[key] 
                for key in ["Header 1", "Header 2", "Header 3"] 
                if key in chunk.metadata
            ]
            
            chunk_id = f"{source_file}_chunk_{i}"
            
            chunk_data = {
                "id": chunk_id,
                "original_url": original_url,
                "scrape_time": scrape_time,
                "headers": headers,
                "chunk": chunk.page_content,
                "source_file": source_file,
                "chunking_strategy_version": chunking_strategy_version
            }
            chunk_results.append(chunk_data)
        return chunk_results

    def process_blob(blob: storage.Blob) -> List[Dict[str, Any]]:
        """Downloads, cleans, and chunks a single GCS blob."""
        if not (blob.name.endswith(".html") or blob.name.endswith(".htm")):
            return []
        
        try:
            content = blob.download_as_text(encoding="utf-8")
            metadata = blob.metadata or {}
            original_url = metadata.get("original_url", "")
            scrape_time = metadata.get("scrape_time", "")
            
            cleaned_html = clean_html(content)
            markdown_content = markdownify.markdownify(
                cleaned_html, 
                heading_style="atx",
                strip=['a']  # Remove link tags but keep the anchor text to reduce noise
            )
            
            chunks = chunk_markdown(markdown_content, blob.name, original_url, scrape_time)
            logger.info(f"Processed {blob.name} with {len(chunks)} chunks.")
            return chunks
        except Exception as error:
            logger.error(f"Failed to process {blob.name}: {error}")
            return []

    html_bucket_name, html_prefix = parse_gcs_uri(gcs_html_uri)

    storage_client = storage.Client()
    html_bucket = storage_client.bucket(html_bucket_name)

    blobs = html_bucket.list_blobs(prefix=html_prefix)
    
    total_chunks_count = 0
    
    with open(output_chunks.path, 'w', encoding='utf-8') as combined_output:
        for blob in blobs:
            document_chunks = process_blob(blob)
            for chunk in document_chunks:
                # Add pointer metadata
                chunk['chunk_gcs_uri'] = output_chunks.uri
                chunk['chunk_line_offset'] = total_chunks_count
                
                # Write to the dataset file
                json_record = json.dumps(chunk)
                combined_output.write(json_record + '\n')
                total_chunks_count += 1
                
    logger.info(f"Chunking complete. Wrote {total_chunks_count} chunks to {output_chunks.uri}.")
