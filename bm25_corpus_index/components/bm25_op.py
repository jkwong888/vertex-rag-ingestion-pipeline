from kfp import dsl

@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pinecone-text"
    ]
)
def build_bm25_index(
    chunks_dataset: dsl.Input[dsl.Dataset],
    bm25_index: dsl.Output[dsl.Model],
):
    import os
    import json
    import logging
    from pinecone_text.sparse import BM25Encoder

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    all_texts = []
    all_ids = []

    logger.info(f"Reading chunks from {chunks_dataset.path}...")
    try:
        with open(chunks_dataset.path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        # Support both "text" and "chunk" keys
                        text = item.get("chunk") or item.get("text")
                        if text:
                            all_texts.append(text)
                            all_ids.append(item["id"])
                    except json.JSONDecodeError:
                        logger.warning("Skipping invalid line in dataset")
    except Exception as e:
        logger.error(f"Error reading dataset: {e}")
        return

    if not all_texts:
        logger.error("No text chunks found to build BM25 index.")
        return

    logger.info(f"Fitting BM25 model on {len(all_texts)} text chunks...")
    bm25 = BM25Encoder.default()
    bm25.fit(all_texts)

    output_dir = bm25_index.path
    os.makedirs(output_dir, exist_ok=True)
    bm25.dump(os.path.join(output_dir, "bm25_params.json"))
    
    with open(os.path.join(output_dir, "doc_ids.json"), "w") as f:
        json.dump(all_ids, f)
        
    logger.info("Model build complete.")
