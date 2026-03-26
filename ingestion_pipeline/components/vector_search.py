from kfp import dsl

@dsl.component(
    base_image='python:3.11',
    packages_to_install=['google-cloud-aiplatform']
)
def create_vector_search_index(
    project: str,
    location: str,
    index_name: str,
    dimensions: int = 768,
    approximate_neighbors_count: int = 150,
    distance_measure_type: str = "DOT_PRODUCT_DISTANCE",
    feature_norm_type: str = "UNIT_L2_NORM"
) -> str:
    """
    Creates a new Vertex AI Vector Search index if it doesn't already exist.

    Args:
        project: Google Cloud Project ID.
        location: Google Cloud region.
        index_name: The display name for the new index.
        dimensions: Number of dimensions for the embedding vectors.
        approximate_neighbors_count: Number of neighbors to find in approximate search.
        distance_measure_type: The distance measure for similarity matching.
        feature_norm_type: The normalization type for vectors.

    Returns:
        The resource name of the created or existing index.
    """
    from google.cloud import aiplatform
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    aiplatform.init(project=project, location=location)

    # Check if index already exists
    matching_indexes = aiplatform.MatchingEngineIndex.list(filter=f'display_name="{index_name}"')
    if matching_indexes:
        logger.info(f"Index '{index_name}' already exists: {matching_indexes[0].resource_name}")
        return matching_indexes[0].resource_name

    logger.info(f"Creating new Vector Search index (Tree-AH / ANN): {index_name}")
    
    # Create the index using Approximate Nearest Neighbor (Tree-AH)
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=index_name,
        dimensions=dimensions,
        distance_measure_type=distance_measure_type,
        feature_norm_type=feature_norm_type,
        approximate_neighbors_count=150,
        leaf_node_emb_count=1000,
        leaf_nodes_to_search_percent=10,
        description="Versioned ANN index created via ingestion pipeline."
    )

    logger.info(f"Index creation operation started: {index.resource_name}")
    index.wait()
    logger.info("Index created successfully.")
    
    return index.resource_name
