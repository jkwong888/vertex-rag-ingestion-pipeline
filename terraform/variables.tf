variable "project_id" {
  description = "The GCP project ID."
  type        = string
}

variable "region" {
  description = "The GCP region where the index will be created."
  type        = string
  default     = "us-central1"
}

variable "index_display_name" {
  description = "The display name of the Vertex AI Vector Search index."
  type        = string
}

variable "index_description" {
  description = "Description of the Vertex AI Vector Search index."
  type        = string
  default     = "A Vertex AI Vector Search index for similarity matching."
}

variable "embedding_dimensions" {
  description = "The number of dimensions of the input vectors."
  type        = number
}

variable "approximate_neighbors_count" {
  description = "The default number of neighbors to find via approximate search."
  type        = number
  default     = 150
}

variable "distance_measure_type" {
  description = "The distance measure type. Options: DOT_PRODUCT_DISTANCE, COSINE_DISTANCE, L2_DISTANCE."
  type        = string
  default     = "DOT_PRODUCT_DISTANCE"
}

variable "feature_norm_type" {
  description = "The feature norm type. Options: UNIT_L2_NORM, NONE."
  type        = string
  default     = "UNIT_L2_NORM"
}

variable "leaf_node_embedding_count" {
  description = "Number of embeddings on each leaf node."
  type        = number
  default     = 1000
}

variable "leaf_node_uncompressed_size" {
  description = "The size of the uncompressed leaf node in bytes."
  type        = number
  default     = 1024
}

variable "index_update_method" {
  description = "The method for updating the index. Options: BATCH_UPDATE, STREAM_UPDATE."
  type        = string
  default     = "BATCH_UPDATE"
}

variable "index_labels" {
  description = "Labels for the Vertex AI Vector Search index."
  type        = map(string)
  default     = {}
}

variable "endpoint_display_name" {
  description = "The display name of the Vertex AI Vector Search Endpoint."
  type        = string
}

variable "endpoint_description" {
  description = "Description of the Vertex AI Vector Search Endpoint."
  type        = string
  default     = "Endpoint for Vertex AI Vector Search index."
}

variable "public_endpoint_enabled" {
  description = "If true, the endpoint will be exposed publicly."
  type        = bool
  default     = true
}

variable "deployed_index_id" {
  description = "The user specified ID of the DeployedIndex."
  type        = string
}

variable "deployment_machine_type" {
  description = "The type of machine used for the deployment."
  type        = string
  default     = "e2-standard-2"
}

variable "min_replica_count" {
  description = "The minimum number of replicas for the deployed index."
  type        = number
  default     = 1
}

variable "max_replica_count" {
  description = "The maximum number of replicas for the deployed index."
  type        = number
  default     = 1
}

variable "chunk_version" {
  description = "The version of the chunking strategy used."
  type        = string
  default     = "v1"
}

variable "embedding_model" {
  description = "The embedding model used for the index."
  type        = string
  default     = "gemini-embedding-001"
}