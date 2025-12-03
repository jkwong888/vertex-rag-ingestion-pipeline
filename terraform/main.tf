data "google_project" "project" {
  project_id = var.project_id
}

resource "google_vertex_ai_index" "vector_search_index" {
  project      = data.google_project.project.project_id
  display_name = var.index_display_name
  description  = var.index_description
  region       = var.region

  metadata {
    config {
      dimensions               = var.embedding_dimensions
      approximate_neighbors_count = var.approximate_neighbors_count
      distance_measure_type    = var.distance_measure_type
      shard_size               = "SHARD_SIZE_SO_DYNAMIC"
      feature_norm_type        = var.feature_norm_type
    }
  }

  index_update_method = var.index_update_method

  labels = var.index_labels
}

resource "google_vertex_ai_index_endpoint" "vector_search_endpoint" {
  project                 = data.google_project.project.project_id
  display_name            = var.endpoint_display_name
  description             = var.endpoint_description
  region                  = var.region
  public_endpoint_enabled = var.public_endpoint_enabled
}

# resource "google_vertex_ai_index_endpoint_deployed_index" "vector_search_deployed_index" {
#   index_endpoint    = google_vertex_ai_index_endpoint.vector_search_endpoint.id
#   index             = google_vertex_ai_index.vector_search_index.id
#   deployed_index_id = var.deployed_index_id
#   display_name      = var.deployed_index_id
#   region            = var.region


#   dedicated_resources {
#     machine_spec {
#       machine_type = var.deployment_machine_type
#     }
#     min_replica_count = var.min_replica_count
#     max_replica_count = var.max_replica_count

#   }
# }