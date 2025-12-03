# Vertex AI Vector Search Terraform Module

This Terraform configuration provisions resources for Google Cloud Vertex AI Vector Search (formerly Matching Engine). It automates the creation of a Vector Search Index and an Index Endpoint.

## Resources Created

*   **`google_vertex_ai_index`**: A Vector Search index configured for storing and searching embeddings.
*   **`google_vertex_ai_index_endpoint`**: An endpoint to which the index can be deployed for online serving.

> **Note:** The resource `google_vertex_ai_index_endpoint_deployed_index` (which deploys the index to the endpoint) is currently commented out in `main.tf`. Storage optimized indexes cannot be provisioned using Terraform.  We supplied `deploy_index.json` as the payload to call the REST API manually to get this to work.

  You can run the following command to deploy the storage optimized index to the endpoint (takes 20-30 minutes):

  ```bash
  curl -X POST \
      -H "Authorization: Bearer $(gcloud auth print-access-token)" \
      -H "Content-Type: application/json; charset=utf-8" \
      -d @deployed_index.json \
      "https://us-central1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/indexEndpoints/${INDEX_ENDPOINT_ID}$:deployIndex"
  ```

## Prerequisites

*   [Terraform](https://www.terraform.io/downloads.html) >= 1.0
*   Google Cloud SDK installed and authenticated (`gcloud auth application-default login`)
*   A Google Cloud Platform project with the **Vertex AI API** enabled.

## Usage

1.  **Initialize Terraform:**
    ```bash
    terraform init
    ```

2.  **Review the Plan:**
    Create a `terraform.tfvars` file to supply values for the required variables (see below), or pass them via the command line.
    ```bash
    terraform plan -var="project_id=YOUR_PROJECT_ID" \
                   -var="index_display_name=my-vector-index" \
                   -var="endpoint_display_name=my-index-endpoint" \
                   -var="embedding_dimensions=768" \
                   -var="deployed_index_id=my_deployed_index"
    ```

3.  **Apply the Configuration:**
    ```bash
    terraform apply
    ```

## Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|:--------:|
| `project_id` | The GCP project ID. | `string` | n/a | **Yes** |
| `region` | The GCP region where resources will be created. | `string` | `us-central1` | No |
| `index_display_name` | Display name for the Vector Search Index. | `string` | n/a | **Yes** |
| `embedding_dimensions` | Number of dimensions for input vectors. | `number` | n/a | **Yes** |
| `endpoint_display_name` | Display name for the Index Endpoint. | `string` | n/a | **Yes** |
| `deployed_index_id` | User-specified ID for the Deployed Index. | `string` | n/a | **Yes** |
| `index_update_method` | Method for updating the index (`BATCH_UPDATE` or `STREAM_UPDATE`). | `string` | `BATCH_UPDATE` | No |
| `distance_measure_type` | Distance measure (`DOT_PRODUCT_DISTANCE`, `COSINE_DISTANCE`, `L2_DISTANCE`). | `string` | `DOT_PRODUCT_DISTANCE` | No |
| `public_endpoint_enabled` | Whether the endpoint is exposed publicly. | `bool` | `true` | No |

*See `variables.tf` for the full list of variables and their descriptions.*

## Files

*   `main.tf`: Defines the main resources (Index, Endpoint).
*   `variables.tf`: Variable definitions.
*   `provider.tf`: Google Cloud provider configuration.
*   `deployed_index.json`: JSON file containing information about deployed indexes (likely used by external scripts or for reference).
