"""
Deployment pipeline definition.

Composes Docker image building → model deployment → endpoint validation
into a single KFP pipeline.
"""

from __future__ import annotations

from kfp import dsl

from kfp_ml_library.components.container.containerized_component import generate_dockerfile_component
from kfp_ml_library.components.container.docker_builder import build_docker_image_cloud_build
from kfp_ml_library.components.deployment.model_deployer import deploy_model_component
from kfp_ml_library.components.deployment.endpoint_manager import manage_endpoint_component


@dsl.pipeline(
    name="ml-deployment-pipeline",
    description="Model deployment pipeline: build container → deploy → validate endpoint",
)
def create_deployment_pipeline(
    project_id: str,
    region: str = "us-central1",
    model_artifact_uri: str = "",
    eval_report_uri: str = "",
    endpoint_name: str = "ml-serving-endpoint",
    deployment_strategy: str = "rolling",
    serving_container_image: str = "",
    machine_type: str = "n1-standard-4",
    min_replicas: int = 1,
    max_replicas: int = 5,
    traffic_percentage: int = 100,
    build_custom_image: bool = False,
    base_image: str = "python:3.10-slim",
    source_gcs_uri: str = "",
    image_name: str = "ml-serving",
    image_tag: str = "latest",
):
    """
    Composable deployment pipeline.

    Steps:
        1. (Optional) Build custom Docker serving image
        2. Deploy model to endpoint
        3. Validate endpoint health
    """

    # Optional: Build custom serving image
    with dsl.If(build_custom_image == True, name="Build Custom Image"):
        dockerfile_task = generate_dockerfile_component(
            base_image=base_image,
            entrypoint="serve.py",
        )
        build_task = build_docker_image_cloud_build(
            source_gcs_uri=source_gcs_uri,
            image_name=image_name,
            image_tag=image_tag,
            project_id=project_id,
            registry=f"{region}-docker.pkg.dev",
        )
        build_task.after(dockerfile_task)

    # Deploy model (this will reference model_artifact and eval_report via URIs)
    # In practice, you'd pass the actual artifacts from a training pipeline.
    # This pipeline accepts URIs for standalone deployment scenarios.
    endpoint_list = manage_endpoint_component(
        project_id=project_id,
        region=region,
        endpoint_name=endpoint_name,
        action="list",
    )
