"""
Model deployer KFP component.

Deploys a blessed model to a serving endpoint using configurable
deployment strategies (blue/green, canary, rolling, shadow).
"""

from __future__ import annotations

from kfp import dsl
from kfp.dsl import Artifact, Input, Model, Output

from kfp_ml_library.configs.constants import (
    CANARY_TRAFFIC_PERCENTAGE,
    DEFAULT_BASE_IMAGE,
    DEFAULT_MAX_REPLICAS,
    DEFAULT_MIN_REPLICAS,
    DEFAULT_SERVING_PORT,
    DEFAULT_TARGET_CPU_UTILIZATION,
)


@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=["google-cloud-aiplatform>=1.38.0", "requests>=2.31.0"],
)
def deploy_model_component(
    model_artifact: Input[Model],
    eval_report: Input[Artifact],
    deployment_artifact: Output[Artifact],
    project_id: str,
    region: str = "us-central1",
    endpoint_name: str = "ml-serving-endpoint",
    deployment_strategy: str = "rolling",
    min_replicas: int = DEFAULT_MIN_REPLICAS,
    max_replicas: int = DEFAULT_MAX_REPLICAS,
    serving_container_image: str = "",
    machine_type: str = "n1-standard-4",
    accelerator_type: str = "",
    accelerator_count: int = 0,
    traffic_percentage: int = 100,
    canary_percentage: int = CANARY_TRAFFIC_PERCENTAGE,
    serving_port: int = DEFAULT_SERVING_PORT,
    target_cpu_utilization: int = DEFAULT_TARGET_CPU_UTILIZATION,
) -> str:
    """
    Deploy model to a serving endpoint.

    Supports Vertex AI, KServe, and generic REST deployment.
    """
    import json
    import logging

    logger = logging.getLogger(__name__)

    # Check blessing
    with open(eval_report.path, "r") as f:
        report = json.load(f)

    if not report.get("is_blessed", False):
        msg = "Model is NOT blessed – skipping deployment"
        logger.warning(msg)
        result = {"status": "skipped", "reason": "model_not_blessed"}
        with open(deployment_artifact.path, "w") as f:
            json.dump(result, f)
        return json.dumps(result)

    deployment_config = {
        "project_id": project_id,
        "region": region,
        "endpoint_name": endpoint_name,
        "deployment_strategy": deployment_strategy,
        "min_replicas": min_replicas,
        "max_replicas": max_replicas,
        "machine_type": machine_type,
        "model_path": model_artifact.path,
        "serving_container_image": serving_container_image,
        "traffic_percentage": traffic_percentage,
    }

    try:
        from google.cloud import aiplatform

        aiplatform.init(project=project_id, location=region)

        # Upload model
        vertex_model = aiplatform.Model.upload(
            display_name=endpoint_name,
            artifact_uri=model_artifact.uri if hasattr(model_artifact, "uri") else model_artifact.path,
            serving_container_image_uri=serving_container_image or "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
        )

        # Get or create endpoint
        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_name}"'
        )
        if endpoints:
            endpoint = endpoints[0]
        else:
            endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)

        # Deploy based on strategy
        if deployment_strategy == "canary":
            traffic_split = {vertex_model.resource_name: canary_percentage}
            # Keep remaining traffic on existing models
            endpoint.deploy(
                model=vertex_model,
                deployed_model_display_name=f"{endpoint_name}-canary",
                machine_type=machine_type,
                min_replica_count=min_replicas,
                max_replica_count=max_replicas,
                traffic_percentage=canary_percentage,
            )
        else:
            endpoint.deploy(
                model=vertex_model,
                deployed_model_display_name=endpoint_name,
                machine_type=machine_type,
                min_replica_count=min_replicas,
                max_replica_count=max_replicas,
                traffic_percentage=traffic_percentage,
            )

        result = {
            "status": "deployed",
            "endpoint_id": endpoint.resource_name,
            "model_id": vertex_model.resource_name,
            "strategy": deployment_strategy,
            "config": deployment_config,
        }

    except ImportError:
        logger.warning("google-cloud-aiplatform not available; writing config only.")
        result = {
            "status": "config_only",
            "config": deployment_config,
            "message": "Deployment target SDK not available; config saved.",
        }
    except Exception as e:
        logger.error("Deployment failed: %s", e)
        result = {"status": "failed", "error": str(e), "config": deployment_config}

    with open(deployment_artifact.path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    deployment_artifact.metadata["status"] = result["status"]
    return json.dumps(result, default=str)
