"""
Endpoint manager KFP component.

Manages serving endpoints: list, update traffic, scale, and clean up.
"""

from __future__ import annotations

from kfp import dsl
from kfp.dsl import Artifact, Output

from kfp_ml_library.configs.constants import DEFAULT_BASE_IMAGE


@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=["google-cloud-aiplatform>=1.38.0"],
)
def manage_endpoint_component(
    endpoint_report: Output[Artifact],
    project_id: str,
    region: str = "us-central1",
    endpoint_name: str = "",
    action: str = "list",
    traffic_split_json: str = "{}",
) -> str:
    """
    Manage serving endpoints.

    Actions: ``list``, ``update_traffic``, ``undeploy_model``, ``delete``.
    """
    import json
    import logging

    logger = logging.getLogger(__name__)

    try:
        from google.cloud import aiplatform

        aiplatform.init(project=project_id, location=region)
        result: dict = {}

        if action == "list":
            endpoints = aiplatform.Endpoint.list()
            result = {
                "endpoints": [
                    {
                        "name": ep.display_name,
                        "id": ep.resource_name,
                        "deployed_models": len(ep.gca_resource.deployed_models),
                    }
                    for ep in endpoints
                ]
            }

        elif action == "update_traffic":
            endpoints = aiplatform.Endpoint.list(
                filter=f'display_name="{endpoint_name}"'
            )
            if endpoints:
                traffic_split = json.loads(traffic_split_json)
                endpoints[0].update(traffic_split=traffic_split)
                result = {"status": "traffic_updated", "endpoint": endpoint_name}
            else:
                result = {"status": "not_found", "endpoint": endpoint_name}

        elif action == "delete":
            endpoints = aiplatform.Endpoint.list(
                filter=f'display_name="{endpoint_name}"'
            )
            if endpoints:
                endpoints[0].undeploy_all()
                endpoints[0].delete()
                result = {"status": "deleted", "endpoint": endpoint_name}
            else:
                result = {"status": "not_found"}

        else:
            result = {"error": f"Unknown action: {action}"}

    except Exception as e:
        logger.error("Endpoint management failed: %s", e)
        result = {"status": "error", "error": str(e)}

    with open(endpoint_report.path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return json.dumps(result, default=str)


@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=["google-cloud-aiplatform>=1.38.0"],
)
def rollback_deployment_component(
    deployment_artifact: Artifact,
    rollback_report: Output[Artifact],
    project_id: str,
    region: str = "us-central1",
    endpoint_name: str = "",
    previous_model_id: str = "",
) -> str:
    """
    Roll back a deployment to a previous model version.
    """
    import json
    import logging

    logger = logging.getLogger(__name__)

    try:
        from google.cloud import aiplatform

        aiplatform.init(project=project_id, location=region)

        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_name}"'
        )
        if not endpoints:
            result = {"status": "error", "message": f"Endpoint {endpoint_name} not found"}
        else:
            endpoint = endpoints[0]
            # Route all traffic to previous model
            if previous_model_id:
                endpoint.update(traffic_split={previous_model_id: 100})
                result = {
                    "status": "rolled_back",
                    "endpoint": endpoint_name,
                    "active_model": previous_model_id,
                }
            else:
                result = {"status": "error", "message": "No previous_model_id provided"}

    except Exception as e:
        logger.error("Rollback failed: %s", e)
        result = {"status": "error", "error": str(e)}

    with open(rollback_report.path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return json.dumps(result, default=str)
