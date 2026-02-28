"""
Docker builder KFP component.

Builds Docker images dynamically for ML training and serving,
optionally pushing them to a container registry.
"""

from __future__ import annotations

from kfp import dsl
from kfp.dsl import Artifact, Output

from kfp_ml_library.configs.constants import DEFAULT_BASE_IMAGE


@dsl.component(
    base_image="gcr.io/kaniko-project/executor:latest",
)
def build_docker_image_kaniko(
    build_report: Output[Artifact],
    dockerfile_content: str,
    context_gcs_path: str,
    image_name: str,
    image_tag: str = "latest",
    registry: str = "gcr.io",
    project_id: str = "",
    build_args: str = "{}",
) -> str:
    """
    Build a Docker image using Kaniko (no Docker daemon required).

    This component is designed to run inside a KFP pipeline where
    the Kaniko executor is the base image.
    """
    import json
    import subprocess
    import os

    full_image = f"{registry}/{project_id}/{image_name}:{image_tag}"
    args = json.loads(build_args) if build_args else {}

    # Write Dockerfile
    dockerfile_path = "/workspace/Dockerfile"
    os.makedirs("/workspace", exist_ok=True)
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)

    cmd = [
        "/kaniko/executor",
        f"--context={context_gcs_path}",
        f"--dockerfile={dockerfile_path}",
        f"--destination={full_image}",
        "--cache=true",
    ]
    for k, v in args.items():
        cmd.append(f"--build-arg={k}={v}")

    result = {
        "image": full_image,
        "command": " ".join(cmd),
        "status": "pending",
    }

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        result["status"] = "success" if proc.returncode == 0 else "failed"
        result["stdout"] = proc.stdout[-2000:] if proc.stdout else ""
        result["stderr"] = proc.stderr[-2000:] if proc.stderr else ""
        result["return_code"] = proc.returncode
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    with open(build_report.path, "w") as f:
        json.dump(result, f, indent=2)

    return json.dumps(result)


@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=["google-cloud-build>=3.22.0"],
)
def build_docker_image_cloud_build(
    build_report: Output[Artifact],
    source_gcs_uri: str,
    image_name: str,
    image_tag: str = "latest",
    project_id: str = "",
    registry: str = "gcr.io",
    dockerfile_path: str = "Dockerfile",
    timeout_seconds: int = 1200,
    machine_type: str = "N1_HIGHCPU_8",
) -> str:
    """
    Build a Docker image using Google Cloud Build.
    """
    import json
    import logging

    logger = logging.getLogger(__name__)
    full_image = f"{registry}/{project_id}/{image_name}:{image_tag}"

    try:
        from google.cloud.devtools import cloudbuild_v1

        client = cloudbuild_v1.CloudBuildClient()

        build_config = cloudbuild_v1.Build(
            source=cloudbuild_v1.Source(
                storage_source=cloudbuild_v1.StorageSource(
                    bucket=source_gcs_uri.replace("gs://", "").split("/")[0],
                    object_=source_gcs_uri.replace("gs://", "").split("/", 1)[1]
                    if "/" in source_gcs_uri.replace("gs://", "")
                    else "",
                )
            ),
            steps=[
                cloudbuild_v1.BuildStep(
                    name="gcr.io/cloud-builders/docker",
                    args=["build", "-t", full_image, "-f", dockerfile_path, "."],
                )
            ],
            images=[full_image],
            options=cloudbuild_v1.BuildOptions(
                machine_type=machine_type,
            ),
            timeout=f"{timeout_seconds}s",
        )

        operation = client.create_build(project_id=project_id, build=build_config)
        build_result = operation.result()

        result = {
            "image": full_image,
            "status": build_result.status.name,
            "build_id": build_result.id,
        }
    except Exception as e:
        logger.error("Cloud Build failed: %s", e)
        result = {"image": full_image, "status": "error", "error": str(e)}

    with open(build_report.path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return json.dumps(result, default=str)
