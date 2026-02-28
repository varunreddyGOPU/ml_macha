"""
Containerized component builder.

Generates a KFP ContainerOp-style component from arbitrary
user code + requirements, encapsulating it in a container.
"""

from __future__ import annotations

from kfp import dsl
from kfp.dsl import Artifact, Output

from kfp_ml_library.configs.constants import DEFAULT_BASE_IMAGE, DOCKERFILE_TEMPLATE


@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=["pyyaml>=6.0.1"],
)
def generate_component_yaml(
    component_spec: Output[Artifact],
    component_name: str,
    description: str = "",
    base_image: str = DEFAULT_BASE_IMAGE,
    command: str = "python",
    args_json: str = "[]",
    packages_json: str = "[]",
    input_specs_json: str = "[]",
    output_specs_json: str = "[]",
) -> str:
    """
    Generate a KFP component YAML specification.

    Creates a reusable component spec that can be loaded with
    ``kfp.components.load_component_from_file()``.
    """
    import json
    import yaml

    args = json.loads(args_json)
    packages = json.loads(packages_json)
    inputs = json.loads(input_specs_json)
    outputs = json.loads(output_specs_json)

    spec = {
        "name": component_name,
        "description": description,
        "inputs": inputs,
        "outputs": outputs,
        "implementation": {
            "container": {
                "image": base_image,
                "command": [command] + args,
            }
        },
    }

    if packages:
        pip_install = " && ".join([f"pip install {p}" for p in packages])
        spec["implementation"]["container"]["command"] = [
            "sh",
            "-c",
            f"{pip_install} && {command} " + " ".join(args),
        ]

    with open(component_spec.path, "w") as f:
        yaml.dump(spec, f, default_flow_style=False)

    return json.dumps(spec, default=str)


@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
)
def generate_dockerfile_component(
    dockerfile_artifact: Output[Artifact],
    base_image: str = DEFAULT_BASE_IMAGE,
    requirements: str = "",
    entrypoint: str = "main.py",
    extra_commands: str = "",
    use_gpu: bool = False,
) -> str:
    """
    Generate a Dockerfile for an ML component.
    """
    import json

    if use_gpu:
        template = '''FROM {base_image}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3 python3-pip && \\
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

{extra_commands}

ENTRYPOINT ["python3", "{entrypoint}"]
'''
    else:
        template = '''FROM {base_image}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

{extra_commands}

ENTRYPOINT ["python", "{entrypoint}"]
'''

    dockerfile_content = template.format(
        base_image=base_image,
        entrypoint=entrypoint,
        extra_commands=extra_commands,
    )

    with open(dockerfile_artifact.path, "w") as f:
        f.write(dockerfile_content)

    result = {
        "base_image": base_image,
        "entrypoint": entrypoint,
        "use_gpu": use_gpu,
        "content_length": len(dockerfile_content),
    }

    return json.dumps(result)
