"""
Compute constraints configuration for KFP pipeline components.

Defines resource requests/limits, node selectors, tolerations,
and accelerator configs that can be attached to any pipeline step.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from kfp_ml_library.configs.constants import (
    DEFAULT_CPU_LIMIT,
    DEFAULT_GPU_LIMIT,
    DEFAULT_GPU_TYPE,
    DEFAULT_MEMORY_LIMIT,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_RETRY_COUNT,
)


@dataclass
class AcceleratorConfig:
    """GPU / TPU accelerator configuration."""

    accelerator_type: str = DEFAULT_GPU_TYPE
    accelerator_count: int = 1
    require_gpu: bool = False


@dataclass
class ResourceRequests:
    """Kubernetes resource requests."""

    cpu: str = "1"
    memory: str = "4Gi"
    ephemeral_storage: str = "10Gi"


@dataclass
class ResourceLimits:
    """Kubernetes resource limits."""

    cpu: str = DEFAULT_CPU_LIMIT
    memory: str = DEFAULT_MEMORY_LIMIT
    gpu: str = DEFAULT_GPU_LIMIT
    ephemeral_storage: str = "50Gi"


@dataclass
class NodeConfig:
    """Node selector and toleration configuration."""

    node_selector: Dict[str, str] = field(default_factory=dict)
    tolerations: List[Dict[str, str]] = field(default_factory=list)
    affinity: Optional[Dict] = None


@dataclass
class ComputeConstraints:
    """
    Complete compute constraints for a pipeline component.

    Encapsulates resource requests/limits, accelerator config,
    node placement, timeouts, and retry behaviour.

    Example::

        constraints = ComputeConstraints(
            requests=ResourceRequests(cpu="2", memory="8Gi"),
            limits=ResourceLimits(cpu="4", memory="16Gi"),
            accelerator=AcceleratorConfig(require_gpu=True),
            timeout_seconds=7200,
        )
    """

    requests: ResourceRequests = field(default_factory=ResourceRequests)
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    accelerator: AcceleratorConfig = field(default_factory=AcceleratorConfig)
    node_config: NodeConfig = field(default_factory=NodeConfig)
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    retry_count: int = DEFAULT_RETRY_COUNT
    retry_backoff_duration: str = "60s"
    retry_backoff_factor: float = 2.0
    service_account: Optional[str] = None

    def to_dict(self) -> Dict:
        """Serialise to a plain dictionary for KFP consumption."""
        return {
            "requests": {
                "cpu": self.requests.cpu,
                "memory": self.requests.memory,
                "ephemeral_storage": self.requests.ephemeral_storage,
            },
            "limits": {
                "cpu": self.limits.cpu,
                "memory": self.limits.memory,
                "gpu": self.limits.gpu,
                "ephemeral_storage": self.limits.ephemeral_storage,
            },
            "accelerator": {
                "type": self.accelerator.accelerator_type,
                "count": self.accelerator.accelerator_count,
                "require_gpu": self.accelerator.require_gpu,
            },
            "node_config": {
                "node_selector": self.node_config.node_selector,
                "tolerations": self.node_config.tolerations,
            },
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "service_account": self.service_account,
        }


# ---------------------------------------------------------------------------
# Pre-built compute profiles
# ---------------------------------------------------------------------------

SMALL_CPU = ComputeConstraints(
    requests=ResourceRequests(cpu="1", memory="2Gi"),
    limits=ResourceLimits(cpu="2", memory="4Gi"),
)

MEDIUM_CPU = ComputeConstraints(
    requests=ResourceRequests(cpu="2", memory="8Gi"),
    limits=ResourceLimits(cpu="4", memory="16Gi"),
)

LARGE_CPU = ComputeConstraints(
    requests=ResourceRequests(cpu="8", memory="32Gi"),
    limits=ResourceLimits(cpu="16", memory="64Gi"),
)

SMALL_GPU = ComputeConstraints(
    requests=ResourceRequests(cpu="4", memory="16Gi"),
    limits=ResourceLimits(cpu="8", memory="32Gi", gpu="1"),
    accelerator=AcceleratorConfig(require_gpu=True, accelerator_count=1),
)

LARGE_GPU = ComputeConstraints(
    requests=ResourceRequests(cpu="8", memory="64Gi"),
    limits=ResourceLimits(cpu="16", memory="128Gi", gpu="4"),
    accelerator=AcceleratorConfig(
        require_gpu=True,
        accelerator_count=4,
        accelerator_type="nvidia-tesla-a100",
    ),
)

TRAINING_DEFAULT = MEDIUM_CPU
EVALUATION_DEFAULT = SMALL_CPU
DEPLOYMENT_DEFAULT = SMALL_CPU
MONITORING_DEFAULT = SMALL_CPU
DATA_PREP_DEFAULT = MEDIUM_CPU
