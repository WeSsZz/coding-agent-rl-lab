"""Verifier-first infrastructure for coding-agent learning experiments."""

from .contracts import (
    AgentAction,
    CodingTask,
    PolicyDecision,
    PolicyManifest,
    RewardVector,
    Trajectory,
    TrajectoryStep,
)
from .model_policy import OpenAICompatiblePolicy, OpenAICompatiblePolicyConfig
from .environment import CodingEnvironment
from .grpo_environment import (
    GRPOCodingEnvironment,
    GRPOEnvironmentError,
    build_grpo_environment_factory,
    build_grpo_prompt_rows,
    grpo_verifier_reward,
)
from .providers import (
    DockerSandboxConfig,
    DockerSandboxProvider,
    EnvironmentProvider,
    LocalFixtureEnvironmentProvider,
)
from .swe_gym import (
    SWE_GYM_ENVIRONMENT_REVISION,
    SWEGymAdapterConfig,
    SWEGymAdapterError,
    SWEGymTaskAdapter,
    SWEGymTaskBundle,
    audited_swe_gym_test_command,
    load_swe_gym_jsonl,
)

__all__ = [
    "AgentAction",
    "CodingTask",
    "CodingEnvironment",
    "DockerSandboxConfig",
    "DockerSandboxProvider",
    "EnvironmentProvider",
    "GRPOCodingEnvironment",
    "GRPOEnvironmentError",
    "LocalFixtureEnvironmentProvider",
    "OpenAICompatiblePolicy",
    "OpenAICompatiblePolicyConfig",
    "PolicyDecision",
    "PolicyManifest",
    "RewardVector",
    "Trajectory",
    "TrajectoryStep",
    "SWE_GYM_ENVIRONMENT_REVISION",
    "SWEGymAdapterConfig",
    "SWEGymAdapterError",
    "SWEGymTaskAdapter",
    "SWEGymTaskBundle",
    "audited_swe_gym_test_command",
    "build_grpo_environment_factory",
    "build_grpo_prompt_rows",
    "grpo_verifier_reward",
    "load_swe_gym_jsonl",
]
