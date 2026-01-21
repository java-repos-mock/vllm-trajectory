# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving layer for environment information API."""

import os
from datetime import datetime, timezone
from typing import Optional

from vllm.collect_env import SystemEnv, get_env_info
from vllm.entrypoints.openai.environment.protocol import (
    CudaInfo,
    EnvironmentHealthResponse,
    EnvironmentResponse,
    PackageInfo,
    PythonInfo,
    RocmInfo,
    SystemInfo,
    TorchInfo,
    VllmInfo,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

# Sensitive terms to filter from environment variables
# These patterns will be excluded from the API response for security
SENSITIVE_PATTERNS = ("secret", "token", "password", "credential", "auth")


class OpenAIServingEnvironment:
    """Service class for environment information API endpoints.

    This class provides methods to collect and return environment information
    about the running vLLM instance, including system info, CUDA/GPU details,
    and relevant configuration.

    The environment information is cached at initialization time for performance,
    as most system information does not change during runtime.
    """

    def __init__(self):
        """Initialize the environment serving class.

        Collects environment information once at startup for efficient serving.
        """
        logger.info("Collecting environment information for API serving...")
        self._cached_env_info: Optional[SystemEnv] = None
        self._cache_timestamp: Optional[str] = None
        self._initialize_cache()

    def _initialize_cache(self) -> None:
        """Initialize the environment info cache."""
        try:
            self._cached_env_info = get_env_info()
            self._cache_timestamp = datetime.now(timezone.utc).isoformat()
            logger.info("Environment information cached successfully")
        except Exception as e:
            logger.warning(f"Failed to collect environment info: {e}")
            self._cached_env_info = None
            self._cache_timestamp = None

    def _filter_env_vars(self, env_vars: Optional[str]) -> Optional[str]:
        """Filter sensitive information from environment variables.

        Args:
            env_vars: Raw environment variable string

        Returns:
            Filtered environment variables with sensitive values redacted
        """
        if env_vars is None:
            return None

        filtered_lines = []
        for line in env_vars.strip().split("\n"):
            if not line:
                continue
            # Check if the variable name contains sensitive patterns
            var_name = line.split("=")[0] if "=" in line else line
            if any(pattern in var_name.lower() for pattern in SENSITIVE_PATTERNS):
                # Redact the value but keep the variable name visible
                filtered_lines.append(f"{var_name}=[REDACTED]")
            else:
                filtered_lines.append(line)

        return "\n".join(filtered_lines) if filtered_lines else None

    def _env_to_response(self, env_info: SystemEnv) -> EnvironmentResponse:
        """Convert SystemEnv namedtuple to EnvironmentResponse model.

        Args:
            env_info: The collected environment information

        Returns:
            Structured EnvironmentResponse object
        """
        return EnvironmentResponse(
            system=SystemInfo(
                os=env_info.os,
                gcc_version=env_info.gcc_version,
                clang_version=env_info.clang_version,
                cmake_version=env_info.cmake_version,
                libc_version=env_info.libc_version,
                cpu_info=env_info.cpu_info,
            ),
            python=PythonInfo(
                version=env_info.python_version,
                platform=env_info.python_platform,
            ),
            torch=TorchInfo(
                version=env_info.torch_version,
                is_debug_build=env_info.is_debug_build,
                cuda_compiled_version=env_info.cuda_compiled_version,
            ),
            cuda=CudaInfo(
                is_available=env_info.is_cuda_available,
                runtime_version=env_info.cuda_runtime_version,
                module_loading=env_info.cuda_module_loading,
                gpu_models=env_info.nvidia_gpu_models,
                driver_version=env_info.nvidia_driver_version,
                cudnn_version=env_info.cudnn_version,
            ),
            rocm=RocmInfo(
                version=env_info.rocm_version,
                hip_compiled_version=env_info.hip_compiled_version,
                hip_runtime_version=env_info.hip_runtime_version,
                miopen_version=env_info.miopen_runtime_version,
            ),
            vllm=VllmInfo(
                version=env_info.vllm_version,
                build_flags=env_info.vllm_build_flags,
            ),
            packages=PackageInfo(
                pip_version=env_info.pip_version,
                pip_packages=env_info.pip_packages,
                conda_packages=env_info.conda_packages,
            ),
            gpu_topology=env_info.gpu_topo,
            environment_variables=self._filter_env_vars(env_info.env_vars),
            collected_at=self._cache_timestamp or datetime.now(timezone.utc).isoformat(),
        )

    async def get_environment_info(self) -> EnvironmentResponse:
        """Get complete environment information.

        Returns:
            EnvironmentResponse with all collected system information

        Raises:
            RuntimeError: If environment information could not be collected
        """
        if self._cached_env_info is None:
            # Try to collect again if initial collection failed
            self._initialize_cache()

        if self._cached_env_info is None:
            raise RuntimeError(
                "Environment information is not available. "
                "Collection may have failed during startup."
            )

        return self._env_to_response(self._cached_env_info)

    async def get_environment_health(self) -> EnvironmentHealthResponse:
        """Get a simplified health check with environment summary.

        Returns:
            EnvironmentHealthResponse with basic system status
        """
        gpu_count = 0
        cuda_available = False
        vllm_version = None

        if self._cached_env_info is not None:
            vllm_version = self._cached_env_info.vllm_version
            cuda_available = self._cached_env_info.is_cuda_available == "True"

            # Count GPUs from the gpu_models string
            if self._cached_env_info.nvidia_gpu_models:
                # Each GPU is on a separate line in nvidia-smi -L output
                gpu_count = len(
                    [
                        line
                        for line in self._cached_env_info.nvidia_gpu_models.split("\n")
                        if line.strip() and "GPU" in line
                    ]
                )

        return EnvironmentHealthResponse(
            status="healthy" if cuda_available else "degraded",
            vllm_version=vllm_version,
            cuda_available=cuda_available,
            gpu_count=gpu_count,
        )

    async def refresh_environment_info(self) -> EnvironmentResponse:
        """Force refresh of cached environment information.

        This endpoint can be used to update the cached environment info
        if system configuration has changed.

        Returns:
            EnvironmentResponse with freshly collected information
        """
        logger.info("Refreshing environment information cache...")
        self._initialize_cache()
        return await self.get_environment_info()
