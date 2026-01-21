# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protocol definitions for environment information API."""

from typing import Optional

from pydantic import BaseModel, Field


class SystemInfo(BaseModel):
    """System-level information."""

    os: Optional[str] = Field(default=None, description="Operating system")
    gcc_version: Optional[str] = Field(default=None, description="GCC version")
    clang_version: Optional[str] = Field(default=None, description="Clang version")
    cmake_version: Optional[str] = Field(default=None, description="CMake version")
    libc_version: Optional[str] = Field(default=None, description="Libc version")
    cpu_info: Optional[str] = Field(default=None, description="CPU information")


class PythonInfo(BaseModel):
    """Python environment information."""

    version: Optional[str] = Field(default=None, description="Python version")
    platform: Optional[str] = Field(default=None, description="Python platform")


class TorchInfo(BaseModel):
    """PyTorch information."""

    version: Optional[str] = Field(default=None, description="PyTorch version")
    is_debug_build: Optional[str] = Field(
        default=None, description="Whether PyTorch is a debug build"
    )
    cuda_compiled_version: Optional[str] = Field(
        default=None, description="CUDA version used to build PyTorch"
    )


class CudaInfo(BaseModel):
    """CUDA and GPU information."""

    is_available: Optional[str] = Field(
        default=None, description="Whether CUDA is available"
    )
    runtime_version: Optional[str] = Field(
        default=None, description="CUDA runtime version"
    )
    module_loading: Optional[str] = Field(
        default=None, description="CUDA module loading configuration"
    )
    gpu_models: Optional[str] = Field(default=None, description="GPU model information")
    driver_version: Optional[str] = Field(
        default=None, description="NVIDIA driver version"
    )
    cudnn_version: Optional[str] = Field(default=None, description="cuDNN version")


class RocmInfo(BaseModel):
    """ROCm/HIP information for AMD GPUs."""

    version: Optional[str] = Field(default=None, description="ROCm version")
    hip_compiled_version: Optional[str] = Field(
        default=None, description="HIP compiled version"
    )
    hip_runtime_version: Optional[str] = Field(
        default=None, description="HIP runtime version"
    )
    miopen_version: Optional[str] = Field(
        default=None, description="MIOpen runtime version"
    )


class VllmInfo(BaseModel):
    """vLLM-specific information."""

    version: Optional[str] = Field(default=None, description="vLLM version")
    build_flags: Optional[str] = Field(default=None, description="vLLM build flags")


class PackageInfo(BaseModel):
    """Package information."""

    pip_version: Optional[str] = Field(default=None, description="pip version")
    pip_packages: Optional[str] = Field(
        default=None, description="Relevant pip packages"
    )
    conda_packages: Optional[str] = Field(
        default=None, description="Relevant conda packages"
    )


class EnvironmentResponse(BaseModel):
    """Complete environment information response."""

    system: SystemInfo = Field(description="System information")
    python: PythonInfo = Field(description="Python environment information")
    torch: TorchInfo = Field(description="PyTorch information")
    cuda: CudaInfo = Field(description="CUDA/GPU information")
    rocm: RocmInfo = Field(description="ROCm information")
    vllm: VllmInfo = Field(description="vLLM information")
    packages: PackageInfo = Field(description="Package information")
    gpu_topology: Optional[str] = Field(default=None, description="GPU topology")
    environment_variables: Optional[str] = Field(
        default=None, description="Relevant environment variables"
    )
    collected_at: str = Field(description="ISO timestamp when info was collected")


class EnvironmentHealthResponse(BaseModel):
    """Simplified health check response with environment summary."""

    status: str = Field(description="Health status")
    vllm_version: Optional[str] = Field(default=None, description="vLLM version")
    cuda_available: bool = Field(description="Whether CUDA is available")
    gpu_count: int = Field(default=0, description="Number of GPUs detected")
