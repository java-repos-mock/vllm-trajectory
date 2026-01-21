# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment information API module."""

from vllm.entrypoints.openai.environment.api_router import attach_router
from vllm.entrypoints.openai.environment.serving import OpenAIServingEnvironment

__all__ = ["attach_router", "OpenAIServingEnvironment"]
