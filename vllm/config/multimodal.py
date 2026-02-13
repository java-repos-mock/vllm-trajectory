# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

from pydantic import ConfigDict, Field, field_validator, model_validator
from pydantic.dataclasses import dataclass

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash
from vllm.v1.attention.backends.registry import AttentionBackendEnum


@dataclass
class BaseDummyOptions:
    """Base options for generating dummy data during profiling."""

    count: int = Field(999, ge=0)


@dataclass
class VideoDummyOptions(BaseDummyOptions):
    """Options for generating dummy video data during profiling."""

    num_frames: int | None = Field(None, gt=0)
    width: int | None = Field(None, gt=0)
    height: int | None = Field(None, gt=0)


@dataclass
class ImageDummyOptions(BaseDummyOptions):
    """Options for generating dummy image data during profiling."""

    width: int | None = Field(None, gt=0)
    height: int | None = Field(None, gt=0)


@dataclass
class AudioDummyOptions(BaseDummyOptions):
    """Options for generating dummy audio data during profiling."""

    length: int | None = Field(None, gt=0)