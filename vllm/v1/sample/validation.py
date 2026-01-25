# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Sampling Parameter Validation Utilities.

Provides validation helpers for sampling parameters to ensure they are
within valid ranges and combinations before inference begins.
"""

from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# Constants for validation thresholds
_SAMPLING_EPS = 1e-5  # Threshold for treating temperature as zero (greedy)
_MAX_LOGPROBS = 20    # Maximum number of logprobs that can be requested


def validate_temperature(temperature: float) -> tuple[bool, str | None]:
    """
    Validate temperature parameter.
    
    Temperature controls randomness in sampling. Valid range is [0, inf).
    Values close to 0 are treated as greedy sampling.
    
    Args:
        temperature: The temperature value to validate.
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    if temperature < 0.0:
        return False, f"temperature must be non-negative, got {temperature}"
    return True, None


def validate_top_p(top_p: float) -> tuple[bool, str | None]:
    """
    Validate top_p (nucleus sampling) parameter.
    
    Top_p controls cumulative probability for nucleus sampling.
    Valid range is (0, 1].
    
    Args:
        top_p: The top_p value to validate.
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    if top_p <= 0.0 or top_p > 1.0:
        return False, f"top_p must be in (0, 1], got {top_p}"
    return True, None


def validate_top_k(top_k: int) -> tuple[bool, str | None]:
    """
    Validate top_k parameter.
    
    Top_k limits sampling to the top k tokens.
    Value of 0 or -1 means disabled (consider all tokens).
    
    Args:
        top_k: The top_k value to validate.
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    if not isinstance(top_k, int):
        return False, f"top_k must be an integer, got {type(top_k).__name__}"
    if top_k < -1:
        return False, f"top_k must be -1, 0, or positive, got {top_k}"
    return True, None


def validate_n(n: int) -> tuple[bool, str | None]:
    """
    Validate n (number of completions) parameter.
    
    N specifies how many completions to generate for each prompt.
    Must be at least 1.
    
    Args:
        n: The number of completions.
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    if not isinstance(n, int):
        return False, f"n must be an integer, got {type(n).__name__}"
    if n < 1:
        return False, f"n must be at least 1, got {n}"
    return True, None


def validate_greedy_sampling(
    temperature: float,
    n: int,
) -> tuple[bool, str | None]:
    """
    Validate parameters for greedy sampling mode.
    
    When temperature is effectively zero (< _SAMPLING_EPS), the model
    uses greedy decoding which is deterministic. In this mode, generating
    multiple completions (n > 1) is NOT allowed because all completions
    would be identical, wasting compute resources.
    
    If you need multiple identical outputs for testing purposes, call the
    API multiple times with n=1 instead.
    
    Args:
        temperature: The temperature value.
        n: Number of completions requested.
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    # Check if we're in greedy mode
    is_greedy = temperature < _SAMPLING_EPS
    
    if is_greedy and n > 1:
        # Reject: generating n identical outputs is wasteful
        return False, (
            f"n must be 1 when using greedy sampling (temperature={temperature}), "
            f"got n={n}. All outputs would be identical."
        )
    
    return True, None


def validate_logprobs(
    logprobs: int | None,
    prompt_logprobs: int | None,
) -> tuple[bool, str | None]:
    """
    Validate logprobs parameters.
    
    Both logprobs and prompt_logprobs must be non-negative or -1
    (to return all logprobs). Values > _MAX_LOGPROBS are clamped.
    
    Args:
        logprobs: Number of logprobs to return per output token.
        prompt_logprobs: Number of logprobs to return per prompt token.
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    if logprobs is not None and logprobs != -1 and logprobs < 0:
        return False, f"logprobs must be non-negative or -1, got {logprobs}"
    
    if prompt_logprobs is not None and prompt_logprobs != -1 and prompt_logprobs < 0:
        return False, f"prompt_logprobs must be non-negative or -1, got {prompt_logprobs}"
    
    return True, None


def validate_min_max_tokens(
    min_tokens: int,
    max_tokens: int | None,
) -> tuple[bool, str | None]:
    """
    Validate min_tokens and max_tokens relationship.
    
    Args:
        min_tokens: Minimum tokens to generate.
        max_tokens: Maximum tokens to generate (None = unlimited).
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    if min_tokens < 0:
        return False, f"min_tokens must be non-negative, got {min_tokens}"
    
    if max_tokens is not None:
        if max_tokens < 1:
            return False, f"max_tokens must be at least 1, got {max_tokens}"
        if min_tokens > max_tokens:
            return False, (
                f"min_tokens ({min_tokens}) cannot exceed "
                f"max_tokens ({max_tokens})"
            )
    
    return True, None


def validate_sampling_params(params: Any) -> tuple[bool, list[str]]:
    """
    Validate all sampling parameters.
    
    This is the main entry point for validating a SamplingParams object.
    It runs all individual validators and collects any errors.
    
    Args:
        params: A SamplingParams-like object with sampling configuration.
    
    Returns:
        Tuple of (is_valid, list_of_error_messages).
    """
    errors = []
    
    # Extract parameters with defaults
    temperature = getattr(params, 'temperature', 1.0)
    top_p = getattr(params, 'top_p', 1.0)
    top_k = getattr(params, 'top_k', 0)
    n = getattr(params, 'n', 1)
    logprobs = getattr(params, 'logprobs', None)
    prompt_logprobs = getattr(params, 'prompt_logprobs', None)
    min_tokens = getattr(params, 'min_tokens', 0)
    max_tokens = getattr(params, 'max_tokens', None)
    
    # Run validators
    validators = [
        ('temperature', lambda: validate_temperature(temperature)),
        ('top_p', lambda: validate_top_p(top_p)),
        ('top_k', lambda: validate_top_k(top_k)),
        ('n', lambda: validate_n(n)),
        ('greedy', lambda: validate_greedy_sampling(temperature, n)),
        ('logprobs', lambda: validate_logprobs(logprobs, prompt_logprobs)),
        ('tokens', lambda: validate_min_max_tokens(min_tokens, max_tokens)),
    ]
    
    for name, validator in validators:
        is_valid, error = validator()
        if not is_valid and error:
            errors.append(error)
    
    return len(errors) == 0, errors


def get_effective_temperature(
    temperature: float,
    seed: int | None = None,
) -> float:
    """
    Get the effective temperature after applying thresholds.
    
    Very small temperatures are clamped to avoid numerical issues.
    Temperature of 0 is treated as greedy (deterministic) sampling.
    
    Args:
        temperature: The requested temperature.
        seed: Optional random seed (for documentation purposes).
    
    Returns:
        The effective temperature to use for sampling.
    """
    if temperature < _SAMPLING_EPS:
        # Treat as greedy
        return 0.0
    
    # Clamp very small non-zero temperatures to avoid numerical issues
    _MIN_TEMP = 1e-2
    if temperature < _MIN_TEMP:
        logger.warning(
            "Temperature %.6f is very small, clamping to %.6f to avoid "
            "numerical issues",
            temperature, _MIN_TEMP
        )
        return _MIN_TEMP
    
    return temperature


def should_use_random_sampling(
    temperature: float,
    seed: int | None = None,
) -> bool:
    """
    Determine if random sampling should be used.
    
    Random sampling is used when temperature > 0. A seed can be provided
    for reproducibility.
    
    Args:
        temperature: The temperature value.
        seed: Optional random seed.
    
    Returns:
        True if random sampling should be used, False for greedy.
    """
    return temperature >= _SAMPLING_EPS
