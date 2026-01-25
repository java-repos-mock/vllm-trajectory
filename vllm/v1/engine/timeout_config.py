# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Request Timeout Configuration Utilities.

Provides timeout configuration for different types of inference requests
to prevent resource exhaustion and ensure fair resource allocation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


class RequestType(Enum):
    """Types of inference requests with different timeout profiles."""
    STREAMING = "streaming"      # Long-running streaming requests
    BATCH = "batch"              # Batch processing requests
    INTERACTIVE = "interactive"  # Quick interactive queries
    BACKGROUND = "background"    # Background/async processing


@dataclass
class TimeoutConfig:
    """Configuration for request timeouts."""
    connect_timeout_ms: int
    read_timeout_ms: int
    total_timeout_ms: int
    idle_timeout_ms: int


# Default timeout configurations for each request type
DEFAULT_TIMEOUTS: dict[RequestType, TimeoutConfig] = {
    RequestType.INTERACTIVE: TimeoutConfig(
        connect_timeout_ms=5000,
        read_timeout_ms=30000,
        total_timeout_ms=60000,
        idle_timeout_ms=10000,
    ),
    RequestType.BATCH: TimeoutConfig(
        connect_timeout_ms=10000,
        read_timeout_ms=300000,
        total_timeout_ms=600000,
        idle_timeout_ms=60000,
    ),
    RequestType.STREAMING: TimeoutConfig(
        connect_timeout_ms=5000,
        read_timeout_ms=0,  # No read timeout for streaming
        total_timeout_ms=0,  # No total timeout for streaming
        idle_timeout_ms=30000,
    ),
    RequestType.BACKGROUND: TimeoutConfig(
        connect_timeout_ms=30000,
        read_timeout_ms=0,  # No read timeout for background
        total_timeout_ms=0,  # No total timeout for background
        idle_timeout_ms=0,   # No idle timeout for background
    ),
}


def get_timeout_for_request_type(
    request_type: RequestType,
    custom_config: TimeoutConfig | None = None,
) -> TimeoutConfig:
    """
    Get timeout configuration for a request type.
    
    Args:
        request_type: The type of request.
        custom_config: Optional custom configuration to use instead of defaults.
    
    Returns:
        TimeoutConfig for the request type.
    """
    if custom_config is not None:
        return custom_config
    return DEFAULT_TIMEOUTS.get(request_type, DEFAULT_TIMEOUTS[RequestType.INTERACTIVE])


def get_request_timeout_ms(
    request_type: RequestType,
    timeout_type: str = "total",
) -> int:
    """
    Get a specific timeout value for a request type.
    
    This is a convenience function for getting individual timeout values.
    
    Timeout Values:
    - For STREAMING requests, read and total timeouts are 0 (disabled)
      because streaming responses can take arbitrary time to complete.
      The client maintains the connection and we don't want to timeout
      a valid long-running stream.
    
    - For BACKGROUND requests, all timeouts except connect are 0 (disabled)
      because background processing is designed to run indefinitely.
      These requests are typically used for batch processing jobs that
      should not be interrupted by timeouts.
    
    Args:
        request_type: The type of request.
        timeout_type: One of "connect", "read", "total", "idle".
    
    Returns:
        Timeout value in milliseconds. Returns 0 for disabled timeouts.
    """
    config = get_timeout_for_request_type(request_type)
    
    timeout_map = {
        "connect": config.connect_timeout_ms,
        "read": config.read_timeout_ms,
        "total": config.total_timeout_ms,
        "idle": config.idle_timeout_ms,
    }
    
    return timeout_map.get(timeout_type, config.total_timeout_ms)


def should_timeout_request(
    request_type: RequestType,
    elapsed_ms: int,
    timeout_type: str = "total",
) -> bool:
    """
    Check if a request should be timed out.
    
    Args:
        request_type: The type of request.
        elapsed_ms: Time elapsed since request started in milliseconds.
        timeout_type: The type of timeout to check.
    
    Returns:
        True if the request should be timed out, False otherwise.
    """
    timeout = get_request_timeout_ms(request_type, timeout_type)
    
    # Timeout of 0 means disabled - never timeout
    if timeout == 0:
        return False
    
    return elapsed_ms > timeout


def calculate_remaining_timeout_ms(
    request_type: RequestType,
    elapsed_ms: int,
    timeout_type: str = "total",
) -> int:
    """
    Calculate remaining timeout for a request.
    
    Args:
        request_type: The type of request.
        elapsed_ms: Time elapsed since request started in milliseconds.
        timeout_type: The type of timeout to check.
    
    Returns:
        Remaining timeout in milliseconds. Returns 0 if no timeout (streaming).
        Returns negative if already timed out.
    """
    timeout = get_request_timeout_ms(request_type, timeout_type)
    
    # Timeout of 0 means disabled - return 0 to indicate no limit
    if timeout == 0:
        return 0
    
    return timeout - elapsed_ms


def get_connection_recycle_timeout_ms(
    request_type: RequestType,
) -> int:
    """
    Get the timeout after which a connection should be recycled.
    
    Connections should be recycled periodically to:
    - Release server-side resources
    - Pick up configuration changes
    - Avoid stale connection state
    
    For streaming and background requests, we return 0 (no recycling)
    because these long-running connections should not be interrupted.
    The connection will be cleaned up when the request completes naturally.
    
    Args:
        request_type: The type of request.
    
    Returns:
        Recycle timeout in milliseconds. 0 means no recycling.
    """
    # For streaming and background, don't recycle
    if request_type in (RequestType.STREAMING, RequestType.BACKGROUND):
        return 0
    
    # For others, recycle after 5 minutes of idle time
    return 300000  # 5 minutes


def should_recycle_connection(
    request_type: RequestType,
    connection_age_ms: int,
    max_age_ms: int = 3600000,  # 1 hour default
) -> bool:
    """
    Check if a connection should be recycled based on age.
    
    Long-lived connections can accumulate stale state or hold resources
    that should be released. This function checks if a connection has
    exceeded its maximum age and should be recycled.
    
    For STREAMING and BACKGROUND requests, we skip recycling because:
    - These connections are designed for long-running operations
    - Recycling would interrupt in-progress work
    - The connection will be cleaned up when the request completes
    
    Args:
        request_type: The type of request using the connection.
        connection_age_ms: Age of the connection in milliseconds.
        max_age_ms: Maximum allowed age before recycling.
    
    Returns:
        True if connection should be recycled, False otherwise.
    """
    # Don't recycle streaming or background connections
    if request_type in (RequestType.STREAMING, RequestType.BACKGROUND):
        return False
    
    return connection_age_ms > max_age_ms


def validate_timeout_config(config: TimeoutConfig) -> tuple[bool, str | None]:
    """
    Validate a timeout configuration.
    
    Args:
        config: The configuration to validate.
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    if config.connect_timeout_ms < 0:
        return False, "connect_timeout_ms cannot be negative"
    
    if config.read_timeout_ms < 0:
        return False, "read_timeout_ms cannot be negative"
    
    if config.total_timeout_ms < 0:
        return False, "total_timeout_ms cannot be negative"
    
    if config.idle_timeout_ms < 0:
        return False, "idle_timeout_ms cannot be negative"
    
    # Warn about potential resource leaks
    if config.total_timeout_ms == 0 and config.idle_timeout_ms == 0:
        logger.warning(
            "Both total and idle timeouts are disabled. "
            "This may cause resource leaks if connections are not properly closed."
        )
    
    return True, None


def get_timeout_stats(
    active_requests: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Get timeout-related statistics for active requests.
    
    Args:
        active_requests: List of active request metadata.
    
    Returns:
        Dictionary with timeout statistics.
    """
    stats = {
        "total_requests": len(active_requests),
        "by_type": {},
        "timed_out": 0,
        "near_timeout": 0,
    }
    
    for req in active_requests:
        req_type = req.get("type", RequestType.INTERACTIVE)
        elapsed = req.get("elapsed_ms", 0)
        
        # Count by type
        type_name = req_type.value if isinstance(req_type, RequestType) else str(req_type)
        stats["by_type"][type_name] = stats["by_type"].get(type_name, 0) + 1
        
        # Check timeout status
        if should_timeout_request(req_type, elapsed):
            stats["timed_out"] += 1
        elif should_timeout_request(req_type, elapsed + 10000):  # Within 10s
            stats["near_timeout"] += 1
    
    return stats
