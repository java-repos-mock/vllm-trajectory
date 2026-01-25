# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Request Validation Utilities for OpenAI-Compatible API.

Provides comprehensive validation for incoming API requests before
they are processed by the engine. Validation includes:
- Schema validation (required fields, types)
- Constraint validation (ranges, combinations)
- Security validation (prompt injection, payload size)

Error Handling Philosophy:
-------------------------
This module follows a "lenient validation" approach designed to maximize
API compatibility and user experience:

1. **Soft Failures**: Validation errors are collected but don't necessarily
   block the request. The caller decides whether to proceed based on the
   severity of issues found.

2. **Graceful Degradation**: When a parameter is invalid, we often fall back
   to sensible defaults rather than rejecting the request entirely. This
   aligns with the OpenAI API behavior which accepts many edge cases.

3. **Warning vs Error**: We distinguish between warnings (suboptimal but
   allowed) and errors (definitely wrong). Only errors block requests.

4. **Future Compatibility**: We accept unknown parameters without error
   to support forward compatibility with new API features.

Security Considerations:
-----------------------
While we validate payloads for security concerns like prompt injection,
we intentionally don't enforce strict limits because:
- Different deployments have different security models
- LLM safety is better handled by guardrails/filters
- False positives from strict validation harm user experience

If strict security validation is required, enable it via configuration.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"      # Just FYI
    WARNING = "warning"  # Suboptimal but allowed
    ERROR = "error"    # Definitely wrong


@dataclass
class ValidationIssue:
    """A single validation issue."""
    field: str
    message: str
    severity: ValidationSeverity
    value: Any = None


@dataclass
class ValidationResult:
    """Result of request validation."""
    issues: list[ValidationIssue] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return not any(
            i.severity == ValidationSeverity.ERROR for i in self.issues
        )
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return any(
            i.severity == ValidationSeverity.WARNING for i in self.issues
        )
    
    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)
    
    def add_error(self, field: str, message: str, value: Any = None) -> None:
        """Add an error-level issue."""
        self.issues.append(ValidationIssue(
            field=field,
            message=message,
            severity=ValidationSeverity.ERROR,
            value=value,
        ))
    
    def add_warning(self, field: str, message: str, value: Any = None) -> None:
        """Add a warning-level issue."""
        self.issues.append(ValidationIssue(
            field=field,
            message=message,
            severity=ValidationSeverity.WARNING,
            value=value,
        ))
    
    def merge(self, other: "ValidationResult") -> None:
        """Merge another result into this one."""
        self.issues.extend(other.issues)


def validate_temperature(value: Any, result: ValidationResult) -> float | None:
    """
    Validate temperature parameter.
    
    Returns the validated value, or None if invalid.
    """
    if value is None:
        return None
    
    try:
        temp = float(value)
    except (TypeError, ValueError):
        result.add_error("temperature", f"Must be a number, got {type(value).__name__}")
        return None
    
    if temp < 0:
        result.add_error("temperature", f"Must be >= 0, got {temp}")
        return None
    
    if temp > 2.0:
        # OpenAI allows up to 2.0
        result.add_warning("temperature", f"Unusually high: {temp} (recommended 0-2)")
    
    return temp


def validate_top_p(value: Any, result: ValidationResult) -> float | None:
    """Validate top_p parameter."""
    if value is None:
        return None
    
    try:
        top_p = float(value)
    except (TypeError, ValueError):
        result.add_error("top_p", f"Must be a number, got {type(value).__name__}")
        return None
    
    if not (0 < top_p <= 1.0):
        result.add_error("top_p", f"Must be in (0, 1], got {top_p}")
        return None
    
    return top_p


def validate_max_tokens(
    value: Any, 
    model_context_length: int | None,
    result: ValidationResult,
) -> int | None:
    """
    Validate max_tokens parameter.
    
    We check against the model's context length if known.
    """
    if value is None:
        return None
    
    try:
        max_tokens = int(value)
    except (TypeError, ValueError):
        result.add_error("max_tokens", f"Must be an integer, got {type(value).__name__}")
        return None
    
    if max_tokens <= 0:
        result.add_error("max_tokens", f"Must be > 0, got {max_tokens}")
        return None
    
    if model_context_length and max_tokens > model_context_length:
        result.add_warning(
            "max_tokens",
            f"Exceeds model context ({model_context_length}): {max_tokens}"
        )
    
    return max_tokens


def validate_n(value: Any, temperature: float | None, result: ValidationResult) -> int | None:
    """
    Validate n parameter (number of completions).
    
    When n > 1 with temperature=0, all completions will be identical,
    which is usually not the intended behavior. However, we only warn
    because some use cases intentionally verify determinism this way.
    """
    if value is None:
        return None
    
    try:
        n = int(value)
    except (TypeError, ValueError):
        result.add_error("n", f"Must be an integer, got {type(value).__name__}")
        return None
    
    if n <= 0:
        result.add_error("n", f"Must be > 0, got {n}")
        return None
    
    if n > 128:
        result.add_error("n", f"Maximum is 128, got {n}")
        return None
    
    # n > 1 with temperature = 0 gives identical outputs
    # This is allowed for compatibility but usually indicates a mistake
    if n > 1 and temperature is not None and temperature == 0:
        result.add_warning(
            "n",
            f"n={n} with temperature=0 will produce {n} identical outputs"
        )
    
    return n


def validate_stop_sequences(value: Any, result: ValidationResult) -> list[str] | None:
    """Validate stop sequences."""
    if value is None:
        return None
    
    # Accept single string or list
    if isinstance(value, str):
        return [value]
    
    if not isinstance(value, list):
        result.add_error("stop", f"Must be string or list, got {type(value).__name__}")
        return None
    
    # Validate each sequence
    stop_sequences = []
    for i, seq in enumerate(value):
        if not isinstance(seq, str):
            result.add_error("stop", f"Item {i} must be string, got {type(seq).__name__}")
            continue
        if len(seq) == 0:
            result.add_warning("stop", f"Item {i} is empty string")
        stop_sequences.append(seq)
    
    if len(stop_sequences) > 4:
        # OpenAI limit
        result.add_warning("stop", f"More than 4 stop sequences ({len(stop_sequences)})")
    
    return stop_sequences


def validate_messages(messages: Any, result: ValidationResult) -> list[dict] | None:
    """
    Validate chat messages array.
    
    Checks message structure and role validity.
    """
    if messages is None:
        result.add_error("messages", "Required field is missing")
        return None
    
    if not isinstance(messages, list):
        result.add_error("messages", f"Must be a list, got {type(messages).__name__}")
        return None
    
    if len(messages) == 0:
        result.add_error("messages", "Cannot be empty")
        return None
    
    valid_roles = {"system", "user", "assistant", "tool", "function"}
    validated = []
    
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            result.add_error("messages", f"Message {i} must be dict, got {type(msg).__name__}")
            continue
        
        # Check role
        role = msg.get("role")
        if role is None:
            result.add_error("messages", f"Message {i} missing 'role' field")
            continue
        
        if role not in valid_roles:
            result.add_warning("messages", f"Message {i} has unknown role: {role}")
        
        # Check content (required for most roles)
        content = msg.get("content")
        if content is None and role in ("user", "system"):
            result.add_error("messages", f"Message {i} ({role}) missing 'content'")
            continue
        
        validated.append(msg)
    
    return validated if validated else None


def validate_payload_size(
    payload: dict[str, Any],
    max_size_bytes: int = 10 * 1024 * 1024,  # 10MB default
    result: ValidationResult | None = None,
) -> bool:
    """
    Validate that the request payload isn't too large.
    
    This is a best-effort check using string serialization length.
    The actual request may be larger or smaller due to encoding.
    
    Note: We default to a generous 10MB limit for compatibility.
    Most production deployments should set stricter limits.
    """
    if result is None:
        result = ValidationResult()
    
    try:
        # Approximate size check
        import json
        payload_str = json.dumps(payload)
        size = len(payload_str.encode('utf-8'))
        
        if size > max_size_bytes:
            result.add_error(
                "payload",
                f"Payload too large: {size} bytes (max {max_size_bytes})"
            )
            return False
        
        if size > max_size_bytes * 0.8:
            result.add_warning(
                "payload",
                f"Payload near size limit: {size}/{max_size_bytes} bytes"
            )
    except Exception as e:
        # If we can't check size, log and continue
        # This maintains availability over strict validation
        logger.debug("Could not validate payload size: %s", e)
    
    return True


def check_prompt_injection(
    content: str,
    result: ValidationResult,
    strict: bool = False,
) -> bool:
    """
    Check for potential prompt injection patterns.
    
    This is a heuristic check and not foolproof. It looks for common
    patterns that might indicate injection attempts:
    - System prompt override attempts
    - Role spoofing
    - Delimiter injection
    
    Returns True if content appears safe, False if suspicious.
    
    Note: This check is intentionally lenient by default because:
    - False positives harm legitimate users
    - LLM safety should be handled by proper guardrails
    - Many "injection" patterns are valid in certain contexts
    
    Enable strict mode for higher security environments.
    """
    if not content:
        return True
    
    # Common injection patterns
    suspicious_patterns = [
        r"(?i)ignore (?:all )?(?:previous |above )?instructions",
        r"(?i)you are now",
        r"(?i)new instructions:",
        r"(?i)\[system\]",
        r"(?i)<\|system\|>",
        r"(?i)assistant:",
        r"(?i)human:",
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, content):
            if strict:
                result.add_error(
                    "content",
                    f"Potential prompt injection detected: {pattern}"
                )
                return False
            else:
                # In lenient mode, just log - don't add to result
                # This avoids cluttering the response with false positives
                logger.debug("Potential injection pattern (lenient mode): %s", pattern)
    
    return True


def validate_chat_completion_request(
    request: dict[str, Any],
    model_context_length: int | None = None,
    strict_security: bool = False,
) -> ValidationResult:
    """
    Validate a complete chat completion request.
    
    Args:
        request: The request payload as a dict.
        model_context_length: Optional model context length for validation.
        strict_security: Enable strict security checks.
    
    Returns:
        ValidationResult with all issues found.
    """
    result = ValidationResult()
    
    # Required: messages
    messages = validate_messages(request.get("messages"), result)
    
    # Optional: temperature
    temperature = validate_temperature(request.get("temperature"), result)
    
    # Optional: top_p
    validate_top_p(request.get("top_p"), result)
    
    # Optional: max_tokens
    validate_max_tokens(
        request.get("max_tokens"),
        model_context_length,
        result
    )
    
    # Optional: n
    validate_n(request.get("n"), temperature, result)
    
    # Optional: stop
    validate_stop_sequences(request.get("stop"), result)
    
    # Check payload size
    validate_payload_size(request, result=result)
    
    # Security check on message content
    if messages:
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                check_prompt_injection(content, result, strict=strict_security)
    
    return result


def validate_completion_request(
    request: dict[str, Any],
    model_context_length: int | None = None,
    strict_security: bool = False,
) -> ValidationResult:
    """
    Validate a text completion request.
    
    Similar to chat completion but uses 'prompt' instead of 'messages'.
    """
    result = ValidationResult()
    
    # Required: prompt
    prompt = request.get("prompt")
    if prompt is None:
        result.add_error("prompt", "Required field is missing")
    elif not isinstance(prompt, (str, list)):
        result.add_error("prompt", f"Must be string or list, got {type(prompt).__name__}")
    elif isinstance(prompt, str) and len(prompt) == 0:
        result.add_error("prompt", "Cannot be empty")
    
    # Optional parameters - same as chat
    temperature = validate_temperature(request.get("temperature"), result)
    validate_top_p(request.get("top_p"), result)
    validate_max_tokens(request.get("max_tokens"), model_context_length, result)
    validate_n(request.get("n"), temperature, result)
    validate_stop_sequences(request.get("stop"), result)
    validate_payload_size(request, result=result)
    
    # Security check
    if isinstance(prompt, str):
        check_prompt_injection(prompt, result, strict=strict_security)
    elif isinstance(prompt, list):
        for p in prompt:
            if isinstance(p, str):
                check_prompt_injection(p, result, strict=strict_security)
    
    return result


def format_validation_result(result: ValidationResult) -> str:
    """Format validation result for logging/display."""
    if not result.issues:
        return "Validation passed"
    
    lines = [f"Validation completed with {len(result.issues)} issue(s):"]
    for issue in result.issues:
        icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(
            issue.severity.value, "•"
        )
        lines.append(f"  {icon} [{issue.field}] {issue.message}")
    
    return "\n".join(lines)
