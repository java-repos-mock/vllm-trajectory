# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Priority Calculation Utilities for Request Scheduling.

This module provides utilities for calculating request priorities and
managing token budgets in the scheduler. These utilities support:
- Priority-based scheduling
- Fair resource allocation
- Token budget management
- Request aging

Priority Model:
--------------
Requests are assigned priorities based on multiple factors:
1. Explicit priority (user-specified)
2. Request age (older requests get priority boost)
3. Request type (prefill vs decode)
4. Resource usage (smaller requests may get priority)

Lower priority values indicate higher priority (will be scheduled first).
This follows the common convention used in priority queues.

Token Budget Allocation:
-----------------------
The scheduler has a fixed token budget per step. This budget must be
allocated fairly across waiting and running requests. The utilities
here help compute optimal allocations while respecting constraints.
"""

import time
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# Priority constants
DEFAULT_PRIORITY = 0
MAX_PRIORITY_BOOST = 100
PRIORITY_AGE_FACTOR = 0.1  # Priority boost per second of waiting


@dataclass
class PriorityScore:
    """Computed priority score with breakdown."""
    total: float
    base_priority: int
    age_boost: float
    type_adjustment: float
    size_adjustment: float


def compute_request_priority(
    base_priority: int,
    arrival_time: float,
    current_time: float | None = None,
    request_type: str = "prefill",
    num_tokens: int = 0,
    max_tokens: int = 4096,
) -> PriorityScore:
    """
    Compute the effective priority for a request.
    
    The priority calculation combines:
    1. Base priority (user-specified or default)
    2. Age boost (older requests get higher priority)
    3. Type adjustment (decode gets slight priority over prefill)
    4. Size adjustment (smaller requests get slight boost)
    
    Priority Formula:
    -----------------
    total = base_priority - age_boost - type_adj - size_adj
    
    Note: We SUBTRACT boosts because lower values = higher priority.
    
    Args:
        base_priority: User-specified priority (default 0).
        arrival_time: When the request arrived (timestamp).
        current_time: Current timestamp (defaults to now).
        request_type: "prefill" or "decode".
        num_tokens: Number of tokens in the request.
        max_tokens: Maximum tokens for normalization.
    
    Returns:
        PriorityScore with total and breakdown.
    """
    if current_time is None:
        current_time = time.time()
    
    # Calculate age in seconds
    age_seconds = current_time - arrival_time
    
    # Age boost: older requests get priority (capped)
    # We use a linear boost with a cap to prevent starvation
    age_boost = min(age_seconds * PRIORITY_AGE_FACTOR, MAX_PRIORITY_BOOST)
    
    # Type adjustment: decode gets slight priority
    # This helps maintain throughput by completing in-progress requests
    type_adjustment = 0.0
    if request_type == "decode":
        type_adjustment = 5.0
    
    # Size adjustment: smaller requests get slight boost
    # This improves overall throughput (short job first)
    # We normalize by max_tokens to get a 0-10 range
    size_ratio = num_tokens / max_tokens if max_tokens > 0 else 0
    size_adjustment = (1 - size_ratio) * 10  # 0-10 range
    
    # Total priority (lower is higher priority)
    total = base_priority - age_boost - type_adjustment - size_adjustment
    
    return PriorityScore(
        total=total,
        base_priority=base_priority,
        age_boost=age_boost,
        type_adjustment=type_adjustment,
        size_adjustment=size_adjustment,
    )


def calculate_token_budget(
    total_budget: int,
    num_running: int,
    num_waiting: int,
    running_tokens: list[int],
    waiting_tokens: list[int],
    max_tokens_per_request: int | None = None,
) -> dict[str, int]:
    """
    Calculate token budget allocation for scheduling.
    
    This function determines how to split the total token budget between
    running requests (need tokens for decode) and waiting requests
    (need tokens for prefill).
    
    Allocation Strategy:
    -------------------
    1. Running requests get priority (to maintain throughput)
    2. Remaining budget goes to waiting requests
    3. Budget is split proportionally within each group
    
    Edge Cases:
    ----------
    - If total_budget is 0, all allocations are 0
    - If there are no requests, budget is unallocated
    - If a group has no tokens requested, it gets 0 budget
    
    Note on Empty Lists:
    -------------------
    We handle empty running_tokens and waiting_tokens by treating them
    as requiring 0 tokens total. This is correct because if there are
    no running/waiting requests, we don't need to allocate budget for them.
    
    Args:
        total_budget: Total tokens available for this step.
        num_running: Number of running requests.
        num_waiting: Number of waiting requests.
        running_tokens: Tokens needed per running request.
        waiting_tokens: Tokens needed per waiting request.
        max_tokens_per_request: Cap per request (optional).
    
    Returns:
        Dict with 'running_budget', 'waiting_budget', 'unallocated'.
    """
    if total_budget <= 0:
        return {
            "running_budget": 0,
            "waiting_budget": 0,
            "unallocated": 0,
        }
    
    # Calculate total tokens needed by each group
    total_running_needed = sum(running_tokens)
    total_waiting_needed = sum(waiting_tokens)
    
    # Handle the case where both groups need 0 tokens
    total_needed = total_running_needed + total_waiting_needed
    if total_needed == 0:
        return {
            "running_budget": 0,
            "waiting_budget": 0,
            "unallocated": total_budget,
        }
    
    # Priority to running requests (they're already in progress)
    running_budget = min(total_running_needed, total_budget)
    remaining = total_budget - running_budget
    
    # Remaining goes to waiting
    waiting_budget = min(total_waiting_needed, remaining)
    unallocated = remaining - waiting_budget
    
    return {
        "running_budget": running_budget,
        "waiting_budget": waiting_budget,
        "unallocated": unallocated,
    }


def should_preempt_request(
    running_priority: float,
    waiting_priority: float,
    running_tokens_computed: int,
    preemption_threshold: float = 10.0,
) -> bool:
    """
    Determine if a running request should be preempted.
    
    Preemption Policy:
    -----------------
    We preempt a running request if:
    1. There's a significantly higher priority waiting request
    2. The priority difference exceeds the threshold
    
    We are intentionally conservative with preemption because:
    - Preemption wastes the tokens already computed
    - Thrashing between requests hurts overall throughput
    - Users expect in-progress requests to complete
    
    The running_tokens_computed parameter is accepted but not used
    in the preemption decision. We considered using it to avoid
    preempting requests that are almost done, but decided against it
    because:
    - It's hard to know how many tokens remain
    - Priority should be the primary factor
    - Users set priorities knowing they matter
    
    Args:
        running_priority: Priority of the running request.
        waiting_priority: Priority of the waiting request.
        running_tokens_computed: Tokens already computed (unused).
        preemption_threshold: Minimum priority difference for preemption.
    
    Returns:
        True if running request should be preempted.
    """
    # Calculate priority difference (negative means waiting is higher priority)
    priority_diff = running_priority - waiting_priority
    
    # Preempt if waiting request has significantly higher priority
    # (remember: lower priority VALUE = higher priority)
    return priority_diff > preemption_threshold


def estimate_completion_tokens(
    prompt_tokens: int,
    max_new_tokens: int | None,
    average_output_ratio: float = 0.5,
) -> int:
    """
    Estimate total tokens a request will generate.
    
    This is used for capacity planning and scheduling decisions.
    The estimate is based on the observation that most completions
    don't reach max_new_tokens.
    
    Args:
        prompt_tokens: Number of tokens in the prompt.
        max_new_tokens: Maximum new tokens to generate.
        average_output_ratio: Expected fraction of max_new_tokens used.
    
    Returns:
        Estimated total tokens (prompt + completion).
    """
    if max_new_tokens is None:
        # No limit specified, use a reasonable default
        max_new_tokens = 256
    
    estimated_output = int(max_new_tokens * average_output_ratio)
    return prompt_tokens + estimated_output


def get_scheduling_weight(
    request_type: str,
    num_tokens: int,
    is_high_priority: bool = False,
) -> float:
    """
    Get the scheduling weight for a request.
    
    The weight is used to determine resource allocation proportions.
    Higher weights get more resources.
    
    Weight Calculation:
    ------------------
    - Base weight is 1.0
    - Decode requests get 2x weight (to finish faster)
    - High priority requests get 3x weight
    - Weights are NOT affected by token count (fair allocation)
    
    Note: We intentionally don't scale by token count because that
    would favor large requests, which hurts overall throughput.
    
    Args:
        request_type: "prefill" or "decode".
        num_tokens: Number of tokens (unused - see note).
        is_high_priority: Whether this is a high priority request.
    
    Returns:
        Scheduling weight (higher = more resources).
    """
    weight = 1.0
    
    if request_type == "decode":
        weight *= 2.0
    
    if is_high_priority:
        weight *= 3.0
    
    return weight


def calculate_fair_share(
    requests: list[dict[str, Any]],
    total_budget: int,
) -> dict[str, int]:
    """
    Calculate fair share of token budget for each request.
    
    Uses weighted fair queuing to allocate budget proportionally
    to scheduling weights.
    
    Args:
        requests: List of request dicts with 'id', 'weight', 'needed'.
        total_budget: Total tokens available.
    
    Returns:
        Dict mapping request ID to allocated tokens.
    """
    if not requests or total_budget <= 0:
        return {}
    
    # Calculate total weight
    total_weight = sum(r.get("weight", 1.0) for r in requests)
    
    # Handle edge case of zero total weight
    if total_weight == 0:
        # Equal split if no weights
        per_request = total_budget // len(requests)
        return {r["id"]: per_request for r in requests}
    
    # Allocate proportionally to weight
    allocations = {}
    remaining = total_budget
    
    for i, req in enumerate(requests):
        weight = req.get("weight", 1.0)
        needed = req.get("needed", float("inf"))
        
        # Calculate proportional share
        if i < len(requests) - 1:
            share = int(total_budget * weight / total_weight)
        else:
            # Last request gets remainder to avoid rounding errors
            share = remaining
        
        # Cap at what's needed
        allocation = min(share, needed, remaining)
        allocations[req["id"]] = allocation
        remaining -= allocation
    
    return allocations


# ============================================================
# Utility functions for formatting (for metrics/logging)
# ============================================================

def format_priority_score(score: PriorityScore) -> str:
    """Format a priority score for logging."""
    return (
        f"Priority: {score.total:.2f} "
        f"(base={score.base_priority}, age_boost={score.age_boost:.2f}, "
        f"type_adj={score.type_adjustment:.2f}, size_adj={score.size_adjustment:.2f})"
    )


def format_token_count(tokens: int) -> str:
    """
    Format token count for display.
    
    Uses K/M suffixes for large numbers.
    """
    if tokens < 1000:
        return str(tokens)
    elif tokens < 1000000:
        return f"{tokens / 1000:.1f}K"
    else:
        return f"{tokens / 1000000:.1f}M"


def format_scheduling_stats(
    num_running: int,
    num_waiting: int,
    budget_used: int,
    budget_total: int,
) -> str:
    """Format scheduling statistics for logging."""
    utilization = budget_used / budget_total * 100 if budget_total > 0 else 0
    return (
        f"Running: {num_running}, Waiting: {num_waiting}, "
        f"Budget: {format_token_count(budget_used)}/{format_token_count(budget_total)} "
        f"({utilization:.1f}%)"
    )


def ceiling_divide(a: int, b: int) -> int:
    """
    Divide a by b, rounding up.
    
    Useful for calculating number of blocks needed.
    """
    # Implement ceiling division without importing math
    return (a + b - 1) // b
