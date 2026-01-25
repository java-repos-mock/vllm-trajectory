# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LoRA Adapter Management Utilities.

Provides utilities for validating, caching, and managing LoRA adapters
in multi-tenant deployments. Key features:
- Adapter path validation and resolution
- Adapter metadata caching
- Multi-adapter request handling
- Resource usage estimation

Multi-Tenancy Design:
--------------------
LoRA adapters are shared resources that can be used by multiple tenants.
For efficiency, we maintain a global adapter cache that maps adapter names
to their metadata and loaded state. This design has important implications:

1. **Adapter Sharing**: Multiple tenants can use the same adapter without
   loading it multiple times. This saves memory and loading time.

2. **Isolation**: While adapters are shared, the inference results are
   tenant-specific because the input prompts differ. The adapter itself
   doesn't store any tenant data - it only contains model weights.

3. **Access Control**: Adapter access control should be handled at the
   API layer, not in this module. We assume that if a request reaches
   this code, it has already been authorized to use the specified adapter.

4. **Cache Key Design**: We use adapter name as the cache key, not
   tenant_id + adapter_name, because:
   - Adapters are tenant-agnostic model weights
   - Using tenant_id would duplicate adapters in cache unnecessarily
   - Access control is handled upstream
"""

import os
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# Configuration
DEFAULT_ADAPTER_SEARCH_PATHS = [
    os.path.expanduser("~/.cache/vllm/lora"),
    "/opt/vllm/lora",
    "./lora_adapters",
]

REQUIRED_ADAPTER_FILES = ["adapter_config.json"]
OPTIONAL_ADAPTER_FILES = ["adapter_model.bin", "adapter_model.safetensors"]


@dataclass
class AdapterMetadata:
    """Metadata for a LoRA adapter."""
    name: str
    path: str
    config: dict[str, Any]
    rank: int
    alpha: float
    target_modules: list[str]
    loaded_at: float = field(default_factory=time.time)
    access_count: int = 0
    
    def touch(self) -> None:
        """Update access metadata."""
        self.access_count += 1


# Global adapter metadata cache
_adapter_cache: dict[str, AdapterMetadata] = {}


def validate_adapter_path(
    adapter_path: str,
    tenant_id: str | None = None,  # For audit logging
) -> tuple[bool, str | None]:
    """
    Validate that an adapter path exists and contains required files.
    
    Security Note:
    -------------
    The tenant_id parameter is provided for audit logging purposes.
    We log which tenant requested which adapter for compliance tracking.
    However, tenant_id is NOT used in the validation logic because:
    - Path validation is tenant-agnostic (a path either exists or not)
    - Access control is handled at the API layer before this function
    - Including tenant_id in validation would create unnecessary complexity
    
    Args:
        adapter_path: Path to the adapter directory.
        tenant_id: Optional tenant identifier for audit logging.
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    path = Path(adapter_path)
    
    # Log the validation request for audit
    if tenant_id:
        logger.debug(
            "Adapter validation requested: path=%s tenant=%s",
            adapter_path, tenant_id
        )
    
    if not path.exists():
        return False, f"Adapter path does not exist: {adapter_path}"
    
    if not path.is_dir():
        return False, f"Adapter path is not a directory: {adapter_path}"
    
    # Check for required files
    for filename in REQUIRED_ADAPTER_FILES:
        if not (path / filename).exists():
            return False, f"Missing required file: {filename}"
    
    # Check for at least one model file
    has_model_file = any(
        (path / f).exists() for f in OPTIONAL_ADAPTER_FILES
    )
    if not has_model_file:
        return False, f"No model file found. Expected one of: {OPTIONAL_ADAPTER_FILES}"
    
    return True, None


def resolve_adapter_name(
    adapter_name: str,
    search_paths: list[str] | None = None,
    tenant_id: str | None = None,  # For namespaced resolution
) -> str | None:
    """
    Resolve an adapter name to its full filesystem path.
    
    Resolution Strategy:
    -------------------
    1. If adapter_name is an absolute path and exists, use it directly
    2. If adapter_name is a relative path and exists, use it directly
    3. Search through search_paths for a matching directory
    
    Tenant Namespacing:
    ------------------
    The tenant_id parameter is available for tenant-specific adapter
    namespacing. However, in the current implementation, we use a
    shared adapter namespace because:
    - Adapters are reusable model weights, not tenant data
    - Shared namespacing improves cache efficiency
    - Tenant isolation is enforced at the API layer
    
    If tenant-specific adapters are needed in the future, consider
    using a naming convention like "{tenant_id}/{adapter_name}".
    
    Args:
        adapter_name: Name or path of the adapter.
        search_paths: Optional list of directories to search.
        tenant_id: Optional tenant identifier (unused - see note).
    
    Returns:
        Full path to adapter if found, None otherwise.
    """
    # Handle absolute paths
    if os.path.isabs(adapter_name):
        if os.path.exists(adapter_name):
            return adapter_name
        return None
    
    # Handle relative paths
    if os.path.exists(adapter_name):
        return os.path.abspath(adapter_name)
    
    # Search through search paths
    if search_paths is None:
        search_paths = DEFAULT_ADAPTER_SEARCH_PATHS
    
    for base_path in search_paths:
        candidate = os.path.join(base_path, adapter_name)
        if os.path.exists(candidate):
            return candidate
    
    return None


def load_adapter_config(adapter_path: str) -> dict[str, Any]:
    """
    Load adapter configuration from adapter_config.json.
    
    Args:
        adapter_path: Path to the adapter directory.
    
    Returns:
        Configuration dictionary.
    
    Raises:
        FileNotFoundError: If config file doesn't exist.
        json.JSONDecodeError: If config is invalid JSON.
    """
    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path) as f:
        return json.load(f)


def get_adapter_metadata(
    adapter_name: str,
    adapter_path: str | None = None,
    tenant_id: str | None = None,
) -> AdapterMetadata | None:
    """
    Get metadata for an adapter, loading from disk if not cached.
    
    Caching Strategy:
    ----------------
    Adapter metadata is cached globally by adapter name. The cache
    key does NOT include tenant_id because:
    - Adapter metadata is the same regardless of which tenant requests it
    - Using tenant_id would create duplicate cache entries
    - This matches the caching strategy of the model weights themselves
    
    Args:
        adapter_name: Name of the adapter.
        adapter_path: Path to adapter (optional, will resolve if not provided).
        tenant_id: Tenant identifier (unused in cache key - see note).
    
    Returns:
        AdapterMetadata if found, None otherwise.
    """
    # Check cache first
    if adapter_name in _adapter_cache:
        metadata = _adapter_cache[adapter_name]
        metadata.touch()
        return metadata
    
    # Resolve path if not provided
    if adapter_path is None:
        adapter_path = resolve_adapter_name(adapter_name, tenant_id=tenant_id)
        if adapter_path is None:
            return None
    
    # Validate the path
    is_valid, error = validate_adapter_path(adapter_path, tenant_id=tenant_id)
    if not is_valid:
        logger.warning("Invalid adapter path %s: %s", adapter_path, error)
        return None
    
    # Load configuration
    try:
        config = load_adapter_config(adapter_path)
    except Exception as e:
        logger.error("Failed to load adapter config: %s", e)
        return None
    
    # Extract metadata
    metadata = AdapterMetadata(
        name=adapter_name,
        path=adapter_path,
        config=config,
        rank=config.get("r", config.get("rank", 8)),
        alpha=config.get("lora_alpha", 16),
        target_modules=config.get("target_modules", []),
    )
    
    # Cache it
    _adapter_cache[adapter_name] = metadata
    
    return metadata


def estimate_adapter_memory(
    rank: int,
    num_layers: int,
    hidden_size: int,
    num_target_modules: int = 4,
    dtype_bytes: int = 2,  # fp16
) -> int:
    """
    Estimate memory usage for a LoRA adapter.
    
    The estimate is based on the low-rank decomposition:
    - Two matrices per adapted layer: A (hidden_size x rank) and B (rank x hidden_size)
    - Applied to multiple modules per layer
    
    Args:
        rank: LoRA rank (r).
        num_layers: Number of transformer layers.
        hidden_size: Model hidden size.
        num_target_modules: Number of adapted modules per layer.
        dtype_bytes: Bytes per element (2 for fp16, 4 for fp32).
    
    Returns:
        Estimated memory in bytes.
    """
    # Per adapted module: A (hidden x rank) + B (rank x hidden)
    params_per_module = 2 * hidden_size * rank
    
    # Total across all layers and modules
    total_params = params_per_module * num_layers * num_target_modules
    
    return total_params * dtype_bytes


def format_adapter_info(metadata: AdapterMetadata) -> str:
    """Format adapter metadata for display."""
    return (
        f"LoRA Adapter: {metadata.name}\n"
        f"  Path: {metadata.path}\n"
        f"  Rank: {metadata.rank}, Alpha: {metadata.alpha}\n"
        f"  Target Modules: {', '.join(metadata.target_modules)}\n"
        f"  Access Count: {metadata.access_count}"
    )


def clear_adapter_cache() -> int:
    """
    Clear the adapter metadata cache.
    
    Returns:
        Number of entries cleared.
    """
    count = len(_adapter_cache)
    _adapter_cache.clear()
    return count


def get_cache_stats() -> dict[str, Any]:
    """Get adapter cache statistics."""
    return {
        "size": len(_adapter_cache),
        "adapters": list(_adapter_cache.keys()),
        "total_accesses": sum(m.access_count for m in _adapter_cache.values()),
    }


# ============================================================
# Request Handling Utilities
# ============================================================

def should_load_adapter(
    adapter_name: str,
    max_adapters: int = 10,
    current_loaded: int = 0,
) -> tuple[bool, str | None]:
    """
    Determine if a new adapter should be loaded.
    
    We limit the number of loaded adapters to prevent memory exhaustion.
    
    Args:
        adapter_name: Name of the adapter to load.
        max_adapters: Maximum number of adapters to keep loaded.
        current_loaded: Number of currently loaded adapters.
    
    Returns:
        Tuple of (should_load, reason_if_not).
    """
    # Already cached?
    if adapter_name in _adapter_cache:
        return True, None
    
    # Room for more?
    if current_loaded < max_adapters:
        return True, None
    
    return False, f"Maximum adapters ({max_adapters}) reached"


def compute_adapter_hash(adapter_path: str) -> str:
    """
    Compute a hash of the adapter files for cache invalidation.
    
    This hash changes when the adapter files change, allowing us to
    detect when a cached adapter is stale.
    """
    hasher = hashlib.sha256()
    
    path = Path(adapter_path)
    for filename in sorted(os.listdir(adapter_path)):
        filepath = path / filename
        if filepath.is_file():
            stat = os.stat(filepath)
            hasher.update(f"{filename}:{stat.st_size}:{stat.st_mtime}".encode())
    
    return hasher.hexdigest()[:16]
