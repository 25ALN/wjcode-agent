"""Compatibility import for ``core.permission.web``."""

from core.permission.web import (
    PendingPermissionRequest,
    PermissionPending,
    WebPermissionController,
)

__all__ = [
    "PermissionPending",
    "PendingPermissionRequest",
    "WebPermissionController",
]

