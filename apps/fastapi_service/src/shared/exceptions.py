"""
ARIES — Domain exceptions mapped to HTTP status codes.
"""

from __future__ import annotations

from fastapi import HTTPException, status


class AriesBaseError(Exception):
    """Base for all ARIES domain errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail: str = "Internal server error"

    def to_http(self) -> HTTPException:
        return HTTPException(status_code=self.status_code, detail=self.detail)


class TenantNotFoundError(AriesBaseError):
    status_code = status.HTTP_404_NOT_FOUND
    detail = "Tenant not found"

    def __init__(self, tenant_id: str) -> None:
        self.detail = f"Tenant '{tenant_id}' not found"
        super().__init__(self.detail)


class ModelNotLoadedError(AriesBaseError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    detail = "Model not loaded; service not ready"

    def __init__(self, pipeline: str) -> None:
        self.detail = f"Model for pipeline '{pipeline}' is not loaded"
        super().__init__(self.detail)


class InvalidPayloadError(AriesBaseError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY

    def __init__(self, reason: str) -> None:
        self.detail = f"Invalid payload: {reason}"
        super().__init__(self.detail)


class TenantIsolationViolation(AriesBaseError):
    status_code = status.HTTP_403_FORBIDDEN
    detail = "Tenant isolation violation"


class S3DownloadError(AriesBaseError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    def __init__(self, key: str, reason: str) -> None:
        self.detail = f"Failed to download '{key}' from S3: {reason}"
        super().__init__(self.detail)
