"""
ARIES — S3/MinIO client for model artifact retrieval.

Uses aioboto3 for async S3 operations. Downloads ONNX model files and
tokenizer directories from MinIO at startup.
"""

from __future__ import annotations

import os
from pathlib import Path

import aioboto3
from botocore.config import Config as BotoConfig

from src.shared.config import ServiceSettings
from src.shared.logging import get_logger

log = get_logger("s3_client")


class S3Client:
    """Async MinIO/S3 wrapper for model artifact management."""

    def __init__(self, settings: ServiceSettings) -> None:
        self._settings = settings
        self._session = aioboto3.Session()
        self._client_kwargs = {
            "service_name": "s3",
            "endpoint_url": settings.s3_endpoint_url,
            "aws_access_key_id": settings.s3_access_key,
            "aws_secret_access_key": settings.s3_secret_key,
            "region_name": settings.s3_region,
            "config": BotoConfig(signature_version="s3v4"),
        }

    async def download_file(self, s3_key: str, local_path: Path) -> Path:
        """Download a single file from S3 to a local path."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._session.client(**self._client_kwargs) as client:
            log.info("downloading_s3_object", bucket=self._settings.s3_bucket_models, key=s3_key)
            await client.download_file(
                Bucket=self._settings.s3_bucket_models,
                Key=s3_key,
                Filename=str(local_path),
            )
        log.info("downloaded_s3_object", key=s3_key, local_path=str(local_path))
        return local_path

    async def download_prefix(self, s3_prefix: str, local_dir: Path) -> Path:
        """Download all objects under an S3 prefix into a local directory."""
        local_dir.mkdir(parents=True, exist_ok=True)
        async with self._session.client(**self._client_kwargs) as client:
            paginator = client.get_paginator("list_objects_v2")
            async for page in paginator.paginate(
                Bucket=self._settings.s3_bucket_models,
                Prefix=s3_prefix,
            ):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    relative = key[len(s3_prefix) :].lstrip("/")
                    if not relative:
                        continue
                    dest = local_dir / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    log.info("downloading_s3_object", key=key, dest=str(dest))
                    await client.download_file(
                        Bucket=self._settings.s3_bucket_models,
                        Key=key,
                        Filename=str(dest),
                    )
        log.info("downloaded_s3_prefix", prefix=s3_prefix, local_dir=str(local_dir))
        return local_dir

    async def ensure_bucket(self) -> None:
        """Create the models bucket if it does not exist."""
        async with self._session.client(**self._client_kwargs) as client:
            try:
                await client.head_bucket(Bucket=self._settings.s3_bucket_models)
            except Exception:
                log.info("creating_s3_bucket", bucket=self._settings.s3_bucket_models)
                await client.create_bucket(Bucket=self._settings.s3_bucket_models)

    async def upload_file(self, local_path: Path, s3_key: str) -> str:
        """Upload a local file to S3. Returns the S3 URI."""
        async with self._session.client(**self._client_kwargs) as client:
            await client.upload_file(
                Filename=str(local_path),
                Bucket=self._settings.s3_bucket_models,
                Key=s3_key,
            )
        uri = f"s3://{self._settings.s3_bucket_models}/{s3_key}"
        log.info("uploaded_s3_object", key=s3_key, uri=uri)
        return uri

    async def list_keys(self, prefix: str) -> list[str]:
        """List all object keys under a prefix."""
        keys: list[str] = []
        async with self._session.client(**self._client_kwargs) as client:
            paginator = client.get_paginator("list_objects_v2")
            async for page in paginator.paginate(
                Bucket=self._settings.s3_bucket_models,
                Prefix=prefix,
            ):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
        return keys
