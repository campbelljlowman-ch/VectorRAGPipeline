from __future__ import annotations
from data_loader import DataLoader
from typing import Optional, Tuple, Union
from io import BytesIO
import re

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError



def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    """
    Parse s3://bucket/key URIs into (bucket, key).
    """
    m = re.match(r"^s3://([^/]+)/?(.*)$", uri)
    if not m:
        raise ValueError(f"Invalid S3 URI: {uri!r}. Expected format s3://bucket/key")
    bucket, key = m.group(1), m.group(2)
    if not key:
        raise ValueError(f"S3 URI missing object key: {uri!r}")
    return bucket, key


class S3DataLoader(DataLoader):
    """
    Load objects from Amazon S3.

    Credentials are resolved by boto3's standard chain (env vars, profile, IMDS/role, etc.)
    unless you supply a specific profile/region.

    Args:
        s3_url: s3://bucket/key
        aws_profile: Optional named profile in ~/.aws/credentials
        region: Optional AWS region override
        client: Optional preconstructed boto3 S3 client (overrides profile/region)
    """
    def __init__(
        self,
        s3_url: str,
        aws_profile: Optional[str] = None,
        region: Optional[str] = None,
        client=None,
    ):
        super().__init__(s3_url)
        self.bucket, self.key = _parse_s3_uri(s3_url)
        if client is not None:
            self.s3_client = client
        else:
            session_kwargs = {}
            if aws_profile:
                session_kwargs["profile_name"] = aws_profile
            if region:
                session_kwargs["region_name"] = region
            session = boto3.Session(**session_kwargs)
            self.s3_client = session.client(
                "s3",
                config=Config(
                    retries={"max_attempts": 10, "mode": "standard"},
                    read_timeout=60,
                    connect_timeout=10,
                ),
            )

    def get_next_chunk(
        self,
        *,
        as_text: bool = False,
        encoding: str = "utf-8",
        byte_range: Optional[Tuple[int, int]] = None,
        version_id: Optional[str] = None,
    ) -> Union[bytes, str]:
        """
        Fetch the object (or byte range) from S3.

        Args:
            as_text: If True, return a decoded string; else return bytes.
            encoding: Text encoding used if as_text=True.
            byte_range: Optional (start, end_inclusive) for ranged get.
            version_id: Optional S3 VersionId to retrieve a specific version.

        Returns:
            Bytes or string content.
        """
        try:
            kwargs = {"Bucket": self.bucket, "Key": self.key}
            if version_id:
                kwargs["VersionId"] = version_id
            if byte_range:
                start, end = byte_range
                kwargs["Range"] = f"bytes={start}-{end}"

            resp = self.s3_client.get_object(**kwargs)
            data = resp["Body"].read()
            return data.decode(encoding) if as_text else data
        except (ClientError, BotoCoreError) as e:
            raise RuntimeError(f"Failed to load S3 object {self.source}: {e}") from e
