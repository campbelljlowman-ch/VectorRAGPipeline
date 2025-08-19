from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional
import zipfile

import boto3
from botocore.config import Config
from boto3.s3.transfer import TransferConfig, S3Transfer


class S3BatchDownloader:
    """
    Iterate through keys in a large S3 bucket and download them in chunks.

    Progress is tracked in-memory and (optionally) persisted to a JSON state file
    so you can resume later without re-listing the whole bucket.

    Typical usage:
        dl = S3BatchDownloader(
            bucket="my-bucket",
            dest_dir="data/",
            prefix="logs/2025/",
            chunk_size=50,
            state_path=".s3_progress.json",
        )

        while True:
            files = dl.get_next_chunk()  # downloads next N objects
            if not files:                # empty => you're done
                break
            print(f"Downloaded {len(files)}: {files[:3]}...")

    Notes:
      - Objects are visited in lexicographic key order.
      - By default, already-existing local files are skipped (size+mtime check disabled
        for speed; override by passing `skip_existing=False`).
    """

    def __init__(
        self,
        bucket: str,
        dest_dir: str,
        prefix: Optional[str] = None,
        chunk_size: int = 20,
        state_path: Optional[str] = None,
        s3_client=None,
        page_size: int = 1000,
        skip_existing: bool = True,
        aws_region: Optional[str] = None,
        max_concurrency: int = 10,
        multipart_threshold_mb: int = 8,
        multipart_chunksize_mb: int = 8,
        request_payer: Optional[str] = None,
        extract_pdf_from_zip: bool = True,
    ):
        self.bucket = bucket
        self.prefix = prefix
        self.dest_dir = Path(dest_dir)
        self.dest_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_size = max(1, int(chunk_size))
        self.page_size = max(1, int(page_size))
        self.skip_existing = bool(skip_existing)
        self.state_path = Path(state_path) if state_path else None
        self.request_payer = request_payer
        self.extract_pdf_from_zip = extract_pdf_from_zip

        if s3_client is None:
            cfg = Config(region_name=aws_region) if aws_region else Config()
            s3_client = boto3.client("s3", config=cfg)
        self.s3 = s3_client
        self.transfer = S3Transfer(
            self.s3,
            config=TransferConfig(
                max_concurrency=max_concurrency,
                multipart_threshold=multipart_threshold_mb * 1024 * 1024,
                multipart_chunksize=multipart_chunksize_mb * 1024 * 1024,
                use_threads=True,
            ),
        )

        self._paginator = self.s3.get_paginator("list_objects_v2")
        self._continuation_token: Optional[str] = None
        self._keys_buffer: List[str] = []
        self._exhausted = False
        self._listed_count = 0
        self._downloaded_count = 0

        if self.state_path and self.state_path.exists():
            self._load_state()

    def get_next_chunk(self, max_files: Optional[int] = None) -> List[str]:
        """
        Download the next batch of objects and return local file paths.
        For .zip objects (when extract_pdf_from_zip=True), returns the extracted
        PDF path if found, otherwise the .zip path.
        Returns [] when there is nothing left.
        """
        if self._exhausted:
            return []

        target = max(1, max_files or self.chunk_size)
        local_paths: List[str] = []

        while len(self._keys_buffer) < target and not self._exhausted:
            self._refill_keys_buffer()

        if not self._keys_buffer:
            self._exhausted = True
            self._save_state()
            return []

        to_take = min(target, len(self._keys_buffer))
        keys = [self._keys_buffer.pop(0) for _ in range(to_take)]

        for key in keys:
            local_path = self._local_path_for_key(key)

            # If we’re extracting PDF from a zip, and the extracted PDF already exists,
            # we can short-circuit on skip_existing.
            if (
                self.extract_pdf_from_zip
                and key.lower().endswith(".zip")
            ):
                preexisting_pdf = self._expected_extracted_pdf_path(local_path)
                if self.skip_existing and preexisting_pdf and preexisting_pdf.exists():
                    local_paths.append(str(preexisting_pdf))
                    continue

            if self.skip_existing and local_path.exists():
                # We already have the zip/file. If it's a zip and extraction is enabled,
                # make sure a PDF is extracted/returned.
                if self.extract_pdf_from_zip and key.lower().endswith(".zip"):
                    extracted = self._maybe_extract_pdf(local_path)
                    local_paths.append(str(extracted or local_path))
                else:
                    local_paths.append(str(local_path))
                continue

            local_path.parent.mkdir(parents=True, exist_ok=True)
            extra_args = {}
            if self.request_payer:
                extra_args["RequestPayer"] = self.request_payer

            self.transfer.download_file(
                bucket=self.bucket,
                key=key,
                filename=str(local_path),
                extra_args=extra_args or None,
            )

            if self.extract_pdf_from_zip and key.lower().endswith(".zip"):
                extracted = self._maybe_extract_pdf(local_path)
                if extracted:
                    self._downloaded_count += 1
                    print(f"Extracted PDF from {key} to {extracted}")
                    local_paths.append([str(extracted), key])

        self._save_state()
        return local_paths

    def done(self) -> bool:
        return self._exhausted

    def stats(self) -> dict:
        return {
            "listed_keys": self._listed_count,
            "downloaded": self._downloaded_count,
            "buffered": len(self._keys_buffer),
            "exhausted": self._exhausted,
            "continuation_token_present": self._continuation_token is not None,
        }

    def remove_data(self) -> None:
        """Remove all downloaded data and reset state."""
        for path in self.dest_dir.glob("**/*"):
            if path.is_file():
                path.unlink()

    def reset_state(self) -> None:
        """Reset the downloader state, keeping the S3 connection."""
        self._continuation_token = None
        self._keys_buffer.clear()
        self._listed_count = 0
        self._downloaded_count = 0
        self._exhausted = False
        if self.state_path and self.state_path.exists():
            self.state_path.unlink()

    # ---------- Internals ----------

    def _refill_keys_buffer(self) -> None:
        if self._exhausted:
            return

        params = {
            "Bucket": self.bucket,
            "Prefix": self.prefix,
            "PaginationConfig": {"PageSize": self.page_size},
        }
        if self._continuation_token:
            params["ContinuationToken"] = self._continuation_token  # type: ignore

        page_iter = self._paginator.paginate(**{k: v for k, v in params.items() if v is not None})

        try:
            page = next(iter(page_iter))
        except StopIteration:
            self._exhausted = True
            return

        contents = page.get("Contents", [])
        keys = [obj["Key"] for obj in contents]
        self._keys_buffer.extend(keys)
        self._listed_count += len(keys)

        if page.get("IsTruncated"):
            self._continuation_token = page.get("NextContinuationToken")
        else:
            self._continuation_token = None
            if not keys:
                self._exhausted = True

    def _local_path_for_key(self, key: str) -> Path:
        return self.dest_dir / key.lstrip("/")

    # ---------- Zip/PDF helpers ----------

    def _expected_extracted_pdf_path(self, zip_local_path: Path) -> Optional[Path]:
        """
        Predict where the first PDF would be extracted (based on existing zip content),
        returning None if the zip isn’t present yet.
        """
        if not zip_local_path.exists():
            return None
        try:
            with zipfile.ZipFile(zip_local_path) as zf:
                pdf_name = next(
                    (n for n in zf.namelist() if n.lower().endswith(".pdf") and not n.endswith("/")),
                    None,
                )
                if pdf_name is None:
                    return None
                return self._safe_extract_path(zip_local_path, pdf_name)
        except zipfile.BadZipFile:
            return None

    def _maybe_extract_pdf(self, zip_local_path: Path) -> Optional[Path]:
        """
        If the zip contains at least one PDF, extract the first one into
        <zip_local_path>.d/ (i.e., a folder named after the zip) and return the PDF path.
        If no PDF is present, return None.
        """
        try:
            with zipfile.ZipFile(zip_local_path) as zf:
                # Find first non-directory PDF entry
                pdf_name = next(
                    (n for n in zf.namelist() if n.lower().endswith(".pdf") and not n.endswith("/")),
                    None,
                )
                if pdf_name is None:
                    return None

                out_path = self._safe_extract_path(zip_local_path, pdf_name)
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # If skipping existing and file is already there, don't re-extract
                if self.skip_existing and out_path.exists():
                    return out_path

                # Extract only the chosen PDF securely (avoid path traversal)
                with zf.open(pdf_name) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
                return out_path
        except zipfile.BadZipFile:
            return None

    def _safe_extract_path(self, zip_local_path: Path, member_name: str) -> Path:
        """
        Compute a safe destination path for a member in the zip to avoid zip-slip.
        Extract into a folder next to the zip: <zip>.d/<member_name_basename>
        (we do not preserve internal directories for the single-PDF use case).
        """
        # Put extracted files under "<zip>.d/"
        extract_root = Path(str(zip_local_path) + ".d")
        target = extract_root / Path(member_name).name

        # Ensure the resolved path stays under dest_dir
        resolved = target.resolve()
        dest_root = self.dest_dir.resolve()
        if dest_root not in resolved.parents and resolved != dest_root:
            # If someone tried to traverse outside, fall back to a safe basename location
            resolved = (dest_root / Path(member_name).name).resolve()
        return resolved

    # ---------- State persistence ----------

    def _save_state(self) -> None:
        if not self.state_path:
            return
        state = {
            "bucket": self.bucket,
            "prefix": self.prefix,
            "dest_dir": str(self.dest_dir),
            "continuation_token": self._continuation_token,
            "keys_buffer": self._keys_buffer,
            "listed_count": self._listed_count,
            "downloaded_count": self._downloaded_count,
            "exhausted": self._exhausted,
            "chunk_size": self.chunk_size,
            "page_size": self.page_size,
            "skip_existing": self.skip_existing,
            "request_payer": self.request_payer,
            "extract_pdf_from_zip": self.extract_pdf_from_zip,
        }
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        os.replace(tmp, self.state_path)

    def _load_state(self) -> None:
        try:
            data = json.loads(self.state_path.read_text())
        except Exception:
            return
        if (
            data.get("bucket") == self.bucket
            and data.get("prefix") == self.prefix
            and Path(data.get("dest_dir", "")) == self.dest_dir
        ):
            self._continuation_token = data.get("continuation_token")
            self._keys_buffer = list(data.get("keys_buffer", []))
            self._listed_count = int(data.get("listed_count", 0))
            self._downloaded_count = int(data.get("downloaded_count", 0))
            self._exhausted = bool(data.get("exhausted", False))
