"""Download the datasets referenced by the XYW-Net repo README.

This script fetches the RCF "augmented" edge datasets:
  - HED-BSDS.tar.gz
  - PASCAL.tar.gz
  - NYUD.tar.gz

These are the same links listed in:
  src/XYW-Net-original/XYW-Net-main/README.md

Notes:
  - Files are large; downloads may take a while.
  - BIPED and Multicue are hosted on Google Drive; this script does not automate those.
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request
import urllib.error
from pathlib import Path


DATASETS = {
    "HED-BSDS": "http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz",
    "PASCAL": "http://mftp.mmcheng.net/liuyun/rcf/data/PASCAL.tar.gz",
    "NYUD": "http://mftp.mmcheng.net/liuyun/rcf/data/NYUD.tar.gz",
}


DEFAULT_HEADERS = {
    # Some hosts return 406 unless a User-Agent is present.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _progress_hook(label: str):
    last_percent = {"v": -1}

    def hook(block_num: int, block_size: int, total_size: int):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        percent = int(min(100, downloaded * 100 / total_size))
        if percent != last_percent["v"] and percent % 2 == 0:
            last_percent["v"] = percent
            sys.stdout.write(f"\r{label}: {percent}%")
            sys.stdout.flush()

    return hook


def _download_stream(url: str, dest: Path, *, headers: dict[str, str]) -> None:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        total_size = int(resp.headers.get("Content-Length") or 0)
        downloaded = 0
        last_percent = -1
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = int(min(100, downloaded * 100 / total_size))
                    if percent != last_percent and percent % 2 == 0:
                        last_percent = percent
                        sys.stdout.write(f"\r{dest.name}: {percent}%")
                        sys.stdout.flush()
    if total_size > 0:
        sys.stdout.write("\n")


def download(url: str, dest: Path, *, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"[skip] {dest.name} already exists")
        return dest

    print(f"[dl] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    try:
        _download_stream(url, tmp, headers=DEFAULT_HEADERS)
    except urllib.error.HTTPError as e:
        # A few hosts respond with 406 for default Python urllib headers.
        # Retry once with a slightly different header set.
        if e.code == 406:
            print("\n[warn] HTTP 406 from host; retrying with alternate headers...")
            alt_headers = dict(DEFAULT_HEADERS)
            alt_headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            _download_stream(url, tmp, headers=alt_headers)
        else:
            raise
    tmp.replace(dest)
    return dest


def extract(tar_gz: Path, out_dir: Path, *, force: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / f".{tar_gz.stem}.extracted"
    if marker.exists() and not force:
        print(f"[skip] already extracted -> {out_dir}")
        return

    print(f"[extract] {tar_gz.name} -> {out_dir}")
    with tarfile.open(tar_gz, "r:gz") as tf:
        tf.extractall(path=out_dir)
    marker.write_text("ok", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Download XYW-Net referenced datasets (RCF links).")
    p.add_argument(
        "--out",
        type=str,
        default=str(Path("datasets") / "RCF"),
        help="Output folder for downloads/extraction (default: datasets/RCF)",
    )
    p.add_argument(
        "--download-only",
        action="store_true",
        help="Only download archives, do not extract",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download / re-extract even if files exist",
    )
    args = p.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    archives_dir = out_root / "archives"
    extract_dir = out_root / "extracted"

    for name, url in DATASETS.items():
        archive = archives_dir / f"{name}.tar.gz"
        download(url, archive, force=args.force)
        if not args.download_only:
            extract(archive, extract_dir, force=args.force)

    print("\nDone.")
    print(f"Archives:  {archives_dir}")
    if not args.download_only:
        print(f"Extracted: {extract_dir}")
        print("\nTip: point your training code to the extracted dataset folders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
