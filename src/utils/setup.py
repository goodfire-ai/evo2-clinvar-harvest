"""Unified setup system for dataset downloads and builds."""

import argparse
import gzip
import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path

from loguru import logger

# --- Path utilities ---


def ensure_dir(path: str | Path, parents: bool = False) -> Path:
    path = Path(path)
    path.mkdir(exist_ok=True, parents=parents)
    return path


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# --- Download utilities ---


def download_file(url: str, dest: Path, compressed: bool = False) -> None:
    """Download file with optional .gz decompression.

    Args:
        url: URL to download from
        dest: Destination path
        compressed: If True, decompress .gz after download
    """
    # Add User-Agent header to avoid 403 Forbidden errors
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
    req = urllib.request.Request(url, headers=headers)

    raw_dest = dest.with_suffix(dest.suffix + ".gz") if compressed else dest
    with urllib.request.urlopen(req) as response, open(raw_dest, "wb") as out_file:
        shutil.copyfileobj(response, out_file)

    if compressed:
        with gzip.open(raw_dest, "rb") as f_in, open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        raw_dest.unlink()


# --- Setup system ---


def _cached_build(output: Path, builder: Callable, action: str, refresh: bool) -> None:
    """Build if output doesn't exist (or refresh forced)."""
    if output.exists() and not refresh:
        logger.info(f"Using cached: {output}")
        return
    ensure_parent(output)
    logger.info(action)
    builder()
    logger.info(f"Saved {output}")


def run_setup(
    name: str,
    description: str,
    *,
    downloads: tuple[tuple[str, Path, bool], ...] = (),
    custom_downloads: tuple[tuple[str, Callable], ...] = (),
    sources: dict[str, tuple[Path, Callable]] | None = None,
    presets: dict[str, tuple[Path, Callable]] | None = None,
    extract_fn: Callable | None = None,
) -> None:
    """Declarative dataset setup: download, build sources/presets, optionally extract.

    Args:
        name: Dataset name (for logging)
        description: CLI help text
        downloads: (url, dest, compressed) tuples for URL downloads
        custom_downloads: (name, callable) tuples for custom download logic
        sources: {name: (output_path, builder)} for source datasets (always built)
        presets: {name: (output_path, builder)} for preset datasets (built on request)
        extract_fn: Optional function(model, presets) for activation extraction

    Example:
        >>> run_setup(
        ...     "clinvar", "Download and build ClinVar datasets",
        ...     downloads=((vcf_url, vcf_path, False),),
        ...     sources={"variants": (path, lambda: parse_vcf(path))},
        ...     presets={name: (path, lambda n=name: metadata(n)) for name in PRESETS},
        ...     extract_fn=extract_activations,
        ... )
    """
    sources = sources or {}
    presets = presets or {}

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("presets", nargs="*", help="Presets to build (default: source only)")
    parser.add_argument("--refresh", action="store_true", help="Force redownload and rebuild")
    parser.add_argument("--extract", type=str, metavar="MODEL",
                        help="Extract activations with MODEL (e.g., ntv3-650m-post, evo2-7b)")
    args = parser.parse_args()
    targets = tuple(args.presets)

    # Downloads
    for url, dest, compressed in downloads:
        if dest.exists() and not args.refresh:
            logger.info(f"Using cached: {dest}")
            continue
        ensure_parent(dest)
        logger.info(f"Downloading: {url}")
        download_file(url, dest, compressed)
        logger.info(f"Saved {dest}")

    for dl_name, downloader in custom_downloads:
        logger.info(f"Running custom download: {dl_name}")
        downloader()

    # Source builds (always)
    for src_name, (output, builder) in sources.items():
        _cached_build(output, builder, f"Building {src_name}...", args.refresh)

    # Preset builds (only if requested)
    for target in targets:
        if target not in presets:
            raise ValueError(f"Unknown preset: {target}. Available: {list(presets.keys())}")
        output, builder = presets[target]
        _cached_build(output, builder, f"Building preset {target}...", args.refresh)

    # Extraction
    if args.extract:
        logger.info(f"Extracting activations with {args.extract}...")
        extract_fn(model=args.extract, presets=targets)
