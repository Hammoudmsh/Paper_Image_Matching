#!/usr/bin/env python3
"""Download and safely extract multiple TUM RGB-D datasets without renaming files."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
import urllib.error
from pathlib import Path
import time

BASE_URL = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1"
BASE_URL2 = "https://cvg.cit.tum.de/rgbd/dataset/freiburg2"
BASE_URL3 = "https://cvg.cit.tum.de/rgbd/dataset/freiburg3"

DATASETS = [
    "rgbd_dataset_freiburg1_360",
    "rgbd_dataset_freiburg1_desk",
    "rgbd_dataset_freiburg1_desk2",
    "rgbd_dataset_freiburg1_floor",
    "rgbd_dataset_freiburg1_room",
    "rgbd_dataset_freiburg1_xyz",
    "rgbd_dataset_freiburg2_360_hemisphere",
    "rgbd_dataset_freiburg2_coke",
    "rgbd_dataset_freiburg2_dishes",
    "rgbd_dataset_freiburg2_flowerbouquet",
    "rgbd_dataset_freiburg2_flowerbouquet_brownbackground",
    "rgbd_dataset_freiburg2_metallic_sphere2",
    "rgbd_dataset_freiburg3_nostructure_texture_near_withloop",
    "rgbd_dataset_freiburg3_structure_notexture_near",
]


def get_url(dataset_name: str) -> str:
    """Get the download URL for a dataset based on its name."""
    if dataset_name.startswith("rgbd_dataset_freiburg1_"):
        return f"{BASE_URL}/{dataset_name}.tgz"
    elif dataset_name.startswith("rgbd_dataset_freiburg2_"):
        return f"{BASE_URL2}/{dataset_name}.tgz"
    elif dataset_name.startswith("rgbd_dataset_freiburg3_"):
        return f"{BASE_URL3}/{dataset_name}.tgz"
    else:
        raise ValueError(f"Unknown dataset prefix: {dataset_name}")


def safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    """Extract tar file safely, preventing path traversal attacks."""
    root = destination.resolve()
    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        if root not in target.parents and target != root:
            raise RuntimeError(f"Unsafe path in archive: {member.name}")
    tar.extractall(destination)


def check_url(url: str, timeout: int = 5) -> bool:
    """Check if a URL is accessible with a timeout."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return False
    except Exception:
        return False


def check_url_with_retry(url: str, max_retries: int = 2, timeout: int = 5) -> tuple[bool, str]:
    """Check URL with retry mechanism."""
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    return True, "OK"
                else:
                    return False, f"HTTP {response.status}"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False, "Not Found (404)"
            return False, f"HTTP Error {e.code}"
        except urllib.error.URLError as e:
            if attempt < max_retries:
                time.sleep(1)
                continue
            return False, f"Connection Error: {str(e.reason)}"
        except TimeoutError:
            if attempt < max_retries:
                time.sleep(1)
                continue
            return False, "Timeout"
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
                continue
            return False, f"Error: {str(e)}"
    return False, "Failed after retries"


def download_dataset(dataset_name: str, output: Path, keep_archive: bool = False, debug: bool = False) -> bool:
    """Download and extract a single dataset."""
    url = get_url(dataset_name)
    archive_name = f"{dataset_name}.tgz"
    archive = output / archive_name
    dataset = output / dataset_name
    
    # Check if already exists
    if dataset.exists():
        print(f"✓ {dataset_name} already exists at {dataset}")
        return True
    
    # Debug mode: just check URLs with better handling
    if debug:
        print(f"[DEBUG] Checking URL: {url}")
        is_valid, status = check_url_with_retry(url)
        if is_valid:
            print(f"  ✓ URL is valid")
            return True
        else:
            print(f"  ✗ URL check failed: {status}")
            return False
    
    # Download the dataset
    print(f"Downloading {dataset_name} from {url}")
    try:
        with urllib.request.urlopen(url, timeout=30) as response, archive.open("wb") as f:
            shutil.copyfileobj(response, f)
    except Exception as e:
        print(f"  ✗ Failed to download {dataset_name}: {e}")
        # Clean up partial download
        if archive.exists():
            archive.unlink(missing_ok=True)
        return False
    
    # Extract the archive
    print(f"Extracting {archive_name}")
    try:
        with tarfile.open(archive, "r:gz") as tar:
            safe_extract(tar, output)
    except Exception as e:
        print(f"  ✗ Failed to extract {dataset_name}: {e}")
        # Clean up corrupted archive
        if archive.exists():
            archive.unlink(missing_ok=True)
        return False
    
    # Clean up archive if not keeping
    if not keep_archive:
        archive.unlink(missing_ok=True)
    
    print(f"  ✓ Successfully downloaded and extracted {dataset_name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Download TUM RGB-D datasets")
    parser.add_argument("--output", default="data", help="Parent output directory")
    parser.add_argument("--keep-archive", action="store_true", help="Keep .tgz files after extraction")
    parser.add_argument("--debug", action="store_true", help="Debug mode: check URLs without downloading")
    parser.add_argument("--dataset", nargs="+", help="Specific datasets to download (default: all)")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout in seconds for URL checks (default: 10)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing datasets without checking")
    args = parser.parse_args()

    # List available datasets
    if args.list:
        print("Available datasets:")
        for i, ds in enumerate(DATASETS, 1):
            print(f"  {i:2d}. {ds}")
        return 0

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    
    # Select datasets to download
    datasets_to_download = args.dataset if args.dataset else DATASETS
    
    # Validate dataset names
    invalid_datasets = [ds for ds in datasets_to_download if ds not in DATASETS]
    if invalid_datasets:
        print(f"Error: Invalid dataset names: {', '.join(invalid_datasets)}")
        print("Use --list to see available datasets")
        return 1
    
    print(f"{'DEBUG MODE' if args.debug else 'Download mode'}")
    print(f"Output directory: {output}")
    print(f"Datasets to process: {len(datasets_to_download)}")
    print("-" * 50)
    
    # Process each dataset
    success_count = 0
    failure_count = 0
    skipped_count = 0
    
    for i, dataset_name in enumerate(datasets_to_download, 1):
        print(f"[{i}/{len(datasets_to_download)}] Processing {dataset_name}...")
        
        dataset_path = output / dataset_name
        if dataset_path.exists() and args.skip_existing:
            print(f"  ⏭ Skipping {dataset_name} (already exists)")
            skipped_count += 1
            print()
            continue
        
        if download_dataset(dataset_name, output, args.keep_archive, args.debug):
            success_count += 1
        else:
            failure_count += 1
        print()
    
    # Summary
    print("-" * 50)
    if args.debug:
        print(f"✓ Valid URLs: {success_count}")
        print(f"✗ Invalid URLs: {failure_count}")
        if skipped_count:
            print(f"⏭ Skipped (already exist): {skipped_count}")
    else:
        print(f"✓ Successfully downloaded: {success_count}")
        print(f"✗ Failed: {failure_count}")
        if skipped_count:
            print(f"⏭ Skipped (already exist): {skipped_count}")
        if failure_count > 0:
            print("\nSome datasets failed to download. Possible reasons:")
            print("  - The dataset doesn't exist on the server")
            print("  - Network connectivity issues")
            print("  - Server timeout (try increasing --timeout)")
            print("\nTip: Use --debug to check which URLs are valid")
    
    return 0 if success_count == len(datasets_to_download) else 1


if __name__ == "__main__":
    raise SystemExit(main())