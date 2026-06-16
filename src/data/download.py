"""
Download Wu et al. (2025) China Coastal GNSS PWV dataset from Zenodo.
"""
import os
import zipfile
import requests
from tqdm import tqdm


ZENODO_URL = "https://zenodo.org/records/17012498/files/Final_pwv.zip"
STATIONS_URL = "https://zenodo.org/records/17012498/files/CGN_sites.txt"


def download_file(url: str, dest_path: str, chunk_size: int = 8192):
    """Download a file with progress bar."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    if os.path.exists(dest_path):
        print(f"[Download] File already exists: {dest_path}")
        return
    
    print(f"[Download] Fetching {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    
    with open(dest_path, "wb") as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"[Download] Saved to {dest_path}")


def unzip_file(zip_path: str, extract_to: str):
    """Unzip archive."""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    
    print(f"[Unzip] Extracted {zip_path} -> {extract_to}")


def download_all(raw_dir: str):
    """Download and extract all dataset files."""
    os.makedirs(raw_dir, exist_ok=True)
    
    # Download PWV data
    pwv_zip = os.path.join(raw_dir, "Final_pwv.zip")
    download_file(ZENODO_URL, pwv_zip)
    
    # Download stations info
    stations_file = os.path.join(raw_dir, "CGN_sites.txt")
    download_file(STATIONS_URL, stations_file)
    
    # Unzip
    unzip_file(pwv_zip, raw_dir)
    
    print("[Done] All dataset files ready.")


if __name__ == "__main__":
    download_all("data/raw")
