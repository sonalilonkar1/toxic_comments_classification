"""Download GloVe pre-trained word embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> None:
    """Download a file with progress bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def main():
    parser = argparse.ArgumentParser(
        description="Download GloVe pre-trained word embeddings"
    )
    parser.add_argument(
        "--dim",
        type=int,
        choices=[50, 100, 200, 300],
        default=100,
        help="Embedding dimension (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/embeddings"),
        help="Output directory for embeddings (default: data/embeddings)",
    )
    args = parser.parse_args()
    
    # GloVe 6B dataset URLs
    # Direct download from Stanford's server
    filename = f"glove.6B.{args.dim}d.txt"
    url = f"https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
    
    output_dir = args.output_dir
    zip_path = output_dir / f"{filename}.zip"
    txt_path = output_dir / filename
    
    # Check if already downloaded
    if txt_path.exists():
        print(f"✅ GloVe {args.dim}d embeddings already exist at: {txt_path}")
        print("   Skipping download.")
        return
    
    print(f"📥 Downloading GloVe {args.dim}d embeddings...")
    print(f"   URL: {url}")
    print(f"   Output: {zip_path}")
    print()
    
    try:
        # Download zip file
        download_file(url, zip_path)
        
        # Extract
        print(f"\n📦 Extracting {zip_path.name}...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Remove zip file to save space
        zip_path.unlink()
        
        print(f"\n✅ Successfully downloaded GloVe {args.dim}d embeddings!")
        print(f"   Location: {txt_path}")
        print(f"   Size: {txt_path.stat().st_size / (1024**2):.1f} MB")
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Download failed: {e}")
        print("\n💡 Alternative: Download manually from:")
        print(f"   {url}")
        print(f"   Then extract to: {output_dir}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

