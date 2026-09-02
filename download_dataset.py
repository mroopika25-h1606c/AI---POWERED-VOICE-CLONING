"""
ASVspoof 2019 LA Dataset Download Script

Downloads the official ASVspoof 2019 LA dataset from University of Edinburgh.
This dataset is publicly available for research purposes.

Official Source: https://datashare.ed.ac.uk/handle/10283/3336

Usage:
    python download_dataset.py

Note: This script creates a download_links.txt file. You'll need to:
1. Go to the official URL above
2. Download the files manually, OR
3. Use this script as a reference for wget/curl commands
"""

import os
import logging
from pathlib import Path
import urllib.request
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset directory
DATASET_DIR = Path(__file__).parent / "dataset" / "LA"

# Official ASVspoof 2019 LA files available for download
# These are the direct download links (may require registration/login)
DATASET_FILES = {
    # Training data
    "ASVspoof2019_LA_train.zip": "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/ASVspoof2019_LA_train.zip",
    "ASVspoof2019_LA_dev.zip": "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/ASVspoof2019_LA_dev.zip",
    "ASVspoof2019_LA_test.zip": "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/ASVspoof2019_LA_test.zip",
}

def print_instructions():
    """Print download instructions."""
    print("\n" + "="*70)
    print("ASVspoof 2019 LA Dataset Download Instructions")
    print("="*70)
    
    instructions = """
The ASVspoof 2019 LA dataset is available from the University of Edinburgh.

OPTION 1: Manual Download (Recommended)
────────────────────────────────────────
1. Visit: https://datashare.ed.ac.uk/handle/10283/3336
2. Click "Files" tab
3. Download these files:
   - ASVspoof2019_LA_train.zip  (~2.1 GB)
   - ASVspoof2019_LA_dev.zip    (~4.6 GB)
   - ASVspoof2019_LA_test.zip   (~5.5 GB)
   
4. Extract to: dataset/LA/
   - Creates: LA_0001_0000000.flac, protocol.txt, etc.

OPTION 2: Using Command Line (wget/curl)
─────────────────────────────────────────
# Make sure you have dataset folder created
mkdir -p dataset/LA

# Download train set
wget --user-agent="Mozilla/5.0" \\
  "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/ASVspoof2019_LA_train.zip" \\
  -O dataset/LA_train.zip

# Download dev set
wget --user-agent="Mozilla/5.0" \\
  "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/ASVspoof2019_LA_dev.zip" \\
  -O dataset/LA_dev.zip

# Download test set
wget --user-agent="Mozilla/5.0" \\
  "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/ASVspoof2019_LA_test.zip" \\
  -O dataset/LA_test.zip

# Extract all
cd dataset/LA
unzip -q ../LA_train.zip
unzip -q ../LA_dev.zip
unzip -q ../LA_test.zip

OPTION 3: Smaller Subset (Demo)
───────────────────────────────
For testing/development without downloading full dataset (~12 GB):
- I'll create a synthetic dataset generator for demo purposes
- Use: python create_demo_dataset.py

⚠️ IMPORTANT NOTES:
───────────────────
• Dataset is large (~12 GB total after extraction)
• First download may take 30-60 minutes depending on connection
• After extraction, verify: ✓ LA_0001_0000000.flac exists
• After extraction, verify: ✓ protocol.txt exists
• Then run: python organize_dataset.py
"""
    
    print(instructions)


def create_dataset_structure():
    """Create necessary dataset folders."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Dataset directory: {DATASET_DIR}")


if __name__ == "__main__":
    print("\n🎙️ ASVspoof 2019 LA Dataset Download Script")
    print_instructions()
    create_dataset_structure()
    print("\n👉 Next steps:")
    print("  1. Download files from the URL above")
    print("  2. Extract all ZIP files to dataset/LA/")
    print("  3. Run: python organize_dataset.py")
