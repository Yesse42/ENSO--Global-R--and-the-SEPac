"""
Merge CERES SYN1deg daily NetCDF subsets into a single compressed file.
Delegates to merge_netcdfs.merge_netcdfs() for the heavy lifting.
"""

import os
import sys

# Allow running from any working directory
sys.path.insert(0, os.path.dirname(__file__))
from merge_netcdfs import merge_netcdfs

CERES_DIR   = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/CERES/daily"
MERGED_FILE = os.path.join(CERES_DIR, "CERES_SYN1deg-Day_merged.nc")

if __name__ == "__main__":
    merge_netcdfs(
        directory=CERES_DIR,
        output_file=MERGED_FILE,
        pattern="CERES_*.nc",
        delete_sources=False,   # keep originals by default; set True to clean up
    )
