"""
Merge yearly ERA5 temperature NetCDF files into a single compressed file.
Delegates to merge_netcdfs.merge_netcdfs() for the heavy lifting.
"""

import os
from merge_netcdfs import merge_netcdfs

OUTPUT_DIR   = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/ERA5/daily_data"
BASE_NAME    = "hourly_t_data"
MERGED_FILE  = os.path.join(OUTPUT_DIR, f"{BASE_NAME}_2000_2025.nc")

if __name__ == "__main__":
    merge_netcdfs(
        directory=OUTPUT_DIR,
        output_file=MERGED_FILE,
        pattern=f"{BASE_NAME}_*.nc",
        delete_sources=True,
    )
