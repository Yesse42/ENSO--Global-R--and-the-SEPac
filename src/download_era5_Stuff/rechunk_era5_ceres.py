"""
Resave the merged ERA5 temperature and CERES daily NetCDF files without compression,
overwriting the originals.
"""

import os
import xarray as xr

ERA5_DIR  = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/ERA5/daily_data"
CERES_DIR = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/CERES/daily"

FILES = [
    os.path.join(ERA5_DIR,  "hourly_t_data_2000_2025.nc"),
    os.path.join(CERES_DIR, "CERES_SYN1deg-Day_merged.nc"),
]


def resave(path: str) -> None:
    print(f"[resave] {os.path.basename(path)} …")
    tmp_path = path + ".tmp"
    ds = xr.open_dataset(path, chunks={})
    encoding = {var: {"zlib": False, "complevel": 0} for var in ds.data_vars}
    ds.to_netcdf(tmp_path, encoding=encoding)
    print(f"[resave] Rewrite Done, Closing + Swapping.")
    ds.close()
    os.remove(path)
    os.rename(tmp_path, path)
    print(f"[resave] Swap Done.")


if __name__ == "__main__":
    for path in FILES:
        resave(path)
    print("\nAll done.")
