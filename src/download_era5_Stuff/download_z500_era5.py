"""
Download ERA5 geopotential height (z) at 500 hPa for 1980–2025,
average the 6-hourly data to daily means, convert to compressed NetCDF,
then merge all yearly files into a single file.
"""

import os
import sys
import zipfile
import shutil
import cdsapi
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_netcdfs import merge_netcdfs

# ── Configuration ────────────────────────────────────────────────────────────

OUTPUT_DIR = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/ERA5/daily_data"
DATASET    = "reanalysis-era5-pressure-levels"
YEARS      = [str(y) for y in range(1980, 2026)]
BASE_NAME  = "daily_z500"

REQUEST_BASE = {
    "product_type": ["reanalysis"],
    "variable": ["geopotential"],
    "pressure_level": ["500"],
    "month": ["01", "02", "03", "04", "05", "06",
               "07", "08", "09", "10", "11", "12"],
    "day":   ["01", "02", "03", "04", "05", "06",
               "07", "08", "09", "10", "11", "12",
               "13", "14", "15", "16", "17", "18",
               "19", "20", "21", "22", "23", "24",
               "25", "26", "27", "28", "29", "30", "31"],
    "time":  ["00:00", "06:00", "12:00", "18:00"],
    "data_format": "grib",
    "download_format": "zip",
}

ENCODING_DEFAULTS = {
    "zlib": True,
    "complevel": 4,
    "shuffle": True,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def zip_path(yr: str) -> str:
    return os.path.join(OUTPUT_DIR, f"{BASE_NAME}_{yr}.zip")

def nc_path(yr: str) -> str:
    return os.path.join(OUTPUT_DIR, f"{BASE_NAME}_{yr}.nc")

def temp_dir(yr: str) -> str:
    return os.path.join(OUTPUT_DIR, f"temp_{BASE_NAME}_{yr}")


def download_year(client: cdsapi.Client, yr: str) -> None:
    target = zip_path(yr)
    if os.path.isfile(target):
        print(f"[download] {os.path.basename(target)} already exists, skipping.")
        return
    # Also skip if the final NetCDF is already present
    if os.path.isfile(nc_path(yr)):
        print(f"[download] {os.path.basename(nc_path(yr))} already exists, skipping download.")
        return
    print(f"[download] Requesting {yr} …")
    request = {**REQUEST_BASE, "year": [yr]}
    try:
        client.retrieve(DATASET, request, target)
        print(f"[download] Saved {os.path.basename(target)}")
    except Exception as exc:
        print(f"[download] FAILED for {yr}: {exc}")


def convert_year(yr: str) -> None:
    infile  = zip_path(yr)
    outfile = nc_path(yr)
    tmpdir  = temp_dir(yr)

    if os.path.isfile(outfile):
        print(f"[convert]  {os.path.basename(outfile)} already exists, skipping.")
        return
    if not os.path.isfile(infile):
        print(f"[convert]  {os.path.basename(infile)} not found, skipping conversion.")
        return

    print(f"[convert]  Extracting {os.path.basename(infile)} …")
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)
    os.makedirs(tmpdir)

    try:
        with zipfile.ZipFile(infile, "r") as zf:
            zf.extractall(tmpdir)

        grib_file = os.path.join(tmpdir, "data.grib")
        if not os.path.exists(grib_file):
            candidates = [f for f in os.listdir(tmpdir) if f.endswith(".grib")]
            if not candidates:
                print(f"[convert]  No .grib file found in archive for {yr}, skipping.")
                return
            grib_file = os.path.join(tmpdir, candidates[0])

        print(f"[convert]  Opening GRIB, computing daily means, writing compressed NetCDF …")
        ds = xr.open_dataset(grib_file, engine="cfgrib", decode_timedelta=True)

        # Convert geopotential (m²/s²) to geopotential height (m)
        if "z" in ds:
            ds["z"] = ds["z"] / 9.80665

        ds_daily = ds.resample(time="1D").mean()
        ds.close()

        ds_daily = ds_daily.astype("float32")

        encoding = {var: ENCODING_DEFAULTS for var in ds_daily.data_vars}
        ds_daily.to_netcdf(outfile, encoding=encoding)
        ds_daily.close()

        print(f"[convert]  Written: {os.path.basename(outfile)}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if os.path.isfile(outfile):
        os.remove(infile)
        print(f"[convert]  Removed {os.path.basename(infile)}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    client = cdsapi.Client()
    for yr in YEARS:
        download_year(client, yr)
        convert_year(yr)

    merged_file = os.path.join(OUTPUT_DIR, f"{BASE_NAME}_1980_2025.nc")
    merge_netcdfs(
        directory=OUTPUT_DIR,
        output_file=merged_file,
        pattern=f"{BASE_NAME}_*.nc",
        delete_sources=True,
        complevel=5,
        chunksizes={"time": 1},
        per_var_encoding={
            "z": {"dtype": "int16", "scale_factor": 0.04, "add_offset": 5400.0},
        },
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
