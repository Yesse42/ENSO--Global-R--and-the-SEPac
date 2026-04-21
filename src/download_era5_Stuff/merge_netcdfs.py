"""
Generic utility for merging a set of NetCDF files in a directory along the
time dimension into a single file.

Verifies the merged data matches the originals before optionally deleting the
source files.
"""

import os
import glob
import numpy as np
import xarray as xr

ENCODING_DEFAULTS = {
    "zlib": False,
    "complevel": 0,
    "shuffle": False,
}


def find_netcdfs(directory: str, pattern: str = "*.nc") -> list[str]:
    """Return sorted list of NetCDF paths matching *pattern* inside *directory*."""
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        print(f"[merge] No files matching '{pattern}' found in {directory}")
    return paths


def verify_merge(
    paths: list[str],
    merged: xr.Dataset,
    tolerance_per_var: dict | None = None,
) -> bool:
    """
    Spot-check the merge by reloading each source file and comparing:
      - time coverage
      - variable names
      - first and last time-slice values (max absolute difference)

    tolerance_per_var maps variable names to acceptable max diff. Defaults to
    1e-4 for any variable not listed (appropriate for lossless encodings).
    For int16-packed variables pass scale_factor / 2 as the tolerance.
    """
    print("[verify] Checking merged dataset against originals …")
    all_ok = True
    tol = tolerance_per_var or {}

    for p in paths:
        label = os.path.basename(p)
        with xr.open_dataset(p, engine="netcdf4", chunks={}) as ds_src:
            src_times = ds_src.time.values
            merged_times = merged.time.values
            missing_times = np.setdiff1d(src_times, merged_times)
            if len(missing_times) > 0:
                print(f"[verify] FAIL {label}: {len(missing_times)} time steps missing from merged file.")
                all_ok = False
                continue

            for var in ds_src.data_vars:
                if var not in merged.data_vars:
                    print(f"[verify] FAIL {label}: variable '{var}' missing from merged file.")
                    all_ok = False
                    continue

                threshold = tol.get(var, 1e-4) * 1.01
                for t in [src_times[0], src_times[-1]]:
                    orig_slice  = ds_src[var].sel(time=t).values
                    merge_slice = merged[var].sel(time=t).values
                    max_diff = np.nanmax(np.abs(orig_slice.astype(float) - merge_slice.astype(float)))
                    if max_diff > threshold:
                        print(f"[verify] FAIL {label} var={var} time={t}: max diff = {max_diff:.6g} (tolerance {threshold:.6g})")
                        all_ok = False

        print(f"[verify] {label}: OK")

    return all_ok


def merge_netcdfs(
    directory: str,
    output_file: str,
    pattern: str = "*.nc",
    delete_sources: bool = True,
    complevel: int = 0,
    chunksizes: dict | None = None,
    per_var_encoding: dict | None = None,
) -> bool:
    """
    Merge all NetCDF files matching *pattern* in *directory* along the time
    dimension and write a single file to *output_file*.

    Parameters
    ----------
    directory:        Directory containing the source NetCDF files.
    output_file:      Full path for the merged output file.
    pattern:          Glob pattern used to select source files (default: '*.nc').
    delete_sources:   If True (default), delete source files after a successful
                      verified merge.
    complevel:        zlib compression level 0–9 (default: 0, no compression).
    chunksizes:       Dict mapping dimension names to chunk sizes. Dimensions
                      omitted get the full dimension size (e.g. {"time": 1} gives
                      one time step per chunk with all spatial points together).
    per_var_encoding: Dict mapping variable names to encoding overrides applied
                      on top of the base options (e.g. int16 packing via
                      {"z": {"dtype": "int16", "scale_factor": 0.02,
                      "add_offset": 5400.0}}).

    Returns
    -------
    True if the merge (and optional cleanup) succeeded, False otherwise.
    """
    # For int16-packed variables the max quantization error is scale_factor / 2.
    tolerance_per_var = {}
    if per_var_encoding:
        for var, enc in per_var_encoding.items():
            if "scale_factor" in enc:
                tolerance_per_var[var] = enc["scale_factor"] / 2

    if os.path.isfile(output_file):
        print(f"[merge] {os.path.basename(output_file)} already exists. Re-verifying …")
        paths = find_netcdfs(directory, pattern)
        if not paths:
            return False
        paths = [p for p in paths if os.path.abspath(p) != os.path.abspath(output_file)]
        print("[verify] Reloading merged file for verification …")
        with xr.open_dataset(output_file, engine="netcdf4", chunks={}) as ds_check:
            ok = verify_merge(paths, ds_check, tolerance_per_var=tolerance_per_var)
        if ok:
            if delete_sources and paths:
                print(f"\n[cleanup] Verification passed. Deleting {len(paths)} source file(s) …")
                for p in paths:
                    os.remove(p)
                    print(f"[cleanup] Removed {os.path.basename(p)}")
                print("[cleanup] Done.")
            else:
                print("\n[merge] Verification passed. Source files kept (delete_sources=False).")
        else:
            print("\n[cleanup] Verification FAILED. Source files kept. Check the merged file manually.")
        return ok

    paths = find_netcdfs(directory, pattern)
    if not paths:
        return False

    # Exclude the output file itself if it happens to live in the same directory
    paths = [p for p in paths if os.path.abspath(p) != os.path.abspath(output_file)]

    print(f"[merge] Lazy-loading {len(paths)} file(s) with dask …")
    for p in paths:
        print(f"[merge]   {os.path.basename(p)}")

    ds_merged = xr.open_mfdataset(
        paths, engine="netcdf4", combine="by_coords", chunks={}
    ).sortby("time")

    base_opts = {**ENCODING_DEFAULTS, "zlib": complevel > 0, "complevel": complevel}

    encoding = {}
    for var in ds_merged.data_vars:
        enc = {**base_opts}
        if chunksizes:
            dims = ds_merged[var].dims
            enc["chunksizes"] = tuple(chunksizes.get(d, ds_merged.dims[d]) for d in dims)
        if per_var_encoding and var in per_var_encoding:
            enc.update(per_var_encoding[var])
        encoding[var] = enc

    print(f"[merge] Writing merged file → {os.path.basename(output_file)} …")
    ds_merged.to_netcdf(output_file, encoding=encoding)
    print(f"[merge] Written: {output_file}")

    print("[verify] Reloading merged file for verification …")
    with xr.open_dataset(output_file, engine="netcdf4", chunks={}) as ds_check:
        ok = verify_merge(paths, ds_check, tolerance_per_var=tolerance_per_var)

    if ok:
        if delete_sources:
            print(f"\n[cleanup] Verification passed. Deleting {len(paths)} source file(s) …")
            for p in paths:
                os.remove(p)
                print(f"[cleanup] Removed {os.path.basename(p)}")
            print("[cleanup] Done.")
        else:
            print("\n[merge] Verification passed. Source files kept (delete_sources=False).")
    else:
        print("\n[cleanup] Verification FAILED. Source files kept. Check the merged file manually.")

    ds_merged.close()
    return ok
