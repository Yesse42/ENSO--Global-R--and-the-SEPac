#!/usr/bin/env python3
"""
plot_vertical_propagation.py
Usage: python plot_vertical_propagation.py <netcdf_path> <output_png>

Reads pre-computed regression maps written by Julia (NCDatasets) and produces a
5-row x 4-column global Robinson-projection figure.

NCDatasets writes Julia column-major arrays to NetCDF such that the first Julia
dimension (lon) becomes the last (fastest-varying) dimension on disk.  Python's
netCDF4 reads in C order, so the array arrives as (lat, lon) — no transpose needed.
"""
import sys
import numpy as np
import netCDF4
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import cartopy.crs as ccrs

_CAP = 5.0  # colorbar cap (K K⁻¹ or W m⁻² K⁻¹); applied when absmax exceeds this

# ── Recreate DAVES_CMAP ───────────────────────────────────────────────────────
_DAVES_COLORS = np.array([
    [ 10,  50, 120],
    [ 15,  75, 165],
    [ 30, 110, 200],
    [ 60, 160, 240],
    [ 80, 180, 250],
    [130, 210, 255],
    [160, 240, 255],
    [220, 250, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 240, 120],
    [255, 192,  60],
    [255, 160,   0],
    [255,  96,   0],
    [255,  50,   0],
    [225,  20,   0],
    [192,   0,   0],
    [165,   0,   0],
], dtype=float) / 255.0

DAVES_CMAP = mcolors.LinearSegmentedColormap.from_list("daves_cmap", _DAVES_COLORS)


def capped_absmax(arrays, cap=_CAP):
    """Return min(max |value| across all arrays, cap)."""
    raw = max(float(np.nanmax(np.abs(a))) for a in arrays)
    return min(raw, cap)


def plot_panel(ax, lon, lat, data, norm, cmap, title):
    """data shape from netCDF4: (lat, lon) — no transpose needed."""
    ax.set_global()
    ax.coastlines(linewidth=0.5)
    ax.pcolormesh(np.asarray(lon), np.asarray(lat), np.asarray(data),
                  transform=ccrs.PlateCarree(),
                  cmap=cmap, norm=norm,
                  shading="auto",
                  rasterized=True)
    ax.set_title(title, fontsize=8)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <netcdf_path> <output_png>", file=sys.stderr)
        sys.exit(1)

    nc_path, out_png = sys.argv[1], sys.argv[2]

    ds = netCDF4.Dataset(nc_path, "r")

    ceres_lon = ds["ceres_lon"][:]
    ceres_lat = ds["ceres_lat"][:]
    era5_lon  = ds["era5_lon"][:]
    era5_lat  = ds["era5_lat"][:]

    # Read all 20 fields: all_data[row][col], 0-indexed
    all_data = []
    for r in range(1, 6):
        row = [np.array(ds[f"r{r}_c{c}"][:], dtype=np.float32) for c in range(1, 5)]
        all_data.append(row)

    # Metadata
    suptitle      = ds.getncattr("suptitle")
    col_titles    = [ds.getncattr(f"col_title_{i}") for i in range(1, 5)]
    row_labels    = [ds.getncattr(f"row_label_{i}") for i in range(1, 6)]
    row_cbar_lbls = [ds.getncattr(f"row_cbar_lbl_{i}") for i in range(1, 5)]

    ds.close()

    # Lat/lon per row: row 1 on CERES grid, rows 2–5 on ERA5 grid
    row_lons = [ceres_lon] + [era5_lon] * 4
    row_lats = [ceres_lat] + [era5_lat] * 4

    proj = ccrs.Robinson(central_longitude=180)
    fig, axs = plt.subplots(5, 4,
                             figsize=(28, 22),
                             subplot_kw={"projection": proj},
                             layout="compressed")

    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)

    # ── Rows 1–4: one shared TwoSlopeNorm per row ────────────────────────────
    # Row 1 (radiation, W m⁻² K⁻¹): span full ±absmax, no cap
    # Rows 2–4 (temperature, K K⁻¹): capped at ±_CAP
    for r in range(4):
        row_maps = all_data[r]
        lon, lat = row_lons[r], row_lats[r]
        if r == 0:
            absmax = min(max(float(np.nanmax(np.abs(a))) for a in row_maps), 30.0)
        else:
            absmax = capped_absmax(row_maps)
        norm = mcolors.TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)
        for c in range(4):
            panel_title = f"{row_labels[r]} | {col_titles[c]}"
            plot_panel(axs[r, c], lon, lat, row_maps[c], norm, DAVES_CMAP, panel_title)
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=DAVES_CMAP),
                     ax=list(axs[r, :]),
                     orientation="horizontal",
                     label=row_cbar_lbls[r],
                     shrink=0.6, pad=0.05)

    # ── Row 5 ─────────────────────────────────────────────────────────────────
    row5 = all_data[4]

    # Col 1: std(t2m) — own colorbar, no cap (std is always ≥ 0, reasonable range)
    absmax_std = float(np.nanmax(np.abs(row5[0])))
    norm_std = mcolors.TwoSlopeNorm(vmin=-absmax_std, vcenter=0.0, vmax=absmax_std)
    plot_panel(axs[4, 0], era5_lon, era5_lat, row5[0], norm_std, DAVES_CMAP,
               f"\u03c3(t2m) | {col_titles[0]}")
    plt.colorbar(cm.ScalarMappable(norm=norm_std, cmap=DAVES_CMAP),
                 ax=axs[4, 0],
                 orientation="horizontal", label="K", shrink=0.8, pad=0.05)

    # Cols 2–4: regression of t2m onto scalar index — shared colorbar, capped
    absmax_reg = capped_absmax(row5[1:4])
    norm_reg = mcolors.TwoSlopeNorm(vmin=-absmax_reg, vcenter=0.0, vmax=absmax_reg)
    for c in range(1, 4):
        plot_panel(axs[4, c], era5_lon, era5_lat, row5[c], norm_reg, DAVES_CMAP,
                   f"t2m | {col_titles[c]}")
    plt.colorbar(cm.ScalarMappable(norm=norm_reg, cmap=DAVES_CMAP),
                 ax=list(axs[4, 1:4]),
                 orientation="horizontal", label="K K\u207b\u00b9", shrink=0.6, pad=0.05)

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close("all")
    print(f"Saved: {out_png}", flush=True)


if __name__ == "__main__":
    main()
