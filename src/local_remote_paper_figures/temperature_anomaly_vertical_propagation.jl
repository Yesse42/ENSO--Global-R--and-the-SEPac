cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../pls_regressor/pls_functions.jl")

using JLD2, Dates, Statistics, NCDatasets
const PYTHON_EXE = "/Users/C837213770/miniconda3/bin/python"
const PLOT_SCRIPT = joinpath(@__DIR__, "plot_vertical_propagation.py")

const visdir_base = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/vertical_temperature_anomaly_propagation"
mkpath(visdir_base)

const mask_dir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/masks"
mkpath(mask_dir)

"""
    generate_and_save_wpwp_masks(era5_coords, ceres_coords, mask_dir)

Generate West Pacific Warm Pool (WPWP) geographic masks for ERA5 and CERES grids and
save them as JLD2 files. The WPWP is defined by the fixed geographic bounds:
  - Longitude: 120°E – 180°
  - Latitude:  10°S – 10°N

Handles both 0–360 and −180–180 longitude conventions. Masks are Boolean arrays
with dimensions (lon, lat) matching the spatial layout of each dataset's data arrays.

Saved files:
  - mask_dir/wpwp_mask_era5.jld2  → keys: "mask", "lat", "lon"
  - mask_dir/wpwp_mask_ceres.jld2 → keys: "mask", "lat", "lon"
"""
function generate_and_save_wpwp_masks(era5_coords, ceres_coords, mask_dir)
    wpwp_lon_min, wpwp_lon_max = 120.0, 180.0
    wpwp_lat_min, wpwp_lat_max = -10.0, 10.0

    function make_mask(lat, lon)
        # Normalise to 0–360 so the bounds work regardless of input convention
        lon_norm = mod.(Float64.(lon), 360.0)
        lat_f    = Float64.(lat)
        lon_mask = (lon_norm .>= wpwp_lon_min) .& (lon_norm .<= wpwp_lon_max)
        lat_mask = (lat_f    .>= wpwp_lat_min) .& (lat_f    .<= wpwp_lat_max)
        # Outer product → (lon, lat) matching array layout
        return lon_mask .& lat_mask'
    end

    era5_mask  = make_mask(era5_coords["latitude"],  era5_coords["longitude"])
    ceres_mask = make_mask(ceres_coords["latitude"], ceres_coords["longitude"])

    jldsave(joinpath(mask_dir, "wpwp_mask_era5.jld2");
            mask=era5_mask, lat=era5_coords["latitude"], lon=era5_coords["longitude"])
    jldsave(joinpath(mask_dir, "wpwp_mask_ceres.jld2");
            mask=ceres_mask, lat=ceres_coords["latitude"], lon=ceres_coords["longitude"])

    println("WPWP masks saved to $mask_dir")
    println("  ERA5 mask:  $(sum(era5_mask)) / $(length(era5_mask)) grid points")
    println("  CERES mask: $(sum(ceres_mask)) / $(length(ceres_mask)) grid points")
    return era5_mask, ceres_mask
end


"""1-component PLS decomposition of scalar time series y into ENSO and non-ENSO parts.
Returns (enso_component, residual) as Float64 vectors."""
function enso_decompose_scalar(enso_matrix, y)
    m = make_pls_regressor(enso_matrix, Float64.(y), 1; print_updates=false)
    e = vec(predict(m, enso_matrix))
    return e, Float64.(y) .- e
end

"""
Pointwise 1-component PLS decomposition of a 3-D field (lon×lat×time) into ENSO
and non-ENSO parts. Returns two Float32 arrays of the same shape.
enso_matrix has size (n_valid_time × n_lags).
"""
function enso_decompose_gridded(field3d, enso_matrix)
    ni, nj, nt = size(field3d)
    field_E  = zeros(Float32, ni, nj, nt)
    field_NE = zeros(Float32, ni, nj, nt)
    @inbounds for j in 1:nj, i in 1:ni
        ts = Float64.(field3d[i, j, :])
        m  = make_pls_regressor(enso_matrix, ts, 1; print_updates=false)
        e  = vec(predict(m, enso_matrix))
        field_E[i, j, :]  .= Float32.(e)
        field_NE[i, j, :] .= Float32.(ts) .- field_E[i, j, :]
    end
    return field_E, field_NE
end

"""Remap 3-D ERA5 array (lon×lat×time) to CERES grid via nearest-neighbour index map."""
function map_era5_to_ceres(era5_3d, idx_map)
    n_time = size(era5_3d, 3)
    [era5_3d[i, j, t] for (i, j) in idx_map, t in 1:n_time]
end

"""
Regress a 3-D field (lon×lat×time) onto a 1-D scalar index.
Returns a 2-D map of β = cov(field[i,j,:], index) / std(index),
i.e. the field response per 1σ of the index.
"""
function regress_grid_on_index(field3d, index1d)
    [cov(field3d[i,j,:], index1d) / var(index1d)
     for i in axes(field3d,1), j in axes(field3d,2)]
end

"""
Pointwise regression of field3d onto ref3d at each gridpoint.
Returns β[i,j] = cov(field3d[i,j,:], ref3d[i,j,:]) / var(ref3d[i,j,:]),
i.e. the field response per 1 K of the local reference.
"""
function regress_grid_pointwise(field3d, ref3d)
    [cov(field3d[i,j,:], ref3d[i,j,:]) / var(ref3d[i,j,:])
     for i in axes(field3d,1), j in axes(field3d,2)]
end

println("Loading ERA5 t2m...")
date_range = (Date(2000, 3), Date(2024, 2, 28))
era5_data, era5_coords = load_era5_data(["t2m"], date_range)

println("Loading ERA5 T on p levels...")
era5_data_pl, era5_coords_pl = load_era5_data(["t"], date_range;
    pressure_level_file="new_pressure_levels.nc")
pressure_levels = era5_coords_pl["pressure_level"]
my_pressure_levels = [700, 500, 250]

p_level_t_arrs = [begin
    p_idx = findfirst(==(plevel), pressure_levels)
    @assert p_idx !== nothing "Pressure level $plevel hPa not found in ERA5 data"
    era5_data_pl["t"][:, :, p_idx, :]
end for plevel in my_pressure_levels]

println("Loading CERES net radiation...")
ceres_data, ceres_coords = load_new_ceres_data(["toa_net_all_mon"], date_range)
ceres_coords["time"] = round_dates_down_to_nearest_month(ceres_coords["time"])
common_time = ceres_coords["time"]
@assert all(common_time .== round_dates_down_to_nearest_month(era5_coords["time"])) "ERA5/CERES time mismatch"
@assert all(common_time .== round_dates_down_to_nearest_month(era5_coords_pl["pressure_time"])) "ERA5 PL/CERES time mismatch"

println("Deseasonalizing and detrending...")
float_time   = calc_float_time.(common_time)
month_groups = groupfind(month, common_time)
for arr in [era5_data["t2m"], ceres_data["toa_net_all_mon"]]
    for sl in eachslice(arr; dims=(1, 2))
        deseasonalize_and_detrend_precalculated_groups_twice!(sl, float_time, month_groups;
            aggfunc=mean, trendfunc=least_squares_fit)
    end
end
for arr in p_level_t_arrs
    for sl in eachslice(arr; dims=(1, 2))
        deseasonalize_and_detrend_precalculated_groups_twice!(sl, float_time, month_groups;
            aggfunc=mean, trendfunc=least_squares_fit)
    end
end

println("Generating WPWP masks...")
wpwp_mask_era5, wpwp_mask_ceres = generate_and_save_wpwp_masks(era5_coords, ceres_coords, mask_dir)

println("Loading region masks...")
jldpath = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/"
const regions_to_inspect = ["SEPac_feedback_definition", "NEPac", "SEAtl"]
era5_region_mask_dict = jldopen(joinpath(jldpath, "stratocumulus_region_masks.jld2"))["regional_masks_era5"]
rad_region_mask_dict  = jldopen(joinpath(jldpath, "stratocumulus_exclusion_masks.jld2"))["stratocum_masks"]
coord_mapping         = JLD2.load(joinpath(jldpath, "era5_ceres_coordinate_mapping.jld2"))
era5_to_ceres_indices = Tuple.(coord_mapping["era5_to_ceres_indices"])

println("Loading ENSO data...")
enso_data, enso_dates_raw = load_enso_data(date_range)
enso_times = round_dates_down_to_nearest_month(enso_dates_raw["time"])
lags = -6:6
n_enso = length(enso_times)
enso_lag_matrix_full = Matrix{Union{Float64,Missing}}(missing, n_enso, length(lags))
for (j, lag) in enumerate(lags)
    col = "oni_lag_$lag"
    haskey(enso_data, col) && (enso_lag_matrix_full[:, j] .= enso_data[col])
end
valid_rows = [all(!ismissing, enso_lag_matrix_full[i, :]) for i in 1:n_enso]
enso_matrix = Float64.(enso_lag_matrix_full[valid_rows, :])
enso_matrix_matched, data_valid_idx = precompute_enso_match(
    common_time, enso_times[valid_rows], enso_matrix)

# ============================================================
# Derived quantities for the vertical propagation figure
# ============================================================

era5_lat  = era5_coords["latitude"]
era5_lon  = era5_coords["longitude"]
ceres_lat = ceres_coords["latitude"]
ceres_lon = ceres_coords["longitude"]

println("Computing scalar t2m indices (WPWP, SEPac, NEPac)...")
wpwp_index  = generate_spatial_mean(era5_data["t2m"], era5_lat, wpwp_mask_era5)
sepac_index = generate_spatial_mean(era5_data["t2m"], era5_lat,
                                    era5_region_mask_dict["SEPac_feedback_definition"])
nepac_index = generate_spatial_mean(era5_data["t2m"], era5_lat,
                                    era5_region_mask_dict["NEPac"])

println("Remapping ERA5 t2m to CERES grid...")
t2m_ceres = map_era5_to_ceres(era5_data["t2m"], era5_to_ceres_indices)

# ============================================================
# ENSO / Non-ENSO Decomposition
# (all quantities restricted to ENSO-valid time steps)
# ============================================================

println("Decomposing scalar t2m indices into ENSO / non-ENSO...")
wpwp_idx_E,  wpwp_idx_NE  = enso_decompose_scalar(enso_matrix_matched, wpwp_index[data_valid_idx])
sepac_idx_E, sepac_idx_NE = enso_decompose_scalar(enso_matrix_matched, sepac_index[data_valid_idx])
nepac_idx_E, nepac_idx_NE = enso_decompose_scalar(enso_matrix_matched, nepac_index[data_valid_idx])

println("Decomposing gridded t2m into ENSO / non-ENSO (pointwise PLS)...")
t2m_valid    = era5_data["t2m"][:, :, data_valid_idx]
t2m_E, t2m_NE = enso_decompose_gridded(t2m_valid, enso_matrix_matched)

println("Remapping decomposed t2m to CERES grid...")
t2m_E_ceres  = map_era5_to_ceres(t2m_E,  era5_to_ceres_indices)
t2m_NE_ceres = map_era5_to_ceres(t2m_NE, era5_to_ceres_indices)

# Restrict other fields to ENSO-valid time steps for ENSO/NE figure calls
R_valid         = ceres_data["toa_net_all_mon"][:, :, data_valid_idx]
t_arrs_valid    = [arr[:, :, data_valid_idx] for arr in p_level_t_arrs]

# ============================================================
# 5×4 Vertical Propagation Figure
# ============================================================

"""
    plot_vertical_propagation_figure(; savepath)

5 rows × 4 columns of global regression maps.

Columns — temperature index:
  1: gridded t2m (pointwise regression)
  2: WPWP-mean t2m
  3: SEPac-mean t2m
  4: NEPac-mean t2m

Rows — variable regressed onto the index:
  1: CERES net radiation        (CERES grid, W m⁻² K⁻¹)
  2: t at 250 hPa               (ERA5 grid,  K K⁻¹)
  3: t at 500 hPa               (ERA5 grid,  K K⁻¹)
  4: t at 700 hPa               (ERA5 grid,  K K⁻¹)
  5: std(t2m) [col 1] or regression of t2m onto scalar index [cols 2–4]  (ERA5 grid)

Each of rows 1–4 has its own shared colorbar (TwoSlopeNorm, ±absmax).
Row 5 has two colorbars: col 1 (std, own) and cols 2–4 (regression, shared).
All colorbars use DAVES_CMAP centred at 0.
"""
function plot_vertical_propagation_figure(;
        t2m        = era5_data["t2m"],
        t2m_ceres  = t2m_ceres,
        R          = ceres_data["toa_net_all_mon"],
        t_arrs     = p_level_t_arrs,          # [t700, t500, t250]
        era5_lat   = era5_lat,
        era5_lon   = era5_lon,
        ceres_lat  = ceres_lat,
        ceres_lon  = ceres_lon,
        wpwp_idx   = wpwp_index,
        sepac_idx  = sepac_index,
        nepac_idx  = nepac_index,
        suptitle   = "Temperature vertical propagation",
        savepath   = joinpath(visdir_base, "vertical_propagation_5x4.png"))

    # ── Column metadata ──────────────────────────────────────────────────
    scalar_indices = [wpwp_idx, sepac_idx, nepac_idx]
    scalar_labels  = ["WPWP", "SEPac", "NEPac"]
    col_titles = vcat(
        ["Gridded t2m"],
        ["$(lbl) t2m index  (σ=$(round(std(idx); sigdigits=3)) K)"
         for (lbl, idx) in zip(scalar_labels, scalar_indices)]
    )

    # ── Row metadata ─────────────────────────────────────────────────────
    # p_level_t_arrs order: [t700, t500, t250]  →  row 2 = 250 hPa, row 4 = 700 hPa
    t_250, t_500, t_700 = t_arrs[3], t_arrs[2], t_arrs[1]
    row_vars_era5  = [(t_250, "250 hPa"), (t_500, "500 hPa"), (t_700, "700 hPa")]
    row_var_labels = ["CERES R", "t 250 hPa", "t 500 hPa", "t 700 hPa", "t2m"]

    # ── Compute all 20 fields ─────────────────────────────────────────────
    println("  Computing row 1: CERES R regression...")
    row1 = Vector{Matrix{Float32}}(undef, 4)
    row1[1] = Float32.(regress_grid_pointwise(R, t2m_ceres))
    for (k, idx) in enumerate(scalar_indices)
        row1[k+1] = Float32.(regress_grid_on_index(R, idx))
    end

    rows2to4 = Vector{Vector{Matrix{Float32}}}(undef, 3)
    for (r, (t_arr, plabel)) in enumerate(row_vars_era5)
        println("  Computing row $(r+1): t@$plabel regression...")
        maps = Vector{Matrix{Float32}}(undef, 4)
        maps[1] = Float32.(regress_grid_pointwise(t_arr, t2m))
        for (k, idx) in enumerate(scalar_indices)
            maps[k+1] = Float32.(regress_grid_on_index(t_arr, idx))
        end
        rows2to4[r] = maps
    end

    println("  Computing row 5: std(t2m) and t2m regressions...")
    row5 = Vector{Matrix{Float32}}(undef, 4)
    row5[1] = Float32.([std(t2m[i,j,:]) for i in axes(t2m,1), j in axes(t2m,2)])
    for (k, idx) in enumerate(scalar_indices)
        row5[k+1] = Float32.(regress_grid_on_index(t2m, idx))
    end

    row_cbar_lbl = ["W m⁻² K⁻¹", "K K⁻¹", "K K⁻¹", "K K⁻¹"]  # rows 1–4

    # ── Save all fields to a temporary NetCDF, then delegate to Python ────
    nc_path = tempname() * ".nc"
    try
        NCDatasets.Dataset(nc_path, "c") do ds
            # Dimensions
            defDim(ds, "ceres_lon", length(ceres_lon))
            defDim(ds, "ceres_lat", length(ceres_lat))
            defDim(ds, "era5_lon",  length(era5_lon))
            defDim(ds, "era5_lat",  length(era5_lat))

            # Coordinate variables
            defVar(ds, "ceres_lon", Float64.(ceres_lon), ("ceres_lon",))
            defVar(ds, "ceres_lat", Float64.(ceres_lat), ("ceres_lat",))
            defVar(ds, "era5_lon",  Float64.(era5_lon),  ("era5_lon",))
            defVar(ds, "era5_lat",  Float64.(era5_lat),  ("era5_lat",))

            # Row 1: CERES grid (lon × lat)
            for c in 1:4
                defVar(ds, "r1_c$c", Float32.(row1[c]), ("ceres_lon", "ceres_lat"))
            end

            # Rows 2–4: ERA5 grid
            for (r_off, row_maps) in enumerate(rows2to4)
                r = r_off + 1
                for c in 1:4
                    defVar(ds, "r$(r)_c$c", Float32.(row_maps[c]), ("era5_lon", "era5_lat"))
                end
            end

            # Row 5: ERA5 grid
            for c in 1:4
                defVar(ds, "r5_c$c", Float32.(row5[c]), ("era5_lon", "era5_lat"))
            end

            # Metadata as global attributes
            ds.attrib["suptitle"] = suptitle
            for (i, t) in enumerate(col_titles)
                ds.attrib["col_title_$i"] = t
            end
            for (i, lbl) in enumerate(row_var_labels)
                ds.attrib["row_label_$i"] = lbl
            end
            for (i, lbl) in enumerate(row_cbar_lbl)
                ds.attrib["row_cbar_lbl_$i"] = lbl
            end
        end

        println("  Running Python plotting subprocess...")
        run(Cmd([PYTHON_EXE, PLOT_SCRIPT, nc_path, savepath]))
        println("Saved: $savepath")
    finally
        isfile(nc_path) && rm(nc_path)
    end
end

println("\nGenerating 5×4 vertical propagation figure (full)...")
plot_vertical_propagation_figure(;
    suptitle = "Temperature vertical propagation — full time series")

println("\nGenerating 5×4 vertical propagation figure (ENSO component)...")
plot_vertical_propagation_figure(;
    t2m       = t2m_E,
    t2m_ceres = t2m_E_ceres,
    R         = R_valid,
    t_arrs    = t_arrs_valid,
    wpwp_idx  = wpwp_idx_E,
    sepac_idx = sepac_idx_E,
    nepac_idx = nepac_idx_E,
    suptitle  = "Temperature vertical propagation — ENSO component",
    savepath  = joinpath(visdir_base, "vertical_propagation_5x4_ENSO.png"))

println("\nGenerating 5×4 vertical propagation figure (non-ENSO component)...")
plot_vertical_propagation_figure(;
    t2m       = t2m_NE,
    t2m_ceres = t2m_NE_ceres,
    R         = R_valid,
    t_arrs    = t_arrs_valid,
    wpwp_idx  = wpwp_idx_NE,
    sepac_idx = sepac_idx_NE,
    nepac_idx = nepac_idx_NE,
    suptitle  = "Temperature vertical propagation — non-ENSO component",
    savepath  = joinpath(visdir_base, "vertical_propagation_5x4_nonENSO.png"))

println("\nAll figures saved to: $visdir_base")