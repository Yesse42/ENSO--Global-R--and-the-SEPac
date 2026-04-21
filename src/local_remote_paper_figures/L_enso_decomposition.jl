# ENSO Decomposition of L = Cov(R̄, Tᵢ) / Var(T̄)
#
# Decomposes both global radiation (R̄) and local temperature (Tᵢ) into ENSO
# and non-ENSO components via PLS regression on the ENSO lag matrix, then
# computes the 4 cross-covariance terms that sum to L.
#
# Global analysis  (L = Cov(R̄, Tᵢ) / Var(T̄)):
#   1. L (net) map with its own colorbar
#   2. 4-pane net L decomposition
#   3. SW / LW  2×5
#   4. SW Clear / Cloud  2×5
#   5. LW Clear / Cloud  2×5
#
# Per-region analysis  (L_region = Cov(Rᵢ, T_region) / Var(T̄)):
#   Same set of plots for each region, plus region contours.

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../pls_regressor/pls_functions.jl")
include("tile_thresh.jl")

using JLD2, Dates, Statistics
@py import numpy as np, matplotlib.pyplot as plt, matplotlib.colors as colors, matplotlib.cm as cm, cmasher as cmr, cartopy.crs as ccrs

visdir_base = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/L_enso_decomposition"
mkpath(visdir_base)

# ============================================================
# Helper Functions
# ============================================================

"""
    compute_L_field(scalar_ts, grid_3d, var_T)

Compute L(x) = Cov(scalar_ts, grid(x)) / var_T at each grid point.
`grid_3d` is (lon × lat × time).  Returns a 2D map (lon × lat).
Since Cov is symmetric this works whether the scalar represents R̄ or T_region.
"""
function compute_L_field(scalar_ts::AbstractVector, grid_3d::AbstractArray{<:Real,3}, var_T::Real)
    cov.(Ref(scalar_ts), vec.(eachslice(grid_3d; dims=(1,2)))) ./ var_T
end

"""
    enso_decompose_scalar(enso_matrix, y)

Decompose scalar time series `y` into its ENSO component and residual
via 1-component PLS on the ENSO lag matrix.
Returns `(enso_component, residual)`.
"""
function enso_decompose_scalar(enso_matrix::Matrix, y::AbstractVector)
    pls_model = make_pls_regressor(enso_matrix, y, 1; print_updates=false)
    enso_comp = vec(predict(pls_model, enso_matrix))
    return enso_comp, y .- enso_comp
end

"""
    compute_full_L_analysis(scalar_total, scalar_enso, scalar_resid,
                            grid_total, grid_enso, grid_resid, var_T)

Compute total L and its 4 ENSO cross-covariance components from pre-decomposed
scalar and gridded fields.  Returns NamedTuple `(; total, EE, E_NE, NE_E, NE_NE)`
where the first subscript refers to the *scalar* component and the second to the
*grid* component.
"""
function compute_full_L_analysis(scalar_total, scalar_enso, scalar_resid,
        grid_total, grid_enso, grid_resid, var_T)
    L_total = compute_L_field(scalar_total, grid_total, var_T)
    L_EE    = compute_L_field(scalar_enso,  grid_enso,  var_T)
    L_E_NE  = compute_L_field(scalar_enso,  grid_resid, var_T)
    L_NE_E  = compute_L_field(scalar_resid, grid_enso,  var_T)
    L_NE_NE = compute_L_field(scalar_resid, grid_resid, var_T)
    return (; total=L_total, EE=L_EE, E_NE=L_E_NE, NE_E=L_NE_E, NE_NE=L_NE_NE)
end

"""
    global_decomp_slices_and_titles(result, label)

For global L (scalar = R̄, grid = Tᵢ):
Return 5 data slices and titles in standard R×T order.
"""
function global_decomp_slices_and_titles(result::NamedTuple, label::String)
    # scalar = R̄, grid = T  →  EE = R̄_E·T_E, E_NE = R̄_E·T_NE, NE_E = R̄_NE·T_E, NE_NE = R̄_NE·T_NE
    slices = [result.total, result.EE, result.E_NE, result.NE_E, result.NE_NE]
    titles = ["$label Total",
              "$label: R̄_E · T_E",   "$label: R̄_E · T_NE",
              "$label: R̄_NE · T_E",  "$label: R̄_NE · T_NE"]
    return slices, titles
end

"""
    regional_decomp_slices_and_titles(result, label)

For regional L (scalar = T_region, grid = Rᵢ):
Return 5 data slices and titles in standard R×T order.
Swaps E_NE and NE_E so labels always read  R_component · T_component.
"""
function regional_decomp_slices_and_titles(result::NamedTuple, label::String)
    # scalar = T, grid = R  →  EE = T_E·R_E, E_NE = T_E·R_NE, NE_E = T_NE·R_E, NE_NE = T_NE·R_NE
    # In R×T convention: EE = R_E·T_E ✓, E_NE → R_NE·T_E, NE_E → R_E·T_NE, NE_NE ✓
    slices = [result.total, result.EE, result.NE_E, result.E_NE, result.NE_NE]
    titles = ["$label Total",
              "$label: Rᵢ_E · T_E",   "$label: Rᵢ_E · T_NE",
              "$label: Rᵢ_NE · T_E",  "$label: Rᵢ_NE · T_NE"]
    return slices, titles
end

"""
    plot_table_heatmap(table_data, row_labels, col_labels; ...)

Plot a heatmap table with annotated cell values, diverging colormap centered at zero.
(Copied from local_remote_paper.jl for self-containedness.)
"""
function plot_table_heatmap(table_data, row_labels, col_labels;
        title="", colorbar_label="", digits=2, separator_rows=[0.5, 2.5],
        figsize=(6, 4), savepath=nothing, clim=nothing)
    absmax = isnothing(clim) ? maximum(abs.(table_data)) : clim
    norm = colors.TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)
    cmap = DAVES_CMAP

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(table_data, cmap=cmap, norm=norm, aspect="auto")

    for i in 0:(size(table_data, 1)-1), j in 0:(size(table_data, 2)-1)
        val = table_data[i+1, j+1]
        text_color = abs(val) > 0.6 * absmax ? "white" : "black"
        ax.text(j, i, string(round(val; digits=digits)), ha="center", va="center",
            fontsize=10, color=text_color, fontweight="bold")
    end

    for y in separator_rows
        ax.axhline(y, color="black", linewidth=2.5)
    end

    ax.set_xticks(pylist(collect(0:length(col_labels)-1)))
    ax.set_xticklabels(pylist(col_labels), fontsize=11)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(pylist(collect(0:length(row_labels)-1)))
    ax.set_yticklabels(pylist(row_labels), fontsize=11)
    ax.tick_params(length=0)

    ax.set_title(title, pad=25, fontsize=13)
    plt.colorbar(im, ax=ax, orientation="vertical", label=colorbar_label, shrink=0.8, pad=0.04)

    if savepath !== nothing
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    end
    return fig, ax
end

"""
    plot_table_heatmap_on_ax!(ax, table_data, row_labels, col_labels, norm, cmap;
                              digits=3, separator_rows=[0.5, 2.5], show_row_labels=true)

Render a single heatmap table on an existing matplotlib axis.
"""
function plot_table_heatmap_on_ax!(ax, table_data, row_labels, col_labels, norm, cmap;
        digits=3, separator_rows=[0.5, 2.5], show_row_labels=true)
    absmax = max(abs(pyconvert(Float64, norm.vmin)), abs(pyconvert(Float64, norm.vmax)))
    im = ax.imshow(table_data, cmap=cmap, norm=norm, aspect="auto")

    for i in 0:(size(table_data, 1)-1), j in 0:(size(table_data, 2)-1)
        val = table_data[i+1, j+1]
        text_color = abs(val) > 0.6 * absmax ? "white" : "black"
        ax.text(j, i, string(round(val; digits=digits)),
            ha="center", va="center", fontsize=8, color=text_color, fontweight="bold")
    end

    for y in separator_rows
        ax.axhline(y, color="black", linewidth=2.0)
    end

    ax.set_xticks(pylist(collect(0:length(col_labels)-1)))
    ax.set_xticklabels(pylist(col_labels), fontsize=9)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(pylist(collect(0:length(row_labels)-1)))
    ax.set_yticklabels(pylist(show_row_labels ? row_labels : fill("", length(row_labels))), fontsize=9)
    ax.tick_params(length=0)
    return im
end

"""
    plot_multi_table_heatmaps(tables, table_titles, row_labels, col_labels; ...)

Plot `N` heatmap tables side-by-side as subplots with a single shared colorbar.
`tables` is a Vector of matrices, `table_titles` is a Vector of strings.
"""
function plot_multi_table_heatmaps(tables, table_titles, row_labels, col_labels;
        suptitle="", colorbar_label="W/m²/K", digits=3, separator_rows=[0.5, 2.5],
        figsize=nothing, savepath=nothing)

    n = length(tables)
    if figsize === nothing
        figsize = (3.6*n + 1.2, 5)
    end

    absmax = maximum(maximum(abs.(t)) for t in tables)
    norm = colors.TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)
    cmap = DAVES_CMAP

    fig, axs = plt.subplots(1, n, figsize=figsize)
    axs_vec = n == 1 ? [axs] : collect(axs)

    for (i, (ax, table, title)) in enumerate(zip(axs_vec, tables, table_titles))
        plot_table_heatmap_on_ax!(ax, table, row_labels, col_labels, norm, cmap;
            digits=digits, separator_rows=separator_rows, show_row_labels=(i==1))
        ax.set_title(title, pad=20, fontsize=10)
    end

    fig.suptitle(suptitle, fontsize=13, y=1.05)
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs,
        orientation="horizontal", label=colorbar_label, shrink=0.6, pad=0.08)

    if savepath !== nothing
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    end
    return fig
end

"""
    compute_L_scalar_decomp(R_total, R_E, R_NE, T_total, T_E, T_NE, var_T)

Compute the 5 scalar L components: total, R_E·T_E, R_E·T_NE, R_NE·T_E, R_NE·T_NE.
"""
function compute_L_scalar_decomp(R_total, R_E, R_NE, T_total, T_E, T_NE, var_T)
    return (
        total = cov(R_total, T_total) / var_T,
        EE    = cov(R_E,  T_E)  / var_T,
        E_NE  = cov(R_E,  T_NE) / var_T,
        NE_E  = cov(R_NE, T_E)  / var_T,
        NE_NE = cov(R_NE, T_NE) / var_T
    )
end

# Ordered radiation labels for table rows
const RAD_ORDER = ["net", "sw", "lw", "sw_clr", "sw_cld", "lw_clr", "lw_cld"]
const RAD_ROW_LABELS = ["Net", "SW", "LW", "SW Clear", "SW Cloud", "LW Clear", "LW Cloud"]
const TABLE_COL_LABELS = ["Total", "Local", "Nonlocal"]
const TABLE_SEPARATOR_ROWS = [0.5, 2.5]
const COMPONENT_TITLES = ["Total", "R_E · T_E", "R_E · T_NE", "R_NE · T_E", "R_NE · T_NE"]
const COMPONENT_KEYS = [:total, :EE, :E_NE, :NE_E, :NE_NE]

"""
    build_region_decomp_tables(rad_grid_decomp, ceres_lat, rad_mask, temp_mask, era5_lat
                               T_total, T_E, T_NE, var_T)

Build 5 tables (7 rad vars × 3 columns) for Total + 4 ENSO cross-terms.
Each row's columns are [Total, f_local*Local, f_nonlocal*Nonlocal].
Returns a Vector of 5 matrices.
"""
function build_region_decomp_tables(rad_grid_decomp, ceres_lat, rad_mask, temp_mask, era5_lat,
        T_total, T_E, T_NE, var_T)

    f_local    = calculate_mask_fractional_area(rad_mask, ceres_lat)
    f_nonlocal = 1.0 - f_local

    temp_mask_frac_area = calculate_mask_fractional_area(temp_mask, era5_lat)

    # Pre-compute spatial means (total, ENSO, resid) × (global, local, nonlocal) for each rad var
    decomp_scalars = Dict{String, NamedTuple}()
    for label in RAD_ORDER
        rd = rad_grid_decomp[label]
        R_glob       = generate_spatial_mean(rd.valid, ceres_lat)
        R_glob_E     = generate_spatial_mean(rd.enso,  ceres_lat)
        R_glob_NE    = generate_spatial_mean(rd.resid, ceres_lat)
        R_local      = generate_spatial_mean(rd.valid, ceres_lat, rad_mask)
        R_local_E    = generate_spatial_mean(rd.enso,  ceres_lat, rad_mask)
        R_local_NE   = generate_spatial_mean(rd.resid, ceres_lat, rad_mask)
        R_nloc       = generate_spatial_mean(rd.valid, ceres_lat, .!rad_mask)
        R_nloc_E     = generate_spatial_mean(rd.enso,  ceres_lat, .!rad_mask)
        R_nloc_NE    = generate_spatial_mean(rd.resid, ceres_lat, .!rad_mask)

        d_glob  = compute_L_scalar_decomp(R_glob,  R_glob_E,  R_glob_NE,  T_total, T_E, T_NE, var_T)
        d_local = compute_L_scalar_decomp(R_local, R_local_E, R_local_NE, T_total, T_E, T_NE, var_T)
        d_nloc  = compute_L_scalar_decomp(R_nloc,  R_nloc_E,  R_nloc_NE,  T_total, T_E, T_NE, var_T)
        decomp_scalars[label] = (; d_glob, d_local, d_nloc)
    end

    tables = Matrix{Float64}[]
    for key in COMPONENT_KEYS
        table = zeros(length(RAD_ORDER), 3)
        for (i, label) in enumerate(RAD_ORDER)
            ds = decomp_scalars[label]
            table[i, 1] = getfield(ds.d_glob,  key)
            table[i, 2] = f_local    * getfield(ds.d_local, key)
            table[i, 3] = f_nonlocal * getfield(ds.d_nloc,  key)
        end

        #Weight by the fractional area of the regions temp mask to get a better sense of the contribution to the local feedback 
        table .*= temp_mask_frac_area

        push!(tables, table)
    end
    return tables
end

"""
    make_and_save_all_L_plots(L_dict, lat, lon, visdir, slices_fn; contour_fn=nothing)

Generate the full suite of L decomposition plots for a given L dictionary
(keyed by "net", "sw", "lw", "sw_clr", "sw_cld", "lw_clr", "lw_cld").
`slices_fn(result, label)` should return `(slices, titles)`.
`contour_fn(ax)` optionally adds region contours.
"""
function make_and_save_all_L_plots(L_dict, lat, lon, visdir, slices_fn;
        contour_fn=nothing, title_prefix="", central_longitude=180)

    proj = ccrs.Robinson(central_longitude=central_longitude)

    function _maybe_contour(fig)
        contour_fn === nothing && return
        for ax in fig.get_axes()
            try contour_fn(ax) catch end
        end
    end

    # 1. L (net) total with its own colorbar
    fig_L = plot_global_heatmap(lat, lon, L_dict["net"].total;
        title  = "$(title_prefix)L Total",
        colorbar_label = "W/m²/K",
        central_longitude = central_longitude)
    contour_fn !== nothing && contour_fn(fig_L.get_axes()[0])
    fig_L.savefig(joinpath(visdir, "L_net_total.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_L)

    # 2. 4-pane net L decomposition
    s, t = slices_fn(L_dict["net"], "Net")
    fig_4 = plot_multiple_levels_rowmajor(lat, lon, s[2:end], (2, 2);
        subtitles = t[2:end], colorbar_label = "W/m²/K", proj = proj)
    fig_4.suptitle("$(title_prefix)Net L Decomposition: ENSO × Non-ENSO", fontsize=14, y=1.02)
    _maybe_contour(fig_4)
    fig_4.savefig(joinpath(visdir, "L_net_4pane_decomposition.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_4)

    # 3. SW / LW 2×5
    sw_s, sw_t = slices_fn(L_dict["sw"], "SW")
    lw_s, lw_t = slices_fn(L_dict["lw"], "LW")
    fig_swlw = plot_multiple_levels_rowmajor(lat, lon,
        vcat(sw_s, lw_s), (2, 5);
        subtitles = vcat(sw_t, lw_t), colorbar_label = "W/m²/K", proj = proj)
    fig_swlw.suptitle("$(title_prefix)L Decomposition: SW and LW", fontsize=14, y=1.02)
    _maybe_contour(fig_swlw)
    fig_swlw.savefig(joinpath(visdir, "L_sw_lw_2x5.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_swlw)

    # 4. SW Clear / Cloud 2×5
    swclr_s, swclr_t = slices_fn(L_dict["sw_clr"], "SW Clr")
    swcld_s, swcld_t = slices_fn(L_dict["sw_cld"], "SW Cld")
    fig_swcc = plot_multiple_levels_rowmajor(lat, lon,
        vcat(swclr_s, swcld_s), (2, 5);
        subtitles = vcat(swclr_t, swcld_t), colorbar_label = "W/m²/K", proj = proj)
    fig_swcc.suptitle("$(title_prefix)L Decomposition: SW Clear vs Cloud", fontsize=14, y=1.02)
    _maybe_contour(fig_swcc)
    fig_swcc.savefig(joinpath(visdir, "L_sw_clear_cloud_2x5.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_swcc)

    # 5. LW Clear / Cloud 2×5
    lwclr_s, lwclr_t = slices_fn(L_dict["lw_clr"], "LW Clr")
    lwcld_s, lwcld_t = slices_fn(L_dict["lw_cld"], "LW Cld")
    fig_lwcc = plot_multiple_levels_rowmajor(lat, lon,
        vcat(lwclr_s, lwcld_s), (2, 5);
        subtitles = vcat(lwclr_t, lwcld_t), colorbar_label = "W/m²/K", proj = proj)
    fig_lwcc.suptitle("$(title_prefix)L Decomposition: LW Clear vs Cloud", fontsize=14, y=1.02)
    _maybe_contour(fig_lwcc)
    fig_lwcc.savefig(joinpath(visdir, "L_lw_clear_cloud_2x5.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_lwcc)
end

"""
    percentile_threshold_func(low_pct, high_pct)

Return a function `f(valid_vals) -> (low, high)` that computes thresholds
from the given percentiles of the data.  Default: 40th and 60th percentile.
"""
function percentile_threshold_func(low_pct::Real=40, high_pct::Real=60)
    return function(valid_vals)
        sorted = sort(valid_vals)
        n = length(sorted)
        low  = sorted[clamp(round(Int, low_pct / 100 * n), 1, n)]
        high = sorted[clamp(round(Int, high_pct / 100 * n), 1, n)]
        return (low, high)
    end
end

const DEFAULT_THRESHOLD_FUNC = percentile_threshold_func(40, 60)

"""
    compute_tiled_L_field(lat, lon, L_field; threshold_func=DEFAULT_THRESHOLD_FUNC, additional_area_factor=1.0)

Create a tiled version of the L field using tile_thresh.jl with pcolormesh.
`threshold_func(valid_vals) -> (low, high)` determines the tiling thresholds.
Each tile color represents the cosine-weighted average of L in that tile,
multiplied by the fractional area of the tile and an optional additional area factor.
"""
function compute_tiled_L_field(lat, lon, L_field; threshold_func=DEFAULT_THRESHOLD_FUNC, additional_area_factor=1.0)
    # Compute threshold from the field being tiled
    valid_vals = L_field[.!isnan.(L_field)]
    low, high = threshold_func(valid_vals)

    # Create criteria function
    criteria_func = standard_criteria_func(low, high)
    
    # Generate tiles
    mask_idxs, masks, mask_criterion_vals = tile_latlon_grid(L_field, criteria_func, standard_adjacency_arr; verbose=false)
    
    # Compute cos weighted average for each tile, multiplied by fractional area
    n_masks = length(masks)
    tiled_field = zeros(size(L_field))
    
    for i in 1:n_masks
        mask = masks[i]
        if !isempty(mask)
            # Create a boolean mask for this tile
            tile_mask = zeros(Bool, size(L_field))
            tile_mask[mask] .= true
            
            # Get L values and corresponding latitudes for this mask
            L_vals = L_field[mask]
            lat_vals = [lat[idx[2]] for idx in mask]  # idx[2] is lat index
            
            # Compute cosine weighted average
            cos_weights = cosd.(lat_vals)
            weighted_avg = sum(L_vals .* cos_weights) / sum(cos_weights)
            
            # Calculate fractional area of this tile
            fractional_area = calculate_mask_fractional_area(tile_mask, lat)
            
            # Multiply weighted average by fractional area and additional factor
            tile_value = weighted_avg * fractional_area * additional_area_factor
            
            # Set all points in this tile to the weighted average times fractional area
            tiled_field[mask] .= tile_value
        end
    end
    
    return tiled_field
end

"""
    plot_tiled_global_heatmap(lat, lon, L_field; title, colorbar_label, threshold_func=DEFAULT_THRESHOLD_FUNC, central_longitude=180, additional_area_factor=1.0, vmax=nothing)

Plot a tiled version of L field using pcolormesh.
"""
function plot_tiled_global_heatmap(lat, lon, L_field; title, colorbar_label, threshold_func=DEFAULT_THRESHOLD_FUNC, central_longitude=180, additional_area_factor=1.0, vmax=nothing)
    # Compute tiled field
    tiled_field = compute_tiled_L_field(lat, lon, L_field; threshold_func=threshold_func, additional_area_factor=additional_area_factor)
    
    # Create figure with Robinson projection
    proj = ccrs.Robinson(central_longitude=central_longitude)
    fig, ax = plt.subplots(subplot_kw=Dict("projection"=>proj))
    
    # Create meshgrid for pcolormesh
    lon_edges = vcat([lon[1] - (lon[2] - lon[1])/2], 
                     [(lon[i] + lon[i+1])/2 for i in 1:length(lon)-1],
                     [lon[end] + (lon[end] - lon[end-1])/2])
    lat_edges = vcat([lat[1] - (lat[2] - lat[1])/2],
                     [(lat[i] + lat[i+1])/2 for i in 1:length(lat)-1], 
                     [lat[end] + (lat[end] - lat[end-1])/2])
    
    LON_mesh, LAT_mesh = np.meshgrid(lon_edges, lat_edges)
    
    # Set color normalization
    if vmax === nothing
        vmax = maximum(abs.(tiled_field[.!isnan.(tiled_field)]))
    end
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    # Plot using pcolormesh
    c = ax.pcolormesh(LON_mesh, LAT_mesh, tiled_field', transform=ccrs.PlateCarree(), 
                     cmap=DAVES_CMAP, norm=norm)
    
    ax.coastlines()
    ax.set_global()
    ax.set_title(title)
    
    plt.colorbar(c, ax=ax, orientation="horizontal", pad=0.05, label=colorbar_label)
    
    return fig
end

"""
    plot_multiple_tiled_levels_rowmajor(lat, lon, data_slices, layout; subtitles, colorbar_label, threshold_func=DEFAULT_THRESHOLD_FUNC, central_longitude=180, additional_area_factor=1.0, vmax=nothing)

Plot multiple tiled data fields in a row-major grid layout using pcolormesh.
"""
function plot_multiple_tiled_levels_rowmajor(lat, lon, data_slices, layout;
        subtitles=nothing, colorbar_label="", threshold_func=DEFAULT_THRESHOLD_FUNC, central_longitude=180, additional_area_factor=1.0, vmax=nothing)
    nrows, ncols = layout
    
    proj = ccrs.Robinson(central_longitude=central_longitude)
    fig, axs = plt.subplots(nrows, ncols,
        subplot_kw=Dict("projection" => proj),
        figsize=(4*ncols, 3*nrows))
    
    # Handle single subplot case
    if nrows * ncols == 1
        axs = [axs]
    elseif nrows == 1 || ncols == 1
        axs = collect(axs)
    else
        axs = collect(Iterators.flatten(axs))
    end
    
    # Create meshgrid for pcolormesh
    lon_edges = vcat([lon[1] - (lon[2] - lon[1])/2], 
                     [(lon[i] + lon[i+1])/2 for i in 1:length(lon)-1],
                     [lon[end] + (lon[end] - lon[end-1])/2])
    lat_edges = vcat([lat[1] - (lat[2] - lat[1])/2],
                     [(lat[i] + lat[i+1])/2 for i in 1:length(lat)-1], 
                     [lat[end] + (lat[end] - lat[end-1])/2])
    
    LON_mesh, LAT_mesh = np.meshgrid(lon_edges, lat_edges)
    
    # Compute maximum absolute value for consistent color scaling
    all_tiled_fields = []
    for i in 1:length(data_slices)
        tiled_field = compute_tiled_L_field(lat, lon, data_slices[i]; threshold_func=threshold_func, additional_area_factor=additional_area_factor)
        push!(all_tiled_fields, tiled_field)
    end
    
    if vmax === nothing
        vmax = maximum(maximum(abs.(field[.!isnan.(field)])) for field in all_tiled_fields if !isempty(field[.!isnan.(field)]))
    end
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    for i in 1:length(data_slices)
        ax = axs[i]
        tiled_field = all_tiled_fields[i]
        
        c = ax.pcolormesh(LON_mesh, LAT_mesh, tiled_field', transform=ccrs.PlateCarree(),
                         cmap=DAVES_CMAP, norm=norm)
        
        ax.coastlines()
        ax.set_global()
        
        if subtitles !== nothing && i <= length(subtitles)
            ax.set_title(subtitles[i])
        end
    end
    
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=DAVES_CMAP),
                ax=axs, orientation="horizontal", label=colorbar_label, 
                pad=0.08, shrink=0.8)
    
    return fig
end

"""
    make_and_save_all_tiled_L_plots(L_dict, lat, lon, visdir, slices_fn; contour_fn=nothing, title_prefix="", central_longitude=180, threshold_func=DEFAULT_THRESHOLD_FUNC, additional_area_factor=1.0, vmax_dict=nothing)

Generate tiled versions of all L decomposition plots using tiles and pcolormesh.
"""
function make_and_save_all_tiled_L_plots(L_dict, lat, lon, visdir, slices_fn;
        contour_fn=nothing, title_prefix="", central_longitude=180, threshold_func=DEFAULT_THRESHOLD_FUNC, additional_area_factor=1.0, vmax_dict=nothing)
    
    function _maybe_contour(fig)
        contour_fn === nothing && return
        for ax in fig.get_axes()
            try contour_fn(ax) catch end
        end
    end

    # 1. Tiled L (net) total with its own colorbar
    fig_L = plot_tiled_global_heatmap(lat, lon, L_dict["net"].total;
        title  = "$(title_prefix)L Total (Tiled)",
        colorbar_label = "W/m²/K",
        threshold_func = threshold_func,
        central_longitude = central_longitude,
        additional_area_factor = additional_area_factor,
        vmax = vmax_dict === nothing ? nothing : vmax_dict["net_total"])
    contour_fn !== nothing && contour_fn(fig_L.get_axes()[0])
    fig_L.savefig(joinpath(visdir, "L_net_total_tiled.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_L)

    # 2. 4-pane tiled net L decomposition
    s, t = slices_fn(L_dict["net"], "Net")
    fig_4 = plot_multiple_tiled_levels_rowmajor(lat, lon, s[2:end], (2, 2);
        subtitles = ["$(t[i]) (Tiled)" for i in 2:length(t)],
        colorbar_label = "W/m²/K", threshold_func = threshold_func,
        central_longitude = central_longitude,
        additional_area_factor = additional_area_factor,
        vmax = vmax_dict === nothing ? nothing : vmax_dict["net_4pane"])
    fig_4.suptitle("$(title_prefix)Net L Decomposition: ENSO × Non-ENSO (Tiled)", fontsize=14, y=1.02)
    _maybe_contour(fig_4)
    fig_4.savefig(joinpath(visdir, "L_net_4pane_decomposition_tiled.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_4)

    # 3. Tiled SW / LW 2×5
    sw_s, sw_t = slices_fn(L_dict["sw"], "SW")
    lw_s, lw_t = slices_fn(L_dict["lw"], "LW")
    all_slices = vcat(sw_s, lw_s)
    all_titles = ["$(t) (Tiled)" for t in vcat(sw_t, lw_t)]
    fig_swlw = plot_multiple_tiled_levels_rowmajor(lat, lon, all_slices, (2, 5);
        subtitles = all_titles, colorbar_label = "W/m²/K",
        threshold_func = threshold_func, central_longitude = central_longitude,
        additional_area_factor = additional_area_factor,
        vmax = vmax_dict === nothing ? nothing : vmax_dict["sw_lw"])
    fig_swlw.suptitle("$(title_prefix)L Decomposition: SW and LW (Tiled)", fontsize=14, y=1.02)
    _maybe_contour(fig_swlw)
    fig_swlw.savefig(joinpath(visdir, "L_sw_lw_2x5_tiled.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_swlw)

    # 4. Tiled SW Clear / Cloud 2×5
    swclr_s, swclr_t = slices_fn(L_dict["sw_clr"], "SW Clr")
    swcld_s, swcld_t = slices_fn(L_dict["sw_cld"], "SW Cld")
    all_slices = vcat(swclr_s, swcld_s)
    all_titles = ["$(t) (Tiled)" for t in vcat(swclr_t, swcld_t)]
    fig_swcc = plot_multiple_tiled_levels_rowmajor(lat, lon, all_slices, (2, 5);
        subtitles = all_titles, colorbar_label = "W/m²/K",
        threshold_func = threshold_func, central_longitude = central_longitude,
        additional_area_factor = additional_area_factor,
        vmax = vmax_dict === nothing ? nothing : vmax_dict["sw_cc"])
    fig_swcc.suptitle("$(title_prefix)L Decomposition: SW Clear vs Cloud (Tiled)", fontsize=14, y=1.02)
    _maybe_contour(fig_swcc)
    fig_swcc.savefig(joinpath(visdir, "L_sw_clear_cloud_2x5_tiled.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_swcc)

    # 5. Tiled LW Clear / Cloud 2×5
    lwclr_s, lwclr_t = slices_fn(L_dict["lw_clr"], "LW Clr")
    lwcld_s, lwcld_t = slices_fn(L_dict["lw_cld"], "LW Cld")
    all_slices = vcat(lwclr_s, lwcld_s)
    all_titles = ["$(t) (Tiled)" for t in vcat(lwclr_t, lwcld_t)]
    fig_lwcc = plot_multiple_tiled_levels_rowmajor(lat, lon, all_slices, (2, 5);
        subtitles = all_titles, colorbar_label = "W/m²/K",
        threshold_func = threshold_func, central_longitude = central_longitude,
        additional_area_factor = additional_area_factor,
        vmax = vmax_dict === nothing ? nothing : vmax_dict["lw_cc"])
    fig_lwcc.suptitle("$(title_prefix)L Decomposition: LW Clear vs Cloud (Tiled)", fontsize=14, y=1.02)
    _maybe_contour(fig_lwcc)
    fig_lwcc.savefig(joinpath(visdir, "L_lw_clear_cloud_2x5_tiled.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_lwcc)
end

# ============================================================
# Load Data  (mirrors local_remote_paper.jl)
# ============================================================

println("Loading ERA5 t2m...")
date_range = (Date(2000, 3), Date(2024, 2, 28))
era5_data, era5_coords = load_era5_data(["t2m"], date_range)

println("Loading CERES radiation...")
ceres_vars_to_load = [
    "toa_net_all_mon", "toa_net_sw_mon", "toa_net_lw_mon",
    "toa_sw_clr_t_mon", "toa_lw_clr_t_mon",
    "toa_sw_all_mon", "toa_lw_all_mon", "solar_mon"]
ceres_data, ceres_coords = load_new_ceres_data(ceres_vars_to_load, date_range)

ceres_coords["time"] = round_dates_down_to_nearest_month(ceres_coords["time"])
common_time = ceres_coords["time"]
@assert all(common_time .== round_dates_down_to_nearest_month(era5_coords["time"])) "ERA5 / CERES time mismatch!"

# Construct cloudy-sky radiation components (sign conventions match local_remote_paper.jl)
set!(ceres_data, "toa_sw_clr_t_mon", ceres_data["solar_mon"] .- ceres_data["toa_sw_clr_t_mon"])
set!(ceres_data, "toa_sw_cld_t_mon", ceres_data["toa_net_sw_mon"] .- ceres_data["toa_sw_clr_t_mon"])
set!(ceres_data, "toa_lw_cld_t_mon", -1 .* (ceres_data["toa_lw_all_mon"] .- ceres_data["toa_lw_clr_t_mon"]))
set!(ceres_data, "toa_lw_clr_t_mon", ceres_data["toa_lw_clr_t_mon"] .* -1)

# Deseasonalize and detrend twice
println("Deseasonalizing and detrending...")
float_time = calc_float_time.(common_time)
month_groups = groupfind(month, common_time)
for datadict in [era5_data, ceres_data]
    for var_data in datadict
        for slice in eachslice(var_data; dims=(1,2))
            deseasonalize_and_detrend_precalculated_groups_twice!(slice, float_time, month_groups;
                aggfunc=mean, trendfunc=least_squares_fit)
        end
    end
end

# Load region masks
println("Loading region masks...")
regions_to_inspect = ["SEPac_feedback_definition", "NEPac", "SEAtl"]
jldpath = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/"
era5_region_mask_dict = jldopen(joinpath(jldpath, "stratocumulus_region_masks.jld2"))["regional_masks_era5"]
rad_region_mask_dict  = jldopen(joinpath(jldpath, "stratocumulus_exclusion_masks.jld2"))["stratocum_masks"]

# Load and prepare ENSO lag matrix
println("Loading ENSO data...")
enso_data, enso_dates_raw = load_enso_data(date_range)
enso_times = round_dates_down_to_nearest_month(enso_dates_raw["time"])
lags = -24:24
lag_columns = ["oni_lag_$lag" for lag in lags]
n_enso = length(enso_times)
enso_lag_matrix_full = Matrix{Union{Float64, Missing}}(missing, n_enso, length(lag_columns))
for (j, col) in enumerate(lag_columns)
    haskey(enso_data, col) && (enso_lag_matrix_full[:, j] .= enso_data[col])
end
valid_enso_rows = [all(!ismissing, enso_lag_matrix_full[i, :]) for i in 1:n_enso]
enso_times_valid = enso_times[valid_enso_rows]
enso_lag_matrix = Float64.(enso_lag_matrix_full[valid_enso_rows, :])
enso_matrix_matched, data_valid_idx = precompute_enso_match(common_time, enso_times_valid, enso_lag_matrix)

# Mapping from short labels to CERES variable names
rad_vars = Dict(
    "net"    => "toa_net_all_mon",
    "sw"     => "toa_net_sw_mon",
    "lw"     => "toa_net_lw_mon",
    "sw_clr" => "toa_sw_clr_t_mon",
    "sw_cld" => "toa_sw_cld_t_mon",
    "lw_clr" => "toa_lw_clr_t_mon",
    "lw_cld" => "toa_lw_cld_t_mon")

# ============================================================
# ENSO Decomposition — Temperature (ERA5 grid)
# ============================================================

println("Performing pointwise PLS on local T (this may take a while)...")
local_T  = era5_data["t2m"]
T_valid  = local_T[:, :, data_valid_idx]
T_enso   = pointwise_pls(enso_matrix_matched, T_valid; n_components=1)
T_resid  = T_valid .- T_enso
println("  Pointwise PLS on T complete.")

# Global mean temperature (NOT decomposed)
global_mean_T = generate_spatial_mean(era5_data["t2m"], era5_coords["latitude"])
var_T = var(global_mean_T[data_valid_idx])

# ============================================================
# ENSO Decomposition — Gridded Radiation (CERES grid)
# ============================================================

# Pre-decompose every gridded radiation variable via pointwise PLS
# (needed for per-region L_region where Rᵢ is the gridded quantity)
rad_grid_decomp = Dict{String, NamedTuple}()
for (label, var_name) in rad_vars
    println("Pointwise PLS on gridded $label...")
    R_valid = ceres_data[var_name][:, :, data_valid_idx]
    R_enso  = pointwise_pls(enso_matrix_matched, R_valid; n_components=1)
    rad_grid_decomp[label] = (; valid=R_valid, enso=R_enso, resid=R_valid .- R_enso)
end
println("  Pointwise PLS on all radiation variables complete.")

# ============================================================
# Global mean radiation — scalar decomposition
# ============================================================

global_rad = Dict(k => generate_spatial_mean(ceres_data[v], ceres_coords["latitude"])
                  for (k, v) in rad_vars)

global_rad_decomp = Dict{String, NamedTuple}()
for (label, _) in rad_vars
    R_valid = global_rad[label][data_valid_idx]
    R_enso, R_resid = enso_decompose_scalar(enso_matrix_matched, R_valid)
    global_rad_decomp[label] = (; valid=R_valid, enso=R_enso, resid=R_resid)
end

# ====================================================================
# Global L analysis:  L(x) = Cov(R̄, Tᵢ(x)) / Var(T̄)
# scalar = R̄ (global mean rad),  grid = Tᵢ (ERA5 temperature)
# ====================================================================

println("\n=== Global L analysis ===")
L_global = Dict{String, NamedTuple}()
for (label, _) in rad_vars
    rd = global_rad_decomp[label]
    L_global[label] = compute_full_L_analysis(
        rd.valid, rd.enso, rd.resid,
        T_valid, T_enso, T_resid, var_T)
    err = maximum(abs.(L_global[label].total .- (L_global[label].EE .+ L_global[label].E_NE .+ L_global[label].NE_E .+ L_global[label].NE_NE)))
    println("  $label  max decomp error: $(round(err; sigdigits=3))")
end

# --- Global L plots ---
println("Generating global L plots...")
make_and_save_all_L_plots(L_global,
    era5_coords["latitude"], era5_coords["longitude"],
    visdir_base, global_decomp_slices_and_titles;
    title_prefix = "[Global] ")

# --- Global L tiled plots ---
println("Generating global L tiled plots...")
make_and_save_all_tiled_L_plots(L_global,
    era5_coords["latitude"], era5_coords["longitude"],
    visdir_base, global_decomp_slices_and_titles;
    title_prefix = "[Global] ")

# --- Global summary table (area-weighted mean of each L field) ---
println("Generating global summary table...")
begin
    era5_lat = era5_coords["latitude"]
    global_table_data = zeros(length(RAD_ORDER), length(COMPONENT_KEYS))
    for (i, label) in enumerate(RAD_ORDER)
        Lr = L_global[label]
        for (j, key) in enumerate(COMPONENT_KEYS)
            field = getfield(Lr, key)
            global_table_data[i, j] = sum(field .* cosd.(era5_lat')) / sum(ones(size(field)) .* cosd.(era5_lat'))
        end
    end
    plot_table_heatmap(global_table_data, RAD_ROW_LABELS, COMPONENT_TITLES;
        title = "Global area-weighted λ = ⟨L⟩  [W/m²/K]",
        colorbar_label = "W/m²/K", digits = 2,
        separator_rows = TABLE_SEPARATOR_ROWS,
        figsize = (8, 4),
        savepath = joinpath(visdir_base, "L_global_summary_table.png"))
end

# ====================================================================
# Per-Region L analysis:  L_region(x) = Cov(Rᵢ(x), T_region) / Var(T̄)
# scalar = T_region (regional mean temp),  grid = Rᵢ (CERES radiation)
# ====================================================================

ceres_lat = ceres_coords["latitude"]
ceres_lon = ceres_coords["longitude"]

# First pass: compute all L_region dictionaries and store metadata
println("\n=== Computing L_region for all regions ===")
region_L_dict = Dict{String, Any}()

for region_name in regions_to_inspect
    println("\n=== Region: $region_name ===")
    
    rad_mask  = rad_region_mask_dict[region_name]
    temp_mask = era5_region_mask_dict[region_name]

    # Regional temperature mean → scalar decomposition
    T_region_full = generate_spatial_mean(era5_data["t2m"], era5_coords["latitude"], temp_mask)
    T_region_valid = T_region_full[data_valid_idx]
    T_region_enso, T_region_resid = enso_decompose_scalar(enso_matrix_matched, T_region_valid)

    # Compute L_region for each radiation variable
    L_region = Dict{String, NamedTuple}()
    for (label, _) in rad_vars
        rd = rad_grid_decomp[label]
        L_region[label] = compute_full_L_analysis(
            T_region_valid, T_region_enso, T_region_resid,
            rd.valid, rd.enso, rd.resid, var_T)
        err = maximum(abs.(L_region[label].total .- (L_region[label].EE .+ L_region[label].E_NE .+ L_region[label].NE_E .+ L_region[label].NE_NE)))
        println("  $label  max decomp error: $(round(err; sigdigits=3))")
    end
    
    # Calculate temperature mask fractional area
    temp_mask_frac_area = calculate_mask_fractional_area(temp_mask, era5_lat)
    
    # Store all data for this region
    region_L_dict[region_name] = Dict(
        "L_region" => L_region,
        "rad_mask" => rad_mask,
        "temp_mask" => temp_mask,
        "temp_mask_frac_area" => temp_mask_frac_area,
        "T_region_valid" => T_region_valid,
        "T_region_enso" => T_region_enso,
        "T_region_resid" => T_region_resid
    )
end

# Compute common vmax values across all regions for each plot type
println("\n=== Computing common colorbars ===")

function compute_common_vmax_for_regions(region_L_dict, slices_fn, ceres_lat, ceres_lon; threshold_func=DEFAULT_THRESHOLD_FUNC)
    vmax_dict = Dict{String, Float64}()
    
    # Helper to compute tiled fields for all regions
    function get_all_tiled_fields(field_getter)
        all_fields = []
        for (region_name, data) in region_L_dict
            L_region = data["L_region"]
            temp_mask_frac_area = data["temp_mask_frac_area"]
            
            field = field_getter(L_region)

            tiled = compute_tiled_L_field(ceres_lat, ceres_lon, field;
                                         threshold_func=threshold_func, additional_area_factor=temp_mask_frac_area)
            push!(all_fields, tiled)
        end
        return all_fields
    end
    
    # Net total
    all_net_total = get_all_tiled_fields(L -> L["net"].total)
    vmax_dict["net_total"] = maximum(maximum(abs.(f[.!isnan.(f)])) for f in all_net_total)
    
    # Net 4-pane decomposition
    all_4pane_fields = []
    for (region_name, data) in region_L_dict
        L_region = data["L_region"]
        temp_mask_frac_area = data["temp_mask_frac_area"]
        s, t = slices_fn(L_region["net"], "Net")

        for field in s[2:end]
            tiled = compute_tiled_L_field(ceres_lat, ceres_lon, field;
                                         threshold_func=threshold_func, additional_area_factor=temp_mask_frac_area)
            push!(all_4pane_fields, tiled)
        end
    end
    vmax_dict["net_4pane"] = maximum(maximum(abs.(f[.!isnan.(f)])) for f in all_4pane_fields)
    
    # SW/LW
    all_swlw_fields = []
    for (region_name, data) in region_L_dict
        L_region = data["L_region"]
        temp_mask_frac_area = data["temp_mask_frac_area"]
        sw_s, sw_t = slices_fn(L_region["sw"], "SW")
        lw_s, lw_t = slices_fn(L_region["lw"], "LW")

        for field in vcat(sw_s, lw_s)
            tiled = compute_tiled_L_field(ceres_lat, ceres_lon, field;
                                         threshold_func=threshold_func, additional_area_factor=temp_mask_frac_area)
            push!(all_swlw_fields, tiled)
        end
    end
    vmax_dict["sw_lw"] = maximum(maximum(abs.(f[.!isnan.(f)])) for f in all_swlw_fields)
    
    # SW Clear/Cloud
    all_swcc_fields = []
    for (region_name, data) in region_L_dict
        L_region = data["L_region"]
        temp_mask_frac_area = data["temp_mask_frac_area"]
        swclr_s, swclr_t = slices_fn(L_region["sw_clr"], "SW Clr")
        swcld_s, swcld_t = slices_fn(L_region["sw_cld"], "SW Cld")

        for field in vcat(swclr_s, swcld_s)
            tiled = compute_tiled_L_field(ceres_lat, ceres_lon, field;
                                         threshold_func=threshold_func, additional_area_factor=temp_mask_frac_area)
            push!(all_swcc_fields, tiled)
        end
    end
    vmax_dict["sw_cc"] = maximum(maximum(abs.(f[.!isnan.(f)])) for f in all_swcc_fields)
    
    # LW Clear/Cloud
    all_lwcc_fields = []
    for (region_name, data) in region_L_dict
        L_region = data["L_region"]
        temp_mask_frac_area = data["temp_mask_frac_area"]
        lwclr_s, lwclr_t = slices_fn(L_region["lw_clr"], "LW Clr")
        lwcld_s, lwcld_t = slices_fn(L_region["lw_cld"], "LW Cld")

        for field in vcat(lwclr_s, lwcld_s)
            tiled = compute_tiled_L_field(ceres_lat, ceres_lon, field;
                                         threshold_func=threshold_func, additional_area_factor=temp_mask_frac_area)
            push!(all_lwcc_fields, tiled)
        end
    end
    vmax_dict["lw_cc"] = maximum(maximum(abs.(f[.!isnan.(f)])) for f in all_lwcc_fields)
    
    return vmax_dict
end

common_vmax = compute_common_vmax_for_regions(region_L_dict, regional_decomp_slices_and_titles, ceres_lat, ceres_lon)

println("Common vmax values:")
for (key, val) in common_vmax
    println("  $key: $(round(val; digits=4))")
end

# Second pass: generate all plots with common colorbars
for region_name in regions_to_inspect
    println("\n=== Generating plots for region: $region_name ===")
    region_visdir = joinpath(visdir_base, region_name)
    mkpath(region_visdir)
    
    # Retrieve stored data
    region_data = region_L_dict[region_name]
    L_region = region_data["L_region"]
    rad_mask = region_data["rad_mask"]
    temp_mask = region_data["temp_mask"]
    temp_mask_frac_area = region_data["temp_mask_frac_area"]
    T_region_valid = region_data["T_region_valid"]
    T_region_enso = region_data["T_region_enso"]
    T_region_resid = region_data["T_region_resid"]

    # Contour helper for this region
    contour_fn = ax -> add_region_contours!(ax,
        ceres_lat, ceres_lon, rad_mask,
        era5_coords["latitude"], era5_coords["longitude"], temp_mask)

    central_lon = region_name == "SEAtl" ? 0 : 180

    println("  Generating plots for $region_name...")
    make_and_save_all_L_plots(L_region,
        ceres_lat, ceres_lon,
        region_visdir, regional_decomp_slices_and_titles;
        contour_fn = contour_fn,
        title_prefix = "[$region_name] ",
        central_longitude = central_lon)
    
    println("  Generating tiled plots for $region_name...")
    make_and_save_all_tiled_L_plots(L_region,
        ceres_lat, ceres_lon,
        region_visdir, regional_decomp_slices_and_titles;
        contour_fn = contour_fn,
        title_prefix = "[$region_name] ",
        central_longitude = central_lon,
        additional_area_factor = temp_mask_frac_area,
        vmax_dict = common_vmax)

    # --- Per-region 5-panel summary table (area-weighted, local/nonlocal split) ---
    println("  Generating decomposition tables for $region_name...")
    tables = build_region_decomp_tables(
        rad_grid_decomp, ceres_lat, rad_mask, temp_mask, era5_lat,
        T_region_valid, T_region_enso, T_region_resid, var_T)

    plot_multi_table_heatmaps(tables, COMPONENT_TITLES, RAD_ROW_LABELS, TABLE_COL_LABELS;
        suptitle = "[$region_name] L Decomposition (area-weighted)  [W/m²/K]",
        digits = 3, separator_rows = TABLE_SEPARATOR_ROWS,
        savepath = joinpath(region_visdir, "L_decomposition_5panel_table.png"))
end

println("\nAll plots saved under: $visdir_base")
