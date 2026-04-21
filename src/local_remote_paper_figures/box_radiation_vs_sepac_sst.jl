# Scatter analysis of area-weighted CERES radiation (Net, SW, LW) in the
# Maritime Continent and Warm Pool boxes against the SEPac SST index,
# with ENSO / non-ENSO decomposition via PLS using both ONI and ELI indices.

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../pls_regressor/pls_functions.jl")

using JLD2, Dates, StatsBase, DataFrames, Printf, Statistics, LinearAlgebra

visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/box_radiation_vs_sepac_sst"
mkpath(visdir)

date_range = (Date(2000, 3), Date(2024, 2, 28))

# ══════════════════════════════════════════════════════════════════════════════
# Define bounding boxes  (lat_min, lon_min, lat_max, lon_max)  —  0-360 lon
# ══════════════════════════════════════════════════════════════════════════════

boxes = Dict(
    "Maritime Continent" => (lat_min=-10.66, lon_min=104.77, lat_max=16.89, lon_max=147.39),
    "Warm Pool"          => (lat_min=-3.50,  lon_min=154.69, lat_max=16.97, lon_max=197.58)   # -162.42+360
)

# ══════════════════════════════════════════════════════════════════════════════
# Load and preprocess CERES radiation data
# ══════════════════════════════════════════════════════════════════════════════

ceres_vars = ["toa_net_all_mon", "toa_net_sw_mon", "toa_net_lw_mon"]
ceres_data, ceres_coords = load_new_ceres_data(ceres_vars, date_range)
ceres_coords["time"] = round_dates_down_to_nearest_month(ceres_coords["time"])
common_time = ceres_coords["time"]

ceres_lat = Float64.(ceres_coords["latitude"])
ceres_lon = Float64.(ceres_coords["longitude"])
ceres_lon_360 = mod.(ceres_lon, 360)

# Deseasonalize and detrend (twice, matching reference script)
float_time = calc_float_time.(common_time)
month_groups = groupfind(month, common_time)
for var_data in ceres_data
    for slice in eachslice(var_data; dims=(1, 2))
        deseasonalize_and_detrend_precalculated_groups_twice!(slice, float_time, month_groups;
            aggfunc=mean, trendfunc=least_squares_fit)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Create box masks
# ══════════════════════════════════════════════════════════════════════════════

function make_box_mask(lat, lon_360, box)
    mask = falses(length(lon_360), length(lat))
    for (j, la) in enumerate(lat), (i, lo) in enumerate(lon_360)
        in_lat = box.lat_min <= la <= box.lat_max
        if box.lon_min <= box.lon_max
            in_lon = box.lon_min <= lo <= box.lon_max
        else
            in_lon = lo >= box.lon_min || lo <= box.lon_max
        end
        mask[i, j] = in_lat && in_lon
    end
    return mask
end

box_masks = Dict(name => make_box_mask(ceres_lat, ceres_lon_360, box) for (name, box) in boxes)

# ══════════════════════════════════════════════════════════════════════════════
# Plot bounding boxes on a map
# ══════════════════════════════════════════════════════════════════════════════

begin
    fig, ax = plt.subplots(figsize=(12, 6),
        subplot_kw=Dict("projection" => ccrs.Robinson(central_longitude=150)))
    ax.set_global()
    ax.coastlines()
    ax.stock_img()

    box_colors = Dict("Maritime Continent" => "red", "Warm Pool" => "blue")

    for (box_name, box) in boxes
        col = box_colors[box_name]
        # Convert to -180/180 for PlateCarree plotting
        l1 = box.lon_min > 180 ? box.lon_min - 360 : box.lon_min
        l2 = box.lon_max > 180 ? box.lon_max - 360 : box.lon_max

        if l1 <= l2
            corners = [(l1, box.lat_min), (l2, box.lat_min),
                       (l2, box.lat_max), (l1, box.lat_max)]
            plot_polygon_on_ax!(ax, corners; color=col, alpha=0.3, linewidth=2)
        else
            # Crosses dateline → split into east and west halves
            plot_polygon_on_ax!(ax,
                [(l1, box.lat_min), (180.0, box.lat_min),
                 (180.0, box.lat_max), (l1, box.lat_max)];
                color=col, alpha=0.3, linewidth=2)
            plot_polygon_on_ax!(ax,
                [(-180.0, box.lat_min), (l2, box.lat_min),
                 (l2, box.lat_max), (-180.0, box.lat_max)];
                color=col, alpha=0.3, linewidth=2)
        end

        mid_lon = (box.lon_min + box.lon_max) / 2
        mid_lon = mid_lon > 180 ? mid_lon - 360 : mid_lon
        ax.text(mid_lon, (box.lat_min + box.lat_max) / 2, box_name,
            transform=ccrs.PlateCarree(), ha="center", va="center",
            fontsize=9, fontweight="bold", color=col,
            bbox=Dict("boxstyle" => "round,pad=0.3", "facecolor" => "white", "alpha" => 0.8))
    end

    ax.set_title("Bounding Boxes: Maritime Continent & Warm Pool")
    fig.savefig(joinpath(visdir, "bounding_boxes.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    println("Saved: bounding_boxes.png")
end

# ══════════════════════════════════════════════════════════════════════════════
# Compute area-weighted means in each box
# ══════════════════════════════════════════════════════════════════════════════

var_labels = Dict(
    "toa_net_all_mon" => "Net",
    "toa_net_sw_mon"  => "SW",
    "toa_net_lw_mon"  => "LW"
)

box_means = Dict{String, Dict{String, Vector{Float64}}}()
for (box_name, mask) in box_masks
    box_means[box_name] = Dict(
        var_name => generate_spatial_mean(ceres_data[var_name], ceres_lat, mask)
        for var_name in ceres_vars
    )
end

# ══════════════════════════════════════════════════════════════════════════════
# Load SEPac SST index and align times
# ══════════════════════════════════════════════════════════════════════════════

sepac_sst_data, sepac_coords = load_sepac_sst_index(date_range; lags=[0])
sepac_sst_times = sepac_coords["time"]
sepac_sst_raw = Float64.(sepac_sst_data["SEPac_SST_Index_Lag0"])

sepac_time_idx = indexin(common_time, sepac_sst_times)
if any(isnothing, sepac_time_idx)
    error("Some CERES times not found in SEPac SST index!")
end
sepac_sst_matched = sepac_sst_raw[sepac_time_idx]

# Deseasonalize and detrend SEPac SST to match other data processing
sepac_sst = copy(sepac_sst_matched)
deseasonalize_and_detrend_precalculated_groups_twice!(sepac_sst, float_time, month_groups;
    aggfunc=mean, trendfunc=least_squares_fit)

# ══════════════════════════════════════════════════════════════════════════════
# Helper functions for ENSO decomposition analysis
# ══════════════════════════════════════════════════════════════════════════════

"""
    build_enso_lag_matrix(enso_index_vector, lags)

Build a lagged matrix from a single ENSO index vector using time_lag from utilfuncs.
Returns matrix of size (n_times, n_lags) where each column is the index at a different lag.
"""
function build_enso_lag_matrix(enso_index_vector::Vector{Float64}, lags)
    n = length(enso_index_vector)
    n_lags = length(lags)
    lag_matrix = Matrix{Float64}(undef, n, n_lags)
    
    for (j, lag) in enumerate(lags)
        lagged_values = time_lag(enso_index_vector, lag)
        # Convert any missing values to NaN
        for i in 1:n
            lag_matrix[i, j] = ismissing(lagged_values[i]) ? NaN : Float64(lagged_values[i])
        end
    end
    
    return lag_matrix
end

"""
    perform_enso_decomposition(enso_matrix, common_time, enso_times, 
                               sepac_sst, box_means, ceres_vars, var_labels, n_components)

Perform PLS decomposition of SEPac SST and box radiation means into ENSO and non-ENSO components.
Returns dictionaries with decomposed components.
"""
function perform_enso_decomposition(enso_matrix_full, enso_times_full, common_time,
                                   sepac_sst, box_means, ceres_vars, var_labels, n_components)
    # Match times and filter out any rows with missing values
    enso_matrix_matched, data_valid_idx = precompute_enso_match(common_time, enso_times_full, enso_matrix_full)
    
    # Decompose SEPac SST
    sepac_sst_for_pls = sepac_sst[data_valid_idx]
    pls_sepac = make_pls_regressor(enso_matrix_matched, sepac_sst_for_pls, n_components; print_updates=false)
    sepac_sst_enso = vec(predict(pls_sepac, enso_matrix_matched))
    sepac_sst_nonenso = sepac_sst_for_pls .- sepac_sst_enso
    
    println("  SEPac SST ENSO PLS (n=$n_components) r = $(round(cor(sepac_sst_enso, sepac_sst_for_pls), digits=3))")
    
    # Decompose box radiation means
    box_enso = Dict{String, Dict{String, Vector{Float64}}}()
    box_nonenso = Dict{String, Dict{String, Vector{Float64}}}()
    
    for (box_name, means) in box_means
        box_enso[box_name] = Dict{String, Vector{Float64}}()
        box_nonenso[box_name] = Dict{String, Vector{Float64}}()
        
        for var_name in ceres_vars
            y = means[var_name][data_valid_idx]
            pls_model = make_pls_regressor(enso_matrix_matched, y, n_components; print_updates=false)
            y_enso = vec(predict(pls_model, enso_matrix_matched))
            y_nonenso = y .- y_enso
            
            box_enso[box_name][var_name] = y_enso
            box_nonenso[box_name][var_name] = y_nonenso
            
            println("  PLS (n=$n_components) r ($box_name, $(var_labels[var_name])): $(round(cor(y_enso, y), digits=3))")
        end
    end
    
    return (sepac_sst_for_pls=sepac_sst_for_pls, 
            sepac_sst_enso=sepac_sst_enso, 
            sepac_sst_nonenso=sepac_sst_nonenso,
            box_enso=box_enso, 
            box_nonenso=box_nonenso,
            data_valid_idx=data_valid_idx)
end

# ══════════════════════════════════════════════════════════════════════════════
# Scatter-plot helper: 4x3 stacked plot (rows: Total, ENSO, Non-ENSO, ENSO PLS Fit; cols: Net, SW, LW)
# ══════════════════════════════════════════════════════════════════════════════

function make_scatter_3x3_stacked(x_total, x_enso, x_nonenso,
                                   y_total_dict, y_enso_dict, y_nonenso_dict,
                                   var_names, var_labels;
                                   suptitle, savepath)
    fig, axs = plt.subplots(4, 3, figsize=(15, 17))
    
    row_labels = ["Total", "ENSO", "Non-ENSO", "ENSO PLS Fit"]
    
    for (col_idx, var_name) in enumerate(var_names)
        # Row 1: Total SEPac SST vs Total Radiation
        ax = axs[0, col_idx - 1]
        x_data = x_total
        y = y_total_dict[var_name]
        r = cor(x_data, y)
        β = cov(x_data, y) / Statistics.var(x_data)
        α = mean(y) - β * mean(x_data)
        
        ax.scatter(x_data, y, s=8, alpha=0.5, color="steelblue")
        x_range = [minimum(x_data), maximum(x_data)]
        ax.plot(x_range, α .+ β .* x_range, "r-", linewidth=1.5)
        
        ax.set_ylabel("$(var_labels[var_name]) (W/m²)")
        title_text = "$(var_labels[var_name])   r=$(round(r, digits=3)), β=$(round(β, digits=2))"
        ax.set_title(title_text)
        
        if col_idx == 1
            ax.text(-0.25, 0.5, row_labels[1]; transform=ax.transAxes,
                   fontsize=11, fontweight="bold", va="center", ha="right", rotation=90)
        end
        
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
        
        # Row 2: ENSO SEPac SST vs ENSO Radiation
        ax = axs[1, col_idx - 1]
        x_data = x_enso
        y = y_enso_dict[var_name]
        r = cor(x_data, y)
        β = cov(x_data, y) / Statistics.var(x_data)
        α = mean(y) - β * mean(x_data)
        
        ax.scatter(x_data, y, s=8, alpha=0.5, color="steelblue")
        x_range = [minimum(x_data), maximum(x_data)]
        ax.plot(x_range, α .+ β .* x_range, "r-", linewidth=1.5)
        
        ax.set_ylabel("$(var_labels[var_name]) (W/m²)")
        ax.set_title("r=$(round(r, digits=3)), β=$(round(β, digits=2))")
        
        if col_idx == 1
            ax.text(-0.25, 0.5, row_labels[2]; transform=ax.transAxes,
                   fontsize=11, fontweight="bold", va="center", ha="right", rotation=90)
        end
        
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
        
        # Row 3: Non-ENSO SEPac SST vs Non-ENSO Radiation
        ax = axs[2, col_idx - 1]
        x_data = x_nonenso
        y = y_nonenso_dict[var_name]
        r = cor(x_data, y)
        β = cov(x_data, y) / Statistics.var(x_data)
        α = mean(y) - β * mean(x_data)
        
        ax.scatter(x_data, y, s=8, alpha=0.5, color="steelblue")
        x_range = [minimum(x_data), maximum(x_data)]
        ax.plot(x_range, α .+ β .* x_range, "r-", linewidth=1.5)
        
        ax.set_ylabel("$(var_labels[var_name]) (W/m²)")
        ax.set_title("r=$(round(r, digits=3)), β=$(round(β, digits=2))")
        
        if col_idx == 1
            ax.text(-0.25, 0.5, row_labels[3]; transform=ax.transAxes,
                   fontsize=11, fontweight="bold", va="center", ha="right", rotation=90)
        end
        
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
        
        # Row 4: ENSO Radiation (PLS fit) vs Total Radiation
        ax = axs[3, col_idx - 1]
        x_data = y_enso_dict[var_name]  # ENSO component of radiation
        y = y_total_dict[var_name]       # Total radiation
        r = cor(x_data, y)
        β = cov(x_data, y) / Statistics.var(x_data)
        α = mean(y) - β * mean(x_data)
        
        ax.scatter(x_data, y, s=8, alpha=0.5, color="steelblue")
        x_range = [minimum(x_data), maximum(x_data)]
        ax.plot(x_range, α .+ β .* x_range, "r-", linewidth=1.5)
        
        ax.set_xlabel("ENSO $(var_labels[var_name]) (W/m²)")
        ax.set_ylabel("Total $(var_labels[var_name]) (W/m²)")
        ax.set_title("r=$(round(r, digits=3)), β=$(round(β, digits=2))")
        
        if col_idx == 1
            ax.text(-0.25, 0.5, row_labels[4]; transform=ax.transAxes,
                   fontsize=11, fontweight="bold", va="center", ha="right", rotation=90)
        end
        
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
    end
    
    # Set x-label for rows 1-3 (SEPac SST)
    for col_idx in 1:3
        axs[2, col_idx - 1].set_xlabel("SEPac SST (K)")
    end
    
    fig.suptitle(suptitle, fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
end

# ══════════════════════════════════════════════════════════════════════════════
# Load and prepare ENSO indices (ONI and ELI) with complete lag data
# ══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("Loading ENSO indices and determining valid time range")
println("="^80)

lags = -24:24

# Load ONI data
enso_data, enso_dates_raw = load_enso_data(date_range)
enso_times = round_dates_down_to_nearest_month(enso_dates_raw["time"])

lag_columns = ["oni_lag_$lag" for lag in lags]
n_enso = length(enso_times)
enso_lag_matrix_full = Matrix{Union{Float64, Missing}}(missing, n_enso, length(lag_columns))
for (j, col) in enumerate(lag_columns)
    if haskey(enso_data, col)
        enso_lag_matrix_full[:, j] .= enso_data[col]
    end
end
valid_oni_rows = [all(!ismissing, enso_lag_matrix_full[i, :]) for i in 1:n_enso]
oni_times_complete = enso_times[valid_oni_rows]
oni_lag_matrix_complete = Float64.(enso_lag_matrix_full[valid_oni_rows, :])

println("ONI: $(length(oni_times_complete)) months with complete lag data")

# Load ELI data (lag 0 only to create our own lags)
eli_data_raw, eli_coords_raw = load_eli_data(date_range; lags=[0])
eli_times_raw = round_dates_down_to_nearest_month(eli_coords_raw["time"])
eli_values_raw = eli_data_raw["ELI_Lag0"]

# Filter out missing ELI values upfront
valid_eli_mask = .!ismissing.(eli_values_raw)
eli_times_nonmissing = eli_times_raw[valid_eli_mask]
eli_values_nonmissing = Float64.(eli_values_raw[valid_eli_mask])

println("ELI: $(length(eli_times_nonmissing)) months with non-missing data")

# Find common time range between ONI complete times, ELI times, and CERES times
max_lag = maximum(abs.(lags))
common_times_candidate = intersect(oni_times_complete, eli_times_nonmissing, common_time)

# Restrict common_time to the valid range
valid_time_mask = in.(common_time, Ref(common_times_candidate))
restricted_time = common_time[valid_time_mask]

println("Restricted time range: $(length(restricted_time)) months")
println("  From: $(minimum(restricted_time))")
println("  To:   $(maximum(restricted_time))")

# Restrict all data to this time range
restricted_sepac_sst = sepac_sst[valid_time_mask]
restricted_box_means = Dict(
    box_name => Dict(
        var_name => means[var_name][valid_time_mask]
        for var_name in ceres_vars
    )
    for (box_name, means) in box_means
)

# Match ELI to restricted time and deseasonalize/detrend
eli_time_idx = indexin(restricted_time, eli_times_nonmissing)
if any(isnothing, eli_time_idx)
    error("Some restricted times not found in ELI data!")
end
eli_matched = eli_values_nonmissing[eli_time_idx]

# Deseasonalize and detrend ELI
restricted_float_time = calc_float_time.(restricted_time)
restricted_month_groups = groupfind(month, restricted_time)
eli_detrended = copy(eli_matched)
deseasonalize_and_detrend_precalculated_groups_twice!(eli_detrended, restricted_float_time, 
    restricted_month_groups; aggfunc=mean, trendfunc=least_squares_fit)

# Build lagged ELI matrix
eli_lag_matrix = build_enso_lag_matrix(eli_detrended, collect(lags))

# Remove rows with NaN values from lagging at boundaries
valid_rows = [!any(isnan, eli_lag_matrix[i, :]) for i in 1:size(eli_lag_matrix, 1)]
if sum(valid_rows) < length(valid_rows)
    println("Removing $(length(valid_rows) - sum(valid_rows)) rows with NaN from boundary effects")
    eli_lag_matrix = eli_lag_matrix[valid_rows, :]
    restricted_time = restricted_time[valid_rows]
    restricted_sepac_sst = restricted_sepac_sst[valid_rows]
    restricted_box_means = Dict(
        box_name => Dict(
            var_name => means[var_name][valid_rows]
            for var_name in ceres_vars
        )
        for (box_name, means) in restricted_box_means
    )
end

# Final check for NaN values
if any(isnan, eli_lag_matrix)
    error("ELI lag matrix still contains NaN values after filtering")
end

println("ELI lag matrix built successfully: $(size(eli_lag_matrix))")
println("Final time range: $(length(restricted_time)) months")

# ══════════════════════════════════════════════════════════════════════════════
# ENSO decomposition via PLS with varying number of components
# ══════════════════════════════════════════════════════════════════════════════

# Loop over different numbers of PLS components
pls_components_to_test = [1, 3]

for n_comp in pls_components_to_test
    println("\n" * "="^80)
    println("ENSO DECOMPOSITION WITH $n_comp PLS COMPONENT(S)")
    println("="^80)
    
    # ──────────────────────────────────────────────────────────────────────────
    # ONI (NINO 3.4)
    # ──────────────────────────────────────────────────────────────────────────
    
    println("\n" * "─"^80)
    println("Performing ENSO decomposition using ONI (NINO 3.4) with n=$n_comp")
    println("─"^80)
    
    oni_results = perform_enso_decomposition(oni_lag_matrix_complete, oni_times_complete, restricted_time,
                                            restricted_sepac_sst, restricted_box_means, ceres_vars, var_labels, n_comp)
    
    # ──────────────────────────────────────────────────────────────────────────
    # ELI (ENSO Longitude Index)
    # ──────────────────────────────────────────────────────────────────────────
    
    println("\n" * "─"^80)
    println("Performing ENSO decomposition using ELI (ENSO Longitude Index) with n=$n_comp")
    println("─"^80)
    
    eli_results = perform_enso_decomposition(eli_lag_matrix, restricted_time, restricted_time,
                                            restricted_sepac_sst, restricted_box_means, ceres_vars, var_labels, n_comp)
    
    # ──────────────────────────────────────────────────────────────────────────
    # Create plots
    # ──────────────────────────────────────────────────────────────────────────
    
    # Function to create plots for a given ENSO decomposition
    function create_plots_for_enso_method(results, enso_method_name, n_components, restricted_box_means, ceres_vars, var_labels)
        println("\nCreating plots for $enso_method_name (n=$n_components)...")
        
        for (box_name, means_total) in restricted_box_means
            fname = replace(box_name, " " => "_")
            fname_method = lowercase(replace(enso_method_name, " " => "_"))
            
            # Get full-length data for total variability
            means_total_matched = Dict(
                var_name => means_total[var_name][results.data_valid_idx]
                for var_name in ceres_vars
            )
            
            make_scatter_3x3_stacked(
                results.sepac_sst_for_pls, results.sepac_sst_enso, results.sepac_sst_nonenso,
                means_total_matched, results.box_enso[box_name], results.box_nonenso[box_name],
                ceres_vars, var_labels;
                suptitle = "$box_name Radiation vs SEPac SST [$enso_method_name, n=$n_components]: Total, ENSO, Non-ENSO",
                savepath = joinpath(visdir, "scatter_$(fname_method)_n$(n_components)_$(fname).png"))
            
            println("  Saved: scatter_$(fname_method)_n$(n_components)_$(fname).png")
        end
    end
    
    # Create plots for ONI
    create_plots_for_enso_method(oni_results, "ONI", n_comp, restricted_box_means, ceres_vars, var_labels)
    
    # Create plots for ELI
    create_plots_for_enso_method(eli_results, "ELI", n_comp, restricted_box_means, ceres_vars, var_labels)
end

println("\n" * "="^80)
println("Done! All figures saved to: $visdir")
println("="^80)
