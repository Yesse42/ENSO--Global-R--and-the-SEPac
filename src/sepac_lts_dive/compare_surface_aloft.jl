"""
Decomposes the relationship between LTS and global rad into the sfc and aloft components.
Similar analysis to generate_nonlocal_radiation_time_series.jl but for atmospheric temperature structure.
"""

cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")

using Statistics, DataFrames, JLD2, CSV, Dates, Plots

# Load region masks to get all available regions
mask_file = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/stratocumulus_region_masks.jld2"
region_data = JLD2.load(mask_file)
base_regions = collect(keys(region_data["regional_masks_ceres"]))

# Add the special ENSO-removed region
valid_regions = vcat(base_regions, ["enso_removed_SEPac_feedback_definition"])

# Define radiation variables and time period
rad_variables = ["gtoa_net_all_mon", "gtoa_net_lw_mon", "gtoa_net_sw_mon"]
date_range = (Date(2002, 3, 1), Date(2022, 3, 31))
is_analysis_time(t) = in_time_period(t, date_range)

# Load global radiation time series
global_rad_data, global_coords = load_new_ceres_data(rad_variables, date_range)

# Load mask area calculations for regional weighting
mask_areas = JLD2.load("/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/mask_area_calculations.jld2")

# Data directories
local_data_dir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/sepac_lts_data/local_region_time_series"
nonlocal_data_dir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/sepac_lts_data/nonlocal_radiation_time_series"
visdir = "../../vis/lts_global_rad/surface_aloft_temperature_correlations"
partitioned_visdir = "../../vis/lts_global_rad/surface_temperature_aloft_correlations"
mkpath(visdir)
mkpath(partitioned_visdir)

for region in valid_regions
    println("Processing region: $region")
    
    # Load ERA5 data with potential temperature variables
    era5_file = joinpath(local_data_dir, "era5_region_avg_$(region).csv")
    era5_df = CSV.read(era5_file, DataFrame)
    
    # Load local and nonlocal radiation data
    local_rad_file = joinpath(nonlocal_data_dir, "$(region)_local_radiation.csv")
    nonlocal_rad_file = joinpath(nonlocal_data_dir, "$(region)_nonlocal_radiation.csv")
    
    # Handle special case for ENSO-removed region
    if region == "enso_removed_SEPac_feedback_definition"
        # Use SEPac_feedback_definition mask areas for the ENSO-removed case
        mask_region = "SEPac_feedback_definition"
        local_rad_file = joinpath(nonlocal_data_dir, "$(mask_region)_local_radiation.csv")
        nonlocal_rad_file = joinpath(nonlocal_data_dir, "$(mask_region)_nonlocal_radiation.csv")

        # Filter out non-lag-0 columns and rename lag-0 columns to plain names
        era5_columns = names(era5_df)
        keep_columns = String[]
        rename_dict = Dict{String, String}()

        for col in era5_columns
            if col == "date"
                push!(keep_columns, col)
            elseif endswith(col, "_lag_0")
                # Keep lag-0 columns and prepare to rename them
                push!(keep_columns, col)
                plain_name = replace(col, "_lag_0" => "")
                rename_dict[col] = plain_name
            end
            # Skip all other columns (non-lag-0)
        end

        # Select only the columns we want to keep
        era5_df = select(era5_df, keep_columns)

        # Rename lag-0 columns to plain names
        rename!(era5_df, rename_dict)
    else
        mask_region = region
    end
    
    local_rad_df = CSV.read(local_rad_file, DataFrame)
    nonlocal_rad_df = CSV.read(nonlocal_rad_file, DataFrame)
    
    # Filter to analysis time period
    filter!(row -> is_analysis_time(row.date), era5_df)
    filter!(row -> is_analysis_time(row.date), local_rad_df)
    filter!(row -> is_analysis_time(row.date), nonlocal_rad_df)
    
    # Get area weights from mask calculations
    local_area = mask_areas["area_results_ceres"][mask_region]["masked_area"]
    total_area = mask_areas["total_areas"]["ceres_total_area"]
    global_area = total_area - local_area  # Complement area
    
    # Prepare time variables for detrending/deseasonalizing
    analysis_times = era5_df.date
    float_times = calc_float_time.(analysis_times)
    months = month.(analysis_times)
    
    # Extract and process potential temperature data
    global foo = era5_df
    theta_1000 = copy(era5_df[!, "θ_1000"])  # Surface potential temperature (1000 hPa)
    theta_700 = copy(era5_df[!, "θ_700"])    # Aloft potential temperature (700 hPa)
    lts_1000 = copy(era5_df[!, "LTS_1000"])  # LTS at 1000 hPa for weighting
    
    # Take negative of surface potential temperature as requested
    neg_theta_1000 = -theta_1000
    
    # Detrend and deseasonalize potential temperature and LTS data
    detrend_and_deseasonalize!(neg_theta_1000, float_times, months)
    detrend_and_deseasonalize!(theta_700, float_times, months)
    detrend_and_deseasonalize!(lts_1000, float_times, months)
    
    # Process local and nonlocal radiation data
    local_sw = copy(local_rad_df[!, "toa_net_sw_mon"])
    local_lw = copy(local_rad_df[!, "toa_net_lw_mon"])
    nonlocal_sw = copy(nonlocal_rad_df[!, "gtoa_net_sw_mon"])
    nonlocal_lw = copy(nonlocal_rad_df[!, "gtoa_net_lw_mon"])
    
    # Detrend and deseasonalize local and nonlocal radiation
    detrend_and_deseasonalize!(local_sw, float_times, months)
    detrend_and_deseasonalize!(local_lw, float_times, months)
    detrend_and_deseasonalize!(nonlocal_sw, float_times, months)
    detrend_and_deseasonalize!(nonlocal_lw, float_times, months)
    
    # Weight potential temperature by local area (representing local atmospheric conditions)
    weighted_surface_theta = neg_theta_1000 .* local_area
    weighted_aloft_theta = theta_700 .* local_area
    
    # Process global radiation data for this time period
    global_rad_filtered = Dict{String, Vector}()
    for (var_name, rad_data) in pairs(global_rad_data)
        # Filter to match ERA5 time period
        global_times = round.(global_coords["time"], Month(1), RoundDown)
        time_mask = [t in analysis_times for t in global_times]
        filtered_rad = rad_data[time_mask]
        
        # Detrend and deseasonalize
        detrend_and_deseasonalize!(filtered_rad, float_times, months)
        
        # Weight by global area
        global_rad_filtered[var_name] = filtered_rad .* global_area
    end
    
    # Calculate correlations and create plots for each radiation variable
    for var_name in rad_variables
        println("  Processing variable: $var_name")
        
        global_rad = global_rad_filtered[var_name]
        
        # Calculate correlations
        surface_corr = cor(weighted_surface_theta, global_rad)
        aloft_corr = cor(weighted_aloft_theta, global_rad)
        lts_corr = cor(lts_1000, global_rad)
        
        # Calculate weights (std of individual theta divided by std of LTS_1000)
        lts_std = std(skipmissing(lts_1000))
        surface_weight = std(skipmissing(neg_theta_1000)) / lts_std
        aloft_weight = std(skipmissing(theta_700)) / lts_std
        
        # Apply weights
        weighted_surface_corr = surface_corr * surface_weight
        weighted_aloft_corr = aloft_corr * aloft_weight
        
        println("    Surface (1000 hPa) correlation: $(round(surface_corr, digits=3))")
        println("    Aloft (700 hPa) correlation: $(round(aloft_corr, digits=3))")
        println("    LTS_1000 correlation: $(round(lts_corr, digits=3))")
        println("    Weighted surface correlation: $(round(weighted_surface_corr, digits=3))")
        println("    Weighted aloft correlation: $(round(weighted_aloft_corr, digits=3))")
        
        # Create comparison plot
        p = plot(title="$region - $var_name: θ Components & LTS Correlations",
                xlabel="Component",
                ylabel="Correlation",
                legend=false,
                size=(700, 400))
        
        # Bar plot showing only weighted correlations for theta and LTS correlation
        categories = ["Surface (neg-θ₁₀₀₀)", "Aloft (θ₇₀₀)", "LTS₁₀₀₀"]
        correlations = [weighted_surface_corr, weighted_aloft_corr, lts_corr]
        
        bar!(p, categories, correlations, color=[:darkblue, :darkred, :green])
        
        # Save plot
        plot_filename = "$(region)_$(var_name)_surface_aloft_correlations.png"
        savefig(p, joinpath(visdir, plot_filename))
        
        println("    Saved plot: $plot_filename")
    end
    
    # Create partitioned impact plots (SW/LW x Surface/Aloft decomposition)
    # Only process net radiation for the partitioned analysis
    net_var_name = "gtoa_net_all_mon"
    if net_var_name in rad_variables
        println("  Creating partitioned impact plot for net radiation")
        
        global_net = global_rad_filtered[net_var_name]
        global_sw = global_rad_filtered["gtoa_net_sw_mon"] 
        global_lw = global_rad_filtered["gtoa_net_lw_mon"]
        
        # Calculate correlations for the four combinations
        sw_surface_corr = cor(global_sw, neg_theta_1000)
        sw_aloft_corr = cor(global_sw, theta_700)
        lw_surface_corr = cor(global_lw, neg_theta_1000)
        lw_aloft_corr = cor(global_lw, theta_700)
        lts_net_corr = cor(lts_1000, global_net)
        
        # Calculate standard deviations for weighting
        std_global_net = std(skipmissing(global_net))
        std_global_sw = std(skipmissing(global_sw))
        std_global_lw = std(skipmissing(global_lw))
        std_theta_1000 = std(skipmissing(neg_theta_1000))
        std_theta_700 = std(skipmissing(theta_700))
        std_lts_1000 = std(skipmissing(lts_1000))
        
        # Calculate weights: std(global_component) * std(theta_component) / (std(global_net) * std(LTS_1000))
        sw_surface_weight = (std_global_sw * std_theta_1000) / (std_global_net * std_lts_1000)
        sw_aloft_weight = (std_global_sw * std_theta_700) / (std_global_net * std_lts_1000)
        lw_surface_weight = (std_global_lw * std_theta_1000) / (std_global_net * std_lts_1000)
        lw_aloft_weight = (std_global_lw * std_theta_700) / (std_global_net * std_lts_1000)
        
        # Apply weights to correlations
        weighted_sw_surface = sw_surface_corr * sw_surface_weight
        weighted_sw_aloft = sw_aloft_corr * sw_aloft_weight
        weighted_lw_surface = lw_surface_corr * lw_surface_weight
        weighted_lw_aloft = lw_aloft_corr * lw_aloft_weight
        
        # Print results
        println("    SW-Surface correlation (weighted): $(round(weighted_sw_surface, digits=3))")
        println("    SW-Aloft correlation (weighted): $(round(weighted_sw_aloft, digits=3))")
        println("    LW-Surface correlation (weighted): $(round(weighted_lw_surface, digits=3))")
        println("    LW-Aloft correlation (weighted): $(round(weighted_lw_aloft, digits=3))")
        println("    LTS-Net correlation: $(round(lts_net_corr, digits=3))")
        
        # Verify that 5-bar weighted sum matches LTS-Net correlation
        weighted_5bar_sum = weighted_sw_surface + weighted_sw_aloft + weighted_lw_surface + weighted_lw_aloft
        relative_error_5bar = abs(weighted_5bar_sum - lts_net_corr) / abs(lts_net_corr)
        println("    5-bar verification: weighted sum = $(round(weighted_5bar_sum, digits=6)), relative error = $(round(relative_error_5bar, digits=8))")
        
        if relative_error_5bar > 1e-3
            @warn "5-bar plot: Weighted sum does not match LTS-Net correlation within tolerance of 1e-3"
        else
            println("    ✓ 5-bar plot verification passed")
        end
        
        # Create partitioned impact plot
        p_part = plot(title="$region: Partitioned LTS Impact on Global Net Radiation",
                     xlabel="Component",
                     ylabel="Weighted Correlation",
                     legend=false,
                     size=(800, 400))
        
        # Five bars: four weighted components + LTS-Net correlation
        categories_part = ["SW-Surface", "SW-Aloft", "LW-Surface", "LW-Aloft", "LTS-Global Net Corr"]
        correlations_part = [weighted_sw_surface, weighted_sw_aloft, weighted_lw_surface, weighted_lw_aloft, lts_net_corr]
        colors_part = [:lightblue, :blue, :pink, :red, :green]
        
        bar!(p_part, categories_part, correlations_part, color=colors_part)
        
        # Save partitioned plot
        part_filename = "$(region)_partitioned_lts_impact.png"
        savefig(p_part, joinpath(partitioned_visdir, part_filename))
        
        println("    Saved partitioned plot: $part_filename")
    end
    
    # Create expanded 9-bar plot with local/nonlocal radiation components
    if net_var_name in rad_variables
        println("  Creating expanded 9-bar plot with local/nonlocal components")
        
        global_net = global_rad_filtered[net_var_name]
        
        # Calculate correlations between radiation components and temperature variables
        # Local radiation correlations
        sw_local_surface_corr = cor(local_sw, neg_theta_1000)
        sw_local_aloft_corr = cor(local_sw, theta_700)
        lw_local_surface_corr = cor(local_lw, neg_theta_1000)
        lw_local_aloft_corr = cor(local_lw, theta_700)
        
        # Nonlocal radiation correlations  
        sw_nonlocal_surface_corr = cor(nonlocal_sw, neg_theta_1000)
        sw_nonlocal_aloft_corr = cor(nonlocal_sw, theta_700)
        lw_nonlocal_surface_corr = cor(nonlocal_lw, neg_theta_1000)
        lw_nonlocal_aloft_corr = cor(nonlocal_lw, theta_700)
        
        # LTS-Global net correlation (unchanged)
        lts_net_corr = cor(lts_1000, global_net)
        
        # Calculate standard deviations for weighting
        std_local_sw = std(skipmissing(local_sw))
        std_local_lw = std(skipmissing(local_lw))
        std_nonlocal_sw = std(skipmissing(nonlocal_sw))
        std_nonlocal_lw = std(skipmissing(nonlocal_lw))
        std_theta_1000 = std(skipmissing(neg_theta_1000))
        std_theta_700 = std(skipmissing(theta_700))
        std_global_net = std(skipmissing(global_net))
        std_lts_1000 = std(skipmissing(lts_1000))
        
        # Calculate weights: std(radiation_component) * std(theta_component) / (std(global_net) * std(LTS_1000))
        sw_local_surface_weight = (std_local_sw * std_theta_1000) / (std_global_net * std_lts_1000)
        sw_local_aloft_weight = (std_local_sw * std_theta_700) / (std_global_net * std_lts_1000)
        lw_local_surface_weight = (std_local_lw * std_theta_1000) / (std_global_net * std_lts_1000)
        lw_local_aloft_weight = (std_local_lw * std_theta_700) / (std_global_net * std_lts_1000)
        
        sw_nonlocal_surface_weight = (std_nonlocal_sw * std_theta_1000) / (std_global_net * std_lts_1000)
        sw_nonlocal_aloft_weight = (std_nonlocal_sw * std_theta_700) / (std_global_net * std_lts_1000)
        lw_nonlocal_surface_weight = (std_nonlocal_lw * std_theta_1000) / (std_global_net * std_lts_1000)
        lw_nonlocal_aloft_weight = (std_nonlocal_lw * std_theta_700) / (std_global_net * std_lts_1000)
        
        # Apply weights to correlations
        weighted_sw_local_surface = sw_local_surface_corr * sw_local_surface_weight
        weighted_sw_local_aloft = sw_local_aloft_corr * sw_local_aloft_weight
        weighted_lw_local_surface = lw_local_surface_corr * lw_local_surface_weight
        weighted_lw_local_aloft = lw_local_aloft_corr * lw_local_aloft_weight
        
        weighted_sw_nonlocal_surface = sw_nonlocal_surface_corr * sw_nonlocal_surface_weight
        weighted_sw_nonlocal_aloft = sw_nonlocal_aloft_corr * sw_nonlocal_aloft_weight
        weighted_lw_nonlocal_surface = lw_nonlocal_surface_corr * lw_nonlocal_surface_weight
        weighted_lw_nonlocal_aloft = lw_nonlocal_aloft_corr * lw_nonlocal_aloft_weight
        
        # Print results
        println("    SW-Local-Surface (weighted): $(round(weighted_sw_local_surface, digits=3))")
        println("    SW-Local-Aloft (weighted): $(round(weighted_sw_local_aloft, digits=3))")
        println("    LW-Local-Surface (weighted): $(round(weighted_lw_local_surface, digits=3))")
        println("    LW-Local-Aloft (weighted): $(round(weighted_lw_local_aloft, digits=3))")
        println("    SW-Nonlocal-Surface (weighted): $(round(weighted_sw_nonlocal_surface, digits=3))")
        println("    SW-Nonlocal-Aloft (weighted): $(round(weighted_sw_nonlocal_aloft, digits=3))")
        println("    LW-Nonlocal-Surface (weighted): $(round(weighted_lw_nonlocal_surface, digits=3))")
        println("    LW-Nonlocal-Aloft (weighted): $(round(weighted_lw_nonlocal_aloft, digits=3))")
        println("    LTS-Net correlation: $(round(lts_net_corr, digits=3))")
        
        # Verify that 9-bar weighted sum matches LTS-Net correlation
        weighted_9bar_sum = weighted_sw_local_surface + weighted_sw_local_aloft + 
                           weighted_lw_local_surface + weighted_lw_local_aloft +
                           weighted_sw_nonlocal_surface + weighted_sw_nonlocal_aloft +
                           weighted_lw_nonlocal_surface + weighted_lw_nonlocal_aloft
        relative_error_9bar = abs(weighted_9bar_sum - lts_net_corr) / abs(lts_net_corr)
        println("    9-bar verification: weighted sum = $(round(weighted_9bar_sum, digits=6)), relative error = $(round(relative_error_9bar, digits=8))")
        
        if relative_error_9bar > 1e-3
            @warn "9-bar plot: Weighted sum does not match LTS-Net correlation within tolerance of 1e-3"
        else
            println("    ✓ 9-bar plot verification passed")
        end
        
        # Create expanded 9-bar plot
        p_expanded = plot(title="$region: Expanded LTS Impact Decomposition",
                         xlabel="Component",
                         ylabel="Weighted Correlation",
                         legend=false,
                         size=(1200, 500),
                         xrotation=8)
        
        # Nine bars: 8 weighted local/nonlocal components + LTS-Net correlation
        categories_exp = ["SW-Local-neg-θ₁₀₀₀", "SW-Local-θ₇₀₀", "LW-Local-neg-θ₁₀₀₀", "LW-Local-θ₇₀₀",
                         "SW-Nonlocal-neg-θ₁₀₀₀", "SW-Nonlocal-θ₇₀₀", "LW-Nonlocal-neg-θ₁₀₀₀", "LW-Nonlocal-θ₇₀₀",
                         "LTS-Global Net Corr"]
        
        correlations_exp = [weighted_sw_local_surface, weighted_sw_local_aloft, 
                           weighted_lw_local_surface, weighted_lw_local_aloft,
                           weighted_sw_nonlocal_surface, weighted_sw_nonlocal_aloft,
                           weighted_lw_nonlocal_surface, weighted_lw_nonlocal_aloft,
                           lts_net_corr]
        
        # Color scheme: light colors for local, dark for nonlocal, green for LTS-Net
        colors_exp = [:lightblue, :lightblue, :pink, :pink, 
                     :blue, :blue, :red, :red, :green]
        
        bar!(p_expanded, categories_exp, correlations_exp, color=colors_exp)

        ylims!(p_expanded, (-0.45, 0.1))
        
        # Save expanded plot
        exp_filename = "$(region)_expanded_9bar_decomposition.png"
        savefig(p_expanded, joinpath(partitioned_visdir, exp_filename))
        
        println("    Saved expanded plot: $exp_filename")
    end
    
    # Create time series comparison plot
    p_ts = plot(title="$region: Time Series Comparison",
               xlabel="Date",
               ylabel="Normalized Values",
               legend=:topright,
               size=(1000, 600))
    
    # Normalize surface and aloft series for comparison
    norm_surface = (weighted_surface_theta .- mean(weighted_surface_theta)) ./ std(weighted_surface_theta)
    norm_aloft = (weighted_aloft_theta .- mean(weighted_aloft_theta)) ./ std(weighted_aloft_theta)
    norm_lts_1000 = (lts_1000 .- mean(lts_1000)) ./ std(lts_1000)
    
    plot!(p_ts, analysis_times, norm_surface, label="Surface (neg-θ₁₀₀₀)", color=:blue, alpha=0.7)
    plot!(p_ts, analysis_times, norm_aloft, label="Aloft (θ₇₀₀)", color=:red, alpha=0.7)
    plot!(p_ts, analysis_times, norm_lts_1000, label="LTS₁₀₀₀", color=:black, lw=2)
    
    # Save time series plot
    ts_filename = "$(region)_surface_aloft_timeseries.png"
    savefig(p_ts, joinpath(visdir, ts_filename))
    
    println("  Completed processing for $region")
    println("  Local area: $(round(local_area, digits=2))")
    println("  Global (complement) area: $(round(global_area, digits=2))")
    println()
end

println("All regions processed successfully!")
