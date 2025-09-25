using CSV, DataFrames, Dates, Statistics, JLD2, Dictionaries, Plots
cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../utils/plot_global.jl")

visdir = "vis/lts_global_rad/local_vs_nonlocal_contributions"
mkpath("../../$visdir")

# Load region masks to get all available regions
mask_file = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/stratocumulus_region_masks.jld2"
region_data = JLD2.load(mask_file)
regions = collect(keys(region_data["regional_masks_ceres"]))

# Define time period
date_range = (Date(2002, 3, 1), Date(2022, 3, 31))
is_analysis_time(t) = in_time_period(t, date_range)

# Data directories
nonlocal_data_dir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/sepac_lts_data/nonlocal_radiation_time_series"
local_data_dir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/sepac_lts_data/local_region_time_series"

for region in regions
    println("Processing region: $region")
    
    # Load radiation data
    global_rad_file = joinpath(nonlocal_data_dir, "$(region)_global_radiation.csv")
    local_rad_file = joinpath(nonlocal_data_dir, "$(region)_local_radiation.csv")
    nonlocal_rad_file = joinpath(nonlocal_data_dir, "$(region)_nonlocal_radiation.csv")
    
    global_rad_df = CSV.read(global_rad_file, DataFrame)
    local_rad_df = CSV.read(local_rad_file, DataFrame)
    nonlocal_rad_df = CSV.read(nonlocal_rad_file, DataFrame)
    
    # Load lagged LTS data
    lts_lagged_file = joinpath(local_data_dir, "era5_region_avg_lagged_$(region).csv")
    lts_df = CSV.read(lts_lagged_file, DataFrame)
    
    # Filter all data to specified time range
    filter!(row -> is_analysis_time(row.date), global_rad_df)
    filter!(row -> is_analysis_time(row.date), local_rad_df)
    filter!(row -> is_analysis_time(row.date), nonlocal_rad_df)
    filter!(row -> is_analysis_time(row.date), lts_df)
    
    # Prepare time variables for detrending/deseasonalizing
    analysis_times = global_rad_df.date
    float_times = calc_float_time.(analysis_times)
    months = month.(analysis_times)
    
    # Get radiation variables (excluding date column)
    rad_vars = names(global_rad_df)[names(global_rad_df) .!= "date"]
    local_rad_vars = replace.(rad_vars, "g" => "")
    
    # Detrend and deseasonalize all radiation time series
    for (var, local_var) in zip(rad_vars, local_rad_vars)
        detrend_and_deseasonalize!(global_rad_df[!, var], float_times, months)
        detrend_and_deseasonalize!(local_rad_df[!, local_var], float_times, months)
        detrend_and_deseasonalize!(nonlocal_rad_df[!, var], float_times, months)
    end
    
    # Process LTS lagged data - extract lag columns and detrend/deseasonalize
    lts_lag_cols = [col for col in names(lts_df) if startswith(col, "LTS_1000_lag_")]
    lts_lagged_dict = Dict{Int, Vector}()
    
    for col in lts_lag_cols
        lag_str = replace(col, "LTS_1000_lag_" => "")
        lag_val = parse(Int, lag_str)
        lts_data = copy(lts_df[!, col])
        detrend_and_deseasonalize!(lts_data, float_times, months)
        lts_lagged_dict[lag_val] = lts_data
    end

    display(lts_lagged_dict)
    
    # Create plots for each radiation variable
    for (var, local_var) in zip(rad_vars, local_rad_vars)
        println("  Processing variable: $var")
        
        # Calculate lagged correlations
        global_rad_data = global_rad_df[!, var]
        local_rad_data = local_rad_df[!, local_var]
        nonlocal_rad_data = nonlocal_rad_df[!, var]
        
        # Calculate correlations
        global_corrs = calculate_lag_correlations(global_rad_data, lts_lagged_dict; lags=-24:24)
        local_corrs = calculate_lag_correlations(local_rad_data, lts_lagged_dict; lags=-24:24)
        nonlocal_corrs = calculate_lag_correlations(nonlocal_rad_data, lts_lagged_dict; lags=-24:24)
        
        # Calculate weights (std of local/nonlocal divided by std of global)
        global_std = std(skipmissing(global_rad_data))
        local_weight = std(skipmissing(local_rad_data)) / global_std
        nonlocal_weight = std(skipmissing(nonlocal_rad_data)) / global_std
        
        # Apply weights to correlations
        weighted_local_corrs = local_corrs .* local_weight
        weighted_nonlocal_corrs = nonlocal_corrs .* nonlocal_weight

        # Create plot
        lags = -24:24
        p = plot(title="$region - $var: Global vs Local/Nonlocal LTS Correlations",
                xlabel="Lag (months), radiation lags to right",
                ylabel="(Weighted) Correlation",
                legend=:topright,
                size=(800, 600))
        
        # Plot correlations
        plot!(p, lags, [global_corrs[lag] for lag in lags], 
              label="Global", color=:black, lw=2)
        plot!(p, lags, [weighted_local_corrs[lag] for lag in lags], 
              label="Local (weighted)", color=:blue, lw=2)
        plot!(p, lags, [weighted_nonlocal_corrs[lag] for lag in lags], 
              label="Nonlocal (weighted)", color=:red, lw=2)
        
        # Save plot
        plot_filename = "$(region)_$(var)_global_vs_weighted_components.png"
        savefig(p, "../../$visdir/$plot_filename")
        
        println("    Saved plot: $plot_filename")
    end
    
    println("  Completed processing for $region")
end

println("All regions processed successfully!")