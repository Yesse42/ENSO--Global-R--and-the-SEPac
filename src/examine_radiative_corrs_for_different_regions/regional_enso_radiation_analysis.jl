using CSV, DataFrames, Dates, Plots, LinearAlgebra, Statistics
using StatsBase # For correlation analysis
gr() # Use GR backend for Plots.jl

# Include required modules
include("regions.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")

println("Starting regional ENSO-radiation analysis...")

# Define analysis parameters
lags = -24:24
time_period = (Date(1979, 1, 1), Date(2025, 1, 1))

# Load ENSO data for all lags
println("Loading ENSO data for lags $lags...")
enso_data, enso_coords = load_enso_data(time_period; lags=collect(lags))
enso_time = enso_coords["time"]

# Convert ENSO data to DataFrame for easier manipulation
enso_df = DataFrame()
enso_df.Date = [Date(year(dt), month(dt), 1) for dt in enso_time]  # Round to nearest month

# Add all ENSO lag columns
for lag in lags
    col_name = "oni_lag_$lag"
    if haskey(enso_data, col_name)
        enso_df[!, Symbol("oni_lag_$lag")] = enso_data[col_name]
    end
end

println("ENSO data loaded. $(nrow(enso_df)) time points available.")

# Function to calculate correlations between all ENSO lags and lag 0 of target variable
function find_optimal_lag_correlation(target_var, enso_df)
    correlations = Float64[]
    lag_values = Int[]
    
    for lag in lags
        oni_col = Symbol("oni_lag_$lag")
        # Convert column names to symbols for comparison
        col_symbols = Symbol.(names(enso_df))
        if oni_col in col_symbols
            # Remove missing values for correlation calculation
            valid_idx = .!ismissing.(target_var) .& .!ismissing.(enso_df[!, oni_col])
            if sum(valid_idx) > 10  # Need at least 10 points
                corr_val = cor(target_var[valid_idx], enso_df[valid_idx, oni_col])
                push!(correlations, abs(corr_val))  # Use absolute correlation to find strongest relationship
                push!(lag_values, lag)
            else
                push!(correlations, 0.0)  # Use 0 instead of NaN
                push!(lag_values, lag)
            end
        else
            push!(correlations, 0.0)  # Use 0 instead of NaN
            push!(lag_values, lag)
        end
    end
    
    # Find lag with maximum absolute correlation
    if length(correlations) > 0
        max_idx = argmax(correlations)
        optimal_lag = lag_values[max_idx]
        max_corr = correlations[max_idx]
        return optimal_lag, max_corr, correlations, lag_values
    else
        return 0, 0.0, correlations, lag_values
    end
end

# Function to remove ENSO signal from time series via linear regression
function remove_enso_signal(target_var, enso_var)
    # Find valid (non-missing) data points
    valid_idx = .!ismissing.(target_var) .& .!ismissing.(enso_var)
    
    if sum(valid_idx) < 10
        @warn "Insufficient valid data points for regression"
        return target_var, (slope=NaN, intercept=NaN)
    end
    
    # Extract valid data
    y_valid = target_var[valid_idx]
    x_valid = enso_var[valid_idx]
    
    # Perform least squares regression
    fit = least_squares_fit(x_valid, y_valid)
    
    # Create residual time series
    residual = copy(target_var)
    residual[valid_idx] .= y_valid .- (fit.slope .* x_valid .+ fit.intercept)
    
    return residual, fit
end

# Function to calculate lagged correlations for plotting
function calculate_lagged_correlations(var1, var2_df, var2_prefix)
    correlations = Float64[]
    lag_values = Int[]
    
    # Debug counter
    found_cols = 0
    missing_cols = 0
    
    for lag in lags
        var2_col = Symbol("$(var2_prefix)_lag_$lag")
        # Convert column names to symbols for comparison
        col_symbols = Symbol.(names(var2_df))
        if var2_col in col_symbols
            found_cols += 1
            valid_idx = .!ismissing.(var1) .& .!ismissing.(var2_df[!, var2_col])
            if sum(valid_idx) > 10
                corr_val = cor(var1[valid_idx], var2_df[valid_idx, var2_col])
                push!(correlations, corr_val)
                push!(lag_values, lag)
            else
                push!(correlations, NaN)
                push!(lag_values, lag)
            end
        else
            missing_cols += 1
            push!(correlations, NaN)
            push!(lag_values, lag)
        end
    end
    
    return correlations, lag_values
end

# Create visualization directory
vis_dir = "../../vis/radiative_corrs_different_regions/local_rad"
if !isdir(vis_dir)
    mkpath(vis_dir)
end

# Process each region
for region_name in region_names
    println("\nProcessing region: $region_name")
    
    # Load regional T2M data
    t2m_file = "../../data/examine_radiative_corrs_for_different_regions/$(region_name)_era5_t2m.csv"
    radiation_file = "../../data/examine_radiative_corrs_for_different_regions/$(region_name)_ceres_radiation.csv"
    
    if !isfile(t2m_file) || !isfile(radiation_file)
        @warn "Data files not found for region $region_name, skipping..."
        continue
    end
    
    # Load T2M data
    t2m_df = CSV.read(t2m_file, DataFrame)
    t2m_df.Date = [Date(year(dt), month(dt), 1) for dt in DateTime.(t2m_df.Date)]
    
    # Load radiation data  
    radiation_df = CSV.read(radiation_file, DataFrame)
    radiation_df.Date = [Date(year(dt), month(dt), 1) for dt in DateTime.(radiation_df.Date)]
    
    # Debug: check date ranges
    println("  T2M date range: $(minimum(t2m_df.Date)) to $(maximum(t2m_df.Date))")
    println("  Radiation date range: $(minimum(radiation_df.Date)) to $(maximum(radiation_df.Date))")
    println("  ENSO date range: $(minimum(enso_df.Date)) to $(maximum(enso_df.Date))")
    
    # Create lagged T2M data
    println("  Creating T2M lags...")
    t2m_lagged_df = DataFrame(Date = t2m_df.Date)
    
    # Use detrended and deseasonalized T2M
    base_t2m = t2m_df.T2M_Detrend_Deseas
    
    for lag in lags
        lagged_t2m = time_lag(base_t2m, lag)
        t2m_lagged_df[!, Symbol("T2M_lag_$lag")] = lagged_t2m
    end
    
    # Similarly create lagged ENSO DataFrame 
    enso_lagged_df = enso_df
    
    # Merge all datasets on date (rounded to month)
    println("  Merging datasets...")
    
    # First merge T2M and radiation data
    merged_df = DataFrames.innerjoin(t2m_df[:, [:Date, :T2M_Detrend_Deseas]], radiation_df, on=:Date)
    
    # Then add ENSO data
    merged_df = DataFrames.innerjoin(merged_df, enso_lagged_df, on=:Date)
    
    # Finally add T2M lags
    merged_df = DataFrames.innerjoin(merged_df, t2m_lagged_df, on=:Date)
    
    println("  Merged dataset has $(nrow(merged_df)) rows and $(ncol(merged_df)) columns")
    
    # Drop rows with any missing data
    println("  Removing missing data...")
    original_rows = nrow(merged_df)
    merged_df = dropmissing(merged_df)
    final_rows = nrow(merged_df)
    println("  Removed $(original_rows - final_rows) rows with missing data. $(final_rows) rows remaining.")
    
    # Debug: check available columns
    println("  Available columns: ", length(names(merged_df)), " total")
    enso_cols = filter(col -> startswith(string(col), "oni"), names(merged_df))
    println("  Found $(length(enso_cols)) ENSO columns")
    
    if final_rows < 50
        @warn "Insufficient data for region $region_name after removing missing values, skipping..."
        continue
    end
    
    # Find optimal ENSO lag for this region's T2M
    println("  Finding optimal ENSO lag...")
    t2m_lag0 = merged_df.T2M_Detrend_Deseas
    optimal_lag, max_corr, all_corrs, lag_vals = find_optimal_lag_correlation(t2m_lag0, merged_df)
    
    println("  Optimal ENSO lag: $optimal_lag (correlation: $(round(max_corr, digits=3)))")
    
    # Perform regression and create residual T2M
    println("  Removing ENSO signal from T2M...")
    optimal_enso_col = Symbol("oni_lag_$optimal_lag")
    regression_fit = nothing
    # Convert column names to symbols for comparison
    col_symbols = Symbol.(names(merged_df))
    if optimal_enso_col in col_symbols
        optimal_enso = merged_df[!, optimal_enso_col]
        t2m_residual, regression_fit = remove_enso_signal(t2m_lag0, optimal_enso)
        merged_df.T2M_Residual = t2m_residual
        
        println("  Regression: T2M = $(round(regression_fit.slope, digits=4)) * ONI + $(round(regression_fit.intercept, digits=4))")
    else
        @warn "Optimal ENSO lag column not found, using original T2M"
        merged_df.T2M_Residual = t2m_lag0
        regression_fit = (slope=NaN, intercept=NaN)
    end
    
    # Calculate correlations for plotting
    println("  Calculating lagged correlations...")
    
    # Correlations with radiation variables
    net_sw_corr_t2m, _ = calculate_lagged_correlations(merged_df.Net_SW_Detrend_Deseas, merged_df, "T2M")
    toa_lw_corr_t2m, _ = calculate_lagged_correlations(merged_df.TOA_LW_Detrend_Deseas, merged_df, "T2M")
    toa_net_corr_t2m, _ = calculate_lagged_correlations(merged_df.TOA_Net_Detrend_Deseas, merged_df, "T2M")
    
    net_sw_corr_oni, _ = calculate_lagged_correlations(merged_df.Net_SW_Detrend_Deseas, merged_df, "oni")
    toa_lw_corr_oni, _ = calculate_lagged_correlations(merged_df.TOA_LW_Detrend_Deseas, merged_df, "oni")
    toa_net_corr_oni, _ = calculate_lagged_correlations(merged_df.TOA_Net_Detrend_Deseas, merged_df, "oni")
    
    net_sw_corr_resid, _ = calculate_lagged_correlations(merged_df.Net_SW_Detrend_Deseas, merged_df, "T2M")
    toa_lw_corr_resid, _ = calculate_lagged_correlations(merged_df.TOA_LW_Detrend_Deseas, merged_df, "T2M")
    toa_net_corr_resid, _ = calculate_lagged_correlations(merged_df.TOA_Net_Detrend_Deseas, merged_df, "T2M")
    
    # For residual T2M correlations, we need to create lagged residual T2M
    residual_lagged_df = DataFrame(Date = merged_df.Date)
    for lag in lags
        lagged_residual = time_lag(merged_df.T2M_Residual, lag)
        residual_lagged_df[!, Symbol("T2M_Residual_lag_$lag")] = lagged_residual
    end
    
    # Calculate correlations with residual T2M
    net_sw_corr_resid, _ = calculate_lagged_correlations(merged_df.Net_SW_Detrend_Deseas, residual_lagged_df, "T2M_Residual")
    toa_lw_corr_resid, _ = calculate_lagged_correlations(merged_df.TOA_LW_Detrend_Deseas, residual_lagged_df, "T2M_Residual")
    toa_net_corr_resid, _ = calculate_lagged_correlations(merged_df.TOA_Net_Detrend_Deseas, residual_lagged_df, "T2M_Residual")
    
    # Create 3-panel correlation plot
    println("  Creating correlation plots...")
    
    # Calculate shared y-limits for all three panels
    all_correlations = vcat(
        net_sw_corr_t2m, toa_lw_corr_t2m, toa_net_corr_t2m,
        net_sw_corr_oni, toa_lw_corr_oni, toa_net_corr_oni,
        net_sw_corr_resid, toa_lw_corr_resid, toa_net_corr_resid
    )
    valid_corrs = all_correlations[.!isnan.(all_correlations)]
    if !isempty(valid_corrs)
        y_min = min(-1.0, minimum(valid_corrs) - 0.1)
        y_max = max(1.0, maximum(valid_corrs) + 0.1)
    else
        y_min, y_max = -1.0, 1.0
    end
    
    # Plot 1: Correlations with T2M
    p1 = plot(title="$region_name: Radiation vs T2M Correlations", 
              xlabel="Lag (months)", ylabel="Correlation", 
              ylims=(y_min, y_max), legend=:topright)
    plot!(p1, collect(lags), net_sw_corr_t2m, label="Net SW", color=:orange, linewidth=2)
    plot!(p1, collect(lags), toa_lw_corr_t2m, label="TOA LW", color=:red, linewidth=2)
    plot!(p1, collect(lags), toa_net_corr_t2m, label="TOA Net", color=:blue, linewidth=2)
    hline!(p1, [0], color=:black, linestyle=:dash, alpha=0.5, label="")
    
    # Plot 2: Correlations with ONI
    p2 = plot(title="$region_name: Radiation vs ONI Correlations", 
              xlabel="Lag (months)", ylabel="Correlation", 
              ylims=(y_min, y_max), legend=:topright)
    plot!(p2, collect(lags), net_sw_corr_oni, label="Net SW", color=:orange, linewidth=2)
    plot!(p2, collect(lags), toa_lw_corr_oni, label="TOA LW", color=:red, linewidth=2)
    plot!(p2, collect(lags), toa_net_corr_oni, label="TOA Net", color=:blue, linewidth=2)
    hline!(p2, [0], color=:black, linestyle=:dash, alpha=0.5, label="")
    
    # Plot 3: Correlations with Residual T2M
    p3 = plot(title="$region_name: Radiation vs Residual T2M Correlations", 
              xlabel="Lag (months)", ylabel="Correlation", 
              ylims=(y_min, y_max), legend=:topright)
    plot!(p3, collect(lags), net_sw_corr_resid, label="Net SW", color=:orange, linewidth=2)
    plot!(p3, collect(lags), toa_lw_corr_resid, label="TOA LW", color=:red, linewidth=2)
    plot!(p3, collect(lags), toa_net_corr_resid, label="TOA Net", color=:blue, linewidth=2)
    hline!(p3, [0], color=:black, linestyle=:dash, alpha=0.5, label="")
    
    # Combine plots
    combined_plot = plot(p1, p2, p3, layout=(3,1), size=(1000, 1200))
    
    # Save plot
    plot_filename = joinpath(vis_dir, "$(region_name)_radiation_correlations.png")
    savefig(combined_plot, plot_filename)
    
    println("  Plot saved: $plot_filename")
    
    # Save correlation data to CSV
    println("  Saving correlation data...")
    corr_data = DataFrame(
        Lag = collect(lags),
        Net_SW_vs_T2M = net_sw_corr_t2m,
        TOA_LW_vs_T2M = toa_lw_corr_t2m,
        TOA_Net_vs_T2M = toa_net_corr_t2m,
        Net_SW_vs_ONI = net_sw_corr_oni,
        TOA_LW_vs_ONI = toa_lw_corr_oni,
        TOA_Net_vs_ONI = toa_net_corr_oni,
        Net_SW_vs_Residual_T2M = net_sw_corr_resid,
        TOA_LW_vs_Residual_T2M = toa_lw_corr_resid,
        TOA_Net_vs_Residual_T2M = toa_net_corr_resid
    )
    
    corr_filename = joinpath(vis_dir, "$(region_name)_correlation_data.csv")
    CSV.write(corr_filename, corr_data)
    
    # Save summary statistics
    summary_data = DataFrame(
        Region = [region_name],
        Optimal_ENSO_Lag = [optimal_lag],
        Max_ENSO_T2M_Correlation = [max_corr],
        Regression_Slope = [regression_fit.slope],
        Regression_Intercept = [regression_fit.intercept],
        Data_Points = [final_rows]
    )
    
    summary_filename = joinpath(vis_dir, "$(region_name)_analysis_summary.csv")
    CSV.write(summary_filename, summary_data)
    
    println("  Analysis complete for $region_name")
end

println("\nRegional ENSO-radiation analysis completed!")
println("Results saved to: $vis_dir")
