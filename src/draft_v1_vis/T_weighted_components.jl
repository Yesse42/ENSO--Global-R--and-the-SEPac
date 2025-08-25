using Plots, Statistics, StatsBase, Dates, SplitApplyCombine
using CSV, DataFrames, Dictionaries

# Include necessary modules
cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/constants.jl")
include("../utils/utilfuncs.jl")
include("../utils/plot_global.jl")

"""
This script will:
Calculate correlations and weighted components of temperature with ENSO and SEPac indices
from the ENSO-SEPac analysis.

The analysis uses:
1. SEPac SST Index (detrended & deseasonalized)
2. Calculated ENSO Index = SEPac SST Index - SEPac SST Residual
3. SEPac SST Residual (after removing ENSO influence)

For each temperature level, the script calculates:
- SEPac weighted component: correlation(SEPac) × std(SEPac)/std(SEPac_residual)
- ENSO weighted component: correlation(ENSO) × std(ENSO)/std(SEPac_residual)
- SEPac SST Residual correlation (unweighted)

The weighting ensures that: weighted_SEPac - weighted_ENSO ≈ correlation(Residual)

Each row of the plot will be a different pressure level.
Each column will show SEPac_weighted/ENSO_weighted/Residual components.
"""

# Set up output directory
visdir = "../../vis/draft_v1_vis/temperature_weighted_components/"
if !isdir(visdir)
    mkpath(visdir)
end

# Define level names
level_names = ["sfc", "850hPa", "700hPa", "500hPa", "250hPa"]
stacked_level_vars = ["t2m", "t"]  # surface temperature and pressure level temperature

# Define the time series names
time_series_names = ["sepac_sst_index", "calculated_enso", "sepac_sst_residual"]
time_series_labels = ["SEPac SST Index", "Calculated ENSO", "SEPac SST Residual"]

function calculate_weighted_temperature_components()
    println("Starting weighted temperature component analysis...")
    
    # Use standard time period from constants.jl
    println("Using standard time period: $(time_period[1]) to $(time_period[2])")
    
    # Load the time series from CSV
    println("Loading time series data from CSV...")
    csv_file = "../../data/v1draft_saved_data/enso_sepac_correlation_results.csv"
    df = CSV.read(csv_file, DataFrame)
    
    # Extract time information and filter for standard time period
    times = DateTime.(df.Date)
    csv_dates = Date.(times)
    
    # Filter CSV data for standard time period
    time_mask = (csv_dates .>= time_period[1]) .& (csv_dates .< time_period[2])
    filtered_df = df[time_mask, :]
    filtered_times = times[time_mask]
    
    println("Filtered to $(length(filtered_times)) time points within standard period")
    
    # Extract time series data
    time_series_data = Dictionary()
    sepac_sst_index = filtered_df[:, "sepac_sst_index"]
    sepac_sst_residual = filtered_df[:, "sepac_sst_residual"]
    
    # Calculate ENSO index as SEPac - Residual
    calculated_enso = sepac_sst_index .- sepac_sst_residual
    
    set!(time_series_data, "sepac_sst_index", sepac_sst_index)
    set!(time_series_data, "calculated_enso", calculated_enso)
    set!(time_series_data, "sepac_sst_residual", sepac_sst_residual)
    
    println("Calculated ENSO index as SEPac - Residual")
    println("Verification: mean(SEPac - ENSO - Residual) = $(round(mean(sepac_sst_index .- calculated_enso .- sepac_sst_residual), digits=8))")
    
    # Load ERA5 temperature data
    println("Loading ERA5 temperature data...")
    era5_data, era5_coords = load_era5_data(stacked_level_vars, time_period)
    
    # Get ERA5 time coordinates
    era5_times = era5_coords["time"]
    println("ERA5 data period: $(minimum(era5_times)) to $(maximum(era5_times))")
    println("Number of ERA5 time points: $(length(era5_times))")
    
    # Prepare ERA5 temperature data
    sfc_name = stacked_level_vars[1]  # "t2m"
    level_name = stacked_level_vars[2]  # "t"
    
    # Use ERA5 data directly (already filtered to standard time period)
    sfc_data_aligned = era5_data[sfc_name]
    level_data_aligned = era5_data[level_name]
    aligned_times = era5_times
    
    # Stack surface and level data
    sfc_dims = size(sfc_data_aligned)
    level_dims = size(level_data_aligned)
    
    # Add level dimension to surface data to match
    sfc_data_reshaped = reshape(sfc_data_aligned, sfc_dims[1], sfc_dims[2], 1, sfc_dims[3])
    
    # Concatenate along level dimension
    vertical_concat_data = cat(sfc_data_reshaped, level_data_aligned; dims=3)
    
    println("Combined temperature data shape: $(size(vertical_concat_data))")
    
    # Preprocess ERA5 data (detrend and deseasonalize)
    println("Preprocessing ERA5 temperature data...")
    float_times = @. year(aligned_times) + (month(aligned_times) - 1) / 12
    month_groups = groupfind(month.(aligned_times))
    
    for slice in eachslice(vertical_concat_data, dims=(1,2,3))
        detrend_and_deseasonalize_precalculated_groups!(slice, float_times, month_groups)
    end
    
    # Calculate correlations and weighted components for each level
    all_contrib_slices = []
    all_contrib_subtitles = []
    all_verification_slices = []
    all_verification_subtitles = []
    
    # Calculate SEPac-ENSO correlation at each grid point for each level (for verification)
    sepac_enso_correlations = []
    
    # Get the time series once
    sepac_data = time_series_data["sepac_sst_index"]
    enso_data = time_series_data["calculated_enso"]
    residual_data = time_series_data["sepac_sst_residual"]
    
    # Calculate standard deviations for weighting (corrected formula)
    println("Calculating statistics for weighting...")
    sepac_std = std(sepac_data)
    enso_std = std(enso_data)
    residual_std = std(residual_data)  # This is the correct denominator
    
    # Loop through columns first (components), then rows (levels) for column-major ordering
    for col_idx in 1:3  # 3 columns: SEPac_weighted, ENSO_weighted, Residual
        for (level_idx, level_name_str) in enumerate(level_names)
            println("Processing column $col_idx, level: $level_name_str...")
            
            # Get temperature data for this level
            level_temp_data = vertical_concat_data[:, :, level_idx, :]
            
            # Calculate correlations with each time series
            sepac_corrs = cor.(eachslice(level_temp_data, dims=(1,2)), Ref(sepac_data))
            enso_corrs = cor.(eachslice(level_temp_data, dims=(1,2)), Ref(enso_data))
            residual_corrs = cor.(eachslice(level_temp_data, dims=(1,2)), Ref(residual_data))
            
            # Convert to Float64 and handle any remaining issues
            sepac_corrs = Float64.(sepac_corrs)
            enso_corrs = Float64.(enso_corrs)
            residual_corrs = Float64.(residual_corrs)
            
            if col_idx == 1  # SEPac Weighted column
                # Calculate weighted component with corrected formula
                sepac_weighted = sepac_corrs .* sepac_std ./ residual_std
                sepac_weighted = replace!(sepac_weighted, NaN => 0, Inf => 0, -Inf => 0)
                
                push!(all_contrib_slices, sepac_weighted)
                push!(all_contrib_subtitles, "SEPac Weighted - $level_name_str")
                
            elseif col_idx == 2  # ENSO Weighted column
                # Calculate weighted component with corrected formula
                enso_weighted = enso_corrs .* enso_std ./ residual_std
                enso_weighted = replace!(enso_weighted, NaN => 0, Inf => 0, -Inf => 0)
                
                push!(all_contrib_slices, enso_weighted)
                push!(all_contrib_subtitles, "ENSO Weighted - $level_name_str")
                
            else  # col_idx == 3, Residual column
                residual_corrs = replace!(residual_corrs, NaN => 0, Inf => 0, -Inf => 0)
                
                push!(all_contrib_slices, residual_corrs)
                push!(all_contrib_subtitles, "Residual - $level_name_str")
            end
        end
    end
    
    # Calculate verification data separately (loop through levels)
    for (level_idx, level_name_str) in enumerate(level_names)
        # Get temperature data for this level
        level_temp_data = vertical_concat_data[:, :, level_idx, :]
        
        # Calculate correlations
        sepac_corrs = cor.(eachslice(level_temp_data, dims=(1,2)), Ref(sepac_data))
        enso_corrs = cor.(eachslice(level_temp_data, dims=(1,2)), Ref(enso_data))
        residual_corrs = cor.(eachslice(level_temp_data, dims=(1,2)), Ref(residual_data))
        
        # Calculate weighted components
        sepac_weighted = sepac_corrs .* sepac_std ./ residual_std
        enso_weighted = enso_corrs .* enso_std ./ residual_std
        
        # Convert to Float64 and clean
        sepac_weighted = Float64.(sepac_weighted)
        enso_weighted = Float64.(enso_weighted)
        residual_corrs = Float64.(residual_corrs)
        
        sepac_weighted = replace!(sepac_weighted, NaN => 0, Inf => 0, -Inf => 0)
        enso_weighted = replace!(enso_weighted, NaN => 0, Inf => 0, -Inf => 0)
        residual_corrs = replace!(residual_corrs, NaN => 0, Inf => 0, -Inf => 0)
        
        # Calculate verification: weighted SEPac - weighted ENSO vs actual residual correlation
        weighted_difference = sepac_weighted .- enso_weighted
        
        # Add verification data (column-major order: Residual, Weighted_Diff, Verification for each level)
        push!(all_verification_slices, residual_corrs)
        push!(all_verification_subtitles, "Residual Correlation - $level_name_str")
        
        push!(all_verification_slices, weighted_difference)
        push!(all_verification_subtitles, "Weighted Difference - $level_name_str")
        
        # Calculate difference for verification
        verification_diff = residual_corrs .- weighted_difference
        push!(all_verification_slices, verification_diff)
        push!(all_verification_subtitles, "Verification Diff - $level_name_str")
        
        # Calculate SEPac-ENSO correlation for this level (constant across space)
        sepac_enso_corr_level = fill(cor(sepac_data, enso_data), size(sepac_corrs))
        push!(sepac_enso_correlations, sepac_enso_corr_level)
    end
    
    # Create the plot layout: rows = levels, columns = SEPac_weighted/ENSO_weighted/Residual
    layout = (length(level_names), 3)  # 5 rows × 3 columns
    
    println("Creating temperature component plot...")
    println("Layout: $(layout[1]) rows (levels) × $(layout[2]) columns (SEPac_weighted/ENSO_weighted/Residual)")
    println("Number of subplots: $(length(all_contrib_slices))")
    
    # Get coordinate information for plotting
    lat = Float64.(era5_coords["latitude"])
    lon = Float64.(era5_coords["longitude"])
    
    # Create the temperature component plot
    contrib_fig = plot_multiple_levels(lat, lon, all_contrib_slices, layout;
                                     subtitles=all_contrib_subtitles,
                                     colorbar_label="Temperature Component")
    
    # Save the plot
    plot_filename = joinpath(visdir, "temperature_weighted_components.png")
    contrib_fig.suptitle("Temperature Components: Weighted SEPac/ENSO & Residual Correlations", fontsize=16)
    contrib_fig.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close(contrib_fig)
    
    println("Plot saved to: $plot_filename")
    
    # Create verification plot
    verification_layout = (length(level_names), 3)  # 5 rows × 3 columns
    
    println("Creating verification plot...")
    println("Verification layout: $(verification_layout[1]) rows (levels) × $(verification_layout[2]) columns (Residual/Weighted_Diff/Verification)")
    println("Number of verification subplots: $(length(all_verification_slices))")
    
    # Create the verification plot
    verification_fig = plot_multiple_levels(lat, lon, all_verification_slices, verification_layout;
                                          subtitles=all_verification_subtitles,
                                          colorbar_label="Correlation Value")
    
    # Save the verification plot
    verification_filename = joinpath(visdir, "temperature_verification.png")
    verification_fig.suptitle("Verification: Residual Correlation vs Weighted Difference", fontsize=16)
    verification_fig.savefig(verification_filename, dpi=300, bbox_inches="tight")
    plt.close(verification_fig)
    
    println("Verification plot saved to: $verification_filename")
    
    # Create SEPac-ENSO correlation plot for all levels
    sepac_enso_layout = (length(level_names), 1)  # 5 rows × 1 column
    sepac_enso_subtitles = ["SEPac-ENSO Correlation - $level" for level in level_names]
    
    println("Creating SEPac-ENSO correlation plot...")
    
    sepac_enso_fig = plot_multiple_levels(lat, lon, sepac_enso_correlations, sepac_enso_layout;
                                        subtitles=sepac_enso_subtitles,
                                        colorbar_label="Correlation Coefficient")
    
    # Save the SEPac-ENSO correlation plot
    sepac_enso_filename = joinpath(visdir, "sepac_enso_correlations.png")
    sepac_enso_fig.suptitle("SEPac-ENSO Correlation by Level", fontsize=16)
    sepac_enso_fig.savefig(sepac_enso_filename, dpi=300, bbox_inches="tight")
    plt.close(sepac_enso_fig)
    
    println("SEPac-ENSO correlation plot saved to: $sepac_enso_filename")
    
    # Print summary statistics
    println("\n=== SUMMARY ===")
    println("Analysis period: $(minimum(aligned_times)) to $(maximum(aligned_times))")
    println("Number of time points used: $(length(aligned_times))")
    println("Time series analyzed:")
    for (i, (name, label)) in enumerate(zip(time_series_names, time_series_labels))
        data = time_series_data[name]
        println("  $i. $label: mean = $(round(mean(data), digits=3)), std = $(round(std(data), digits=3))")
    end
    
    # Print correlation between time series
    sepac_data = time_series_data["sepac_sst_index"]
    enso_data = time_series_data["calculated_enso"]
    residual_data = time_series_data["sepac_sst_residual"]
    
    println("\nTime series correlations:")
    println("  SEPac-ENSO: $(round(cor(sepac_data, enso_data), digits=4))")
    println("  SEPac-Residual: $(round(cor(sepac_data, residual_data), digits=4))")
    println("  ENSO-Residual: $(round(cor(enso_data, residual_data), digits=4))")
    
    # Print verification statistics for each level
    for (level_idx, level_name_str) in enumerate(level_names)
        # For column-major ordering (columns first, then rows):
        # Column 1 (SEPac Weighted): indices 1, 6, 11, 16, 21 (level_idx + 0*5)
        # Column 2 (ENSO Weighted): indices 2, 7, 12, 17, 22 (level_idx + 1*5) 
        # Column 3 (Residual): indices 3, 8, 13, 18, 23 (level_idx + 2*5)
        sepac_weighted_idx = level_idx + 0 * length(level_names)  # level_idx + 0*5
        enso_weighted_idx = level_idx + 1 * length(level_names)   # level_idx + 1*5
        residual_idx = level_idx + 2 * length(level_names)        # level_idx + 2*5
        
        verification_residual_idx = level_idx * 3 - 2  # 1, 4, 7, 10, 13
        verification_diff_idx = level_idx * 3 - 1      # 2, 5, 8, 11, 14
        verification_check_idx = level_idx * 3          # 3, 6, 9, 12, 15
        
        sepac_weighted = all_contrib_slices[sepac_weighted_idx]
        enso_weighted = all_contrib_slices[enso_weighted_idx]
        residual_corr = all_contrib_slices[residual_idx]
        
        # Get verification data
        residual_verification = all_verification_slices[verification_residual_idx]
        weighted_diff_verification = all_verification_slices[verification_diff_idx]
        verification_diff = all_verification_slices[verification_check_idx]
        
        # Calculate statistics (excluding NaN/zeros)
        residual_valid = residual_corr[isfinite.(residual_corr) .& (residual_corr .!= 0)]
        verification_valid = verification_diff[isfinite.(verification_diff)]
        
        println("\n$level_name_str results:")
        if !isempty(residual_valid)
            println("  Residual: mean = $(round(mean(residual_valid), digits=4)), std = $(round(std(residual_valid), digits=4))")
        end
        
        if !isempty(verification_valid)
            println("  Verification (Residual - Weighted_Diff): mean = $(round(mean(verification_valid), digits=6)), std = $(round(std(verification_valid), digits=6)), max_abs = $(round(maximum(abs.(verification_valid)), digits=6))")
        end
    end
    
    println("\nFormulas used:")
    println("SEPac Weighted = correlation(SEPac) × std(SEPac) / std(SEPac_residual)")
    println("ENSO Weighted = correlation(ENSO) × std(ENSO) / std(SEPac_residual)")
    println("Residual = correlation(Residual) [unweighted]")
    println("Verification: SEPac_Weighted - ENSO_Weighted ≈ correlation(Residual)")
    
    return contrib_fig, verification_fig, sepac_enso_fig, time_series_data, aligned_times, all_contrib_slices, all_verification_slices
end

# Run the analysis
println("Starting Temperature Weighted Component Analysis")
println("="^60)
calculate_weighted_temperature_components()
