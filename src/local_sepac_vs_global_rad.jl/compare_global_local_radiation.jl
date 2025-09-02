"""
This script compares global CERES radiation variables with local SEPac radiation variables.
It calculates correlations between:
- Global net radiation vs Local SEPac net radiation
- Global SW radiation vs Local SEPac SW radiation  
- Global LW radiation vs Local SEPac LW radiation

Results are saved to data/txt_results/local_global_rad_comparison.txt
"""

using Plots, Statistics, StatsBase, Dates, DataFrames, CSV
using LinearAlgebra

# Include necessary modules
olddir = pwd()
cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/constants.jl")
include("../utils/utilfuncs.jl")
cd(olddir)

"""
    preprocess_data(data, time_points)

Detrend and deseasonalize time series data using existing utility functions.
"""
function preprocess_data(data, time_points)
    # Make a copy to avoid modifying the original
    processed_data = copy(data)
    
    # Convert time points to float times for detrending
    float_times = @. year(time_points) + (month(time_points) - 1) / 12
    months = month.(time_points)
    
    # Use the existing utility function to detrend and deseasonalize
    detrend_and_deseasonalize!(processed_data, float_times, months)
    
    return processed_data
end

"""
    load_global_radiation_data(time_period)

Load global CERES radiation data using load_funcs.jl functions.
"""
function load_global_radiation_data(time_period)
    println("Loading global CERES radiation data...")
    
    # Load global CERES variables as specified in the requirements
    variables = ["gtoa_net_all_mon", "global_net_sw", "gtoa_lw_all_mon"]
    ceres_data, coords = load_ceres_data(variables, time_period)
    
    # Extract the data arrays
    global_net = ceres_data["gtoa_net_all_mon"]
    global_sw = ceres_data["global_net_sw"] 
    global_lw = ceres_data["gtoa_lw_all_mon"] .* -1  # Multiply by -1 as specified
    
    return global_net, global_sw, global_lw, coords["time"]
end

"""
    load_local_sepac_radiation_data()

Load local SEPac radiation data from CSV file.
"""
function load_local_sepac_radiation_data()
    println("Loading local SEPac CERES radiation data...")
    
    # Load local SEPac data
    sepac_path = "../../data/SEPac_SST/sepac_ceres_flux_time_series.csv"
    sepac_df = CSV.read(sepac_path, DataFrame)
    
    # Extract radiation variables
    local_net = sepac_df[!, "SEPac_Net_Radiation"]
    local_sw = sepac_df[!, "SEPac_Net_SW"]
    local_lw = sepac_df[!, "SEPac_Minus_LW"]
    
    # Parse dates
    local_time = DateTime.(sepac_df[!, "Date"])
    
    return local_net, local_sw, local_lw, local_time
end

"""
    align_time_series(global_data, global_time, local_data, local_time)

Align global and local time series to common time points.
"""
function align_time_series(global_data, global_time, local_data, local_time)
    # Convert to year-month pairs for alignment
    global_ym = [(year(gt), month(gt)) for gt in global_time]
    local_ym = [(year(lt), month(lt)) for lt in local_time]
    
    # Find common year-month pairs
    common_ym = intersect(global_ym, local_ym)
    sort!(common_ym)
    
    if length(common_ym) == 0
        error("No overlapping time periods between global and local data")
    end
    
    println("Found $(length(common_ym)) overlapping time points")
    println("Time range: $(minimum(common_ym)) to $(maximum(common_ym))")
    
    # Find indices for alignment
    global_indices = [findfirst(==(ym), global_ym) for ym in common_ym]
    local_indices = [findfirst(==(ym), local_ym) for ym in common_ym]
    
    # Extract aligned data
    global_aligned = global_data[global_indices]
    local_aligned = local_data[local_indices]
    
    return global_aligned, local_aligned, common_ym
end

"""
    calculate_correlations_and_save_results()

Main function to compare global and local radiation data.
"""
function calculate_correlations_and_save_results()
    println("Starting comparison of global vs local SEPac radiation...")
    
    # Load global radiation data for the standard time period
    global_net, global_sw, global_lw, global_time = load_global_radiation_data(time_period)
    
    # Load local SEPac radiation data
    local_net, local_sw, local_lw, local_time = load_local_sepac_radiation_data()
    
    # Align time series
    println("\nAligning time series...")
    
    # Align net radiation
    global_net_aligned, local_net_aligned, common_time = align_time_series(
        global_net, global_time, local_net, local_time)
    
    # Align SW radiation (should have same time alignment)
    global_sw_aligned, local_sw_aligned, _ = align_time_series(
        global_sw, global_time, local_sw, local_time)
    
    # Align LW radiation (should have same time alignment)
    global_lw_aligned, local_lw_aligned, _ = align_time_series(
        global_lw, global_time, local_lw, local_time)
    
    # Preprocess data (detrend and deseasonalize)
    println("Preprocessing data (detrend and deseasonalize)...")
    
    # Create DateTime objects for preprocessing (need day for the function)
    common_datetime = [DateTime(ym[1], ym[2], 15) for ym in common_time]
    
    # Preprocess global data
    global_net_processed = preprocess_data(global_net_aligned, common_datetime)
    global_sw_processed = preprocess_data(global_sw_aligned, common_datetime)
    global_lw_processed = preprocess_data(global_lw_aligned, common_datetime)
    
    # Preprocess local data
    local_net_processed = preprocess_data(local_net_aligned, common_datetime)
    local_sw_processed = preprocess_data(local_sw_aligned, common_datetime)
    local_lw_processed = preprocess_data(local_lw_aligned, common_datetime)
    
    # Calculate correlations
    println("Calculating correlations...")
    
    corr_net = cor(global_net_processed, local_net_processed)
    corr_sw = cor(global_sw_processed, local_sw_processed)
    corr_lw = cor(global_lw_processed, local_lw_processed)
    
    # Calculate correlations with raw (unprocessed) data for comparison
    corr_net_raw = cor(global_net_aligned, local_net_aligned)
    corr_sw_raw = cor(global_sw_aligned, local_sw_aligned)
    corr_lw_raw = cor(global_lw_aligned, local_lw_aligned)
    
    # Calculate basic statistics
    println("Calculating basic statistics...")
    
    # Global data stats (processed)
    global_net_mean = mean(global_net_processed)
    global_net_std = std(global_net_processed)
    global_sw_mean = mean(global_sw_processed)
    global_sw_std = std(global_sw_processed)
    global_lw_mean = mean(global_lw_processed)
    global_lw_std = std(global_lw_processed)
    
    # Local data stats (processed)
    local_net_mean = mean(local_net_processed)
    local_net_std = std(local_net_processed)
    local_sw_mean = mean(local_sw_processed)
    local_sw_std = std(local_sw_processed)
    local_lw_mean = mean(local_lw_processed)
    local_lw_std = std(local_lw_processed)
    
    # Create output directory
    output_dir = "../../data/txt_results"
    mkpath(output_dir)
    
    # Save results to text file
    output_file = joinpath(output_dir, "local_global_rad_comparison.txt")
    
    open(output_file, "w") do io
        println(io, "="^80)
        println(io, "COMPARISON OF GLOBAL VS LOCAL SEPac RADIATION DATA")
        println(io, "="^80)
        println(io, "")
        println(io, "Analysis Date: $(now())")
        println(io, "")
        println(io, "DATASET INFORMATION:")
        println(io, "-"^40)
        println(io, "Global Data Variables:")
        println(io, "  - Net Radiation: gtoa_net_all_mon")
        println(io, "  - SW Radiation: global_net_sw") 
        println(io, "  - LW Radiation: gtoa_lw_all_mon * (-1)")
        println(io, "")
        println(io, "Local Data Variables:")
        println(io, "  - Net Radiation: SEPac_Net_Radiation")
        println(io, "  - SW Radiation: SEPac_Net_SW")
        println(io, "  - LW Radiation: SEPac_Minus_LW")
        println(io, "")
        println(io, "TIME PERIOD:")
        println(io, "-"^40)
        println(io, "Common time points: $(length(common_time))")
        println(io, "Start: $(minimum(common_time))")
        println(io, "End: $(maximum(common_time))")
        println(io, "")
        println(io, "DATA PROCESSING:")
        println(io, "-"^40)
        println(io, "All data was detrended and deseasonalized before correlation calculation")
        println(io, "")
        println(io, "CORRELATION RESULTS (Processed Data):")
        println(io, "="^50)
        println(io, "Net Radiation:  $(round(corr_net, digits=4))")
        println(io, "SW Radiation:   $(round(corr_sw, digits=4))")
        println(io, "LW Radiation:   $(round(corr_lw, digits=4))")
        println(io, "")
        println(io, "CORRELATION RESULTS (Raw Data):")
        println(io, "="^50)
        println(io, "Net Radiation:  $(round(corr_net_raw, digits=4))")
        println(io, "SW Radiation:   $(round(corr_sw_raw, digits=4))")
        println(io, "LW Radiation:   $(round(corr_lw_raw, digits=4))")
        println(io, "")
        println(io, "BASIC STATISTICS (Processed Data):")
        println(io, "="^50)
        println(io, "")
        println(io, "GLOBAL RADIATION:")
        println(io, "Net - Mean: $(round(global_net_mean, digits=3)), Std: $(round(global_net_std, digits=3))")
        println(io, "SW  - Mean: $(round(global_sw_mean, digits=3)), Std: $(round(global_sw_std, digits=3))")
        println(io, "LW  - Mean: $(round(global_lw_mean, digits=3)), Std: $(round(global_lw_std, digits=3))")
        println(io, "")
        println(io, "LOCAL SEPac RADIATION:")
        println(io, "Net - Mean: $(round(local_net_mean, digits=3)), Std: $(round(local_net_std, digits=3))")
        println(io, "SW  - Mean: $(round(local_sw_mean, digits=3)), Std: $(round(local_sw_std, digits=3))")
        println(io, "LW  - Mean: $(round(local_lw_mean, digits=3)), Std: $(round(local_lw_std, digits=3))")
        println(io, "")
        println(io, "INTERPRETATION:")
        println(io, "="^50)
        println(io, "Correlation values range from -1 to +1:")
        println(io, "  - Values close to +1 indicate strong positive correlation")
        println(io, "  - Values close to -1 indicate strong negative correlation")
        println(io, "  - Values close to 0 indicate weak or no linear correlation")
        println(io, "")
        if abs(corr_net) > 0.7
            println(io, "Net radiation shows strong correlation between global and local SEPac data.")
        elseif abs(corr_net) > 0.3
            println(io, "Net radiation shows moderate correlation between global and local SEPac data.")
        else
            println(io, "Net radiation shows weak correlation between global and local SEPac data.")
        end
        
        if abs(corr_sw) > 0.7
            println(io, "SW radiation shows strong correlation between global and local SEPac data.")
        elseif abs(corr_sw) > 0.3
            println(io, "SW radiation shows moderate correlation between global and local SEPac data.")
        else
            println(io, "SW radiation shows weak correlation between global and local SEPac data.")
        end
        
        if abs(corr_lw) > 0.7
            println(io, "LW radiation shows strong correlation between global and local SEPac data.")
        elseif abs(corr_lw) > 0.3
            println(io, "LW radiation shows moderate correlation between global and local SEPac data.")
        else
            println(io, "LW radiation shows weak correlation between global and local SEPac data.")
        end
        println(io, "")
        println(io, "="^80)
    end
    
    println("\nResults saved to: $output_file")
    
    # Print summary to console
    println("\n" * "="^60)
    println("CORRELATION SUMMARY (Processed Data)")
    println("="^60)
    println("Net Radiation: $(round(corr_net, digits=4))")
    println("SW Radiation:  $(round(corr_sw, digits=4))")
    println("LW Radiation:  $(round(corr_lw, digits=4))")
    println("="^60)
    
    return corr_net, corr_sw, corr_lw
end

# Run the analysis
if true
    cd(@__DIR__)
    calculate_correlations_and_save_results()
end
