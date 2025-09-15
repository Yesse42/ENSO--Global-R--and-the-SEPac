"""
This script aims to calculate the area of different masks (regions) on the Earth's surface using latitude and longitude data.
"""

function get_differential_area(lat, dlon, dlat)
    return dlon * dlat * cosd(lat)
end

function get_area_bool_mask(bool_mask, lons, lats)
    dlon = abs(deg2rad(lons[2] - lons[1]))
    dlat = abs(deg2rad(lats[2] - lats[1]))

    expanded_lats = repeat(lats', size(bool_mask, 1), 1)

    area_array = get_differential_area.(expanded_lats, dlon, dlat)
    masked_area = sum(area_array[bool_mask])
    return masked_area
end

using JLD2
using Dates

mask_file = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/stratocumulus_region_masks.jld2"
region_data = JLD2.load(mask_file)

# Extract coordinate arrays
lats_ceres = region_data["latitude_ceres"]
lons_ceres = region_data["longitude_ceres"]
lats_era5 = region_data["latitude_era5"]
lons_era5 = region_data["longitude_era5"]

# Get the masks
masks_ceres = region_data["regional_masks_ceres"]
masks_era5 = region_data["regional_masks_era5"]

println("Calculating areas for CERES and ERA5 grids...")
println("=" ^ 60)

# Initialize dictionaries to store results
area_results_ceres = Dict{String, Dict{String, Float64}}()
area_results_era5 = Dict{String, Dict{String, Float64}}()

# Calculate areas for each region and its complement
for (region_name, mask_ceres) in masks_ceres
    println("\nRegion: $region_name")
    println("-" ^ 40)
    
    # CERES calculations
    mask_era5 = masks_era5[region_name]
    
    # Calculate area of the masked region
    area_masked_ceres = get_area_bool_mask(mask_ceres, lons_ceres, lats_ceres)
    area_masked_era5 = get_area_bool_mask(mask_era5, lons_era5, lats_era5)
    
    # Calculate area of the complement (not masked region)
    complement_ceres = .!mask_ceres
    complement_era5 = .!mask_era5
    
    area_complement_ceres = get_area_bool_mask(complement_ceres, lons_ceres, lats_ceres)
    area_complement_era5 = get_area_bool_mask(complement_era5, lons_era5, lats_era5)
    
    area_masked_ceres_km2 = area_masked_ceres * 1
    area_masked_era5_km2 = area_masked_era5 * 1
    area_complement_ceres_km2 = area_complement_ceres * 1
    area_complement_era5_km2 = area_complement_era5 * 1
    
    # Store results in dictionaries
    area_results_ceres[region_name] = Dict(
        "masked_area" => area_masked_ceres_km2,
        "complement_area" => area_complement_ceres_km2,
        "total_area" => area_masked_ceres_km2 + area_complement_ceres_km2
    )
    
    area_results_era5[region_name] = Dict(
        "masked_area" => area_masked_era5_km2,
        "complement_area" => area_complement_era5_km2,
        "total_area" => area_masked_era5_km2 + area_complement_era5_km2
    )
    
    println("CERES Grid:")
    println("  Masked region area: $(round(area_masked_ceres_km2, digits=2)) km²")
    println("  Complement area: $(round(area_complement_ceres_km2, digits=2)) km²")
    println("  Total area: $(round(area_masked_ceres_km2 + area_complement_ceres_km2, digits=2)) km²")
    
    println("ERA5 Grid:")
    println("  Masked region area: $(round(area_masked_era5_km2, digits=2)) km²")
    println("  Complement area: $(round(area_complement_era5_km2, digits=2)) km²")
    println("  Total area: $(round(area_masked_era5_km2 + area_complement_era5_km2, digits=2)) km²")
end

# Sanity check: calculate area of all-true mask (should be Earth's surface area)
println("\n" * "=" ^ 60)
println("SANITY CHECK: All-true mask areas")
println("=" ^ 60)

all_true_ceres = trues(size(first(values(masks_ceres))))
all_true_era5 = trues(size(first(values(masks_era5))))

total_area_ceres = get_area_bool_mask(all_true_ceres, lons_ceres, lats_ceres)
total_area_era5 = get_area_bool_mask(all_true_era5, lons_era5, lats_era5)

total_area_ceres_km2 = total_area_ceres * 1
total_area_era5_km2 = total_area_era5 * 1

println("CERES Grid total surface area: $(round(total_area_ceres_km2, digits=2)) km²")
println("ERA5 Grid total surface area: $(round(total_area_era5_km2, digits=2)) km²")
println("Theoretical Earth surface area: $(round(4 * π, digits=2)) km²")

# Calculate percentage coverage for each region
println("\n" * "=" ^ 60)
println("PERCENTAGE COVERAGE")
println("=" ^ 60)

percentage_coverage = Dict{String, Dict{String, Float64}}()

for (region_name, mask_ceres) in masks_ceres
    mask_era5 = masks_era5[region_name]
    
    area_masked_ceres = get_area_bool_mask(mask_ceres, lons_ceres, lats_ceres)
    area_masked_era5 = get_area_bool_mask(mask_era5, lons_era5, lats_era5)
    
    # Calculate complement areas for percentage calculation
    complement_ceres = .!mask_ceres
    complement_era5 = .!mask_era5
    area_complement_ceres = get_area_bool_mask(complement_ceres, lons_ceres, lats_ceres)
    area_complement_era5 = get_area_bool_mask(complement_era5, lons_era5, lats_era5)
    
    pct_ceres = (area_masked_ceres / total_area_ceres) * 100
    pct_era5 = (area_masked_era5 / total_area_era5) * 100
    pct_complement_ceres = (area_complement_ceres / total_area_ceres) * 100
    pct_complement_era5 = (area_complement_era5 / total_area_era5) * 100
    
    # Store percentage coverage for both masked and complement regions
    percentage_coverage[region_name] = Dict(
        "ceres_masked_percentage" => pct_ceres,
        "era5_masked_percentage" => pct_era5,
        "ceres_complement_percentage" => pct_complement_ceres,
        "era5_complement_percentage" => pct_complement_era5
    )
    
    println("$region_name:")
    println("  CERES Masked: $(round(pct_ceres, digits=3))%")
    println("  CERES Complement: $(round(pct_complement_ceres, digits=3))%")
    println("  ERA5 Masked: $(round(pct_era5, digits=3))%")
    println("  ERA5 Complement: $(round(pct_complement_era5, digits=3))%")
end

# Save the results to dictionaries and export to JLD2
println("\n" * "=" ^ 60)
println("SAVING RESULTS")
println("=" ^ 60)

# Create comprehensive results dictionary
results_dict = Dict(
    "area_results_ceres" => area_results_ceres,
    "area_results_era5" => area_results_era5,
    "percentage_coverage" => percentage_coverage,
    "total_areas" => Dict(
        "ceres_total_area" => total_area_ceres_km2,
        "era5_total_area" => total_area_era5_km2
    ),
    "metadata" => Dict(
        "units" => "steradians (raw areas), percentage (coverage)",
        "description" => "Area calculations for stratocumulus region masks",
        "calculation_date" => string(today()),
        "source_file" => mask_file
    )
)

# Save to JLD2 file in the same directory as the original
output_file = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/mask_area_calculations.jld2"
JLD2.save(output_file, results_dict)

println("Results saved to: $output_file")
println("Dictionary contains the following keys:")
for key in keys(results_dict)
    println("  - $key")
end
