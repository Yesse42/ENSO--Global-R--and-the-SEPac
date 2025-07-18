using JLD2, Statistics, StatsBase, Dates, SplitApplyCombine

# Include necessary modules
cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/constants.jl")
include("../utils/utilfuncs.jl")
include("../pls_regressor/pls_functions.jl")
include("../pls_regressor/pls_structs.jl")
include("../utils/plot_global.jl")

"""
This script compares the matrices from make_matrix_to_multiply_by_X_to_get_Y 
for components 1-5 of both full and ENSO deflated PLSs.
It creates side-by-side visualizations showing the differences between
basic atmospheric PLS and ENSO-deflated PLS regression coefficients.
"""

# Set up output directory
visdir = "../../vis/enso_deflated_comparison/"
if !isdir(visdir)
    mkpath(visdir)
end

PLSdir = "../../data/PLSs/"

# Load PLS models
println("Loading PLS models...")
atmospheric_pls_data = jldopen(joinpath(PLSdir, "atmospheric_ceres_net_pls.jld2"), "r")
deflated_pls_data = jldopen(joinpath(PLSdir, "enso_deflated_net_pls.jld2"), "r")

atmospheric_pls = atmospheric_pls_data["pls_model"]
deflated_pls = deflated_pls_data["pls_model"]

atmospheric_idxs = atmospheric_pls_data["predictor_indices"]
deflated_idxs = deflated_pls_data["predictor_idxs"]

atmospheric_shapes = atmospheric_pls_data["predictor_shapes"]
deflated_shapes = deflated_pls_data["predictor_shapes"]

# Get coordinate information
coords = atmospheric_pls_data["coordinates"]
lat = Float64.(coords["latitude"])
lon = Float64.(coords["longitude"])
p_level = Float64.(coords["pressure_level"])

stacked_var_names = ["temp", "press_geopotential"]
stacked_level_vars = Dictionary(stacked_var_names, [["t2m", "t"], ["msl", "z"]])

level_names = ["sfc", "850hPa", "700hPa", "500hPa", "250hPa"]


function plot_comparison(atmospheric_pls, deflated_pls, atmospheric_idxs, deflated_idxs, atmospheric_shapes, deflated_shapes, 
                        lon, lat, sfc_var, level_var, level_names, visdir, components)
    #First generate the relevant matrices and plop them into an array
    atmospheric_matrices = Array{Array{Float64,2}}(undef, (length(level_names), length(components)))
    deflated_matrices = Array{Array{Float64,2}}(undef, (length(level_names), length(components)))

    deflated_sfc_var = "deflated_" * sfc_var
    deflated_level_var = "deflated_" * level_var
    
    atmospheric_matrix_names = Array{String,2}(undef, (length(level_names), length(components)))
    deflated_matrix_names = Array{String,2}(undef, (length(level_names), length(components)))

    for (j, comp) in enumerate(components)
        atmospheric_unreshaped = make_matrix_to_multiply_by_X_to_get_Y(atmospheric_pls; components=1:comp)
        deflated_unreshaped = make_matrix_to_multiply_by_X_to_get_Y(deflated_pls; components=1:comp)

        # Now reshape them
        atmospheric_reshaped = reconstruct_spatial_arrays(atmospheric_unreshaped, atmospheric_idxs, atmospheric_shapes)
        deflated_reshaped = reconstruct_spatial_arrays(deflated_unreshaped, deflated_idxs, deflated_shapes)
        for (i, level_name) in enumerate(level_names)
            if i == 1
                atmospheric_matrices[i,j] = atmospheric_reshaped[sfc_var][:,:,]
                deflated_matrices[i,j] = deflated_reshaped[deflated_sfc_var][:,:]

                atmospheric_matrix_names[i,j] = "$(sfc_var)_sfc_comp$(comp)"
                deflated_matrix_names[i,j] = "$(sfc_var)_sfc_comp$(comp)_deflated"
            else
                atmospheric_matrices[i,j] = atmospheric_reshaped[level_var][:,:,i-1]
                deflated_matrices[i,j] = deflated_reshaped[deflated_level_var][:,:,i-1]

                atmospheric_matrix_names[i,j] = "$(level_var)_$(level_name)_comp$(comp)"
                deflated_matrix_names[i,j] = "$(level_var)_$(level_name)_comp$(comp)_deflated"
            end
        end
    end

    #Plots got permuted
    atmospheric_matrices = permutedims(atmospheric_matrices, (2, 1))
    deflated_matrices = permutedims(deflated_matrices, (2, 1))
    atmospheric_matrix_names = permutedims(atmospheric_matrix_names, (2, 1))
    deflated_matrix_names = permutedims(deflated_matrix_names, (2, 1))

    #Divide by the norms for better visualization
    atmosphere_norms = [norm(reduce(vcat, matrow)) for matrow in eachrow(atmospheric_matrices)]
    deflated_norms = [norm(reduce(vcat, matrow)) for matrow in eachrow(deflated_matrices)]

    for (row, norm) in zip(eachrow(atmospheric_matrices), atmosphere_norms)
        for mat in row
            mat ./= norm
        end
    end

    for (row, norm) in zip(eachrow(deflated_matrices), deflated_norms)
        for mat in row
            mat ./= norm
        end
    end

    #Now plot them
    plot_size = (length(level_names), length(components) * 2)

    combined_matrices = reduce(hcat, [hcat(atmos_col, reduced_col) for (atmos_col, reduced_col) in zip(eachcol(atmospheric_matrices), eachcol(deflated_matrices))])
    combined_titles = reduce(hcat, hcat(atmos_name_col, deflated_name_col) for (atmos_name_col, deflated_name_col) in zip(eachcol(atmospheric_matrix_names), eachcol(deflated_matrix_names)))

    #Plotting
    fig = plot_multiple_levels(lat, lon, combined_matrices, plot_size; 
        subtitles = combined_titles, 
        colorbar_label = "Regression Coefficient", 
        cmap = cmr.prinsenvlag.reversed(), proj = ccrs.Robinson(central_longitude=180))

    fig.savefig(joinpath(visdir, "comparison_$(sfc_var)_$(level_var)_comp$(join(components, "_")).png"), dpi=500)
    plt.close()
end

# Plot comparisons for all variable combinations
for (stacked_var, var_names) in pairs(stacked_level_vars)
    sfc_var = var_names[1]
    level_var = var_names[2]
    
    println("Plotting comparison for $(sfc_var) and $(level_var)...")
    plot_comparison(atmospheric_pls, deflated_pls, atmospheric_idxs, deflated_idxs, 
                   atmospheric_shapes, deflated_shapes, lon, lat, sfc_var, level_var, 
                   level_names, visdir, 1:5)
end