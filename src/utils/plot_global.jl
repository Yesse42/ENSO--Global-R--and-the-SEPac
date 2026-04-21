using PythonCall, NCDatasets, Statistics, Plots
@py import cartopy.crs as ccrs, matplotlib.pyplot as plt, matplotlib.path as mpath, matplotlib.colors as colors, matplotlib.cm as cm, ssl as ssl, cmasher as cmr

# Custom colormap
CUSTOM_CMAP_COLORS = [
     10  50 120
     15  75 165
     30 110 200
     60 160 240
     80 180 250
    130 210 255
    160 240 255
    220 250 255
    255 255 255
    255 255 255
    255 240 120
    255 192  60
    255 160   0
    255  96   0
    255  50   0
    225  20   0
    192   0   0
    165   0   0
] ./ 255

# Function to create the custom colormap

DAVES_CMAP = colors.LinearSegmentedColormap.from_list("custom", pylist(collect(eachrow(CUSTOM_CMAP_COLORS))))



function plot_global_heatmap(lat, lon, data_matrix; title, colorbar_label, cmap = cmr.prinsenvlag.reversed(), central_longitude=180, colornorm = nothing)
    fig, ax = plt.subplots(subplot_kw=Dict("projection"=>ccrs.Robinson(;central_longitude)))
    ax.set_global()
    ax.coastlines()
    ax.set_title(title)

    if isnothing(colornorm)
        absmax = max(abs(minimum(data_matrix)), abs(maximum(data_matrix)))
        colornorm = colors.Normalize(vmin=-absmax, vmax=absmax)
    end

    # Convert on the Julia side first to avoid memory leaks with non-Array{Float64} inputs
    lon_f64  = convert(Array{Float64}, lon)
    lat_f64  = convert(Array{Float64}, lat)
    data_f64 = convert(Array{Float64}, collect(data_matrix'))
    _np = pyimport("numpy")
    c = ax.pcolormesh(_np.asarray(lon_f64),
                      _np.asarray(lat_f64),
                      _np.asarray(data_f64);
        transform = ccrs.PlateCarree(), cmap = cmap, norm = colornorm, shading = "auto")

    # Add a colorbar
    plt.colorbar(c, ax=ax, orientation="horizontal", pad=0.05, label = colorbar_label)

    return fig
end

"""
    plot_global_heatmap_on_ax!(ax, lat, lon, data_matrix; ..., alpha_matrix=nothing)

Draw an imshow on a Cartopy axis.  When `alpha_matrix` (same lon×lat shape as
`data_matrix`) is supplied each cell's opacity is set to the corresponding value
(expected in [0, 1]).  This is used to encode e.g. R² so that low-signal cells fade out.
"""
function plot_global_heatmap_on_ax!(ax, lat, lon, data_matrix;
        cmap = cmr.prinsenvlag.reversed(), colornorm = nothing, title,
        alpha_matrix = nothing)
    ax.set_global()
    ax.coastlines()

    if isnothing(colornorm)
        absmax = max(abs(minimum(data_matrix)), abs(maximum(data_matrix)))
        colornorm = colors.Normalize(vmin=-absmax, vmax=absmax)
    end

    # Sort lat ascending so origin="lower" is correct
    lat_f64   = convert(Array{Float64}, lat)
    lon_f64   = convert(Array{Float64}, lon)
    lat_order = sortperm(lat_f64)
    data_f64  = convert(Array{Float64}, collect(data_matrix[:, lat_order]'))  # (lat, lon)
    extent    = [minimum(lon_f64), maximum(lon_f64), minimum(lat_f64), maximum(lat_f64)]

    np = pyimport("numpy")

    if isnothing(alpha_matrix)
        c = ax.imshow(np.array(data_f64, copy=true);
            transform = ccrs.PlateCarree(), cmap = cmap, norm = colornorm,
            extent = extent, origin = "lower", aspect = "auto")
    else
        alpha_f64 = convert(Array{Float64}, collect(clamp.(alpha_matrix[:, lat_order]', 0.0, 1.0)))
        c = ax.imshow(np.array(data_f64, copy=true);
            transform = ccrs.PlateCarree(), cmap = cmap, norm = colornorm,
            extent = extent, origin = "lower", aspect = "auto",
            alpha = np.array(alpha_f64, copy=true))
    end

    ax.set_title(title)
    return c
end

function create_multi_projection_plot(shape, projection)
    # Use plt.subplots with subplot_kw to set projections
    fig, axs = plt.subplots(shape[1], shape[2], 
                           figsize=(6*shape[2], 4*shape[1]),
                           subplot_kw=Dict("projection" => projection),
                           layout = "compressed")
    # Flatten the array of axes
    if shape[1] ≠ 1 && shape[2] ≠ 1
        axes_flat = reduce(vcat, [pyconvert(Array, axvec) for axvec in axs])
    else
        axes_flat = collect(axs)
    end 

    axes_flat = vec(permutedims(reshape(axes_flat, shape[2], shape[1]))) #Ensure column major
    
    return fig, axs, axes_flat
end

function plot_multiple_levels(lat, lon, data_slices, layout; subtitles, colorbar_label, cmap = cmr.prinsenvlag.reversed(), proj = ccrs.Robinson(central_longitude=180), colornorm = nothing)
    fig, axs, axes_flat = create_multi_projection_plot(layout, proj)
    
    if colornorm === nothing
        all_data = vcat([vec(slice) for slice in data_slices]...)
        absmax = max(abs(minimum(all_data)), abs(maximum(all_data)))
        colornorm = colors.Normalize(vmin=-absmax, vmax=absmax)
    else
        # Calculate all_data even when colornorm is provided
        all_data = vcat([vec(slice) for slice in data_slices]...)
    end

    #Transpose the subtitles if needed
    if subtitles isa Matrix
        subtitles = permutedims(subtitles)
    else
        subtitles = permutedims(reshape(subtitles, layout[1], layout[2]))
    end

    for (i, ax) in enumerate(axes_flat[1:length(data_slices)])
        plot_global_heatmap_on_ax!(ax, lat, lon, data_slices[i]; cmap=cmap, colornorm=colornorm, title=subtitles[i])
    end

    plt.colorbar(cm.ScalarMappable(norm=colornorm, cmap=cmap), ax=axs, orientation="horizontal", label=colorbar_label)
    return fig
end

"""
    plot_multiple_levels_rowmajor(lat, lon, data_slices, layout; subtitles, colorbar_label, ...)

Like `plot_multiple_levels` but correctly preserves row-major ordering so that
`data_slices[i]` and `subtitles[i]` always correspond, laid out left-to-right
then top-to-bottom (reading order).

The original `plot_multiple_levels` has a bug for multi-row multi-column layouts:
it permutes axes and subtitles to column-major but leaves data_slices in input order,
causing data/title mismatches.
"""
function plot_multiple_levels_rowmajor(lat, lon, data_slices, layout;
        subtitles, colorbar_label, cmap = cmr.prinsenvlag.reversed(),
        proj = ccrs.Robinson(central_longitude=180), colornorm = nothing)

    nrows, ncols = layout
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(6*ncols, 4*nrows),
                            subplot_kw=Dict("projection" => proj),
                            layout = "compressed")

    # Flatten axes in row-major order (Python's natural iteration order)
    if nrows > 1 && ncols > 1
        axes_flat = reduce(vcat, [pyconvert(Array, row) for row in axs])
    elseif nrows == 1 && ncols == 1
        axes_flat = [axs]
    else
        axes_flat = collect(axs)
    end

    if colornorm === nothing
        all_vals = filter(!isnan, vcat([vec(Float64.(s)) for s in data_slices]...))
        absmax = maximum(abs, all_vals)
        colornorm = colors.Normalize(vmin=-absmax, vmax=absmax)
    end

    for (i, ax) in enumerate(axes_flat[1:length(data_slices)])
        plot_global_heatmap_on_ax!(ax, lat, lon, data_slices[i];
            cmap=cmap, colornorm=colornorm, title=subtitles[i])
    end

    plt.colorbar(cm.ScalarMappable(norm=colornorm, cmap=cmap),
                 ax=axs, orientation="horizontal", label=colorbar_label)
    return fig
end

function add_region_contours!(ax, rad_lat, rad_lon, rad_mask, temp_lat, temp_lon, temp_mask)
    ax.contour(convert(Array{Float64}, rad_lon),
               convert(Array{Float64}, rad_lat),
               convert(Array{Float64}, collect(Float64.(rad_mask)'));
        transform=ccrs.PlateCarree(), levels=pylist([0.5]), colors=pylist(["red"]), linewidths=pylist([1.5]))
    ax.contour(convert(Array{Float64}, temp_lon),
               convert(Array{Float64}, temp_lat),
               convert(Array{Float64}, collect(Float64.(temp_mask)'));
        transform=ccrs.PlateCarree(), levels=pylist([0.5]), colors=pylist(["blue"]), linewidths=pylist([1.5]), linestyles=pylist(["--"]))
    return ax
end

function plot_polygon_on_ax!(ax, lonlat_points; color="red", alpha=0.5, linewidth=1, linestyle="-", facecolor=nothing, edgecolor=nothing, kwargs...)
    # Extract longitude and latitude arrays
    lons = [point[1] for point in lonlat_points]
    lats = [point[2] for point in lonlat_points]
    
    # Close the polygon by adding the first point at the end if not already closed
    if lons[1] != lons[end] || lats[1] != lats[end]
        push!(lons, lons[1])
        push!(lats, lats[1])
    end
    
    # Set default colors if not provided
    if isnothing(facecolor)
        facecolor = color
    end
    if isnothing(edgecolor)
        edgecolor = color
    end
    
    # Plot the polygon
    ax.fill(lons, lats, transform=ccrs.PlateCarree(), 
           facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, 
           linewidth=linewidth, linestyle=linestyle; kwargs...)
    
    return ax
end