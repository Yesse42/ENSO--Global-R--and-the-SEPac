using PythonCall, NCDatasets, Statistics, Plots
@py import cartopy.crs as ccrs, matplotlib.pyplot as plt, matplotlib.path as mpath, matplotlib.colors as colors, matplotlib.cm as cm, ssl as ssl, cmasher as cmr



function plot_global_heatmap(lat, lon, data_matrix; title, colorbar_label, cmap = cmr.prinsenvlag.reversed(), central_longitude=180, colornorm = nothing)
    fig, ax = plt.subplots(subplot_kw=Dict("projection"=>ccrs.Robinson(;central_longitude)))
    ax.set_global()
    ax.coastlines()
    ax.set_title(title)

    if isnothing(colornorm)
        absmax = max(abs(minimum(data_matrix)), abs(maximum(data_matrix)))
        colornorm = colors.Normalize(vmin=-absmax, vmax=absmax)
    end

    # Plot the data
    c = ax.contourf(lon, lat, data_matrix'; transform=ccrs.PlateCarree(), cmap, levels = 21, norm = colornorm)

    # Add a colorbar
    plt.colorbar(c, ax=ax, orientation="horizontal", pad=0.05, label = colorbar_label)

    return fig
end

function plot_global_heatmap_on_ax!(ax, lat, lon, data_matrix; cmap = cmr.prinsenvlag.reversed(), colornorm = nothing, title)
    ax.set_global()
    ax.coastlines()

    if isnothing(colornorm)
        absmax = max(abs(minimum(data_matrix)), abs(maximum(data_matrix)))
        colornorm = colors.Normalize(vmin=-absmax, vmax=absmax)
    end

    # Plot the data
    c = ax.contourf(lon, lat, data_matrix'; transform=ccrs.PlateCarree(), cmap, levels = 21, norm = colornorm)

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

    for (i, ax) in enumerate(axes_flat[1:length(data_slices)])
        plot_global_heatmap_on_ax!(ax, lat, lon, data_slices[i]; cmap=cmap, colornorm=colornorm, title=subtitles[i])
    end

    plt.colorbar(cm.ScalarMappable(norm=colornorm, cmap=cmap), ax=axs, orientation="horizontal", label=colorbar_label)
    return fig
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