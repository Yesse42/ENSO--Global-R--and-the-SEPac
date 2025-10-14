using PythonCall

# Import necessary Python modules
mpl = pyimport("matplotlib")
plt = pyimport("matplotlib.pyplot")
ccrs = pyimport("cartopy.crs")
cfeature = pyimport("cartopy.feature")

visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/refs"

# Set up the figure with Robinson projection and central longitude at 180°
fig = plt.figure(figsize=(12, 8), dpi=500)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=180))

# Add coastlines
ax.coastlines(color="black", linewidth=0.5)

# Add thin gridlines every 5 degrees (no labels)
gl_thin = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=false, 
                       linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
gl_thin.xlocator = mpl.ticker.MultipleLocator(5)
gl_thin.ylocator = mpl.ticker.MultipleLocator(5)

# Add bold gridlines with labels every 15 degrees
gl_bold = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=true, 
                       linewidth=1.0, color="gray", alpha=0.8, linestyle="-")
gl_bold.xlocator = mpl.ticker.MultipleLocator(15)
gl_bold.ylocator = mpl.ticker.MultipleLocator(15)
gl_bold.xlabel_style = Dict("size" => 10, "color" => "black")
gl_bold.ylabel_style = Dict("size" => 10, "color" => "black")

# Add title
plt.title("Robinson Projection Map (Central Longitude: 180°)", fontsize=14, pad=20)

# Show the plot
plt.tight_layout()
plt.savefig(joinpath(visdir, "robinson_projection_map.png"); dpi = 500)

plt.close()
