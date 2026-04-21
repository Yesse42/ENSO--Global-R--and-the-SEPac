#This code seeks to tile a grid of values into a collection of masks that contain values meeting certain criteria

using SplitApplyCombine, Dictionaries, Random, Graphs

function standard_criteria_func(low, high)
    function curryfunc(x)
        return (x >= low) + (x > high)
    end
end

lazy_mod_idx1(idx, modulus) = rem(idx - 1, modulus, RoundDown) + 1

standard_adjacency_arr = [0 1 0; 1 0 1; 0 1 0]

"assumes the grid is lonxlat, and is periodic in the longitude direction, and that the adjacency array is odd sized"
function tile_latlon_grid(grid, criteria_func, adjacency_arr; verbose = true)
    #First generate the needed data
    data = criteria_func.(grid)
    visited = zeros(Int, size(data)) #Begins as zeros, will be set to a unique integer value for each mask we generate

    #Now we need to generate some helper objects and functions to make indexing better
    if !all(isodd.(size(adjacency_arr)))
        error("This code only works for odd sized adjacency arrays, sorry")
    end

    idx_offsets_centered = CartesianIndices(size(adjacency_arr)) .- CartesianIndex(ceil.(Int, size(adjacency_arr) ./ 2))
    cartesian_idxs_to_check = idx_offsets_centered[adjacency_arr .!= 0]

    function map_cartesian_idx_into_valid_range(idx)
        idx = Tuple(idx)
        idx = (lazy_mod_idx1(idx[1], size(data, 1)), clamp(idx[2], 1, size(data, 2)))
        return CartesianIndex(idx)
    end

    #Now that we have the helper functions write our code to generate the masks. We will do this by iterating through the data, and whenever we find a value that meets the criteria and has not been visited, we will generate a mask for it. If we find an adjacent value meets the same criterion and is part of a different mask, we will merge the two. After the whole array has been iterated over
    current_n_masks = 0

    # Create a graph to track which masks need to be merged
    # We'll dynamically add vertices as needed
    mask_graph = SimpleGraph()
    
    total_points_to_consider = length(data)
    points_considered = 0
    check_in_fracs = [0, 0.25, 0.5, 0.75]
    check_in_idxs = round.(Int, check_in_fracs .* total_points_to_consider)

    for I in CartesianIndices(data)
        if verbose && points_considered in check_in_idxs
            println("Considered $points_considered out of $total_points_to_consider points")
        end
        points_considered += 1
        current_val = data[I]
        cartesian_idxs_to_check_mapped = map_cartesian_idx_into_valid_range.(I .+ cartesian_idxs_to_check)
        adjacent_vals = data[cartesian_idxs_to_check_mapped]

        #First check if any of the adjacent values should be in the same mask 
        adjacent_idxs_with_same_value = cartesian_idxs_to_check_mapped[adjacent_vals .== current_val]

        #Check if any of the adjacent values with the same value are already in a mask
        adjacent_and_matching_and_in_mask = visited[adjacent_idxs_with_same_value] .!= 0

        adjacent_point_mask_vals = visited[adjacent_idxs_with_same_value][adjacent_and_matching_and_in_mask]

        unique_masks_adjacent = sort!(unique(adjacent_point_mask_vals))

        if isempty(adjacent_point_mask_vals)
            #If none of the matching adjacent points are in a mask, we give them our current point's mask. If the current point is not in a mask, we create a new mask for it.
            if visited[I] == 0
                current_n_masks += 1
                # Ensure the graph has enough vertices for this mask
                while nv(mask_graph) < current_n_masks
                    add_vertex!(mask_graph)
                end
                visited[I] = current_n_masks
                visited[adjacent_idxs_with_same_value] .= current_n_masks
            else
                #If the current point is already in a mask, we set all adjacent points with the same value to be in that mask
                visited[adjacent_idxs_with_same_value] .= visited[I]
            end
        #If the adjacent points are only in one mask, just set all matching points including the current one to be in that mask
        elseif length(unique_masks_adjacent) == 1
            visited[I] = only(unique_masks_adjacent)
            visited[adjacent_idxs_with_same_value] .= only(unique_masks_adjacent)
        else
            #If the adjacent points are in multiple masks, add edges to the graph to indicate they should be merged
            # First ensure all mask vertices exist in the graph
            max_mask = maximum(unique_masks_adjacent)
            while nv(mask_graph) < max_mask
                add_vertex!(mask_graph)
            end
            
            # Add edges between all pairs of masks that need to be merged
            for i in 1:(length(unique_masks_adjacent)-1)
                for j in (i+1):length(unique_masks_adjacent)
                    add_edge!(mask_graph, unique_masks_adjacent[i], unique_masks_adjacent[j])
                end
            end
            
            # Assign current point to first mask (actual merging happens later)
            mask_to_merge_into = first(unique_masks_adjacent)
            visited[I] = mask_to_merge_into
            visited[adjacent_idxs_with_same_value] .= mask_to_merge_into
        end 
    end

    # Now use connected components to determine which masks should be merged
    components = connected_components(mask_graph)
    
    # Create a remapping dictionary: each mask value maps to the minimum value in its component
    remapping_dict = Dict{Int, Int}()
    for component in components
        if !isempty(component)
            representative = minimum(component)
            for mask_val in component
                remapping_dict[mask_val] = representative
            end
        end
    end
    
    # Apply the remapping to visited array
    for I in CartesianIndices(visited)
        if visited[I] != 0
            visited[I] = get(remapping_dict, visited[I], visited[I])
        end
    end
    
    # Relabel to sequential integers starting from 1
    unique_mask_vals = sort!(unique(visited[visited .!= 0]))
    sequential_dict = Dictionary(unique_mask_vals, collect(1:length(unique_mask_vals)))
    for I in CartesianIndices(visited)
        if visited[I] != 0
            visited[I] = sequential_dict[visited[I]]
        end
    end

    #Now return both the raw array of mask values, as well as a vector of the masks themselves, and then another vector of the value of the criteria func for each mask
    masks = groupfind(visited)
    mask_criterion_vals = [data[first(mask)] for mask in masks]

    return visited, masks, mask_criterion_vals
end

#Now write a brief test dataset
#=n_lons = 20
n_lats = 20
data = randn(n_lons, n_lats)
lons = LinRange(0, 360, n_lons)
lats = LinRange(-90, 90, n_lats)
filter_size = 3
high = 0.5/filter_size
low = -high

#Now to smear the points together a bit, replace each point with the weighted mean of itself given weight 0.125 and the weighted mean of all surrounding points with total weight 0.875. To calculate the weighted mean, use 1/r^2 weighting of all points where r is the haversine distance.
using ImageFiltering
data = imfilter(data, Kernel.gaussian((filter_size, filter_size)), "circular")

using Plots, GraphPlot, Cairo, Fontconfig

display(heatmap(data))

criteria_func = standard_criteria_func(low, high)
mask_idxs, masks, mask_criterion_vals = tile_latlon_grid(data, criteria_func, standard_adjacency_arr)

display(heatmap(mask_idxs))
display(heatmap(criteria_func.(data)))=#

