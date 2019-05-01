module RandomForestRegression
using Statistics
using StatsBase

mutable struct Node
    feature    :: Int64 
    comp_value :: Float64
    data_idxs  :: Vector{Int64}
    mean       :: Float64
    left_child :: Union{Nothing, Node}
    right_child:: Union{Nothing, Node}
    
    Node() = new()
end

mutable struct Tree
    root                :: Node
    Tree() = new()
end

mutable struct SplitObj
    feature     :: Int64
    min         :: Float64
    max         :: Float64
    gain        :: Float64
    split_val   :: Float64
    left_idxs   :: Vector{Int64}
    right_idxs  :: Vector{Int64}
    SplitObj() = new() 
end

function new_split_obj(feature, min_val, max_val,  gain, split_val, left_idxs, right_idxs)
    s = SplitObj()
    s.feature = feature
    s.min = min_val
    s.max = max_val
    s.gain = gain
    s.split_val = split_val
    s.left_idxs = left_idxs
    s.right_idxs = right_idxs  
    return s
end

"""
    cstd(a)

Custom stddev as std([1]) is NaN but should be 0 for most
"""
function cstd(a)
    if length(a) == 1
        return 0.0
    end
    return std(a)
end

"""
    get_best_split(feature_matrix, left_ys, right_ys, left_idxs, right_idxs, feature_idx, train_ys; run_no=1)

Get the best split object by choosing a random feature and then gets the best split. 
If no split is found repeat this for 5 times
Return the split object
"""
function get_best_split(feature_matrix, left_ys, right_ys, left_idxs, right_idxs, feature_idx, train_ys; run_no=1)
    nfeatures = length(feature_idx)
    len_train_xs = size(feature_matrix)[2]
    t = time()
    rand_feature = rand(1:nfeatures)
    fct_vals = feature_matrix[rand_feature,:]
    min_val = minimum(fct_vals)
    max_val = maximum(fct_vals)

    best_gain = Inf
    best_split_val = -Inf
    if min_val != max_val
        best_left_idxs = nothing
        best_right_idxs = nothing
        right_end = 0
        left_end = 0
        # checking 200 different split values and choose the best
        # by minimzing the sum of the standard deviation of both children
        for split_val in LinRange(min_val, max_val, 20) 
            right_end = 0
            left_end = 0
            for data_idx = 1:len_train_xs
                if fct_vals[data_idx] < split_val
                    left_end += 1
                    left_idxs[left_end] = data_idx
                    left_ys[left_end] = train_ys[data_idx]
                else
                    right_end += 1
                    right_idxs[right_end] = data_idx
                    right_ys[right_end] = train_ys[data_idx]
                end
            end
            
            if left_end != 0 && right_end != 0
                @views left_ys_sub = left_ys[1:left_end]
                @views right_ys_sub = right_ys[1:right_end]
                gain = cstd(left_ys_sub)+cstd(right_ys_sub)
                if  gain < best_gain
                    best_gain = gain
                    best_split_val = split_val
                    # needs to be copied
                    best_left_idxs = left_idxs[1:left_end]
                    best_right_idxs = right_idxs[1:right_end]
                end
            end
        end
    end

    std_before = cstd(train_ys)
    max_split_c = 15

    split_obj = SplitObj()
    split_obj.feature = feature_idx[rand_feature]
    split_obj.split_val = best_split_val
    split_obj.min = min_val
    split_obj.max = max_val
    if !isinf(best_split_val)
        split_obj = new_split_obj(feature_idx[rand_feature], min_val, max_val, best_gain, best_split_val, best_left_idxs, best_right_idxs)
        return split_obj
    end

    if isinf(best_split_val) && run_no < max_split_c
        split_obj = get_best_split(feature_matrix, left_ys, right_ys, left_idxs, right_idxs, feature_idx, train_ys; run_no=run_no+1)
    end

    return split_obj
end

function queue_compute_nodes!(queue::Vector{Node}, left_node::Node, right_node::Node, train_ys)
    if length(left_node.data_idxs) > 1 #&& cstd(train_ys[left_node.data_idxs]) > 10000
        push!(queue, left_node)
    end
    if length(right_node.data_idxs) > 1 #&& cstd(train_ys[right_node.data_idxs]) > 10000
        push!(queue, right_node)
    end
end


function compute_node!(tree::Tree, node::Node, feature_matrix, left_ys, right_ys, left_idxs, right_idxs, feature_idx, train_ys)
    # last three params if split needs to be restarted
    @views split_obj = get_best_split(feature_matrix[:,node.data_idxs], left_ys, right_ys, left_idxs, right_idxs, feature_idx, train_ys[node.data_idxs])
   
    node.feature = split_obj.feature
    node.comp_value = split_obj.split_val
    node.mean = mean(train_ys[node.data_idxs])
    std_of_train_node_idx = cstd(train_ys[node.data_idxs])

    node.left_child = nothing
    node.right_child = nothing
    if !isinf(split_obj.split_val)
        left_node = Node()
        right_node = Node()
        left_node.data_idxs = node.data_idxs[split_obj.left_idxs]
        right_node.data_idxs = node.data_idxs[split_obj.right_idxs]
        left_node.mean = mean(train_ys[node.data_idxs[split_obj.left_idxs]])
        right_node.mean = mean(train_ys[node.data_idxs[split_obj.right_idxs]])
        left_node.feature = -1
        right_node.feature = -1
        left_node.left_child = nothing
        left_node.right_child = nothing
        right_node.left_child = nothing
        right_node.right_child = nothing
        left_node.comp_value = 0.0
        right_node.comp_value = 0.0

        node.left_child = left_node
        node.right_child = right_node
        return left_node, right_node
    end
    return nothing, nothing
end

function create_root_node!(tree::Tree, feature_matrix, left_ys, right_ys, left_idxs, right_idxs, feature_idx, train_ys)
    
    split_obj = get_best_split(feature_matrix, left_ys, right_ys, left_idxs, right_idxs, feature_idx, train_ys)
    queue = Vector{Node}()
    if !isinf(split_obj.split_val)
        left_node = Node()
        left_node.feature = -1
        left_node.left_child = nothing
        left_node.right_child = nothing
        left_node.mean = mean(train_ys)
        left_node.comp_value = NaN
        left_node.data_idxs = []

        right_node = Node()
        right_node.feature = -1
        right_node.left_child = nothing
        right_node.right_child = nothing
        right_node.mean = mean(train_ys)
        right_node.comp_value = NaN
        right_node.data_idxs = []

        left_node.data_idxs = split_obj.left_idxs
        right_node.data_idxs = split_obj.right_idxs
        
        node = Node()
        node.feature = split_obj.feature
        node.comp_value = split_obj.split_val
        node.data_idxs = collect(1:size(feature_matrix)[2])
        node.mean = mean(train_ys)
        node.left_child = left_node
        node.right_child = right_node

        queue_compute_nodes!(queue, left_node, right_node, train_ys)
        return node, queue
    end
    node = Node()
    node.left_child = nothing
    node.right_child = nothing
    return node, queue
end

function create_random_tree(glob_feature_matrix, feature_idx, cols, train_ys, total_nfeatures)
    tree = Tree() 
    @views feature_matrix = glob_feature_matrix[feature_idx, cols]
    left_ys = zeros(Float64, length(cols))
    right_ys = zeros(Float64, length(cols))
    left_idxs = zeros(Int64, length(cols))
    right_idxs = zeros(Int64, length(cols))
    root, queue = create_root_node!(tree, feature_matrix, left_ys, right_ys, left_idxs, right_idxs, feature_idx, train_ys[cols])
    s_time = time()
    while length(queue) > 0
        node = popfirst!(queue)
        left_node, right_node = compute_node!(tree, node, feature_matrix, left_ys, right_ys, left_idxs, right_idxs, feature_idx, train_ys[cols])
        if left_node != nothing
            queue_compute_nodes!(queue, left_node, right_node, train_ys[cols])
        end
    end
    println("time for creating one tree: ", time()-s_time)
    tree.root = root
    glob_feature_matrix = nothing

    return tree
end

function create_random_forest(feature_matrix, train_ys, ntrees)
    forest = Vector{Tree}()
    tc = 1
    total_nfeatures = size(feature_matrix)[1]
    nfeatures_per_tree = 8
    ncols_per_tree = convert(Int64,floor(length(train_ys)/2))

    while tc < ntrees
        start_time = time()
                        
        feature_idx = sample(1:total_nfeatures, nfeatures_per_tree; replace=false)
        cols = sample(1:length(train_ys), ncols_per_tree; replace=false)

        tree = create_random_tree(feature_matrix, feature_idx, cols, train_ys, total_nfeatures)
        if tree.root.left_child != nothing
            push!(forest,tree)
            tc += 1
        else
            println("Couldn't create tree!!")
        end
        flush(stdout)
    end
    return forest
end

function predict_single_tree(tree::Tree, feature_matrix; check_range=false)
    pred_ys = zeros(size(feature_matrix)[2])
    for r =1:length(pred_ys)
        node = tree.root
        early_break = false
        while node.feature != -1 && node.left_child != nothing
            split_value = node.comp_value
            @views val = feature_matrix[node.feature,r]
            if val < split_value
                node = node.left_child                  
            else
                node = node.right_child
            end          
        end
        pred_ys[r] = node.mean
    end
    return pred_ys
end

function predict_forest(forest::Vector{Tree}, feature_matrix; default=1)
    predictions = Vector{Vector{Float64}}()
    tc = 0
    function get_next_tree()
        tc += 1
        if tc > length(forest)
            return nothing, true, tc
        end
        return forest[tc], false, tc
    end

    while true
        tree, dobreak, ptc = get_next_tree()
        if dobreak
            break
        end
        start_time = time()
        pred = predict_single_tree(tree, feature_matrix)
        push!(predictions, pred)
        if ptc % 20 == 0
            println("Finished pred tree #", ptc)
            flush(stdout)
        end
    end
  
    result = zeros(Float64, size(feature_matrix)[2])
    divider = zeros(Float64, size(feature_matrix)[2])
    for p_idx=1:length(predictions)
        tp = predictions[p_idx]
        for c=1:size(feature_matrix)[2]
            if !isnan(tp[c])
                result[c] += tp[c]
                divider[c] += 1
            end
        end
    end
    result ./= divider

    for idx in findall(i->isnan(i), result)
        result[idx] = default
        println("HAD TO SET DEFAULT")
    end
    return result
end

end # end module
