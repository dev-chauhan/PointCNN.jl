using Flux
using NearestNeighbors
using StatsBase
using Zygote: @nograd

cd(@__DIR__)

include("xconv.jl")

struct PointCNN
    n_p::Int
    k::Int
    dilation::Int
    xconv::Xconv
end

function get_knn(P, n_p, dilation, k)
    p_idx = sample(1:size(P)[1], n_p; replace=false)
    ball_tree = [BallTree(transpose(P[:, 1, :, i])) for i in 1:size(P)[end]]
    return P[p_idx, :, :, :], [knn(ball_tree[i], transpose(P[p_idx, 1, :, i]), dilation * k)[1] for i in 1:size(P)[end]]
end

@nograd get_knn

PointCNN(n_p, cin, cout, cmid, k; dilation=1) = PointCNN(n_p, k, dilation, Xconv(cin, cout, 3, k, n_p, cmid, Int(floor(cout/(cin + cmid)))))

function (m::PointCNN)(P, F)
    p2, idxs = get_knn(P, m.n_p, m.dilation, m.k)
    p1 = cat([cat([reshape(P[idxs[i][j][1:m.dilation:(m.k * m.dilation)],1,:,i], (1, m.k, :)) for j in 1:m.n_p]..., dims=1) for i in 1:size(P)[end]]..., dims=4)
    if F != nothing
        F1 = cat([cat([reshape(F[idxs[i][j][1:m.dilation:(m.k * m.dilation)],1,:,i], (1, m.k, :)) for j in 1:m.n_p]..., dims=1) for i in 1:size(P)[end]]..., dims=4)
    else
        F1 = F
    end
    return p2, m.xconv(p2, p1, F1)
end

Flux.@functor PointCNN