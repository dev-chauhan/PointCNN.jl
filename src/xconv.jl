using Flux

struct Xconv
    n_p::Int
    k::Int
    mlp1
    mlp2_dense
    mlp2_conv1
    mlp2_conv2
    conv
end

function Xconv(c1::Int,
    c2::Int,
    dim::Int,
    k::Int,
    n_p::Int,
    c_del::Int,
    dm::Int)
    Xconv(n_p,
        k,
        Chain(Conv((1, 1), dim=>c_del, elu), Conv((1, 1), c_del=>c_del, elu)),
        Conv((1, k), dim=>k*k, elu),
        DepthwiseConv((1, k), k=>k*k, elu),
        DepthwiseConv((1, k), k=>k*k),
#                     Chain(Conv((3, k), ) ),
        Chain(Conv((1, k), (c1+c_del)=>dm*(c1+c_del)), Conv((1, 1), dm*(c1+c_del)=>c2))
    )
end
function (m::Xconv)(p, P, F=nothing)
    # P : (n_p, k, 3, batch)
    # p : (n_p, 1, 3, batch)
    p_shifted = P .- p

    F_del = m.mlp1(p_shifted) # F_del : (n_p, k, c_del, batch)
    # println("F_del: ", size(F_del))
    F_star = F != nothing ? cat(F_del, F, dims=3) : F_del # F_star : (n_p, k, c_del + c1, batch)
    # println("F_star: ", size(F_star))
    # X = m.mlp2_conv2(reshape(m.mlp2_conv1(reshape(m.mlp2_dense(p_shifted), (m.n_p, m.k, m.k, :))), (m.n_p, m.k, m.k, :)))
    # X = reshape(X, (size(X)[1], m.k, m.k, :))
    X = reshape(
        m.mlp2_conv2(reshape(
            m.mlp2_conv1(reshape(
                m.mlp2_dense(p_shifted),
                (m.n_p, m.k, m.k, :)
            )),
            (m.n_p, m.k, m.k, :)
        )),
        (m.n_p, m.k, m.k, :)
    )
    # println("X: ", size(X))
    F_X = cat([cat([reshape(X[i,:,:,j]*F_star[i,:,:,j], (1, size(X)[2], size(F_star)[3])) for j in 1:size(X)[end]]..., dims=4) for i in 1:size(X)[1]]..., dims=1);
    # println("F_X: ", size(F_X))
    F_p = m.conv(F_X)
    # println("F_p: ", size(F_p))
    return F_p
end
    
Flux.@functor Xconv