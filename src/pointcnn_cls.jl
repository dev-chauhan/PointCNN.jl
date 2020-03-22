using Flux

cd(@__DIR__)

include("pointcnn.jl")

struct Classifier
    cls::Int
    pcnn1::PointCNN
    pcnn2::PointCNN
    pcnn3::PointCNN
    pcnn4::PointCNN
    fc1::Dense
    fc2::Dense
end

Classifier(cls::Int) = Classifier(
    cls,
    PointCNN(1024, 0, 32, 16, 8),
    PointCNN(384, 32, 64, 8, 12;dilation=2),
    PointCNN(128, 64, 128, 16, 16;dilation=2),
    PointCNN(128, 128, 256, 32, 16;dilation=3),
    Dense(256, 128),
    Dense(128, cls)
    )

function (m::Classifier)(pts)
    # pts : (n_p, 1, 3, batch)
    p1, f1 = m.pcnn1(pts, nothing)
    p2, f2 = m.pcnn2(p1, f1)
    p3, f3 = m.pcnn3(p2, f2)
    p4, f4 = m.pcnn4(p3, f3)
    f = reshape(f4, (size(f4)[1], size(f4)[3], size(f4)[end]))
    fp = permutedims(f, [2, 1, 3])
    logpred = reshape(sum(cat(
        [m.fc2(m.fc1(fp[:,:,i])) for i in 1:size(fp)[3]]..., dims=3
    ), dims=2), (m.cls, :))

    return logpred
end

Flux.@functor Classifier