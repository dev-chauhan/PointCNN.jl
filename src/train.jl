using Flux

cd(@__DIR__)

include("pointcnn_cls.jl")

model = Classifier(40)|>gpu

include("data.jl")

file1 = h5open(train_files[1], "r")

batch_size = 32
train_data_len = 2048

# batches : Array of minibatches of size (2048, 1, 3, batch_size) = (n_p, 1, dimentions, batch_size)
# y : array of (40, batch_size) multidimentional arrays

batches = [
    cat(
        [reshape(transpose(file1["data"][:,:,i] |> gpu), (2048, 1, 3)) for i=j:j+batch_size-1]...,
        dims=4
    ) for j=1:batch_size:Int(floor(train_data_len / batch_size))*batch_size
];

y = [
    Flux.onehotbatch(file1["label"][1,j:j+batch_size-1])|>gpu for j=1:batch_size:Int(floor(train_data_len / batch_size))*batch_size
]

# optimizer
opt = Flux.ADAM()

function loss(p, y)
    logpred = model(p)
    Flux.logitcrossentropy(logpred, y)
end

Flux.train!(loss, Flux.params(model), zip(batches, y), opt)