using HDF5

data_dir = "../data/modelnet40_ply_hdf5_2048"

cd(@__DIR__)
test_files = readlines("../data/modelnet40_ply_hdf5_2048/test_files.txt")
train_files = readlines("../data/modelnet40_ply_hdf5_2048/train_files.txt")
