# PointCNN.jl
Implementation of PointCNN for classification task of modelnet40 dataset in julia

## Data

run following command in shell to get modelnet40 dataset

```
$ wget https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
$ mkdir data
$ unzip modelnet40_ply_hdf5_2048.zip
$ mv modelnet40_ply_hdf5_2048 ./data
$ rm modelnet40_ply_hdf5_2048.zip
```

## train classifier

To train classifier for modelnet40 dataset

```
$ julia --project=. src/train.jl
```
