## Rethinking Bayesian Deep Learning Methods for Semi-Supervised Volumetric Medical Image Segmentation (CVPR 2022)

A Pytorch implementation of our CVPR 2022 paper "Rethinking Bayesian Deep Learning Methods for Semi-Supervised Volumetric Medical Image Segmentation".



Build
-----

please run with the following command:

```
conda env create -f GBDL_env.yaml
conda activate GBDL
```


Preparation
-----
The datasets can be downloaded from their official sites. We also provide them here:

Baidu Disk: <a href="https://pan.baidu.com/s/16tpuXNGwm5ssJagP_yHAtw">download</a>  (code: zr4s)   

Note that they are saved as 'png', which are extracted from their original datasets without any further preprocessing. 

Plus, please prepare the training and the testing text files. Each file has the following format:

```
/Path/to/the/image/files /Path/to/the/label/map/files
...
...
```
We provide two example files, i.e., 'train_AtriaSeg.txt' and 'test_AtriaSeg.txt'
