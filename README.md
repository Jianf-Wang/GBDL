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

Baidu Disk: <a href="https://pan.baidu.com/s/1yOGMBZOzlZ5UJ2EGh9y8CQ">download</a>  (code: zr4s)   

Google Drive: <a href="https://drive.google.com/drive/folders/1JprKNLCGQtaCXuVziNHz7HyOMbqzsXrM?usp=sharing">download</a>  

Note that they are saved as 'png', which are extracted from their original datasets without any further preprocessing. 

Plus, please prepare the training and the testing text files. Each file has the following format:

```
/Path/to/the/image/files /Path/to/the/label/map/files
...
...
```
We provide two example files, i.e., 'train_AtriaSeg.txt' and 'test_AtriaSeg.txt'


Training
-----
After the preperation, you can start to train your model. We provide an example file "train_AtriaSeg_16.sh", please run with:

```
sh train_AtriaSeg_16.sh
```
After the training, the latest model will be saved, which is used for testing.

Testing
-----

We provide an example file "test_AtriaSeg_16.sh", please run with:

```
sh test_AtriaSeg_16.sh
```

Citation
-----------------

  ```
  @inproceedings{wang2022rethinking,
  title={Rethinking Bayesian Deep Learning Methods for Semi-Supervised Volumetric Medical Image Segmentation},
  author={Wang, Jianfeng and Lukasiewicz, Thomas},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={182--190},
  year={2022}
}
  ```
