# Covariant Particle Transformer

This repository contains the code and pretrained models for ["A Holistic Approach to Predicting Top Quark Kinematic Properties with the Covariant Particle Transformer"](https://arxiv.org/pdf/2203.05687.pdf).

## Getting Started
Follow these instructions to set up the environment, download pretrained models, and download datasets used in the paper:

1. **Set up conda environment and install dependencies**:
```
sh setup.sh
conda activate cpt
```
2. **(Optional) Download pretrained $ttH$ and $tttt$ models**:

The script also downloads model predictions on all datasets used in the paper.
```
sh download_pretrained.sh
```

3. **(Optional) Download datasets**:
Note the $ttH$ and $tttt$ dataset is ~30GB in size each, while the other datasets are smaller. You can comment out the lines in `download_datasets.sh` to download only the datasets you need.

```
sh download_datasets.sh
```
The current implementation of our datasets are not well-optimized for speed or size. We encourage making your own datasets and modifying the data processing code in ```dataset.py``` and ```utils.py``` to suit your needs.

## Usage
To use the pretrained models, refer to the Jupyter notebooks ```pretrained_ttH.ipynb``` or ```pretrained_tttt.ipynb```.
To train a new model, use the ```train.ipynb``` Jupyter notebook.

## Citation
Please cite our paper as:
```
@article{qiu2022holistic,
  title={A Holistic Approach to Predicting Top Quark Kinematic Properties with the Covariant Particle Transformer},
  author={Qiu, Shikai and Han, Shuo and Ju, Xiangyang and Nachman, Benjamin and Wang, Haichen},
  journal={arXiv preprint arXiv:2203.05687},
  year={2022}
}
```