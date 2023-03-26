# 4DProximal

4D Post-stack seismic inversion with Proximal solvers. Examples on synthetic Hess model and field Sleipner dataset.

## Project structure

This repository is organized as follows:

- :open_file_folder: **data**: for
- :open_file_folder: **prox4d**: ALgorithm to compute joint inversion-segmentation of 4D post-stack seismic data.

## Notebooks

The following notebooks are provided:

- :orange_book: ``4D_Inversion_approaches.ipynb``: In this notebook we compare different approaches for 4D post-stack
  seismic Inversion. These approaches involve using different regularization terms (i.e. L2-Reg or TV-Reg or using
  segmentation terms) with different arguments (i.e. $m_1$, $m_2$ or $m_2-m_1$). ...;
- :orange_book: ``Creating_4D_Synthetic_seismic_data.ipynb``: In this notebook the original Hess model is modified to
  create two new models: baseline and monitor

- :orange_book: ``Experiments_TV.ipynb``: In this notebook we compare the effect of implementing isotropic and
  anisotropic TV in 4D post-stack seismic Inversion

## Getting started :space_invader: :robot:

To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:

```
./install_env.sh
```

It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. After that you can
simply install your package:

```
pip install -e .
```

Remember to always activate the environment by typing:

```
conda activate prox4d
```

Similarly, you can use `environment_gpu.yml` and `install_env_gpu.sh` to create an environment that runs the 4D JIS 
algorithm in GPU. 


**Disclaimer:** For computer time, this research used the resources of the Supercomputing Laboratory at KAUST in Thuwal,
Saudi Arabia. All experiments have been carried on an AMD EPYC 7713P 64-Core Processor equipped with a single NVIDIA 
TESLA A100 . Different environment configurations may be required for different combinations of workstation and GPU.



