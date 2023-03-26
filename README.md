# 4DProximal 
4D Post-stack seismic inversion with Proximal solvers. Examples on synthetic Hess model and field Sleipner dataset.


## Project structure 
This repository is organized as follows:

- :open_file_folder: **data**: for
- :open_file_folder: **prox4d**: ALgorithm to compute joint inversion-segmentation of 4D post-stack seismic data.


## Notebooks 
The following notebooks are provided:

- :orange_book: ``4D_Inversion_approaches.ipynb``: In this notebook we compare different approaches for 4D post-stack seismic Inversion. These approaches involve using different regularization terms (i.e. L2-Reg or TV-Reg or using segmentation terms) with different arguments (i.e. $m_1$, $m_2$ or $m_2-m_1$). ...;
- :orange_book: ``Creating_4D_Synthetic_seismic_data.ipynb``: In this notebook the original Hess model is modified to create two new models: baseline and monitor

- :orange_book: ``Experiments_TV.ipynb``: In this notebook we compare the effect of implementing isotropic and anisotropic TV in 4D post-stack seismic Inversion

## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

